import math

from accelerate.data_loader import BatchSamplerShard
from torch.utils.data import BatchSampler
import pytest

from scripts.train_gawm_epochs import EpochPermutationSampler, exclude_trajectories


def test_full_epochs_cover_every_frame_on_35_ranks_with_only_tail_padding():
    size, batch, world = 1027, 8, 35
    sampler = EpochPermutationSampler(range(size), seed=42)
    previous = None
    for epoch in range(12):
        sampler.set_epoch(epoch)
        combined = []
        for rank in range(world):
            sharded = BatchSamplerShard(
                BatchSampler(sampler, batch_size=batch, drop_last=False),
                num_processes=world, process_index=rank,
            )
            rows = list(sharded)
            assert len(rows) == math.ceil(size / (world * batch))
            assert all(len(row) == batch for row in rows)
            combined.extend(i for row in rows for i in row)
        assert set(combined) == set(range(size))
        assert len(combined) - size == math.ceil(size / (world * batch)) * world * batch - size
        assert combined != previous
        previous = combined


def test_sampler_reproduces_epoch_without_advancing_rng():
    first = EpochPermutationSampler(range(101), seed=42)
    second = EpochPermutationSampler(range(101), seed=42)
    first.set_epoch(7)
    second.set_epoch(7)
    assert list(first) == list(second) == list(first)


def test_only_listed_corrupt_episode_is_excluded_without_changing_source():
    class Frames:
        all_steps = [(1, 0), (1, 1), (82, 0), (82, 1), (83, 0)]

        def __len__(self):
            return len(self.all_steps)

        def __getitem__(self, index):
            return self.all_steps[index]

    source = Frames()
    filtered = exclude_trajectories(source, [82])
    assert list(filtered) == [(1, 0), (1, 1), (83, 0)]
    assert len(source) == 5
    assert exclude_trajectories(source, []) is source
    with pytest.raises(ValueError, match="Unknown excluded episodes"):
        exclude_trajectories(source, [99])


def test_resume_skips_batch_indices_without_loading_old_samples():
    from accelerate import PartialState
    from accelerate.data_loader import prepare_data_loader
    from torch.utils.data import DataLoader
    from scripts.train_gawm_epochs import EpochTrainer

    PartialState(cpu=True)

    class CountingDataset:
        def __init__(self):
            self.visited = []

        def __len__(self):
            return 101

        def __getitem__(self, index):
            self.visited.append(index)
            return index

    dataset = CountingDataset()
    sampler = EpochPermutationSampler(dataset, seed=42)
    loader = prepare_data_loader(
        DataLoader(dataset, sampler=sampler, batch_size=2),
        num_processes=3, process_index=1,
    )
    trainer = EpochTrainer.__new__(EpochTrainer)
    trainer.epoch_sampler = sampler
    trainer.vla_train_dataloader = loader
    full = [row.tolist() for row in trainer._start_epoch(7)]
    dataset.visited.clear()
    resumed = [row.tolist() for row in trainer._start_epoch(7, offset=4)]
    assert resumed == full[4:]
    assert dataset.visited == [i for row in resumed for i in row]
