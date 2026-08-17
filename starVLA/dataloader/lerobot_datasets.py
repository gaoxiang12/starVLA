# Copyright 2025 NVIDIA Corp. and affiliates. All rights reserved.
# Modified by [Fangjing Wang/ SUST University] in [2025]. 
# Modification: [return raw data and suport multi-dataset mixture].
# Modified by [Jinhui YE/ HKUST University] in [2025]. 
# Modification: [suport topdowm processing, suport param from config].

import logging
import math
from pathlib import Path
from typing import Iterator, Mapping, Sequence
from omegaconf import OmegaConf
import numpy as np
from torch.utils.data import Sampler

from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset, LeRobotMixtureDataset
from starVLA.dataloader.gr00t_lerobot.registry import (
    ROBOT_TYPE_CONFIG_MAP,
    DATASET_NAMED_MIXTURES,
    EmbodimentTag,
)

logger = logging.getLogger(__name__)

def collate_fn(batch):
    return batch


class EmbodimentBatchSampler(Sampler[list[tuple[int, int]]]):
    """Yield batches whose samples all share one embodiment tag.

    ``LeRobotMixtureDataset`` normally chooses a dataset independently for each
    item, which cannot be collated when embodiments have different action/state
    shapes.  This sampler first chooses an embodiment for the complete batch,
    then chooses datasets within that embodiment using the mixture weights.
    The yielded tuple is interpreted by ``LeRobotMixtureDataset.__getitem__`` as
    ``(forced_dataset_index, deterministic_sample_index)``.
    """

    def __init__(
        self,
        dataset: LeRobotMixtureDataset,
        batch_size: int,
        *,
        embodiment_weights: Mapping[str, float] | None = None,
        drop_last: bool = True,
        seed: int = 42,
    ) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0

        groups: dict[str, list[int]] = {}
        for index, child in enumerate(dataset.datasets):
            groups.setdefault(child.tag, []).append(index)
        self.embodiment_tags = tuple(sorted(groups))
        self.dataset_indices_by_tag = groups

        if embodiment_weights is None:
            raw_tag_weights = np.asarray(
                [
                    dataset.dataset_sampling_weights[groups[tag]].sum()
                    for tag in self.embodiment_tags
                ],
                dtype=np.float64,
            )
        else:
            unknown = set(embodiment_weights) - set(self.embodiment_tags)
            if unknown:
                raise ValueError(
                    f"embodiment_sampling_weights contains unknown tags: {sorted(unknown)}"
                )
            raw_tag_weights = np.asarray(
                [float(embodiment_weights.get(tag, 0.0)) for tag in self.embodiment_tags],
                dtype=np.float64,
            )
        if np.any(raw_tag_weights < 0) or raw_tag_weights.sum() <= 0:
            raise ValueError("embodiment sampling weights must be non-negative and non-zero")
        self.embodiment_weights = raw_tag_weights / raw_tag_weights.sum()

        self.dataset_weights_by_tag: dict[str, np.ndarray] = {}
        for tag, indices in groups.items():
            weights = np.asarray(
                dataset.dataset_sampling_weights[indices], dtype=np.float64
            )
            self.dataset_weights_by_tag[tag] = weights / weights.sum()

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        return math.ceil(len(self.dataset) / self.batch_size)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self.dataset.set_epoch(self.epoch)

    def __iter__(self) -> Iterator[list[tuple[int, int]]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        total = len(self.dataset)
        sample_cursor = 0
        for batch_index in range(len(self)):
            current_batch_size = min(self.batch_size, total - sample_cursor)
            if current_batch_size <= 0:
                break
            tag = str(rng.choice(self.embodiment_tags, p=self.embodiment_weights))
            dataset_indices = self.dataset_indices_by_tag[tag]
            chosen = rng.choice(
                dataset_indices,
                size=current_batch_size,
                p=self.dataset_weights_by_tag[tag],
            )
            yield [
                (int(dataset_index), sample_cursor + offset)
                for offset, dataset_index in enumerate(chosen)
            ]
            sample_cursor += current_batch_size

def make_LeRobotSingleDataset(
    data_root_dir: Path | str,
    data_name: str,
    robot_type: str,
    delete_pause_frame: bool = False,
    data_cfg: dict | None = None,
) -> LeRobotSingleDataset:
    """
    Make a LeRobotSingleDataset object.

    :param data_root_dir: The root directory of the dataset.
    :param data_name: The name of the dataset.
    :param robot_type: The robot type config to use.
    :param crop_obs_camera: Whether to crop the observation camera images.
    :return: A LeRobotSingleDataset object.
    """
    
    data_config = ROBOT_TYPE_CONFIG_MAP[robot_type]
    modality_config = data_config.modality_config()
    transforms = data_config.transform()
    dataset_path = data_root_dir / data_name
    embodiment_tag = getattr(data_config, "embodiment_tag", None)
    if embodiment_tag is None:
        print(f"Warning: DataConfig for robot_type={robot_type!r} has no embodiment_tag, using {EmbodimentTag.NEW_EMBODIMENT} as default")
        embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
    
    video_backend = data_cfg.get("video_backend", "decord") if data_cfg else "torchvision_av"

    # Opt-in factory hook: a DataConfig may define ``make_dataset(dataset_name=..., **ds_kwargs)``
    # to swap in a custom dataset class (e.g. with per-task filtering / chunk stride).
    # When absent, fall through to the default LeRobotSingleDataset construction below.
    if hasattr(data_config, "make_dataset"):
        dataset = data_config.make_dataset(
            dataset_path=dataset_path,
            modality_configs=modality_config,
            transforms=transforms,
            embodiment_tag=embodiment_tag,
            video_backend=video_backend,
            delete_pause_frame=delete_pause_frame,
            data_cfg=data_cfg,
            dataset_name=data_name,
        )

    else:
        dataset = LeRobotSingleDataset(
            dataset_path=dataset_path,
            modality_configs=modality_config,
            transforms=transforms,
            embodiment_tag=embodiment_tag,
            video_backend=video_backend, # decord is more efficiency | torchvision_av for video.av1
            delete_pause_frame=delete_pause_frame,
            data_cfg=data_cfg,
        )

    # Keep routing/schema information next to each concrete dataset.  This is
    # intentionally metadata rather than tensor padding: model heads are chosen
    # by semantic action space, not merely by its numeric width.
    dataset.robot_type = robot_type
    dataset.action_spec_id = getattr(data_config, "action_spec_id", robot_type)
    dataset.state_spec_id = getattr(data_config, "state_spec_id", robot_type)
    dataset.control_hz = getattr(data_config, "control_hz", None)
    dataset.future_time_offsets_s = getattr(
        data_config, "future_time_offsets_s", None
    )
    return dataset

def get_vla_dataset(
    data_cfg: dict,
    mode: str = "train",
    balance_dataset_weights: bool = False,
    balance_trajectory_weights: bool = False,
    seed: int = 42,
    **kwargs: dict,
) -> LeRobotMixtureDataset:
    """
    Get a LeRobotMixtureDataset object.
    """
    data_root_dir = data_cfg.data_root_dir
    data_mix = data_cfg.data_mix
    delete_pause_frame = data_cfg.get("delete_pause_frame", False)
    mixture_spec = DATASET_NAMED_MIXTURES[data_mix]
    logger.info(f"[dataloader] Using mixture '{data_mix}': {[(d, w, r) for d, w, r in mixture_spec]}")
    included_datasets, filtered_mixture_spec = set(), []
    for d_name, d_weight, robot_type in mixture_spec:  
        dataset_key = (d_name, robot_type)  
        if dataset_key in included_datasets:
            print(f"Skipping Duplicate Dataset: `{(d_name, d_weight, robot_type)}`")
            continue

        included_datasets.add(dataset_key)
        filtered_mixture_spec.append((d_name, d_weight, robot_type))

    dataset_mixture = []
    for d_name, d_weight, robot_type in filtered_mixture_spec:
        dataset_mixture.append((make_LeRobotSingleDataset(Path(data_root_dir), d_name, robot_type, delete_pause_frame=delete_pause_frame, data_cfg=data_cfg), d_weight))

    return LeRobotMixtureDataset(
        dataset_mixture,
        mode=mode,
        balance_dataset_weights=balance_dataset_weights,
        balance_trajectory_weights=balance_trajectory_weights,
        seed=seed,
        data_cfg=data_cfg,
        **kwargs,
    )



if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="./examples/LIBERO/train_files/bar/starvla_cotrain_libero.yaml", help="Path to YAML config")
    parser.add_argument("--data_mix", type=str, default=None, help="Override data_mix from config")
    parser.add_argument("--data_root_dir", type=str, default=None, help="Override data_root_dir from config")
    args = parser.parse_args()

    if os.getenv("DEBUGPY_ENABLE", "0") == "1":
        import debugpy
        debugpy.listen(("0.0.0.0", 10092))
        print("Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    cfg = OmegaConf.load(args.config_yaml)
    vla_dataset_cfg = cfg.datasets.vla_data
    vla_dataset_cfg.data_root_dir = Path(vla_dataset_cfg.data_root_dir)
    if args.data_mix is not None:
        vla_dataset_cfg.data_mix = args.data_mix
    if args.data_root_dir is not None:
        vla_dataset_cfg.data_root_dir = Path(args.data_root_dir)

    dataset = get_vla_dataset(data_cfg=vla_dataset_cfg)
    from torch.utils.data import DataLoader
    train_dataloader = DataLoader(
        dataset,
        batch_size=2,
        num_workers=1, # For Debug
        collate_fn=collate_fn,
    )

    cfg.output_dir = "./results/debug"
    output_dir = Path(cfg.output_dir)
    dataset.save_dataset_statistics(output_dir / "dataset_statistics.json")

    from tqdm import tqdm
    count = 0
    for batch in tqdm(train_dataloader, desc="Processing Batches"):
        if count > 3:
            break
        count += 1
        pass
