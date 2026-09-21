import json
from types import SimpleNamespace

import pytest

from starVLA.dataloader.training_anchor_bounds import attach_training_anchor_bounds


class Dataset(SimpleNamespace):
    @property
    def all_steps(self):
        return self._all_steps


def setup(tmp_path):
    dataset = Dataset(dataset_path=tmp_path, trajectory_ids=[3, 7], trajectory_lengths=[100, 80],
        delta_indices={'action.joints': list(range(1, 17)), 'video.head': [0, 6, 12]},
        _all_steps=[(ep, i) for ep, n in [(3, 100), (7, 80)] for i in range(n-1)])
    document = dict(format_version=1, dataset=str(tmp_path), maximum_target_offset=16,
                    episodes=[dict(episode_index=3, target_end_exclusive=63),
                              dict(episode_index=7, target_end_exclusive=41)])
    path = tmp_path/'anchors.json'
    path.write_text(json.dumps(document))
    return dataset, document, path


def test_every_action_and_future_target_stays_inside_prefix(tmp_path):
    dataset, _, path = setup(tmp_path)
    attach_training_anchor_bounds(dataset, {'training_anchor_manifest': str(path), 'episode_split': 'train'})
    assert dataset.training_anchor_limits == {3:47, 7:25}
    assert len(dataset.all_steps) == 72
    for episode, anchor in dataset.all_steps:
        assert anchor+16 < dataset.training_target_ends[episode]
        assert anchor+12 < dataset.training_target_ends[episode]
    assert (3, 46) in dataset.all_steps and (3, 47) not in dataset.all_steps


@pytest.mark.parametrize('change', ['missing_episode', 'horizon', 'short_prefix'])
def test_reject_incompatible_manifest(tmp_path, change):
    dataset, document, path = setup(tmp_path)
    if change == 'missing_episode':
        document['episodes'].pop()
    elif change == 'horizon':
        document['maximum_target_offset'] = 8
    else:
        document['episodes'][0]['target_end_exclusive'] = 16
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError):
        attach_training_anchor_bounds(dataset, {'training_anchor_manifest': str(path)})


def test_validation_and_default_sampling_are_unchanged(tmp_path):
    dataset, _, path = setup(tmp_path)
    original = dataset.all_steps.copy()
    attach_training_anchor_bounds(dataset, {})
    attach_training_anchor_bounds(dataset, {'training_anchor_manifest': str(path), 'episode_split': 'validation'})
    assert dataset.all_steps == original and not hasattr(dataset, 'training_anchor_limits')
