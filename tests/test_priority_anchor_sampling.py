"""Priority replay must preserve full trajectory support and validation behavior."""
from types import SimpleNamespace

import numpy as np
import pytest

from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotMixtureDataset


def mixture(priority=None):
    labels = {'event_anchors': np.arange(10),
              'priority_anchors': np.arange(50, 60) if priority is None else priority}
    child = SimpleNamespace(trajectory_ids=np.array([0]), trajectory_lengths=np.array([101]),
        minimum_action_offset=1, data_cfg={'event_sampling_probability': .35},
        _spatial_episode=lambda _: labels)
    mix = LeRobotMixtureDataset.__new__(LeRobotMixtureDataset)
    mix.datasets = [child]
    mix._dataset_sampling_weights = np.array([1.])
    mix._trajectory_sampling_weights = [np.array([1.])]
    mix.epoch, mix.seed, mix.mode = 0, 42, 'train'
    return mix, child, labels


def test_priority_exposure_keeps_full_support_and_valid_tail():
    mix, child, _ = mixture()
    child.data_cfg['priority_sampling_probability'] = .5
    anchors = np.array([mix.sample_step(i)[2] for i in range(5000)])
    assert set(anchors) == set(range(100))
    assert .49 < ((anchors >= 50) & (anchors < 60)).mean() < .57
    assert .17 < (anchors < 10).mean() < .24


def test_disabled_priority_and_validation_preserve_legacy_sequence():
    mix, child, _ = mixture()
    original = [mix.sample_step(i)[2] for i in range(200)]
    child.data_cfg['priority_sampling_probability'] = 0
    assert original == [mix.sample_step(i)[2] for i in range(200)]
    mix.mode = 'validation'
    original = [mix.sample_step(i)[2] for i in range(200)]
    child.data_cfg['priority_sampling_probability'] = .9
    child._spatial_episode = lambda _: pytest.fail('Validation must not read priority sidecars')
    assert original == [mix.sample_step(i)[2] for i in range(200)]


def test_empty_or_only_invalid_tail_priority_falls_back_to_existing_sampler():
    mix, child, labels = mixture(np.array([], dtype=np.int64))
    original = [mix.sample_step(i)[2] for i in range(200)]
    child.data_cfg['priority_sampling_probability'] = .5
    assert original == [mix.sample_step(i)[2] for i in range(200)]
    labels['priority_anchors'] = np.array([100])
    assert original == [mix.sample_step(i)[2] for i in range(200)]


@pytest.mark.parametrize('anchors', [np.array([-1]), np.array([101]), np.array([2, 2]),
                                    np.array([1.5]), np.array([[2]])])
def test_malformed_priority_anchors_rejected(anchors):
    mix, child, _ = mixture(anchors)
    child.data_cfg['priority_sampling_probability'] = .5
    with pytest.raises(ValueError):
        mix.sample_step(0)


@pytest.mark.parametrize('probability', [-.1, 1., float('nan')])
def test_invalid_priority_probability_rejected(probability):
    mix, child, _ = mixture()
    child.data_cfg['priority_sampling_probability'] = probability
    with pytest.raises(ValueError, match='nonzero uniform'):
        mix.sample_step(0)


def test_priority_requires_audited_sidecar():
    mix, child, labels = mixture()
    del labels['priority_anchors']
    child.data_cfg['priority_sampling_probability'] = .5
    with pytest.raises(ValueError, match='audited'):
        mix.sample_step(0)
