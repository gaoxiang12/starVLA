"""Train-only recovery data must never leak into the held-out evaluator."""
from omegaconf import OmegaConf
import pytest

import starVLA.dataloader.lerobot_datasets as loader
from starVLA.training.trainer_utils.config_tracker import wrap_config


def config():
    return OmegaConf.create(dict(data_root_dir='/tmp', data_mix='test_recovery_options',
        episode_split='train', episode_split_manifest='original_split.json',
        spatial_supervision_dir='original_labels', validation_episode_stride=20,
        normalization_statistics_path='frozen_stats.json', dataset_options={
            'recovery': dict(splits=['train'], episode_split_manifest=None,
                             validation_episode_stride=0, spatial_supervision_dir='recovery_labels')}))


@pytest.mark.parametrize('tracked', [False, True])
def test_training_override_isolated_from_original_and_normalization(tracked):
    cfg = config()
    cfg.dataset_options.recovery.priority_sampling_probability = .5
    argument = wrap_config(cfg) if tracked else cfg
    assert loader._dataset_config_for_split(argument, 'original') is argument
    recovery = loader._dataset_config_for_split(argument, 'recovery')
    assert recovery.episode_split_manifest is None
    assert recovery.validation_episode_stride == 0
    assert recovery.spatial_supervision_dir == 'recovery_labels'
    assert recovery.priority_sampling_probability == .5
    assert 'priority_sampling_probability' not in cfg
    assert recovery.normalization_statistics_path == 'frozen_stats.json'
    assert cfg.episode_split_manifest == 'original_split.json'
    assert cfg.spatial_supervision_dir == 'original_labels'
    cfg.episode_split = 'validation'
    assert loader._dataset_config_for_split(argument, 'recovery') is None


def test_public_dataset_builder_excludes_recovery_before_opening_it(monkeypatch):
    cfg = config()
    monkeypatch.setitem(loader.DATASET_NAMED_MIXTURES, cfg.data_mix,
                        [('original', .9, 'fake'), ('recovery', .1, 'fake')])
    opened = []
    def make(root, name, robot, **kwargs):
        opened.append((name, kwargs['data_cfg']))
        return name
    monkeypatch.setattr(loader, 'make_LeRobotSingleDataset', make)
    monkeypatch.setattr(loader, 'LeRobotMixtureDataset', lambda datasets, **kwargs: datasets)
    assert loader.get_vla_dataset(cfg) == [('original', .9), ('recovery', .1)]
    assert opened[1][1].episode_split_manifest is None
    opened.clear()
    cfg.episode_split = 'validation'
    assert loader.get_vla_dataset(cfg, mode='validation') == [('original', .9)]
    assert [name for name, _ in opened] == ['original']


@pytest.mark.parametrize('options', [dict(splits='train'), dict(splits=['typo']),
                                    dict(splits=[]), dict(normalization_statistics_path='different.json')])
def test_invalid_or_semantic_overrides_rejected(options):
    cfg = config()
    cfg.dataset_options.recovery = options
    with pytest.raises(ValueError):
        loader._dataset_config_for_split(cfg, 'recovery')


def test_mistyped_dataset_key_rejected(monkeypatch):
    cfg = config()
    monkeypatch.setitem(loader.DATASET_NAMED_MIXTURES, cfg.data_mix, [('original', 1., 'fake')])
    with pytest.raises(ValueError, match='outside this mixture'):
        loader.get_vla_dataset(cfg)
