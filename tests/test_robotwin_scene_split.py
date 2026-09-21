import json

import pytest

from starVLA.dataloader.lerobot_datasets import episode_split_blacklist


def fixture_manifest(tmp_path):
    dataset = tmp_path / 'dataset'
    (dataset / 'meta').mkdir(parents=True)
    (dataset / 'meta/episodes.jsonl').write_text(''.join(json.dumps({'episode_index': i})+'\n' for i in range(6)))
    manifest = dict(dataset=str(dataset), train_episode_ids=[1, 2, 3], validation_episode_ids=[0, 4], excluded_episode_ids=[5])
    path = tmp_path / 'split.json'
    path.write_text(json.dumps(manifest))
    return dataset, path, manifest


def test_manifest_excludes_scene_overlap_without_changing_training(tmp_path):
    dataset, path, manifest = fixture_manifest(tmp_path)
    cfg = dict(episode_split_manifest=str(path), validation_episode_stride=20)
    assert episode_split_blacklist(dataset, dict(cfg, episode_split='train')) == [0, 4, 5]
    assert episode_split_blacklist(dataset, dict(cfg, episode_split='validation')) == [1, 2, 3, 5]


@pytest.mark.parametrize('mutation', ['overlap', 'unknown', 'duplicate', 'wrong_dataset'])
def test_manifest_rejects_invalid_or_misrouted_partitions(tmp_path, mutation):
    dataset, path, manifest = fixture_manifest(tmp_path)
    if mutation == 'overlap':
        manifest['validation_episode_ids'].append(1)
    elif mutation == 'unknown':
        manifest['excluded_episode_ids'].append(7)
    elif mutation == 'duplicate':
        manifest['train_episode_ids'].append(1)
    else:
        manifest['dataset'] = str(tmp_path/'other')
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        episode_split_blacklist(dataset, dict(episode_split_manifest=str(path)))
