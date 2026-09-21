"""Opt-in bounded training anchors, retaining every requested target inside a prefix."""
import json
from pathlib import Path


def attach_training_anchor_bounds(dataset, data_cfg):
    path = data_cfg.get('training_anchor_manifest') if data_cfg is not None else None
    if not path or data_cfg.get('episode_split', 'train') != 'train':
        return
    manifest = json.loads(Path(path).read_text())
    if manifest.get('format_version') != 1 or Path(manifest['dataset']).resolve() != Path(dataset.dataset_path).resolve():
        raise ValueError('Training anchor manifest version or dataset mismatch')
    rows = manifest['episodes']
    ends = {int(row['episode_index']):int(row['target_end_exclusive']) for row in rows}
    if len(ends) != len(rows) or set(ends) != set(map(int, dataset.trajectory_ids)):
        raise ValueError('Anchor manifest must cover the selected training episodes exactly once')
    offsets = [int(offset) for key, values in dataset.delta_indices.items()
               if key.startswith(('action.', 'video.')) for offset in values]
    if not offsets:
        raise ValueError('Cannot determine action/video target horizons')
    maximum_offset = max(offsets)
    if maximum_offset != int(manifest['maximum_target_offset']):
        raise ValueError('Training anchor manifest does not match the configured horizons')
    lengths = dict(zip(map(int, dataset.trajectory_ids), map(int, dataset.trajectory_lengths)))
    limits = {}
    for episode, end in ends.items():
        if not maximum_offset < end <= lengths[episode]:
            raise ValueError(f'Invalid target prefix length for episode {episode}: {end}')
        # Last legal anchor is end-1-maximum_offset; every target stays < end.
        limits[episode] = end-maximum_offset
    selected = [(episode, step) for episode, step in dataset.all_steps if step < limits[int(episode)]]
    if not selected or {int(episode) for episode, _ in selected} != set(limits):
        raise ValueError('Training anchor restriction leaves an empty episode')
    dataset._all_steps = selected
    dataset.training_anchor_limits = limits
    dataset.training_target_ends = ends
