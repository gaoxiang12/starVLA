"""Audit candidate red-label safeguard on the existing fixed training sample."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import h5py
import numpy as np

from starVLA.dataloader.rgb_object_supervision_core import candidates_with_core


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Use a new output file')
    data = json.loads(args.source.read_text())
    rows = [r for r in data['rows'] if r['camera'] == 'head_camera']
    counts = {c: dict(old=0, new=0) for c in ('red', 'green', 'blue')}
    changed, frames = [], []
    for row in rows:
        ep, step = row['episode'], row['step']
        assert ep in data['train_episode_ids']
        path = Path('/data/gaoxiang/RoboTwinGenerated_raw/Clean/blocks_ranking_rgb/demo_clean/data') / f'episode{ep}.hdf5'
        with h5py.File(path) as f:
            rgb = cv2.imdecode(np.frombuffer(f['observation/head_camera/rgb'][step].tobytes(), np.uint8), cv2.IMREAD_COLOR)
        key = dict(episode=ep, step=step, native_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest())
        frames.append(key)
        for old, new in zip(row['objects'], candidates_with_core(rgb)):
            assert all(new[k] == v for k, v in old.items() if k != 'accepted')
            assert old['accepted'] == new['original_accepted']
            counts[new['color']]['old'] += int(old['accepted'])
            counts[new['color']]['new'] += int(new['accepted'])
            if old['accepted'] != new['accepted']:
                assert not new['accepted'] and not new['visibility_supervision_valid']
                changed.append(dict(**key, proposal=new))
    helper = Path('starVLA/dataloader/rgb_object_supervision_core.py')
    result = dict(frames=len(rows), train_episode_ids=data['train_episode_ids'],
        source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
        helper_sha256=hashlib.sha256(helper.read_bytes()).hexdigest(),
        frame_records=frames, counts=counts, changed=changed,
        note='Candidate red-only label safeguard. No integration into running training/inference. '
             'Rejected uncertain positive visibility targets are masked, not labelled absent. '
             'Train sample only, not independent label-accuracy benchmark or task performance.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(counts=counts, changed=[(r['episode'], r['step'], r['proposal']['color']) for r in changed])))


if __name__ == '__main__':
    main()
