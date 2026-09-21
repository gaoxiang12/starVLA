"""Inspect candidate image-only object labels on training trajectories, without training."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import h5py
import numpy as np
from PIL import Image, ImageDraw

from starVLA.dataloader.rgb_object_supervision import COLORS, candidates
CAMERAS = ('head_camera', 'left_camera', 'right_camera')




def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, default=Path('/data/gaoxiang/RoboTwinGenerated_raw/Clean/blocks_ranking_rgb/demo_clean/data'))
    parser.add_argument('--split', type=Path, default=Path('examples/Robotwin/audits/rgb_scene_safe_validation_20260907.json'))
    parser.add_argument('--episodes', type=int, default=40)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    split = json.loads(args.split.read_text())
    available = [i for i in split['train_episode_ids'] if (args.raw / f'episode{i}.hdf5').is_file()]
    selected = np.random.default_rng(42).choice(available, args.episodes, replace=False).tolist()
    if set(selected) & set(split['validation_episode_ids']):
        raise ValueError('Validation data may not enter the candidate label audit')
    args.output.mkdir(exist_ok=False, parents=True)
    rows, tiles, event_rows = [], [], []
    label_root = Path('playground/Checkpoints/gawm_rgb_focus_20260907/labels')
    for episode in selected:
        with h5py.File(args.raw / f'episode{episode}.hdf5') as raw, np.load(label_root / f'episode_{episode:06d}.npz') as labels:
            n = len(raw['joint_action/vector'])
            close_steps = sorted(set(labels['target_step'][labels['kind'] == 0].tolist()))
            # Fixed coverage plus ten samples before each expert closing event.
            indices = sorted(set(np.linspace(0, n - 1, 5, dtype=int).tolist() + [max(0, t - 10) for t in close_steps]))
            for step in indices:
                for camera in CAMERAS:
                    # These historical JPEG buffers were encoded from RGB using
                    # cv2.imencode, so cv2.imdecode yields the native RGB here.
                    rgb = cv2.imdecode(np.frombuffer(raw[f'observation/{camera}/rgb'][step], np.uint8), cv2.IMREAD_COLOR)
                    if rgb is None or rgb.shape != (240, 320, 3):
                        raise ValueError(f'Invalid image: {episode}/{step}/{camera}')
                    objects = candidates(rgb)
                    rows.append(dict(episode=episode, step=int(step), progress=float(step / (n - 1)), camera=camera, objects=objects))
                    if camera == 'head_camera' and step in [max(0, t - 10) for t in close_steps]:
                        upcoming = [t for t in close_steps if t >= step]
                        target_step = upcoming[0]
                        ordinal = close_steps.index(target_step)
                        # Original expert play_once always grasps red/green/blue.
                        if len(close_steps) == 3 and labels['valid'][step, 0]:
                            point = labels['xy'][step, 0] * [320, 240] - .5
                            distances = [float(np.linalg.norm(point - o['center_xy'])) if o['accepted'] else None for o in objects]
                            present = [i for i, distance in enumerate(distances) if distance is not None]
                            nearest = min(present, key=lambda i: distances[i]) if present else None
                            event_rows.append(dict(episode=episode, step=int(step), expected_color=COLORS[ordinal],
                                nearest_color=COLORS[nearest] if nearest is not None else None,
                                distance_px=distances, tcp_target_xy=point.tolist()))
                    if episode in selected[:2] or (len(tiles) < 30 and camera == 'head_camera' and not all(o['accepted'] for o in objects)):
                        tile = Image.fromarray(rgb)
                        draw = ImageDraw.Draw(tile)
                        draw.rectangle((0, 0, 319, 18), fill='black')
                        draw.text((3, 3), f'ep{episode} t{step} {camera}', fill='white')
                        for obj in objects:
                            if obj['box_xywh'] is not None:
                                x, y, w, h = obj['box_xywh']
                                draw.rectangle((x, y, x+w, y+h), outline=obj['color'], width=2)
                                draw.text((x, max(19, y-12)), obj['color'] + (' OK' if obj['accepted'] else ' ?'), fill='white')
                        tiles.append(tile)
    for page, start in enumerate(range(0, len(tiles), 24)):
        canvas = Image.new('RGB', (320*4, 240*6), '#202020')
        for index, tile in enumerate(tiles[start:start+24]):
            canvas.paste(tile, ((index%4)*320, (index//4)*240))
        canvas.save(args.output / f'contact_sheet_{page:02d}.jpg')
    summary = {}
    for camera in CAMERAS:
        camera_rows = [r for r in rows if r['camera'] == camera]
        summary[camera] = dict(frames=len(camera_rows),
            accepted_by_color={color: sum(r['objects'][i]['accepted'] for r in camera_rows) for i, color in enumerate(COLORS)},
            all_colors_accepted=sum(all(o['accepted'] for o in r['objects']) for r in camera_rows))
    report = dict(state='candidate_audited_not_training_enabled', train_episode_ids=selected,
        split_sha256=hashlib.sha256(args.split.read_bytes()).hexdigest(), frames=len(rows),
        summary=summary, expert_preclose_checks=event_rows,
        expected_is_nearest=sum(r['expected_color'] == r['nearest_color'] for r in event_rows),
        rows=rows, note='Heuristic color-component proposals only; acceptance is not verified accuracy. '
        'Visible component centroids may shift under occlusion. No labels, segmentation, scene state, '
        'or expert event identity are supplied to the running policies. This audit uses training episodes only.')
    (args.output / 'audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(summary=summary, preclose_checks=len(event_rows), expected_is_nearest=report['expected_is_nearest'], contact_sheets=(len(tiles)+23)//24), indent=2))


if __name__ == '__main__':
    main()
