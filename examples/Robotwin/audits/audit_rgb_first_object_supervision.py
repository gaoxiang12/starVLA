"""Check first-waypoint color and first-placement color in the training videos."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import cv2
import numpy as np

from examples.Robotwin.audits.audit_rgb_scene_and_color import colors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--release-offsets', type=int, nargs='+', default=[12])
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not args.release_offsets or min(args.release_offsets) < 1:
        raise ValueError('Release offsets must be positive recorded-frame counts')
    manifest = json.loads((args.labels / 'manifest.json').read_text())
    assert Path(manifest['dataset']).resolve() == args.dataset.resolve()
    cv2.setNumThreads(1)

    def audit(record):
        episode = record['episode']
        row = dict(episode=episode, raw_aligned=record['raw_aligned'])
        events = record['events']
        closes = [e for e in events if e[2] == 0]
        if not closes:
            return dict(**row, error='No closing event')
        first = closes[0]
        opens = [e for e in events if e[2] == 1 and e[1] == first[1] and e[0] > first[0]]
        if not opens:
            return dict(**row, error='No subsequent opening of first grasping arm')
        later_closes = [e[0] for e in closes if e[0] > opens[0][0]]
        limit = min(record['frames'] - 1, later_closes[0] - 1 if later_closes else record['frames'] - 1)
        frame_indices = sorted({min(limit, opens[0][0] + offset) for offset in args.release_offsets})
        row.update(first_close=first, first_open=opens[0], after_release_frames=frame_indices)
        path = args.dataset / f'videos/chunk-000/observation.images.cam_high/episode_{episode:06d}.mp4'
        cap = cv2.VideoCapture(str(path))
        try:
            if int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) != record['frames']:
                return dict(**row, error='Video/label frame count mismatch')
            observed = []
            for index in (0, *frame_indices):
                cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, frame_bgr = cap.read()
                if not ok:
                    return dict(**row, error=f'Cannot decode frame {index}')
                decoded_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                # Exactly the loader's video_channel_order=bgr correction.
                corrected_rgb = decoded_rgb[..., ::-1]
                observed.append(colors(corrected_rgb))
        finally:
            cap.release()
        initial = observed[0]
        row.update(initial_colors=initial, placement_observations=[])
        confident_colors = set()
        for index, released in zip(frame_indices, observed[1:]):
            observation = dict(frame=index, colors=released, placed_color='ambiguous')
            if len(initial) == len(released) == 3:
                motion = {c: float(np.linalg.norm(np.asarray(initial[c]['xy']) - released[c]['xy']))
                          for c in ('red', 'green', 'blue')}
                ordered = sorted(motion, key=motion.get, reverse=True)
                confident = motion[ordered[0]] > 20 and motion[ordered[1]] < 10
                observation.update(color_displacements_px=motion,
                                   placed_color=ordered[0] if confident else 'ambiguous')
                if confident:
                    confident_colors.add(ordered[0])
            row['placement_observations'].append(observation)
        row['first_placed_color'] = next(iter(confident_colors)) if len(confident_colors) == 1 else 'ambiguous'
        if record['raw_aligned']:
            with np.load(args.labels / f'episode_{episode:06d}.npz', allow_pickle=False) as label:
                if label['valid'][0, 0] and label['kind'][0] == 0 and len(initial) == 3:
                    point = label['xy'][0, 0] * [320, 240] - .5
                    distances = {c: float(np.linalg.norm(point - initial[c]['xy'])) for c in initial}
                    ordered = sorted(distances, key=distances.get)
                    confident = distances[ordered[0]] < 20 and distances[ordered[1]] - distances[ordered[0]] > 10
                    row.update(initial_waypoint_xy_pixels=point.tolist(),
                               waypoint_distances_px=distances,
                               initial_waypoint_color=ordered[0] if confident else 'ambiguous')
                else:
                    row['initial_waypoint_color'] = 'ambiguous'
        return row

    with ThreadPoolExecutor(max_workers=2) as pool:
        rows = list(pool.map(audit, manifest['episodes']))
    summary = {}
    for name, subset in (('original', rows[:500]), ('appended', rows[500:])):
        summary[name] = dict(episodes=len(subset), errors=sum('error' in row for row in subset),
             first_placed_color=dict(Counter(row.get('first_placed_color', 'error') for row in subset)),
             initial_waypoint_color=dict(Counter(row.get('initial_waypoint_color', 'unavailable') for row in subset)))
    report = dict(dataset=str(args.dataset.resolve()), summary=summary, rows=rows,
         release_offsets=args.release_offsets,
         method='Corrected head-camera RGB. First-placement identity: exactly one color moves >20px after '
                'the first grasping arm opens (configured recorded-frame offsets, before next closing), '
                'other colors each move <10px. If multiple observed offsets are confident they must agree. '
                'Initial geometric waypoint identity: raw-aligned labels only, nearest center <20px and '
                'next-nearest margin >10px. Largest saturated components; uncertain cases remain ambiguous.',
         limitation='Heuristic image audit, not a physics contact measurement or proof that every trajectory '
                    'has correct supervision throughout. No inferred color labels are supplied to training.')
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
