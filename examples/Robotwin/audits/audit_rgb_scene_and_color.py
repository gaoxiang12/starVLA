"""Audit terminal color order and cross-source initial-scene matches for RGB."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import av
import cv2
import numpy as np
from scipy.spatial import cKDTree

ROOT = Path('/data/gaoxiang/RoboTwinGenerated/Clean/blocks_ranking_rgb')
OUT = Path(__file__).resolve().parent / 'rgb_scene_and_color_20260907.json'


def colors(frame):
    a = frame.astype(np.int16)
    result = {}
    for name, channel in (('red', 0), ('green', 1), ('blue', 2)):
        others = [i for i in range(3) if i != channel]
        mask = ((a[..., channel] > 100) & (a[..., channel] - a[..., others].max(-1) > 60)).astype(np.uint8)
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
        if count > 1:
            index = 1 + int(stats[1:, cv2.CC_STAT_AREA].argmax())
            area = int(stats[index, cv2.CC_STAT_AREA])
            if area >= 50:
                result[name] = dict(xy=centroids[index].tolist(), area=area)
    return result


def audit_episode(episode):
    path = ROOT / f'videos/chunk-000/observation.images.cam_high/episode_{episode:06d}.mp4'
    try:
        with av.open(str(path)) as video:
            stream = video.streams.video[0]
            stream.codec_context.thread_count = 1
            first = next(video.decode(video=0)).to_ndarray(format='rgb24')
            if stream.duration:
                video.seek(max(0, stream.duration - int(2 / float(stream.time_base))), stream=stream)
            last = first
            for frame in video.decode(video=0):
                last = frame.to_ndarray(format='rgb24')
        start, end = colors(first), colors(last)
        order = sorted(end, key=lambda name: end[name]['xy'][0]) if len(end) == 3 else None
        return dict(episode=episode, initial=start, terminal=end, terminal_order=order)
    except Exception as exc:
        return dict(episode=episode, error=repr(exc))


def main():
    cv2.setNumThreads(1)
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(audit_episode, range(1000)))
    halves = []
    for start in (0, 500):
        selected = [r for r in rows[start:start+500] if len(r.get('initial', {})) == 3]
        coords = np.asarray([[r['initial'][c]['xy'] for c in ('red', 'green', 'blue')] for r in selected])
        halves.append((selected, coords))
    left, right = halves
    tree = cKDTree(left[1].reshape(-1, 6))
    distance, nearest = tree.query(right[1].reshape(-1, 6))
    matches = []
    for i, j in enumerate(nearest):
        per_block = np.linalg.norm(right[1][i] - left[1][j], axis=-1)
        area_ratios = [right[0][i]['initial'][c]['area'] / left[0][j]['initial'][c]['area'] for c in ('red', 'green', 'blue')]
        if per_block.max() <= 1.5 and all(.8 <= ratio <= 1.25 for ratio in area_ratios):
            a, b = left[0][j]['episode'], right[0][i]['episode']
            matches.append(dict(first_source_episode=a, appended_source_episode=b,
                                per_block_center_difference_px=per_block.tolist(), area_ratios=area_ratios,
                                crosses_current_train_validation_split=(a % 20 == 0) != (b % 20 == 0)))
    report = dict(dataset=str(ROOT), episodes=1000,
                  method='Largest saturated R/G/B components in head-camera first and final frames. Cross-source scene match: each block center within 1.5 px and area ratio 0.8-1.25. These are scene matches, not proof of identical trajectories.',
                  errors=[r for r in rows if 'error' in r],
                  terminal_order_by_source=[dict(source=name,
                      bgr=sum(r.get('terminal_order') == ['blue', 'green', 'red'] for r in subset),
                      rgb=sum(r.get('terminal_order') == ['red', 'green', 'blue'] for r in subset),
                      other_or_ambiguous=sum(r.get('terminal_order') not in (['blue','green','red'], ['red','green','blue']) for r in subset))
                      for name, subset in [('original', rows[:500]), ('appended', rows[500:])]],
                  matched_initial_scenes=len(matches), cross_split_matches=sum(m['crosses_current_train_validation_split'] for m in matches),
                  matches=matches, episodes_detail=rows)
    OUT.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('matches', 'episodes_detail')}, indent=2))
    print('first matched pairs', matches[:5])


if __name__ == '__main__':
    main()
