"""Compare completed RGB rollouts on identical initial block positions."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np


PREDICATES = (
    'correct_color_order', 'adjacent_x_within_tolerance',
    'adjacent_y_within_tolerance', 'left_gripper_open', 'right_gripper_open',
)


def read_rows(path, episodes):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows) != episodes or {r['trial'] for r in rows} != set(range(episodes)):
        raise ValueError(f'Expected exactly {episodes} distinct completed trials: {path}')
    rows.sort(key=lambda row: row['trial'])
    for row in rows:
        arrangement = row['rgb_arrangement']
        if row['task'] != 'blocks_ranking_rgb':
            raise ValueError(f'Wrong task in {path}')
        if bool(row['success']) != all(arrangement[key] for key in PREDICATES):
            raise ValueError(f'Terminal success/predicate mismatch: {path}, trial {row["trial"]}')
        positions = np.asarray(row['initial_block_positions_m'])
        if positions.shape != (3, 3) or not np.isfinite(positions).all():
            raise ValueError(f'Invalid initial positions in {path}')
    return rows


def summarize(rows):
    failed = [r for r in rows if not r['success']]
    all_lifted_failed = [r for r in failed if r['blocks_lifted'] == 3]
    first_attempt_without_lift = [r for r in rows
        if r['first_close_step'] is not None and not r['first_attempt_lifted']]
    details = []
    for row in failed:
        arrangement = row['rgb_arrangement']
        details.append(dict(trial=row['trial'], blocks_ever_lifted=row['blocks_lifted'],
            first_close_step=row['first_close_step'], first_lift_step=row['first_lift_step'],
            final_color_order=arrangement['final_left_to_right_colors'],
            failed_predicates=[key for key in PREDICATES if not arrangement[key]]))
    return dict(episodes=len(rows), successes=sum(r['success'] for r in rows),
        successful_trials=[r['trial'] for r in rows if r['success']],
        blocks_ever_lifted_histogram=dict(sorted(Counter(r['blocks_lifted'] for r in rows).items())),
        first_attempt_lifted=sum(r['first_attempt_lifted'] for r in rows),
        first_attempt_without_lift=len(first_attempt_without_lift),
        later_lift_after_first_attempt_without_lift=sum(r['any_block_lifted'] for r in first_attempt_without_lift),
        successes_after_first_attempt_without_lift=sum(r['success'] for r in first_attempt_without_lift),
        all_three_ever_lifted_but_failed=len(all_lifted_failed),
        failed_predicate_counts={key: sum(not r['rgb_arrangement'][key] for r in failed) for key in PREDICATES},
        all_three_lifted_failure_predicates={key: sum(not r['rgb_arrangement'][key] for r in all_lifted_failed)
                                            for key in PREDICATES},
        failed_trials=details)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group', action='append', required=True, metavar='NAME=METRICS_JSONL')
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    paths = {}
    for item in args.group:
        name, path = item.split('=', 1)
        if not name or name in paths:
            raise ValueError('Each group needs a unique nonempty name')
        paths[name] = Path(path).resolve()
    if len(paths) < 2 or args.episodes < 1:
        raise ValueError('Need at least two groups and a positive episode count')
    groups = {name: read_rows(path, args.episodes) for name, path in paths.items()}
    reference_name = next(iter(groups))
    reference = groups[reference_name]
    paired = {}
    for name, rows in groups.items():
        for a, b in zip(reference, rows):
            if not np.array_equal(a['initial_block_positions_m'], b['initial_block_positions_m']):
                raise ValueError(f'Initial scene mismatch: {name}, trial {a["trial"]}')
        paired[name] = dict(
            wins=[b['trial'] for a, b in zip(reference, rows) if b['success'] and not a['success']],
            losses=[b['trial'] for a, b in zip(reference, rows) if a['success'] and not b['success']])
    report = dict(state='complete', reference=reference_name, episodes=args.episodes,
        initial_xyz_identical=True,
        inputs={name: dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                for name, path in paths.items()},
        groups={name: summarize(rows) for name, rows in groups.items()}, paired_success=paired,
        note='Descriptive screening only. Lift means >4 cm above initial z at any time, not correct placement. '
             'First attempt uses commanded gripper thresholds; absence of lift does not prove an empty grasp. '
             'Failure predicates overlap. Initial XYZ matching alone does not check model/protocol equivalence; '
             'retain campaign manifests for that comparison.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps({name: {key: value for key, value in report['groups'][name].items()
                            if key != 'failed_trials'} for name in groups}, indent=2))


if __name__ == '__main__':
    main()
