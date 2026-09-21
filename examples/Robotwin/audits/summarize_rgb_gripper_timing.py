"""Compare commanded gripper timing on exactly matched teacher observations."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', action='append', required=True, help='name=trajectory_report.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    reports = {}
    for specification in args.report:
        name, path = specification.split('=', 1)
        if name in reports:
            raise ValueError(f'Duplicate report: {name}')
        reports[name] = json.loads(Path(path).read_text())
    key = lambda row: (row['episode'], row['event'], row['arm'], row['lookahead'])
    reference = next(iter(reports.values()))['rows']
    expected = [key(row) for row in reference]
    assert len(set(expected)) == len(expected)
    result = {}
    for name, report in reports.items():
        assert [key(row) for row in report['rows']] == expected
        derived = []
        for row, ref in zip(report['rows'], reference):
            target = np.asarray(row['target_gripper_trajectory'])
            prediction = np.asarray(row['predicted_gripper_trajectory'])
            valid = np.asarray(row['action_valid_mask'], dtype=bool)
            assert target.shape == prediction.shape == valid.shape == (16,)
            np.testing.assert_array_equal(target, ref['target_gripper_trajectory'])
            np.testing.assert_array_equal(valid, ref['action_valid_mask'])
            assert np.isfinite(target[valid]).all() and np.isfinite(prediction[valid]).all()
            event_index = row['lookahead'] - 1
            truth_crossings = np.flatnonzero(valid & (target < .2))
            assert len(truth_crossings) and truth_crossings[0] == event_index
            assert target[event_index] == row['target_gripper']
            assert prediction[event_index] == row['predicted_gripper']
            crossings = np.flatnonzero(valid & (prediction < .2))
            delta = int(crossings[0]) - event_index if len(crossings) else None
            derived.append(dict(episode=row['episode'], event=row['event'], arm=row['arm'],
                                lookahead=row['lookahead'], first_crossing_offset_steps=delta,
                                at_event_error=float(prediction[event_index]-target[event_index]),
                                predicted_closed_at_event=bool(prediction[event_index] < .2),
                                valid_steps_after_event=int(valid[event_index+1:].sum())))
        groups = {}
        for h in (1, 8, 16):
            rows = [row for row in derived if row['lookahead'] == h]
            deltas = [row['first_crossing_offset_steps'] for row in rows
                      if row['first_crossing_offset_steps'] is not None]
            groups[str(h)] = dict(
                count=len(rows), no_crossing_within_chunk=len(rows)-len(deltas),
                earlier=sum(d < 0 for d in deltas), same_step=sum(d == 0 for d in deltas),
                later=sum(d > 0 for d in deltas),
                median_offset_among_observed_crossings=float(np.median(deltas)) if deltas else None,
                closed_at_event=sum(row['predicted_closed_at_event'] for row in rows),
                mean_signed_gripper_error_at_event=float(np.mean([row['at_event_error'] for row in rows])),
                mean_absolute_gripper_error_at_event=float(np.mean([abs(row['at_event_error']) for row in rows])))
        result[name] = dict(checkpoint=report['checkpoint'], by_lookahead=groups, rows=derived)
    output = dict(reports=result, note='Open-loop command predictions at teacher observations. '
                  'First crossing uses command <0.2, not measured jaw contact or grasp success. '
                  'No crossing is censored at the end of the 16-step chunk, especially at lookahead 16. '
                  'Offsets are recorded action steps; do not convert to simulator steps without alignment.')
    with args.output.open('x') as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
