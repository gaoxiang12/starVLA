"""Summarize preclosure alignment and displacement on an existing paired snapshot."""
import argparse
import hashlib
import json
from pathlib import Path

from audit_preclosure_object_push import inspect_case
from measure_grasp_precision import bin_mm


def outcome_counts(rows):
    return {
        'episodes': len(rows),
        'successes': sum(r['success'] for r in rows),
        'first_attempt_successes': sum(r['first_attempt_success'] for r in rows),
        'seeds': [r['seed'] for r in rows],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paired', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raw = args.paired.read_bytes()
    paired = json.loads(raw)
    groups = {}
    for side in ('reference', 'candidate'):
        rows = []
        for pair in paired['records']:
            precision = pair[side]
            row = dict(seed=pair['seed'], source_case=precision['source_case'],
                       success=precision['scored_result']['success'],
                       first_attempt_success=precision['scored_result']['first_attempt_success'],
                       first_closing_arm=precision['first_closing_arm'],
                       first_arm_is_success_arm=(not precision['scored_result']['success'] or
                           precision['first_closing_arm'] == precision['scored_result']['success_arm']),
                       cube_edge_mm=precision['cube_edge_mm'],
                       preclosure_bin=precision['preclosure_bin'],
                       preclosure_xy_mm=None, preclosure_xy_over_half_edge=None,
                       red_xy_displacement_at_preclosure_mm=None,
                       red_xy_ever_displaced_ge_20mm_before_closure=None,
                       moving_finger_contact_before_closure=None,
                       first_preclosure_contact=None)
            if precision['closing_command_observed']:
                detail = inspect_case(precision['source_case'])
                assert detail['seed'] == pair['seed']
                assert detail['scored_result'] == precision['scored_result']
                event = detail['at_preclosure']
                offset = event['tcp_to_current_red_xy_mm']
                assert bin_mm(offset) == precision['preclosure_bin']
                row.update(
                    preclosure_xy_mm=offset,
                    preclosure_xy_over_half_edge=offset / (precision['cube_edge_mm'][0] / 2),
                    red_xy_displacement_at_preclosure_mm=event['red_xy_displacement_mm'],
                    red_xy_ever_displaced_ge_20mm_before_closure=(
                        detail['first_displacement_threshold_events_mm']['20'] is not None),
                    moving_finger_contact_before_closure=detail['first_moving_finger_contact'] is not None,
                    first_preclosure_contact=detail['first_moving_finger_contact'])
            rows.append(row)
        bins = {name: outcome_counts([r for r in rows if r['preclosure_bin'] == name])
                for name in ('le_10mm', '10_to_20mm', '20_to_30mm', 'gt_30mm', 'no_closure')}
        assert sum(item['episodes'] for item in bins.values()) == len(rows)
        groups[side] = dict(
            overall=outcome_counts(rows),
            preclosure_alignment_bins=bins,
            displacement_ge_20mm=outcome_counts([
                r for r in rows if r['red_xy_ever_displaced_ge_20mm_before_closure'] is True]),
            preclosure_finger_contact=outcome_counts([
                r for r in rows if r['moving_finger_contact_before_closure'] is True]),
            records=rows)
    report = dict(
        state=paired['state'], paired_completed_scenes=paired['paired_completed_scenes'],
        planned_scenes=paired['planned_scenes'], pending_scene_ids=paired['pending_scene_ids'],
        paired_source=str(args.paired.resolve()), paired_source_sha256=hashlib.sha256(raw).hexdigest(),
        checkpoints=paired['checkpoints'], checkpoint_sha256=paired['checkpoint_sha256'],
        script_sha256={str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in (Path(__file__), Path(__file__).with_name('audit_preclosure_object_push.py'))},
        groups=groups,
        limitations=[
            'These are the same completed development scenes in protocol order, not a random sample or final test.',
            'Each episode contributes once. Pending scenes are unscored; no-closure episodes remain in the overall counts.',
            'Contact and displacement are measured only through the preclosure event; they do not prove causal force.',
            'Contact trace covers moving finger links only; it does not include palm or other robot links.',
            'XY offset is TCP to current object center, not an independently labeled optimal grasp pose error.',
            'Offset/half-edge is a scale diagnostic, not a collision or grasp feasibility boundary.',
            'Preclosure geometry belongs to the first closing arm; overall success may occur on a later attempt.',
            'A no-closure episode has no defined preclosure window; its contact and displacement values are unknown.',
        ])
    temp = args.output.with_suffix('.tmp')
    temp.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    temp.replace(args.output)
    print(json.dumps({side: {k: v for k, v in group.items() if k != 'records'}
                      for side, group in groups.items()}, indent=2))


if __name__ == '__main__':
    main()
