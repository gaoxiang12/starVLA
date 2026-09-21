"""Measure alignment at descending TCP height crossings before first closure."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def measure(precision):
    directory = Path(precision['source_case'])
    result = dict(seed=precision['seed'], source_case=str(directory),
                  scored_result=precision['scored_result'], crossings={})
    if not precision['closing_command_observed']:
        result['state'] = 'no_closure_window'
        return result
    arm = precision['first_closing_arm']
    with np.load(directory / 'pregrasp_physics_poses.npz', allow_pickle=False) as poses:
        times = poses['sim_s']
        end = int(np.searchsorted(times, precision['preclosure']['sim_s']))
        assert abs(times[end] - precision['preclosure']['sim_s']) < 1e-9
        times = times[:end + 1]
        block = poses['block_world_matrix'][:end + 1, 0, :3, 3]
        tcp = poses['tcp_world_matrix'][:end + 1, arm, :3, 3]
        drive = (poses['finger_drive_targets'][:end + 1, arm, 0] + .01) / .055
    with np.load(directory / 'physics_scoring_trace.npz', allow_pickle=False) as scoring:
        samples = scoring['samples'][:end]
    np.testing.assert_allclose(times[1:], samples[:, 0], atol=1e-9, rtol=0)
    contacts = samples[:, 4:16].reshape(-1, 3, 2, 2)[:, 0].any(axis=(1, 2))
    delta = (tcp - block) * 1000
    assert np.isfinite(delta).all() and np.all(np.diff(times) > 0)

    def event(index):
        return dict(sim_s=float(times[index]),
                    horizontal_offset_mm=float(np.linalg.norm(delta[index, :2])),
                    tcp_minus_red_xyz_mm=delta[index].tolist(),
                    red_xy_displacement_mm=float(np.linalg.norm(block[index, :2] - block[0, :2]) * 1000),
                    finger_drive=float(drive[index]),
                    red_finger_contact_seen_through_event=bool(contacts[:index].any()))

    for height in (100, 80, 60, 40, 20):
        indices = np.flatnonzero((delta[:-1, 2] > height) & (delta[1:, 2] <= height)) + 1
        result['crossings'][str(height)] = dict(
            crossing_count=len(indices),
            first=event(int(indices[0])) if len(indices) else None,
            last=event(int(indices[-1])) if len(indices) else None)
    result.update(state='measured', first_closing_arm=arm, preclosure=event(end))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paired', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raw = args.paired.read_bytes()
    paired = json.loads(raw)
    rows = [{side: measure(row[side]) for side in ('reference', 'candidate')}
            for row in paired['records']]
    report = dict(state=paired['state'], paired_completed_scenes=len(rows),
                  planned_scenes=paired['planned_scenes'],
                  paired_source=str(args.paired.resolve()),
                  paired_source_sha256=hashlib.sha256(raw).hexdigest(),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  records=rows,
                  limitations=[
                      'Height is actual TCP z minus current red-block center z, not fingertip clearance.',
                      'Crossings are sampled physical ticks without interpolation, measured through the first preclosure event.',
                      'All downward crossings are counted; both first and last are retained. Missing crossings remain unknown.',
                      'A downward relative-height crossing can also reflect object motion; displacement and contact history are reported.',
                      'Contacts cover moving fingers only. No contact here does not exclude palm or other-link contact.',
                      'Same protocol-order development prefix; these are diagnostics, not new independent trials.',
                  ])
    temp = args.output.with_suffix('.tmp')
    temp.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    temp.replace(args.output)
    for row in rows:
        print(json.dumps({side: dict(seed=r['seed'], success=r['scored_result']['success'],
                                    at_80mm=r['crossings'].get('80'))
                          for side, r in row.items()}))


if __name__ == '__main__':
    main()
