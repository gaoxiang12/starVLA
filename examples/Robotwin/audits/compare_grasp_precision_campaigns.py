"""Pair completed development scenes without comparing unequal denominators."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def transitions(rows,key):
    output={name:[] for name in ('gained','lost','both_success','both_failure')}
    for row in rows:
        before,after=[row[side]['scored_result'][key] for side in ('reference','candidate')]
        name='both_success' if before and after else 'gained' if after else 'lost' if before else 'both_failure'
        output[name].append(row['seed'])
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    statuses=[];manifests=[];hashes=[]
    for directory in (args.reference,args.candidate):
        raw=(directory/'status.json').read_bytes()
        statuses.append(json.loads(raw));hashes.append(hashlib.sha256(raw).hexdigest())
        manifests.append(json.loads((directory/'manifest.json').read_text()))
    # An early single-case probe intentionally lacks this complete scene manifest.
    seeds=[[row['seed'] for row in m['scenes']] for m in manifests]
    assert seeds[0]==seeds[1] and len(seeds[0])==20 and len(set(seeds[0]))==20
    assert all(row['split']=='development' for m in manifests for row in m['scenes'])
    assert all(m['mode']=='full' and m['execute_horizon']==16 and m['precision']=='float32' for m in manifests)
    indexes=[{r['seed']:r for r in status['results']} for status in statuses]
    assert all(len(index)==len(status['results']) for index,status in zip(indexes,statuses))
    common=[seed for seed in seeds[0] if all(seed in index for index in indexes)]
    rows=[]
    for seed in common:
        before,after=[index[seed] for index in indexes]
        cases=[json.loads((Path(row['source_case'])/'result.json').read_text()) for row in (before,after)]
        assert all(c['state']=='complete' and c['seed']==seed for c in cases)
        assert cases[0]['initial_scene_audit']['initial_rgb_sha256']==cases[1]['initial_scene_audit']['initial_rgb_sha256']
        for key in ('initial_block_positions_m','initial_block_quaternions','block_half_extents'):
            np.testing.assert_array_equal(cases[0][key],cases[1][key])
        rows.append(dict(seed=seed,reference=before,candidate=after,
            same_first_closing_arm=before['first_closing_arm']==after['first_closing_arm'],
            first_arm_is_success_arm=[not r['scored_result']['success'] or r['first_closing_arm']==r['scored_result']['success_arm'] for r in (before,after)]))
    geometry={}
    for phase in ('preclosure','0.95','0.8','0.5','0.2','0.05'):
        values=[];available=[]
        for row in rows:
            if not row['same_first_closing_arm'] or not all(row['first_arm_is_success_arm']):continue
            events=[row[k]['preclosure'] if phase=='preclosure' else row[k]['drive_threshold_events'].get(phase) for k in ('reference','candidate')]
            if all(event is not None for event in events):
                values.append([event['horizontal_offset_mm'] for event in events]);available.append(row['seed'])
        values=np.asarray(values)
        geometry[phase]=dict(paired_valid_scenes=available,
            reference_median_mm=float(np.median(values[:,0])) if len(values) else None,
            candidate_median_mm=float(np.median(values[:,1])) if len(values) else None,
            paired_delta_median_mm=float(np.median(values[:,1]-values[:,0])) if len(values) else None)
    report=dict(state='complete' if all(s['state']=='complete' for s in statuses) else 'partial_paired_snapshot',
        planned_scenes=20,paired_completed_scenes=len(rows),
        pending_scene_ids=[seed for seed in seeds[0] if seed not in common],
        source_states=[s['state'] for s in statuses],source_status_sha256=hashes,
        checkpoints=[m['checkpoint'] for m in manifests],checkpoint_sha256=[m['checkpoint_sha256'] for m in manifests],
        success_transitions=transitions(rows,'success'),first_attempt_transitions=transitions(rows,'first_attempt_success'),
        paired_success_counts=[sum(row[side]['scored_result']['success'] for row in rows) for side in ('reference','candidate')],
        no_closure_scenes={side:[row['seed'] for row in rows if not row[side]['closing_command_observed']] for side in ('reference','candidate')},
        geometry=geometry,records=rows,
        interpretation='Compare only the same completed scenes; pending scenes remain unscored. Protocol-order prefixes are not representative random samples. Geometry summaries use shared observed closure stages and the same measured arm; binary success keeps every paired scene. This excludes single-scene early probes.')
    temp=args.output.with_suffix('.tmp');temp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');temp.replace(args.output)
    print(json.dumps({k:v for k,v in report.items() if k not in ('records','geometry')}),flush=True)


if __name__=='__main__':main()
