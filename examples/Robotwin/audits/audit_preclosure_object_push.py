"""Locate measured object displacement and finger contact before first closure."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def inspect_case(directory):
    directory=Path(directory)
    case=json.loads((directory/'result.json').read_text())
    precision=json.loads((directory/'precision.json').read_text())
    assert case['state']=='complete' and precision['closing_command_observed']
    with np.load(directory/'pregrasp_physics_poses.npz',allow_pickle=False) as data:
        times=data['sim_s']
        tcp=data['tcp_world_matrix'][:,:,:3,3]
        blocks=data['block_world_matrix'][:,:,:3,3]
        drive=(data['finger_drive_targets'][:,:,0]+.01)/.055
        ids=data['block_entity_ids']
    with np.load(directory/'physics_scoring_trace.npz',allow_pickle=False) as data:
        samples=data['samples']
    np.testing.assert_array_equal(ids,case['block_entity_ids'])
    np.testing.assert_allclose(times[1:],samples[:,0],atol=1e-9,rtol=0)
    contacts=samples[:,4:16].reshape(-1,3,2,2).astype(bool)
    arm=precision['first_closing_arm']
    end=int(np.searchsorted(times,precision['preclosure']['sim_s']))
    displacement=np.linalg.norm(blocks[:,0,:2]-blocks[0,0,:2],axis=-1)*1000

    def event(index):
        delta=(tcp[index,arm]-blocks[index,0])*1000
        original_delta=(tcp[index,arm]-blocks[0,0])*1000
        return dict(sim_s=float(times[index]),
            red_xy_displacement_mm=float(displacement[index]),
            tcp_to_current_red_xy_mm=float(np.linalg.norm(delta[:2])),
            tcp_to_initial_red_xy_mm=float(np.linalg.norm(original_delta[:2])),
            tcp_minus_current_red_xyz_mm=delta.tolist(),
            tcp_minus_initial_red_xyz_mm=original_delta.tolist(),
            red_displacement_xyz_mm=((blocks[index,0]-blocks[0,0])*1000).tolist(),
            first_closing_arm_drive=float(drive[index,arm]),
            both_arm_drive_targets=drive[index].tolist(),
            red_moving_finger_contacts=contacts[index-1,0].tolist() if index>0 else None)

    movements={}
    for threshold in (2,5,10,20,50):
        selected=np.flatnonzero(displacement[:end+1] >= threshold)
        movements[str(threshold)]=event(int(selected[0])) if len(selected) else None
    fingers={}
    for a,label in enumerate(('left','right')):
        for f in (0,1):
            # Contacts row i corresponds to pose i+1.
            selected=np.flatnonzero(contacts[:end,0,a,f])
            fingers[f'{label}_finger_{f}']=event(int(selected[0])+1) if len(selected) else None
    any_contact=np.flatnonzero(contacts[:end,0].any(axis=(1,2)))
    first=int(any_contact[0])+1 if len(any_contact) else None
    report=dict(seed=case['seed'],checkpoint=case['policy'],source_case=str(directory.resolve()),
        scored_result=case['result'],first_closing_arm=arm,
        scope='initialization through last drive>=95% before first closure',
        at_preclosure=event(end),
        first_moving_finger_contact=event(first) if first is not None else None,
        just_before_first_moving_finger_contact=event(first-1) if first is not None else None,
        first_contact_per_finger=fingers,first_displacement_threshold_events_mm=movements,
        cube_edge_mm=precision['cube_edge_mm'],
        limitations=['Contact trace only covers moving finger links; palm and other-link contacts are not recorded.',
            'TCP vertical offset is not fingertip clearance; no claim of fingertip height follows from TCP z alone.',
            'Motion thresholds identify physical displacement, not an independently inferred causal force.'])
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    reports=[inspect_case(p) for p in (args.reference,args.candidate)]
    assert reports[0]['seed']==reports[1]['seed']
    sources=[json.loads((p/'result.json').read_text()) for p in (args.reference,args.candidate)]
    assert sources[0]['initial_scene_audit']['initial_rgb_sha256']==sources[1]['initial_scene_audit']['initial_rgb_sha256']
    report=dict(state='paired_preclosure_push_measured',records=reports,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scene_initial_rgb_exact=True,
        note='Post-hoc diagnostic on an existing development failure; not additional independent evaluation episodes.')
    with args.output.open('x') as stream:
        stream.write(json.dumps(report,indent=2,allow_nan=False)+'\n')
    for record in reports:
        print(json.dumps({k:record[k] for k in ('checkpoint','at_preclosure','first_moving_finger_contact')}),flush=True)


if __name__=='__main__':
    main()
