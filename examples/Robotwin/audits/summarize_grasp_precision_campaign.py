"""Separate target alignment from action-end tracking in recorded grasp campaigns."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import torch
import yaml

from examples.Robotwin.audits.measure_grasp_precision import bin_mm, summarize
from examples.Robotwin.audits.run_grasp_lift_development import digest
from starVLA.model.modules.robotwin_kinematics import SerialTCPKinematics


def stats(values):
    values = np.asarray(values)
    if not len(values):
        return dict(count=0,median=None,p90=None,maximum=None)
    assert np.isfinite(values).all()
    return dict(count=len(values),median=float(np.median(values)),
        p90=float(np.quantile(values,.9)),maximum=float(values.max()))


def tracking(row):
    directory = Path(row['source_case'])
    case = json.loads((directory/'result.json').read_text())
    actions = json.loads((directory/'action_trace.json').read_text())
    poses = np.load(directory/'pregrasp_physics_poses.npz', allow_pickle=False)
    config_path = Path('/data/gaoxiang/Code/RoboTwin/assets/embodiments/aloha-agilex/config.yml')
    config = yaml.safe_load(config_path.read_text())
    assert len(config['robot_pose']) == 1 and config['gripper_bias'] == .12
    root_pose = np.asarray(config['robot_pose'][0])
    rotation = Rotation.from_quat(root_pose[[4,5,6,3]]).as_matrix()
    urdf = (config_path.parent/config['urdf_path']).resolve()
    arm = row['first_closing_arm']
    if arm is None:
        arm = ('left','right').index(case['initial_target_side'])
    prefix = ('fl','fr')[arm]
    chain = SerialTCPKinematics(urdf,f'{prefix}_link6',[f'{prefix}_joint{i}' for i in range(1,7)])
    commands = np.asarray([a['requested_robot_action'] for a in actions])
    times = poses['sim_s']
    action_times = np.asarray([a['sim_s'] for a in actions])
    indices = np.searchsorted(times,action_times)
    np.testing.assert_allclose(times[indices],action_times,atol=1e-9,rtol=0)
    tcp = poses['tcp_world_matrix'][:,arm,:3,3]
    blocks = poses['block_world_matrix'][:,0,:3,3]
    names = list(poses['active_joint_names'][0])
    columns = [names.index(f'{prefix}_joint{i}') for i in range(1,7)]
    actual_q = poses['articulation_qpos'][indices,0][:,columns]
    selection = slice(0,6) if arm == 0 else slice(7,13)
    with torch.no_grad():
        fk = chain(torch.tensor(actual_q,dtype=torch.float64)).numpy() @ rotation.T+root_pose[:3]
        command_tcp = chain(torch.tensor(commands[:,selection],dtype=torch.float64)).numpy() @ rotation.T+root_pose[:3]
    fk_error = float(np.abs(fk-tcp[indices]).max())
    assert fk_error < 2e-6, 'Measured qpos FK does not match actual TCP'
    # The next observation follows execution of the current requested action.
    endpoint_error = np.linalg.norm(command_tcp[:-1]-tcp[indices[1:]],axis=-1)*1000
    if row['closing_window'] is not None:
        window = row['closing_window']
        selected = np.flatnonzero((action_times[:-1] >= window['start_sim_s']) &
                                 (action_times[:-1] <= window['end_sim_s']))
    else:
        selected = np.arange(len(actions)-1)
    command_xy = np.linalg.norm(command_tcp[:,:2]-blocks[indices,:2],axis=-1)*1000
    actual_xy = np.linalg.norm(tcp[indices,:2]-blocks[indices,:2],axis=-1)*1000
    records = [dict(action_index=int(i), sim_s=float(action_times[i]),
        requested_target_xy_to_current_red_mm=float(command_xy[i]),
        actual_tcp_xy_to_current_red_mm=float(actual_xy[i]),
        actual_action_end_to_requested_target_mm=float(endpoint_error[i])) for i in selected]
    return dict(arm=arm,scope='first_closure' if row['closing_window'] else 'full_rollout_no_closure_target_side_arm',
        action_end_tracking_error_mm=stats(endpoint_error[selected]),
        requested_target_xy_offset_mm=stats(command_xy[selected]),
        actual_tcp_xy_offset_mm=stats(actual_xy[selected]),records=records,
        measured_qpos_fk_max_difference_m=fk_error, urdf_sha256=digest(urdf),
        note='Tracking is measured after action execution, not during interpolation. '
             'Residual motion/contact can contribute; this is not a controller-identification experiment.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    status_bytes = (args.campaign/'status.json').read_bytes()
    status = json.loads(status_bytes)
    rows = status['results']
    assert rows
    report = summarize(rows)
    stage_bins = {}
    for threshold in ('0.95','0.8','0.5','0.2','0.05'):
        counts = {key:0 for key in ('le_10mm','10_to_20mm','20_to_30mm','gt_30mm','no_closure','not_reached')}
        for row in rows:
            event = row['drive_threshold_events'].get(threshold)
            key = 'no_closure' if not row['closing_command_observed'] else 'not_reached' if event is None else bin_mm(event['horizontal_offset_mm'])
            counts[key] += 1
        stage_bins[threshold] = counts
    records = [dict(precision=row,tracking=tracking(row)) for row in rows]
    report.update(state='complete' if status['state']=='complete' else 'partial_snapshot',
        checkpoint=rows[0]['checkpoint'],stage_episode_bins=stage_bins,records=records,
        source_status_sha256=hashlib.sha256(status_bytes).hexdigest(), source_status_time=status['time'],
        interpretation='Each threshold contains one measurement per episode. The same episode can improve or worsen during closure; do not interpret the 95% snapshot as final grasp alignment. Pending scenes are not successes or failures.')
    with args.output.open('x') as stream:
        stream.write(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='records'}),flush=True)


if __name__ == '__main__':
    main()
