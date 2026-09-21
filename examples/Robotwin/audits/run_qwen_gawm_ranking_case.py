"""One fixed full-sorting scene using the unmodified RoboTwin success predicate."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import av
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(ROOT/'examples/Robotwin/eval_files')]
from examples.Robotwin.audits.smoke_rgb_recovery_teacher import load_task_config
from examples.Robotwin.audits.run_grasp_lift_case import ObservedScene, bottom_z, save
from ranking_outcome_diagnostics import rgb_arrangement
from robotwin_eval_runner import _load_robotwin_evaluator


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--port',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    out=args.output.resolve(); out.mkdir(parents=True,exist_ok=False)
    os.chdir(ROOT.parent/'RoboTwin')
    evaluator=_load_robotwin_evaluator(ROOT.parent/'RoboTwin')
    from test_render import Sapien_TEST
    from model2robotwin_interface import get_model, reset_model, eval as policy_step
    Sapien_TEST()
    env=model=container=stream=scene=None
    frames=0
    report=dict(state='starting',seed=args.seed,checkpoint=str(args.checkpoint.resolve()),
                training_eligible=False,scoring='Unmodified blocks_ranking_rgb.check_success',execute_horizon=16)
    trace=[]
    try:
        cfg,_=load_task_config(evaluator,out)
        env=evaluator.class_decorator('blocks_ranking_rgb')
        env.setup_demo(now_ep_num=0,seed=args.seed,is_test=True,**cfg)
        blocks=[env.block1,env.block2,env.block3]
        obs=env.get_obs()
        report['initial']=dict(positions=[b.get_pose().p.tolist() for b in blocks],
            quaternions=[b.get_pose().q.tolist() for b in blocks],
            extents=[list(b.config['extents']) for b in blocks],
            rgb_sha256=[hashlib.sha256(obs['observation'][v]['rgb'].tobytes()).hexdigest()
                        for v in ('head_camera','left_camera','right_camera')])
        # Keep the earlier visual duplicate check, because seed disjointness alone is insufficient.
        from examples.Robotwin.audits.audit_rgb_scene_and_color import colors
        current=colors(obs['observation']['head_camera']['rgb'][...,::-1])
        if set(current)!={'red','green','blue'}:
            raise RuntimeError('Cannot audit initial scene colors')
        audit=json.loads((ROOT/'examples/Robotwin/audits/rgb_scene_and_color_20260907.json').read_text())
        matches=[]
        for row in audit['episodes_detail']:
            ref=row.get('initial',{})
            if set(ref)!=set(current):continue
            if max(np.linalg.norm(np.asarray(current[c]['xy'])-ref[c]['xy']) for c in current)<=1.5 and all(
                .8<=current[c]['area']/ref[c]['area']<=1.25 for c in current):matches.append(row['episode'])
        report['training_scene_matches']=matches
        if matches:raise RuntimeError(f'Evaluation scene matches dataset episodes: {matches}')
        env.set_instruction('blocks ranking rgb')
        env.eval_success=False; env.take_action_cnt=0
        assert env.step_lim==1200
        report['step_limit']=env.step_lim
        model=get_model(dict(policy_ckpt_path=str(args.checkpoint.resolve()),host='127.0.0.1',
            port=args.port,unnorm_key='aloha',action_mode='abs',execute_horizon=16))
        assert Path(model.client.get_server_metadata()['ckpt_path']).resolve()==args.checkpoint.resolve()
        assert model.action_chunk_size==16 and model.execute_horizon==16
        reset_model(model)
        scene=env.scene
        initial_bottom=np.array([bottom_z(b) for b in blocks])
        max_lift=np.zeros(3); sim_time=0.
        contacts=np.zeros((3,2,2),bool)
        held=np.zeros(3,bool); held_seconds=np.zeros((3,2))
        first_hold_order=[]
        finger_lookup={int(j.child_link.entity.per_scene_id):(a,f) for a,group in enumerate(
            (env.robot.left_gripper,env.robot.right_gripper)) for f,(j,_,_) in enumerate(group)}
        block_lookup={int(b.actor.per_scene_id):i for i,b in enumerate(blocks)}

        def observe_physics():
            nonlocal sim_time, max_lift, held_seconds
            dt=float(scene.get_timestep()); sim_time+=dt
            lift=np.array([bottom_z(b) for b in blocks])-initial_bottom
            max_lift=np.maximum(max_lift,lift); contacts.fill(False)
            for contact in scene.get_contacts():
                ids=[int(body.entity.per_scene_id) for body in contact.bodies]
                for b,f in (ids,ids[::-1]):
                    if b in block_lookup and f in finger_lookup and any(p.separation<=.001 for p in contact.points):
                        a,fi=finger_lookup[f]; contacts[block_lookup[b],a,fi]=True
            held_seconds=np.where(contacts.all(-1)&(lift[:,None]>=.05),held_seconds+dt,0.)
            for i in np.flatnonzero((held_seconds>=1.).any(-1)&~held):
                held[i]=True; first_hold_order.append(['red','green','blue'][int(i)])

        env.scene=ObservedScene(scene,observe_physics)
        raw_take=env.take_action
        def record_action(action, action_type='qpos'):
            if not np.isfinite(action).all():raise ValueError('Nonfinite policy action')
            trace.append(dict(step=int(env.take_action_cnt),sim_s=sim_time,action=np.asarray(action).tolist(),
                tcp=[env.robot.get_left_tcp_pose(),env.robot.get_right_tcp_pose()],
                block_positions=[b.get_pose().p.tolist() for b in blocks]))
            return raw_take(action,action_type=action_type)
        env.take_action=record_action
        def frame(obs):
            nonlocal container,stream,frames
            rgb=np.concatenate([obs['observation'][v]['rgb'] for v in ('head_camera','left_camera','right_camera')],axis=1)
            if container is None:
                container=av.open(str(out/'rollout.mp4'),'w'); stream=container.add_stream('libx264',rate=20)
                stream.width=rgb.shape[1];stream.height=rgb.shape[0];stream.pix_fmt='yuv420p'
                stream.options={'crf':'22','preset':'veryfast','threads':'2'}
            for packet in stream.encode(av.VideoFrame.from_ndarray(rgb,format='rgb24')):container.mux(packet)
            frames+=1
        frame(obs)
        while env.take_action_cnt<env.step_lim and not env.eval_success:
            policy_step(env,model,obs)
            obs=env.get_obs(); frame(obs)
            if env.take_action_cnt%16==0:
                save(out/'status.json',dict(report,state='running',actions=int(env.take_action_cnt),sim_s=sim_time))
        arrangement=rgb_arrangement([b.get_pose().p for b in blocks],env.is_left_gripper_open(),env.is_right_gripper_open())
        success=bool(env.eval_success)
        assert success==bool(env.check_success())==arrangement['all_success_predicates']
        report.update(state='complete',success=success,actions=int(env.take_action_cnt),sim_s=sim_time,
            termination='success' if success else 'action_budget',maximum_block_lift_m=max_lift.tolist(),
            held_red_green_blue=held.tolist(),first_hold_order=first_hold_order,arrangement=arrangement,
            frames=frames,diagnostic_note='Lift/contact metrics are passive and do not terminate the original task.')
        save(out/'result.json',report)
    except BaseException as error:
        report.update(state='failed',error=repr(error));raise
    finally:
        save(out/'status.json',report)
        save(out/'action_trace.json',trace)
        if container is not None:
            for packet in stream.encode():container.mux(packet)
            container.close()
        if model is not None:model.client.close()
        if env is not None:
            if scene is not None:env.scene=scene
            env.close_env(clear_cache=True)


if __name__=='__main__':main()
