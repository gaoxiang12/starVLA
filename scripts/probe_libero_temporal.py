"""Read-only temporal diagnostics on fixed contiguous LIBERO demonstration clips."""
import argparse,json,sys,hashlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--steps',default='80000,160000');p.add_argument('--out',type=Path,required=True);p.add_argument('--manifest',type=Path);args=p.parse_args()
run=args.run.resolve();out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(run/'source_snapshot'))
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from starVLA.model.framework.WM4A.GAWM import GAWM
from starVLA.training.recipe import prepare_parameter_precision
from starVLA.dataloader.gr00t_lerobot.video import get_frames_by_timestamps
from deployment.model_server.policy_norm_processor import PolicyNormProcessor

torch.set_num_threads(2);torch.manual_seed(42);torch.backends.mha.set_fastpath_enabled(False)
cfg=OmegaConf.load(run/'config.full.yaml');root=Path(cfg.datasets.vla_data.data_root_dir)
if args.manifest:
    manifest=json.loads(args.manifest.read_text())
else:
    clips=[]
    for suite in ['spatial','object','goal','10']:
        name=f'libero_{suite}_no_noops_1.0.0_lerobot';path=root/name;seen=set()
        for line in (path/'meta/episodes.jsonl').read_text().splitlines():
            ep=json.loads(line);task=ep['tasks'][0]
            if task in seen or ep['length']<40 or (suite=='goal' and ep['episode_index']==82):continue
            seen.add(task);eid=ep['episode_index'];df=pd.read_parquet(path/f'data/chunk-{eid//1000:03d}/episode_{eid:06d}.parquet')
            gripper=np.stack(df['action'])[:,-1];events=np.flatnonzero(np.abs(np.diff(gripper))>.5)+1
            length=min(96,len(df)-8);center=int(events[0]) if len(events) else len(df)//2
            start=max(0,min(center-length//2,len(df)-length-8))
            clips.append(dict(dataset=name,episode=eid,task=task,start=start,length=length,suite=suite,group='calibration' if len(seen)%2 else 'report'))
        assert len(seen)==10,(suite,len(seen))
    manifest=dict(clips=clips,scope='40 fixed training-demo clips, NOT held-out trajectories. Recorded 20 Hz timestamps; no-op deletion may remove original physical intervals.',camera_order=['observation.images.image','observation.images.wrist_image'])
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')

def smooth_stats(x,mask=None):
    # Scale-free temporal roughness: second-difference RMS / first-difference RMS.
    x=x.float();dx=x[1:]-x[:-1];dd=dx[1:]-dx[:-1]
    one=dx.square().flatten(1).mean(1);two=dd.square().flatten(1).mean(1)
    if mask is not None:two=two[mask]
    speed=one.mean().sqrt();acc=two.mean().sqrt() if len(two) else x.new_tensor(float('nan'))
    return dict(delta_rms=float(speed),curvature_rms=float(acc),roughness=float(acc/speed.clamp_min(1e-8)),across_time_std=float(x.var(0,unbiased=False).mean().sqrt()))

for step in map(int,args.steps.split(',')):
    model=GAWM(cfg);prepare_parameter_precision(model,cfg)
    weights=torch.load(run/f'checkpoints/steps_{step}_pytorch_model.pt',map_location='cpu',weights_only=True);model.load_state_dict(weights,strict=True);del weights
    model.cuda().eval();reports=[]
    processor=PolicyNormProcessor(str(run/f"checkpoints/steps_{step}_pytorch_model.pt"),unnorm_key="franka")
    with torch.inference_mode():
      for j,clip in enumerate(manifest['clips']):
        path=root/clip['dataset'];eid=clip['episode'];df=pd.read_parquet(path/f'data/chunk-{eid//1000:03d}/episode_{eid:06d}.parquet')
        start=clip['start'];n=clip['length'];times=df['timestamp'].to_numpy()[start:start+n+8]
        assert np.all(np.diff(times)>0)
        videos=[get_frames_by_timestamps(str(path/f'videos/chunk-{eid//1000:03d}/{cam}/episode_{eid:06d}.mp4'),times,'torchvision_av',{'num_threads':1}) for cam in manifest['camera_order']]
        zs=[];teachers=[]
        for a in range(0,len(times),8):
            frames=[[[v[i] for v in videos]] for i in range(a,min(a+8,len(times)))]
            with torch.autocast('cuda',dtype=torch.bfloat16):feat,teacher=model.backbone.encode_patch_frames(frames,return_teacher=True)
            zs.append(model.visual_token_pooler(feat.float())[:,0]);teachers.append(teacher[:,0].float())
        z=torch.cat(zs);teacher=torch.cat(teachers);preds=[];losses=[];coarse=[];actions=[]
        states=torch.as_tensor(processor.apply_state(np.stack(df["observation.state"])[start:start+n]),device=z.device,dtype=torch.float32)
        for a in range(0,n,8):
            count=min(8,n-a);goal=model._condition_task_on_embodiment(model._embed_task([clip['task']]*count,z.device,robot_tag='franka'),'franka')
            pred=model.world_model.regress_future(z[a:a+count,None],goal);preds.append(pred)
            action=model._predict_action_chunk(model.action_models["franka"],torch.cat([z[a:a+count,None],pred],1),states[a:a+count])
            actions.append(action[:,0])
            target=torch.stack([teacher[a+4:a+count+4],teacher[a+8:a+count+8]],1)
            decoded=model.world_model.feature_decoder(pred)
            losses.append((1-torch.nn.functional.cosine_similarity(decoded.float(),target.float(),dim=-1)).flatten(1).mean(1))
            coarse.append((pred[:,1]-2*pred[:,0]+z[a:a+count]).square().flatten(1).mean(1))
        pred=torch.cat(preds);grip=np.stack(df['action'])[start:start+n,-1];events=np.abs(np.diff(grip))>.5
        near=np.zeros(n,dtype=bool)
        for e in np.flatnonzero(events)+1:near[max(0,e-2):min(n,e+3)]=True
        normal=torch.tensor(~near[1:-1],device=z.device)
        teacher_unit=torch.nn.functional.normalize(teacher[:n],dim=-1)
        latent_unit=torch.nn.functional.normalize(z[:n],dim=-1)
        row={**clip,'timestamp_dt_min':float(np.diff(times).min()),'timestamp_dt_max':float(np.diff(times).max()),'gripper_switches':int(events.sum()),
             'current':smooth_stats(latent_unit),'teacher':smooth_stats(teacher_unit),
             'current_away_gripper':smooth_stats(latent_unit,normal),
             'pred_h1':smooth_stats(torch.nn.functional.normalize(pred[:,0],dim=-1)),
             'pred_h2':smooth_stats(torch.nn.functional.normalize(pred[:,1],dim=-1)),
             'coarse_prediction_curvature_rms':float(torch.cat(coarse).mean().sqrt()),
             'realized_coarse_curvature_rms':float((z[8:n+8]-2*z[4:n+4]+z[:n]).square().mean().sqrt()),
             'same_target_reprediction_rms':float((pred[:-4,1]-pred[4:,0]).square().mean().sqrt()),
             'dino_future_loss':float(torch.cat(losses).mean())}
        action=torch.cat(actions)
        physical=processor.unapply_actions(action.cpu().numpy())
        pred_sign=physical[:,-1]>0
        predicted_events=np.flatnonzero(pred_sign[1:]!=pred_sign[:-1])+1
        signed_delays=[]
        for event in np.flatnonzero(events)+1:
            direction=np.sign(grip[event]-grip[event-1])
            same=[e for e in predicted_events if np.sign(int(pred_sign[e])-int(pred_sign[e-1]))==direction and abs(e-event)<=8]
            if same:signed_delays.append(int(min(same,key=lambda e:abs(e-event))-event))
        row.update(action_first_continuous=smooth_stats(action[:,:6]),
            gripper_prediction_switches=int(len(predicted_events)),gripper_matched_events=len(signed_delays),
            gripper_signed_delays_frames=signed_delays,
            gripper_sign_accuracy=float(np.mean(pred_sign==(grip>0))))
        reports.append(row)
        np.savez_compressed(out/f'step{step}_clip{j:02d}.npz',latent_delta=(z[1:n]-z[:n-1]).square().mean((1,2)).sqrt().cpu().numpy(),latent_curvature=(z[2:n]-2*z[1:n-1]+z[:n-2]).square().mean((1,2)).sqrt().cpu().numpy(),gripper_switch=events,times=times[:n])
        print(json.dumps({'step':step,'clip':j+1,'current_roughness':row['current']['roughness'],'pred_roughness':row['pred_h1']['roughness']}),flush=True)
    summary={}
    for name in ('current','teacher','current_away_gripper','pred_h1','pred_h2','action_first_continuous'):
        summary[name]={k:float(np.nanmean([r[name][k] for r in reports])) for k in reports[0][name]}
    for k in ('coarse_prediction_curvature_rms','realized_coarse_curvature_rms','same_target_reprediction_rms','dino_future_loss'):
        summary[k]=float(np.mean([r[k] for r in reports]))
    delays=[x for r in reports for x in r['gripper_signed_delays_frames']]
    summary['gripper_event_proxy']=dict(true_switches=sum(r['gripper_switches'] for r in reports),
        predicted_switches=sum(r['gripper_prediction_switches'] for r in reports),matched_events=len(delays),
        mean_signed_delay_frames=float(np.mean(delays)) if delays else None,
        median_abs_delay_frames=float(np.median(np.abs(delays))) if delays else None,
        sign_accuracy=float(np.mean([r['gripper_sign_accuracy'] for r in reports])))
    (out/f'step{step}.json').write_text(json.dumps(dict(step=step,run=str(run),summary=summary,clips=reports,scope=manifest['scope']),indent=2)+'\n')
    print('SUMMARY',step,json.dumps(summary),flush=True)
    del model;torch.cuda.empty_cache()
