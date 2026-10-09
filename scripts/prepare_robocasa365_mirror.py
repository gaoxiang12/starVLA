"""Audit verified RoboCasa365 mirrors and prepare lossless training views.

Raw snapshots remain unchanged. Video/parquet payloads are hard-linked; only
metadata, statistics and training indices are generated in a separate directory.
"""
import argparse,hashlib,json,os,pickle,shutil,sys,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import av

ROOT=Path(__file__).resolve().parents[1]
CAMERAS=['robot0_agentview_left','robot0_agentview_right','robot0_eye_in_hand']

def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)

def audit_prepare(source,destination,expected_tasks):
    proof=json.loads((source/'DOWNLOAD_VERIFIED.json').read_text())
    info=json.loads((source/'meta/info.json').read_text())
    assert info['codebase_version']=='v3.0' and info['fps']==20
    assert info['features']['action']['shape']==[12] and info['features']['observation.state']['shape']==[16]
    episodes=pd.concat([pd.read_parquet(p) for p in sorted((source/'meta/episodes').glob('*/*.parquet'))],ignore_index=True)
    tasks=pd.read_parquet(source/'meta/tasks.parquet').reset_index()
    if 'task' not in tasks:tasks=tasks.rename(columns={tasks.columns[0]:'task'})
    task_text=dict(zip(tasks['task_index'].astype(int),tasks['task']))
    assert len(episodes)==info['total_episodes'] and episodes.episode_index.is_unique
    assert int(episodes.length.sum())==info['total_frames']
    by_id=episodes.set_index('episode_index');classes=sorted({Path(p).parts[-2] for p in episodes.source_prefix})
    assert len(classes)==expected_tasks,(classes,expected_tasks)
    destination.mkdir(parents=True,exist_ok=True)
    # Never mutate shared payloads; metadata is copied, and all checks precede readiness.
    for path in source.rglob('*'):
        rel=path.relative_to(source)
        if not path.is_file() or rel.parts[0] not in ('meta','data','videos'):continue
        if path.suffix not in ('.json','.parquet','.mp4'):continue
        target=destination/rel;target.parent.mkdir(parents=True,exist_ok=True)
        if rel.parts[0]=='meta':shutil.copy2(path,target)
        elif not target.exists():os.link(path,target)
        elif not os.path.samefile(path,target):raise ValueError(f'Unexpected existing payload: {target}')
    modality_path=ROOT/'playground/Code/robocasa365/robocasa/models/assets/groot_dataset_assets/PandaOmron_modality.json'
    shutil.copy2(modality_path,destination/'meta/modality.json')
    total=info['total_frames'];work=destination/'meta/audit_work';work.mkdir(exist_ok=True)
    values={k:np.memmap(work/(k.replace('.','_')+'.f64'),mode='w+',dtype=np.float64,shape=(total,dim)) for k,dim in [('action',12),('observation.state',16)]}
    seen=set();offset=0
    for path in sorted((source/'data').glob('*/*.parquet')):
        df=pd.read_parquet(path);n=len(df)
        assert offset+n<=total
        for key,dim in [('action',12),('observation.state',16)]:
            x=np.stack(df[key]);assert x.shape==(n,dim) and np.isfinite(x).all(),(path,key)
            values[key][offset:offset+n]=x
        for eid,ep in df.groupby('episode_index',sort=False):
            eid=int(eid);assert eid not in seen;seen.add(eid);meta=by_id.loc[eid]
            assert len(ep)==int(meta.length)
            expected=source/info['data_path'].format(chunk_index=int(meta['data/chunk_index']),file_index=int(meta['data/file_index']))
            assert path==expected
            np.testing.assert_array_equal(ep.frame_index.to_numpy(),np.arange(len(ep)))
            np.testing.assert_allclose(ep.timestamp.to_numpy(),np.arange(len(ep))/20,atol=2e-3,rtol=0)
            langs=np.asarray(ep['annotation.human.task_description'].tolist()).reshape(-1)
            assert all(int(i) in task_text and isinstance(task_text[int(i)],str) and task_text[int(i)].strip() for i in np.unique(langs))
        offset+=n
    assert offset==total and seen==set(map(int,episodes.episode_index))
    statistics={}
    for key,x in values.items():
        statistics[key]={name:[] for name in ('mean','std','min','max','q01','q99')}
        for i in range(x.shape[1]):
            col=np.asarray(x[:,i]);quant=np.quantile(col,[.01,.99])
            for name,v in [('mean',col.mean()),('std',col.std()),('min',col.min()),('max',col.max()),('q01',quant[0]),('q99',quant[1])]:statistics[key][name].append(float(v))
        x.flush()
    values.clear()
    for p in work.iterdir():p.unlink()
    work.rmdir()
    save(destination/'meta/stats.json',statistics)
    save(destination/'meta/stats_gr00t.json',dict(__format_version=2,__cache_config={'mode':'abs'},statistics=statistics))
    # Check every episode's per-camera video coordinates against the actual stream.
    streams={}
    for row in episodes.to_dict('records'):
        for cam in CAMERAS:
            key='observation.images.'+cam;prefix='videos/'+key
            path=source/info['video_path'].format(video_key=key,chunk_index=int(row[prefix+'/chunk_index']),file_index=int(row[prefix+'/file_index']))
            if str(path) not in streams:
                with av.open(str(path)) as container:
                    stream=container.streams.video[0]
                    assert (stream.width,stream.height)==(256,256)
                    assert abs(float(stream.average_rate)-20)<1e-4
                    duration=float(stream.duration*stream.time_base) if stream.duration is not None else float(container.duration/av.time_base)
                    streams[str(path)]=dict(frames=stream.frames,duration=duration)
            start=float(row[prefix+'/from_timestamp']);end=float(row[prefix+'/to_timestamp'])
            assert 0<=start<end<=streams[str(path)]['duration']+.051,(path,start,end)
            assert abs((end-start)*20-int(row['length']))<.02,(path,row['episode_index'])
    # Precomputed index is safe only because all episode lengths and languages were audited.
    steps=[(int(row.episode_index),i) for row in episodes.itertuples() for i in range(int(row.length))]
    config={'delete_pause_frame':False,'dataset_name':destination.name}
    key=hashlib.md5(str(sorted(config.items())).encode()).hexdigest()[:12]
    with (destination/'meta/steps_data_index.pkl').open('wb') as f:pickle.dump(dict(config_key=key,steps=steps,num_trajectories=len(episodes),total_steps=total,delete_pause_frame=False),f,protocol=pickle.HIGHEST_PROTOCOL)
    del steps
    # Exercise the actual training loader, normalization, camera order, and tail masks.
    from omegaconf import OmegaConf
    from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset
    from examples.Robocasa_365.train_files.data_registry.data_config import PandaOmronRoboCasa365MirrorWMDataConfig
    cfg=OmegaConf.create(dict(lerobot_version='v3.0',action_mode='abs',include_state=True,future_obs_frames=True,future_obs_valid_mask=True,action_valid_mask=True,target_num_views=3,strict_camera_views=True,packed_image_size=[256,256],packed_numeric_dtype='float32',image_resize_resample='opencv_linear',task_language_mode='metadata'))
    spec=PandaOmronRoboCasa365MirrorWMDataConfig()
    ds=LeRobotSingleDataset(destination,spec.modality_config(),spec.embodiment_tag,video_backend='torchvision_av',video_backend_kwargs={'num_threads':1},transforms=spec.transform(),data_cfg=cfg,task_language_mode='metadata')
    assert len(ds)==total
    samples=[];done=set();begin=0
    action_order=[5,6,7,8,9,10,11,0,1,2,3,4]
    low=np.array(statistics['action']['min'])[action_order];high=np.array(statistics['action']['max'])[action_order]
    for row in episodes.itertuples():
        task=Path(row.source_prefix).parts[-2]
        if task not in done:
            done.add(task);anchor=min(8,int(row.length)-1);sample=ds[begin+anchor]
            assert len(sample['image'])==3 and len(sample['future_images'])==2
            assert np.asarray(sample['state']).shape==(1,16) and sample['action'].shape==(16,12)
            assert sample['view_valid_mask']==[True]*3 and all(im.size==(256,256) for im in sample['image'])
            raw=ds.curr_traj_data.iloc[anchor]['action'];raw=np.asarray(raw)[action_order]
            span=high-low;expected=np.where(span>1e-6,2*(raw-low)/np.where(span>1e-6,span,1)-1,0)
            # The repository transform clips values after normalization; this checks semantic order.
            np.testing.assert_allclose(sample['action'][0],np.clip(expected,-1,1),atol=1e-5)
            assert sample['lang'].strip() and np.isfinite(sample['state']).all()
            samples.append(dict(task=task,episode=int(row.episode_index),anchor=anchor,language=sample['lang']))
        begin+=int(row.length)
    tail=ds[len(ds)-1];assert np.asarray(tail['action_valid_mask']).sum()==1
    assert list(tail['future_frame_valid_mask'])==[True,False,False]
    report=dict(status='passed',source=str(source),destination=str(destination),revision=proof['revision'],episodes=len(episodes),frames=total,tasks=classes,physical_cameras=CAMERAS,state_dim=16,action_dim=12,recorded_fps=20,future_recorded_offsets=[0,8,16],future_time_offsets_s=[0.,.4,.8],video_files=len(streams),sample_checks=samples,video_reencoded=False,normalization_statistics='computed from all training frames',action_storage_order='base4,mode1,eef_position3,eef_rotation3,gripper1',action_model_order='eef_position3,eef_rotation3,gripper1,base4,mode1',raw_snapshots_unchanged=True)
    save(destination/'PREPARATION_COMPLETE.json',report);return report

def extract_assets(source):
    proof=json.loads((source/'DOWNLOAD_VERIFIED.json').read_text())
    base=ROOT/'playground/Code/robocasa365/robocasa/models/assets'
    destinations={'textures.zip':base,'generative_textures.zip':base,'fixtures.zip':base,'aigen_objs.zip':base/'objects','objaverse.zip':base/'objects','lightwheel.zip':base/'objects'}
    extracted=[]
    for filename,dest in destinations.items():
        dest.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(source/filename) as archive:
            for info in archive.infolist():
                path=Path(info.filename)
                if path.is_absolute() or '..' in path.parts or (info.external_attr>>16)&0o170000==0o120000:raise ValueError('Unsafe zip member')
                target=dest/path
                if target.exists() and target.is_file() and not info.is_dir():
                    # Do not silently overwrite checked source files.
                    existing=target.read_bytes();incoming=archive.read(info)
                    if existing!=incoming:raise ValueError(f'Asset conflicts with source: {target}')
                    continue
                archive.extract(info,dest)
        extracted.append(filename)
    save(ROOT/'.cache/robocasa_setup/assets_verified.json',dict(status='extracted',revision=proof['revision'],archives=extracted))

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--destination',type=Path);p.add_argument('--expected-tasks',type=int);p.add_argument('--assets',action='store_true');args=p.parse_args()
    if args.assets:extract_assets(args.source)
    else:
        if not args.destination or not args.expected_tasks:p.error('Dataset preparation requires destination and expected-tasks')
        audit_prepare(args.source,args.destination,args.expected_tasks)
if __name__=='__main__':main()
