import json
from pathlib import Path
import av
import numpy as np
import pandas as pd
from scripts.prepare_robocasa365_mirror import audit_prepare, CAMERAS


def test_v3_preparation_preserves_video_and_reorders_actions(tmp_path):
    source=tmp_path/'source';dest=tmp_path/'prepared'
    (source/'meta/episodes/chunk-000').mkdir(parents=True)
    (source/'data/chunk-000').mkdir(parents=True)
    frames=40;length=20
    features={'action':{'dtype':'float64','shape':[12]},'observation.state':{'dtype':'float64','shape':[16]}}
    rows=[]
    for eid in range(2):
        row={'episode_index':eid,'length':length,'tasks':['open drawer'],'source_prefix':'target/atomic/OpenDrawer/20250816','data/chunk_index':0,'data/file_index':0}
        for cam in CAMERAS:
            key='observation.images.'+cam;prefix='videos/'+key
            row.update({prefix+'/chunk_index':0,prefix+'/file_index':0,prefix+'/from_timestamp':eid*length/20,prefix+'/to_timestamp':(eid+1)*length/20})
        rows.append(row)
    for view,cam in enumerate(CAMERAS):
        key='observation.images.'+cam
        features[key]={'dtype':'video','shape':[256,256,3],'names':['height','width','channel'],'video_info':{'video.fps':20}}
        path=source/f'videos/{key}/chunk-000/file-000.mp4';path.parent.mkdir(parents=True)
        with av.open(str(path),'w') as output:
            stream=output.add_stream('libx264',rate=20);stream.width=256;stream.height=256;stream.pix_fmt='yuv420p'
            for t in range(frames):
                image=np.full((256,256,3),(30+view*70,40+t,100),np.uint8)
                for packet in stream.encode(av.VideoFrame.from_ndarray(image,format='rgb24')):output.mux(packet)
            for packet in stream.encode():output.mux(packet)
    info=dict(codebase_version='v3.0',fps=20,chunks_size=1000,total_episodes=2,total_frames=frames,total_tasks=1,features=features,data_path='data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet',video_path='videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4')
    (source/'meta/info.json').write_text(json.dumps(info));(source/'meta/stats.json').write_text('{}')
    pd.DataFrame(rows).to_parquet(source/'meta/episodes/chunk-000/file-000.parquet')
    pd.DataFrame({'task_index':[0]},index=pd.Index(['open drawer'],name='task')).to_parquet(source/'meta/tasks.parquet')
    action=np.arange(frames)[:,None]*np.arange(1,13)[None,:]/100;action[:,0]=0
    state=np.arange(frames)[:,None]*np.arange(1,17)[None,:]/100
    pd.DataFrame({'episode_index':np.repeat([0,1],length),'frame_index':np.tile(np.arange(length),2),'timestamp':np.tile(np.arange(length)/20,2),'action':list(action),'observation.state':list(state),'annotation.human.task_description':np.zeros(frames,dtype=np.int64)}).to_parquet(source/'data/chunk-000/file-000.parquet')
    (source/'DOWNLOAD_VERIFIED.json').write_text(json.dumps({'revision':'synthetic-test'}))
    result=audit_prepare(source,dest,1)
    assert result['status']=='passed' and result['frames']==frames
    assert (source/'meta/stats.json').read_text()=='{}'
    assert not (source/'meta/modality.json').exists()
    assert (source/'data/chunk-000/file-000.parquet').samefile(dest/'data/chunk-000/file-000.parquet')

    # Second episode shares the physical video with episode zero. Its local
    # timestamp must be offset by one second before decoding all three views.
    from omegaconf import OmegaConf
    from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset
    from examples.Robocasa_365.train_files.data_registry.data_config import PandaOmronRoboCasa365MirrorWMDataConfig
    spec=PandaOmronRoboCasa365MirrorWMDataConfig()
    cfg=OmegaConf.create(dict(lerobot_version='v3.0',include_state=True,future_obs_frames=True,
        future_obs_valid_mask=True,action_valid_mask=True,packed_image_size=[256,256],
        packed_numeric_dtype='float32',target_num_views=3,strict_camera_views=True))
    ds=LeRobotSingleDataset(dest,spec.modality_config(),spec.embodiment_tag,
        transforms=spec.transform(),data_cfg=cfg,video_backend='torchvision_av',
        video_backend_kwargs={'num_threads':1})
    sample=ds[28]
    for views,frame in zip([sample['image'],*sample['future_images']],[28,36,39]):
        for view,image in enumerate(views):
            np.testing.assert_allclose(np.asarray(image)[128,128],[30+view*70,40+frame,100],atol=4)
    assert sample['state'].shape==(1,16)
    assert sample['future_frame_valid_mask'].tolist()==[True,True,False]
