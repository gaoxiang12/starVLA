"""Audit selected LeRobot data; derive train-only normalization and task VTTs.

No dataset conversion or separate training loader. Output assets live outside
the repository. Every selected episode is numerically checked; representative
episodes (deterministically selected per task) provide statistics and VTTs.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf
from transformers import AutoModel

from starVLA.dataloader.lerobot_datasets import (
    DATASET_NAMED_MIXTURES, make_LeRobotSingleDataset, episode_split_blacklist,
)
from starVLA.model.framework.WM4A.LiLaWAMTrain import task_key
from starVLA.task_language import configured_task_language_mode, resolve_task_language


def write_json(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False)); tmp.replace(path)


def modality_array(child, table, modality):
    arrays = []
    for key in child.modality_keys[modality]:
        field = getattr(child.lerobot_modality_meta, modality)[key.split('.', 1)[1]]
        arrays.append(np.stack(table[field.original_key or key.split('.', 1)[1]]).astype(np.float32)[:, field.start:field.end])
    return np.concatenate(arrays, axis=1)


def stats(array):
    return {k: v.tolist() for k,v in dict(min=array.min(0), max=array.max(0),
        mean=array.mean(0, dtype=np.float64), std=array.std(0, dtype=np.float64),
        q01=np.quantile(array,.01,axis=0), q99=np.quantile(array,.99,axis=0)).items()}


@torch.no_grad()
def cls_difference(encoder, frames, size, device):
    # Main-view VTT follows the official design; wrist views condition the DiT.
    imgs=[np.asarray(Image.fromarray(np.asarray(f)).resize(size, Image.Resampling.BILINEAR)).copy() for f in frames]
    x=torch.from_numpy(np.stack(imgs)).permute(0,3,1,2).to(device=device,dtype=torch.float32)/255
    x=(x-x.new_tensor([.485,.456,.406])[None,:,None,None])/x.new_tensor([.229,.224,.225])[None,:,None,None]
    out=encoder(pixel_values=x.to(next(encoder.parameters()).dtype)).last_hidden_state[:,0].float()
    return (out[1]-out[0]).cpu().numpy()


def prepare(config_path, device='cpu', max_episodes_per_task=20):
    cfg=OmegaConf.load(config_path); data=cfg.datasets.vla_data
    if data.get('episode_split') != 'train': raise ValueError('Preparation requires the training partition')
    if max_episodes_per_task and max_episodes_per_task < 20:
        raise ValueError('Use at least 20 train episodes per task, or 0 for all')
    model_cfg=cfg.framework.lila
    output=Path(model_cfg.task_vectors_path).parent
    output.mkdir(parents=True,exist_ok=True)
    audit={'format_version':1,'status':'running','config':str(Path(config_path).resolve()),
           'representative_episodes_per_task':max_episodes_per_task,'seed':int(cfg.seed),
           'statistics_split':'train','vtt_split':'train','future_offset_unit':'recorded_frame',
           'datasets':[], 'errors':[], 'metadata_repairs':[]}
    write_json(output/'preparation_audit.json',audit)
    children=[]; groups=defaultdict(list); hashes={}
    try:
        for name,weight,robot_type in DATASET_NAMED_MIXTURES[data.data_mix]:
            path=Path(data.data_root_dir)/name
            # Some existing v2.1 exports only ship episodes_stats.jsonl plus
            # the shared loader's aggregated raw-column cache. Add the missing
            # compatibility file without changing any trajectory or old file.
            stats_path=path/'meta/stats.json'
            source_stats=path/'meta/stats_gr00t.json'
            if not stats_path.exists() and source_stats.is_file():
                cached=json.loads(source_stats.read_text())
                raw=cached.get('statistics',cached)
                if not all(k in raw for k in ('action','observation.state')):
                    raise ValueError(f'{name}: cannot recover raw-column stats.json from cache')
                with stats_path.open('x') as handle: json.dump(raw,handle,indent=2)
                audit['metadata_repairs'].append(dict(path=str(stats_path),source=str(source_stats),
                    note='Existing aggregate metadata only; training uses separately recomputed train-only statistics.'))
            required=['info.json','episodes.jsonl','tasks.jsonl','modality.json','stats.json','stats_gr00t.json']
            missing=[f for f in required if not (path/'meta'/f).is_file()]
            if missing: raise ValueError(f'{name}: missing metadata {missing}')
            # Build/read a complete cache before applying the split. Avoid the
            # shared loader's first-use train-only cache hiding validation IDs.
            unfiltered=OmegaConf.create(OmegaConf.to_container(data,resolve=True))
            unfiltered.validation_episode_stride=0
            unfiltered.pop('episode_split_manifest',None)
            child=make_LeRobotSingleDataset(Path(data.data_root_dir),name,robot_type,data_cfg=unfiltered)
            excluded=set(episode_split_blacklist(path,data))
            selected=set(map(int,child.trajectory_ids))-excluded
            coverage={int(ep) for ep,_ in child.all_steps}
            if not selected <= coverage: raise ValueError(f'{name}: existing steps cache is incomplete')
            record={'name':name,'robot_type':robot_type,'train_episode_ids':sorted(selected),
                    'excluded_from_training':sorted(excluded),'checked_episodes':0,
                    'observed_views':len(child.modality_keys['video']),
                    'metadata_sha256':{f:hashlib.sha256((path/'meta'/f).read_bytes()).hexdigest() for f in required}}
            audit['datasets'].append(record); children.append(child)
            lengths=dict(zip(map(int,child.trajectory_ids),map(int,child.trajectory_lengths)))
            for ep in sorted(selected):
                table=child.get_trajectory_data(ep)
                if len(table)!=lengths[ep] or len(table)<2: raise ValueError(f'{name}/{ep}: length mismatch')
                if 'frame_index' in table and not np.array_equal(table.frame_index.to_numpy(),np.arange(len(table))):
                    raise ValueError(f'{name}/{ep}: nonsequential frame_index')
                child.curr_traj_data=table
                lang=child.get_language(ep,child.modality_keys['language'][0],0)[0]
                mode=configured_task_language_mode(data,child.tag)
                lang=resolve_task_language(lang,child.dataset_name,mode)
                key=task_key(child.tag,lang,data)
                action=modality_array(child,table,'action'); state=modality_array(child,table,'state')
                spec=cfg.framework.action_model.embodiment_heads[child.tag]
                if action.shape[1]!=spec.action_dim or state.shape[1]!=spec.state_dim:
                    raise ValueError(f'{name}/{ep}: state/action schema mismatch')
                if not np.isfinite(action).all() or not np.isfinite(state).all():
                    raise ValueError(f'{name}/{ep}: NaN/Inf')
                digest=hashlib.sha256(action.tobytes()+state.tobytes()).hexdigest()
                if digest in hashes: raise ValueError(f'Duplicate trajectory: {name}/{ep} and {hashes[digest]}')
                hashes[digest]=f'{name}/{ep}'
                for video_key in child.modality_keys['video']:
                    if not child.get_video_path(ep,video_key.removeprefix('video.')).is_file():
                        raise ValueError(f'{name}/{ep}: missing video {video_key}')
                groups[key].append((len(children)-1,ep,len(table),digest))
                record['checked_episodes']+=1
            write_json(output/'preparation_audit.json',audit)
        small={key:len(values) for key,values in groups.items() if len(values)<20}
        if small: raise ValueError(f'Tasks with fewer than 20 train episodes; explicitly exclude before training: {small}')
        encoder=AutoModel.from_pretrained(model_cfg.vision_encoder_path,local_files_only=True,
                    torch_dtype=torch.bfloat16 if device.startswith('cuda') else torch.float32).to(device).eval()
        encoder.requires_grad_(False)
        values=defaultdict(lambda:defaultdict(list)); vectors={}; provenance={}
        rng=np.random.default_rng(int(cfg.seed))
        for key,episodes in sorted(groups.items()):
            order=rng.permutation(len(episodes))
            chosen=[episodes[i] for i in order[:max_episodes_per_task or len(episodes)]]
            diffs=[]
            for child_id,ep,length,digest in chosen:
                child=children[child_id]; table=child.get_trajectory_data(ep);child.curr_traj_data=table
                for modality in ('action','state'): values[child.tag][modality].append(modality_array(child,table,modality))
                frames=[]
                for anchor in (0,length-1):
                    # Decode each actual camera at the endpoints as a video
                    # smoke check. Only primary-camera CLS differences form VTT.
                    for i,video_key in enumerate(child.modality_keys['video']):
                        images=child.get_video(ep,video_key,anchor)
                        if images.shape[0]<1: raise ValueError(f'{child.dataset_name}/{ep}: empty decoded video')
                        if i==0: frames.append(images[0])
                diffs.append(cls_difference(encoder,frames,tuple(model_cfg.image_size),device))
            vectors[key]=np.stack(diffs).mean(0).tolist()
            provenance[key]=[dict(dataset=children[c].dataset_name,episode=ep,frames=n,sha256=h) for c,ep,n,h in chosen]
            print(f'Prepared VTT {key}: {len(chosen)}/{len(episodes)} train episodes',flush=True)
        statistics={tag:{m:stats(np.concatenate(arrays)) for m,arrays in modalities.items()} for tag,modalities in values.items()}
        write_json(data.normalization_statistics_path,statistics)
        write_json(model_cfg.task_vectors_path,dict(format_version=1,vectors=vectors,
            encoder_path=str(model_cfg.vision_encoder_path),image_size=list(model_cfg.image_size),
            feature='mean(last_frame_CLS-first_frame_CLS), primary camera, last encoder layer',
            provenance=provenance))
        audit.update(status='prepared',num_tasks=len(vectors),
            quality_score=80,quality_scale=100,
            quality_reason='Train numeric/schema/dedup/path checks passed; all cameras decoded at representative episode endpoints. Full video decode and closed-loop validation remain separate checks.',
            normalization_statistics_path=str(data.normalization_statistics_path),task_vectors_path=str(model_cfg.task_vectors_path))
        write_json(output/'preparation_audit.json',audit)
    except Exception as error:
        audit['status']='failed';audit['errors'].append(repr(error))
        write_json(output/'preparation_audit.json',audit)
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--max-episodes-per-task',type=int,default=20)
    args=parser.parse_args();prepare(args.config,args.device,args.max_episodes_per_task)
