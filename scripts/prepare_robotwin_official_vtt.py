"""Validate official RoboTwin HDF5 and compute all-episode primary-view VTTs.

Independent task shards can run on separate GPUs. Atomic per-task outputs are
merged only after every audited task succeeds. Source HDF5 files are read-only.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import cv2
import h5py
import numpy as np
import torch
from transformers import AutoModel


def write_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pixels(image):
    if image.shape != (240,320,3) or image.dtype != np.uint8:
        raise ValueError(f'Expected physical RGB 320x240, got {image.shape}/{image.dtype}')
    x=image.astype(np.float32)/255.
    x=(x-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)
    return x.transpose(2,0,1).copy()


def read_episode(root, record):
    path=root/record['path'];before=path.stat()
    with h5py.File(path,'r') as f:
        action=f['joint_action/vector'][:]
        state=np.concatenate([f['endpose/left_endpose'][:],f['endpose/left_gripper'][:].reshape(-1,1),
                              f['endpose/right_endpose'][:],f['endpose/right_gripper'][:].reshape(-1,1)],axis=1)
        length=record['frames']
        if action.shape!=(length,14) or state.shape!=(length,16) or length<2:
            raise ValueError(f'Invalid state/action shapes: {path}')
        if not np.isfinite(action).all() or not np.isfinite(state).all():
            raise ValueError(f'Nonfinite state/action: {path}')
        digest=hashlib.sha256(action.tobytes()+state.tobytes()).hexdigest()
        if digest!=record['state_action_sha256']:
            raise ValueError(f'State/action hash mismatch: {path}')
        frames=[];head_hash=hashlib.sha256()
        for camera in ('head_camera','front_camera'):
            stream=f[f'observation/{camera}/rgb']
            if len(stream)!=length:
                raise ValueError(f'Camera count mismatch: {path}/{camera}')
            for index in (0,length-1):
                encoded=np.frombuffer(stream[index],np.uint8)
                image=cv2.imdecode(encoded,cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError(f'Invalid JPEG: {path}/{camera}/{index}')
                value=pixels(image)  # Official RGB-as-OpenCV encoding: no BGR2RGB.
                if camera=='head_camera':
                    frames.append(value);head_hash.update(encoded.tobytes())
        cameras=sorted(f['observation'].keys())
    after=path.stat()
    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
        raise ValueError(f'Source changed while reading: {path}')
    provenance=dict(path=record['path'],frames=length,state_action_sha256=digest,
                    head_endpoint_jpeg_sha256=head_hash.hexdigest(),bytes=before.st_size,
                    mtime_ns=before.st_mtime_ns,cameras=cameras)
    return frames,provenance


@torch.inference_mode()
def run_task(root, task, encoder, device, batch_images):
    audit_path=root/'audits'/f'{task}.json'
    report=json.loads(audit_path.read_text())
    if report['status']!='passed' or report['task']!=task:raise ValueError(f'Unaudited task: {task}')
    records=report['records'];expected={r['path'] for r in records}
    actual={str(p.relative_to(root)) for p in (root/'data'/task).glob('*/data/*.hdf5')}
    if expected!=actual or len(expected)!=len(records):raise ValueError(f'Inventory mismatch: {task}')
    total=np.zeros(encoder.config.hidden_size,np.float64)
    pending=[];rows=[];count=0
    def consume():
        nonlocal total,count
        batch=torch.from_numpy(np.stack(pending)).to(device=device,dtype=next(encoder.parameters()).dtype)
        cls=encoder(pixel_values=batch,return_dict=True).last_hidden_state[:,0].float().cpu().numpy()
        differences=cls[1::2]-cls[0::2]
        if not np.isfinite(differences).all():raise ValueError(f'Nonfinite DINO features: {task}')
        total+=differences.sum(0,dtype=np.float64);count+=len(differences);pending.clear()
    for record in records:
        frames,provenance=read_episode(root,record)
        pending.extend(frames);rows.append(provenance)
        if len(pending)>=batch_images:consume()
    if pending:consume()
    if count!=len(records) or sum(r['frames'] for r in rows)!=report['frames']:
        raise ValueError(f'Incomplete task: {task}')
    return dict(status='passed',task=task,episodes=count,frames=report['frames'],
                primary_camera='head_camera',camera_names=['head_camera','front_camera'],
                decoded_endpoint_images=count*4,vector=(total/count).astype(np.float32).tolist(),
                task_audit_sha256=fingerprint(audit_path),provenance=rows)



def validate(args):
    root=Path(args.root);out=Path(args.output)
    cv2.setNumThreads(1);torch.set_num_threads(2)
    audit=json.loads((root/'dataset_audit.json').read_text())
    if audit['status']!='passed':raise ValueError('Dataset audit failed')
    totals=dict(tasks=0,episodes=0,frames=0,endpoint_images_decoded=0)
    for summary in sorted(audit['task_summaries'],key=lambda r:r['task']):
        task=summary['task'];report=json.loads((root/'audits'/f'{task}.json').read_text())
        if report['status']!='passed' or report['task']!=task:raise ValueError('Task audit failed')
        records=report['records'];expected={r['path'] for r in records}
        actual={str(p.relative_to(root)) for p in (root/'data'/task).glob('*/data/*.hdf5')}
        if expected!=actual or len(expected)!=len(records):raise ValueError(f'Inventory mismatch: {task}')
        rows=[read_episode(root,r)[1] for r in records]
        count=len(rows);frames=sum(r['frames'] for r in rows)
        if count!=summary['episodes'] or frames!=summary['frames']:raise ValueError('Task count mismatch')
        write_json(out/'validation'/f'{task}.json',dict(status='passed',task=task,provenance=rows))
        totals['tasks']+=1;totals['episodes']+=count;totals['frames']+=frames;totals['endpoint_images_decoded']+=4*count
        print(json.dumps(dict(task=task,**totals)),flush=True)
    if any(totals[k]!=audit[k] for k in ('tasks','episodes','frames')):raise ValueError('Root count mismatch')
    write_json(out/'source_validation.json',dict(status='passed',root=str(root),**totals,
        state_dim=16,action_dim=14,camera_names=['head_camera','front_camera'],image_channel_order='rgb',
        image_size=[320,240],full_image_decode=False,numeric_hashes_checked=totals['episodes'],
        dataset_audit_sha256=fingerprint(root/'dataset_audit.json'),
        statistics_sha256=fingerprint(root/'training_metadata/stat-500-all.json'),
        excluded_new_generated_batch=True))


def shard(args):
    root=Path(args.root);out=Path(args.output)
    if args.batch_images<2 or args.batch_images%2:raise ValueError('batch-images must be positive/even')
    cv2.setNumThreads(1);torch.set_num_threads(2)
    audit=json.loads((root/'dataset_audit.json').read_text())
    if audit['status']!='passed':raise ValueError('Dataset audit failed')
    protocol=dict(dataset_audit_sha256=fingerprint(root/'dataset_audit.json'),
                  statistics_sha256=fingerprint(root/'training_metadata/stat-500-all.json'),
                  encoder_config_sha256=fingerprint(Path(args.encoder)/'config.json'),
                  preparation_script_sha256=fingerprint(__file__),torch_version=torch.__version__,encoder_dtype='bfloat16',
                  image_size=[320,240],image_channel_order='rgb',feature='mean(last_frame_CLS-first_frame_CLS)',split='train')
    tasks=sorted(x['task'] for x in audit['task_summaries'])[args.shard::args.shards]
    encoder=AutoModel.from_pretrained(args.encoder,local_files_only=True,torch_dtype=torch.bfloat16).to(args.device).eval()
    encoder.requires_grad_(False);started=time.time()
    for task in tasks:
        path=out/'tasks'/f'{task}.json'
        if path.exists():
            previous=json.loads(path.read_text())
            if previous.get('status')=='passed' and previous.get('protocol')==protocol:continue
            raise ValueError(f'Existing task output has different protocol: {path}')
        result=run_task(root,task,encoder,args.device,args.batch_images)
        result['protocol']=protocol;write_json(path,result)
        print(json.dumps(dict(task=task,episodes=result['episodes'],elapsed_seconds=time.time()-started)),flush=True)
    write_json(out/f'shard_{args.shard}.json',dict(status='complete',tasks=tasks,protocol=protocol,elapsed_seconds=time.time()-started))


def merge(args):
    root=Path(args.root);out=Path(args.output)
    audit=json.loads((root/'dataset_audit.json').read_text())
    rows=[json.loads((out/'tasks'/f"{r['task']}.json").read_text()) for r in sorted(audit['task_summaries'],key=lambda r:r['task'])]
    if audit['status']!='passed' or len(rows)!=audit['tasks']:raise ValueError('Invalid root audit')
    hidden=json.loads((Path(args.encoder)/'config.json').read_text())['hidden_size']
    for summary, row in zip(sorted(audit['task_summaries'],key=lambda r:r['task']), rows):
        if any(row[k]!=summary[k] for k in ('task','episodes','frames')):
            raise ValueError('Task totals differ from root audit')
        vector=np.asarray(row['vector'])
        if vector.shape!=(hidden,) or not np.isfinite(vector).all():raise ValueError('Invalid task vector')
        if row['task_audit_sha256']!=fingerprint(root/'audits'/f"{row['task']}.json"):
            raise ValueError('Task audit changed')
    protocol=rows[0]['protocol']
    if any(r['status']!='passed' or r['protocol']!=protocol for r in rows):raise ValueError('Task protocol mismatch')
    if protocol['dataset_audit_sha256']!=fingerprint(root/'dataset_audit.json'):raise ValueError('Source audit changed')
    if sum(r['episodes'] for r in rows)!=audit['episodes'] or sum(r['frames'] for r in rows)!=audit['frames']:
        raise ValueError('Merged counts differ from audited dataset')
    vectors={};provenance={}
    for r in rows:
        key='aloha:'+' '.join(r['task'].replace('-','_').split('_')).strip().lower()
        if key in vectors:raise ValueError('Task-key collision')
        vectors[key]=r['vector'];provenance[key]=r['provenance']
    payload=dict(format_version=1,split='train',vectors=vectors,provenance=provenance,protocol=protocol,
                 image_size=[320,240],camera_names=['head_camera','front_camera'],encoder_path=args.encoder,
                 feature='mean(last_frame_CLS-first_frame_CLS), primary camera, last_hidden_state')
    write_json(out/'robotwin_official_train_vtt.json',payload)
    summary=dict(status='ready_for_head_front_training',root=str(root),tasks=len(rows),episodes=audit['episodes'],frames=audit['frames'],
                 camera_names=['head_camera','front_camera'],state_dim=16,action_dim=14,
                 numeric_hashes_checked=audit['episodes'],endpoint_images_decoded=sum(r['decoded_endpoint_images'] for r in rows),
                 full_image_decode=False,three_wrist_view_ready=False,excluded_new_generated_batch=True,protocol=protocol)
    write_json(out/'data_readiness.json',summary)
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True);parser.add_argument('--encoder',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--device',default='cuda:0');parser.add_argument('--shard',type=int,default=0);parser.add_argument('--shards',type=int,default=1)
    parser.add_argument('--batch-images',type=int,default=64);parser.add_argument('--merge',action='store_true')
    parser.add_argument('--validate-only',action='store_true')
    args=parser.parse_args()
    if args.shards<1 or not 0<=args.shard<args.shards:raise ValueError('Invalid shard index/count')
    (validate if args.validate_only else merge if args.merge else shard)(args)
