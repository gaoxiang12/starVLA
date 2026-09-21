"""Read-only paired-episode comparison of official HDF5 and existing LeRobot data."""
import io
import json
from pathlib import Path

import cv2
import h5py
import numpy as np
from PIL import Image
import pyarrow.parquet as pq


def align(source, target, tolerance=1e-6):
    indices, cursor = [], 0
    for row in source:
        if cursor >= len(target):
            return None
        hits = np.flatnonzero(np.max(np.abs(target[cursor:]-row),axis=1) <= tolerance)
        if not len(hits):
            return None
        cursor += int(hits[0])
        indices.append(cursor)
        cursor += 1
    return np.array(indices)


def compare(path, task, split):
    index=int(path.stem.replace('episode','').replace('_',''))
    local=Path('/data/gaoxiang/RoboTwin')/split/task/'data/chunk-000'/f'episode_{index:06d}.parquet'
    tab=pq.read_table(local)
    state=np.asarray(tab['observation.state'].to_pylist(),np.float32)
    action=np.asarray(tab['action'].to_pylist(),np.float32)
    result={'task':task,'split':split,'episode':index,'official_path':str(path),'local_path':str(local),
            'local_frames':len(action),'local_action_equals_state':bool(np.array_equal(action,state))}
    with h5py.File(path) as f:
        original=f['joint_action/vector'][:].astype(np.float32)
        result['official_frames']=len(original)
        result['official_cameras']=list(f['observation'].keys())
        result['local_cameras']=[k for k in tab.column_names if k.startswith('observation.images.')]
        result['same_length_action_max_abs']=float(np.max(np.abs(original-action))) if len(original)==len(action) else None
        matches={key:align(original,value) for key,value in [('action',action),('state',state)]}
        result['official_is_subsequence_of_local']={k:v is not None for k,v in matches.items()}
        result['nearest_action_max_errors']=[]
        for i in sorted({0,len(original)//2,len(original)-1}):
            errors=np.max(np.abs(action-original[i]),axis=1)
            result['nearest_action_max_errors'].append({'official_frame':i,'nearest_local_frame':int(errors.argmin()),'max_abs_error':float(errors.min())})
        mapping=matches['action'] if matches['action'] is not None else matches['state']
        comparisons=[]
        for i in sorted({0,len(original)//2,len(original)-1}):
            j=int(mapping[i]) if mapping is not None else (0 if i==0 else len(action)-1 if i==len(original)-1 else len(action)//2)
            ob=np.frombuffer(f['observation/head_camera/rgb'][i],np.uint8)
            off=cv2.imdecode(ob,cv2.IMREAD_COLOR)
            item=tab['observation.images.cam_high'][j].as_py()
            loc=np.asarray(Image.open(io.BytesIO(item['bytes'])).convert('RGB'))
            initial_shape=list(off.shape)
            if off.shape!=loc.shape:
                off=cv2.resize(off,(loc.shape[1],loc.shape[0]),interpolation=cv2.INTER_LINEAR)
            err=np.abs(off.astype(np.float32)-loc.astype(np.float32))
            reverse=np.abs(off[:,:,::-1].astype(np.float32)-loc.astype(np.float32))
            comparisons.append({'official_frame':i,'local_frame':j,'action_aligned':mapping is not None,
                                'official_shape':initial_shape,'local_shape':list(loc.shape),
                                'pixel_equal':bool(np.array_equal(off,loc)), 'mae_rgb':float(err.mean()),
                                'mae_reverse_channels':float(reverse.mean()), 'max_abs_rgb':float(err.max())})
        result['head_image_comparisons']=comparisons
        if mapping is not None:
            target=action if matches['action'] is not None else state
            result['aligned_action_bitwise_equal_float32']=bool(np.array_equal(original,target[mapping]))
            result['aligned_action_max_abs']=float(np.max(np.abs(original-target[mapping])))
    return result


def main():
    cv2.setNumThreads(1)
    root=Path('/data/gaoxiang/LiLaWAM_RoboTwin_Official')
    tasks=sorted(p.stem for p in (root/'audits').glob('*.json') if not p.name.endswith('.extraction.json'))
    rows=[]
    for task in tasks:
        for split,official_split in [('Clean','demo_clean'),('Randomized','demo_randomized')]:
            files=sorted((root/'data'/task/official_split/'data').glob('*.hdf5'),key=lambda p:int(p.stem.replace('episode','').replace('_','')))
            for index in sorted({0,len(files)//2,len(files)-1}):
                rows.append(compare(files[index],task,split))
    result={'scope':'Three episodes per task/split among fully audited downloads; first/middle/last head images per episode',
            'tasks':tasks,'episodes_compared':len(rows),'rows':rows}
    out=Path('examples/LiLaWAM/audits/official_vs_local_data_20260914.json')
    out.write_text(json.dumps(result,indent=2)+'\n')
    summary={'tasks':tasks,'episodes':len(rows),'different_frame_counts':sum(r['official_frames']!=r['local_frames'] for r in rows),
             'official_action_subsequences':sum(r['official_is_subsequence_of_local']['action'] for r in rows),
             'official_state_subsequences':sum(r['official_is_subsequence_of_local']['state'] for r in rows),
             'exact_float32_actions_after_alignment':sum(r.get('aligned_action_bitwise_equal_float32',False) for r in rows),
             'identical_head_images':sum(x['pixel_equal'] for r in rows for x in r['head_image_comparisons']),
             'head_images_compared':sum(len(r['head_image_comparisons']) for r in rows), 'report':str(out)}
    print(json.dumps(summary,indent=2))
    for row in rows[:4]:
        print(json.dumps(row))


if __name__=='__main__':
    main()
