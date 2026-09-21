"""Exercise real checkpoint -> websocket -> RoboTwin client with native RGB."""
import json
import os
from pathlib import Path
import subprocess
import time

import av
import numpy as np
import pyarrow.parquet as pq

from examples.Robotwin.eval_files.model2robotwin_interface import ModelClient, ROBOTWIN_TO_MODEL_JOINT_ORDER


def main():
    root=Path(__file__).resolve().parents[3]
    camp=root/'playground/Checkpoints/gawm_rgb_focus_20260907'
    checkpoint=root/'playground/Checkpoints/gawm_rgb_focus_local_smoke_v2_20260907/final_model/pytorch_model.pt'
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='5',PYTHONPATH=str(root),NO_ALBUMENTATIONS_UPDATE='1',OMP_NUM_THREADS='4')
    with (camp/'deployment_smoke_server.log').open('w') as out:
        server=subprocess.Popen([str(root.parent/'.venvs/starVLA/bin/python'),str(root/'deployment/model_server/server_policy.py'),
                                 '--ckpt_path',str(checkpoint),'--port','6105'],cwd=root,env=env,stdout=out,stderr=subprocess.STDOUT)
    client=None
    try:
        for attempt in range(60):
            if server.poll() is not None:raise RuntimeError('Deployment server failed; inspect log')
            try:
                import socket
                with socket.create_connection(('127.0.0.1',6105),timeout=.5):break
            except OSError:time.sleep(1)
        else:raise RuntimeError('Server startup timeout')
        client=ModelClient(str(checkpoint),unnorm_key='aloha',port=6105)
        dataset=Path('/data/gaoxiang/RoboTwinGenerated/Clean/blocks_ranking_rgb')
        images=[]
        for camera in ('cam_high','cam_left_wrist','cam_right_wrist'):
            with av.open(str(dataset/f'videos/chunk-000/observation.images.{camera}/episode_000000.mp4')) as video:
                # Match REAL RGB camera observations, not legacy decoded BGR.
                images.append(next(video.decode(video=0)).to_ndarray(format='rgb24')[...,::-1].copy())
        action=np.array(pq.read_table(dataset/'data/chunk-000/episode_000000.parquet',columns=['action'])['action'][0].as_py(),np.float32)
        example=dict(image=images,lang='blocks ranking rgb',state=action[ROBOTWIN_TO_MODEL_JOINT_ORDER])
        result=client.step(example,step=0)
        assert result.shape==(14,) and np.isfinite(result).all()
        assert client.raw_actions.shape==(16,14)
        assert np.isfinite(client.raw_actions).all()
        (camp/'deployment_smoke.json').write_text(json.dumps(dict(checkpoint=str(checkpoint),native_wh=[320,240],
            action_shape=list(result.shape),chunk_shape=list(client.raw_actions.shape),finite=True,full_model_bf16=False),indent=2)+'\n')
        print('Deployment smoke passed',flush=True)
    finally:
        if client is not None:client.client.close()
        server.terminate()
        try:server.wait(timeout=10)
        except subprocess.TimeoutExpired:server.kill();server.wait()


if __name__=='__main__':main()
