"""Check real GPU websocket inference repeats after episode reset, using train images."""
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

import numpy as np
from omegaconf import OmegaConf
from websockets.sync.client import connect

from deployment.model_server.tools import msgpack_numpy
from examples.Robotwin.audits.run_grasp_lift_development import ROOT, PYTHON, digest, save
from starVLA.dataloader.lerobot_datasets import get_vla_dataset


def main():
    out = ROOT/'playground/Checkpoints/gawm_grasp_precision_training_20260909/seeded_server_smoke'
    out.mkdir(exist_ok=False)
    run = ROOT/'playground/Checkpoints/gawm_grasp_precision_compact_flow_smoke20_20260909'
    checkpoint = run/'checkpoints/steps_20_pytorch_model.pt'
    assert checkpoint.is_file()
    cfg = OmegaConf.load(run/'config.full.yaml')
    child = get_vla_dataset(cfg.datasets.vla_data, mode='train').datasets[0]
    episode = int(child.trajectory_ids[0])
    child.transforms.eval()
    sample = child._pack_sample(child.transforms(child.get_step_data(episode, 0)))
    payload = dict(image=[np.asarray(x) for x in sample['image']],
                   native_images=[np.asarray(x) for x in sample['native_images']],
                   state=np.asarray([[0.] * 12 + [1., 1.]], dtype=np.float32), lang=sample['lang'])
    port = 6895
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', port))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='5', PYTHONPATH=str(ROOT),
               PYTHONNOUSERSITE='1', OMP_NUM_THREADS='4', NO_ALBUMENTATIONS_UPDATE='1')
    server = None
    try:
        with (out/'server.log').open('x') as log:
            server = subprocess.Popen([str(PYTHON), str(ROOT/'deployment/model_server/server_policy.py'),
                '--ckpt_path', str(checkpoint), '--port', str(port), '--idle_timeout', '-1',
                '--policy-seed', '20260909', '--policy-seed-log', str(out/'policy_rng.jsonl')],
                cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True)
        deadline = time.monotonic()+180
        while True:
            assert server.poll() is None, 'Server failed; inspect server.log'
            try:
                ws = connect(f'ws://127.0.0.1:{port}', proxy=None, open_timeout=2, max_size=None)
                break
            except OSError:
                assert time.monotonic() < deadline, 'Server startup timeout'
                time.sleep(2)
        with ws:
            metadata = msgpack_numpy.unpackb(ws.recv(timeout=10))
            assert metadata['policy_rng']['base_seed'] == 20260909
            assert metadata['action_chunk_sizes']['aloha'] == 16
            assert metadata['action_specs']['aloha']['action_horizon'] == 16
            actions = []
            for start in (True, False, True, False):
                ws.send(msgpack_numpy.Packer().pack(dict(
                    examples=[dict(payload, episode_start=start)], unnorm_key='aloha')))
                result = msgpack_numpy.unpackb(ws.recv(timeout=60))
                assert result['ok'], result
                action = np.asarray(result['data']['actions'])
                assert action.shape == (1, 16, 14) and np.isfinite(action).all()
                actions.append(action)
        np.testing.assert_array_equal(actions[0], actions[2])
        np.testing.assert_array_equal(actions[1], actions[3])
        assert not np.array_equal(actions[0], actions[1]), 'Query seed did not change flow noise'
        logs = [json.loads(line) for line in (out/'policy_rng.jsonl').read_text().splitlines()]
        assert [row['query_index'] for row in logs] == [0, 1, 0, 1]
        assert [row['episode_index'] for row in logs] == [0, 0, 1, 1]
        save(out/'report.json', dict(state='passed', checkpoint=str(checkpoint),
            checkpoint_sha256=digest(checkpoint), metadata=metadata, train_source_episode=episode,
            gpu='5', precision='float32', same_episode_reset_max_difference=0.,
            adjacent_query_max_difference=float(np.abs(actions[0]-actions[1]).max()),
            note='Websocket inference engineering test, not grasp success or accuracy evidence.'))
    finally:
        if server is not None and server.poll() is None:
            os.killpg(server.pid, signal.SIGTERM)
            try:
                server.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait(timeout=5)


if __name__ == '__main__':
    main()
