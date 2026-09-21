"""Regenerate selected training scenes and verify against their original data."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import time

import cv2
import h5py
import numpy as np
from PIL import Image

from examples.Robotwin.audits.audit_rgb_scene_and_color import colors
from examples.Robotwin.audits.smoke_rgb_recovery_teacher import (
    VIEWS, _load_robotwin_evaluator, load_task_config, save,
)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def verify_source(env, record, output):
    """Check geometry and colored objects; JPEG compression prevents byte equality."""
    assert digest(record['source_hdf5']) == record['source_hdf5_sha256']
    observation = env.get_obs()
    result = dict(source_episode=record['source_episode'], scene_seed=record['scene_seed'], views={})
    with h5py.File(record['source_hdf5']) as source:
        state = np.asarray(observation['joint_action']['vector'])
        truth = source['joint_action/vector'][0]
        result['state_max_difference'] = float(np.abs(state-truth).max())
        np.testing.assert_allclose(state, truth, atol=1e-6, rtol=0)
        for view in VIEWS:
            current = observation['observation'][view]
            original = source['observation'][view]
            # Legacy RoboTwin calls cv2.imencode on an RGB array directly.
            # cv2.imdecode reproduces that RGB array (up to JPEG loss).
            raw_image = cv2.imdecode(np.frombuffer(original['rgb'][0], dtype=np.uint8), cv2.IMREAD_COLOR)
            assert raw_image is not None and raw_image.shape == current['rgb'].shape
            pixel_mae = float(np.abs(raw_image.astype(float)-current['rgb'].astype(float)).mean())
            swapped_mae = float(np.abs(raw_image[..., ::-1].astype(float)-current['rgb'].astype(float)).mean())
            differences = {}
            for field in ('intrinsic_cv', 'extrinsic_cv', 'cam2world_gl'):
                expected = original[field][0]
                actual = np.asarray(current[field])
                np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=0)
                differences[field] = float(np.abs(actual-expected).max())
            result['views'][view] = dict(pixel_mae=pixel_mae, swapped_pixel_mae=swapped_mae,
                                         camera_max_differences=differences)
            assert pixel_mae < 3.0, result['views'][view]
            if view == 'head_camera':
                current_colors, original_colors = colors(current['rgb']), colors(raw_image)
                assert len(current_colors) == len(original_colors) == 3
                color_errors = {color: float(np.linalg.norm(np.asarray(current_colors[color]['xy'])-
                                                            original_colors[color]['xy']))
                                for color in ('red', 'green', 'blue')}
                assert max(color_errors.values()) < 2.0, color_errors
                assert pixel_mae < swapped_mae
                result['color_center_errors_px'] = color_errors
                Image.fromarray(current['rgb']).save(output / f"episode_{record['source_episode']:06d}_regenerated.png")
                Image.fromarray(raw_image).save(output / f"episode_{record['source_episode']:06d}_original.png")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--robotwin-root', type=Path, required=True)
    parser.add_argument('--sources', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sources_path, output = args.sources.resolve(), args.output.resolve()
    sources = json.loads(sources_path.read_text())
    split = json.loads(Path(sources['split']).read_text())
    assert digest(sources['split']) == sources['split_sha256']
    assert digest(sources['seed_file']) == sources['seed_file_sha256']
    seed_list = [int(x) for x in Path(sources['seed_file']).read_text().split()]
    for record in sources['records']:
        assert record['source_episode'] in split['train_episode_ids']
        assert record['source_episode'] not in split['validation_episode_ids']
        assert seed_list[record['source_episode']] == record['scene_seed']
    output.mkdir(parents=True, exist_ok=False)
    report = dict(state='starting', source_manifest=str(sources_path), pid=os.getpid(), verified=[])
    env = None

    def status():
        save(output / 'status.json', dict(report, time=time.strftime('%Y-%m-%d %H:%M:%S')))

    def stop(signum, frame):
        raise RuntimeError(f'Signal {signum}')

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop)
    status()
    try:
        os.chdir(args.robotwin_root.resolve())
        evaluator = _load_robotwin_evaluator(args.robotwin_root.resolve())
        from test_render import Sapien_TEST
        Sapien_TEST()
        config, _ = load_task_config(evaluator, output)
        env = evaluator.class_decorator('blocks_ranking_rgb')
        for record in sources['records']:
            report.update(state='verifying', source_episode=record['source_episode'])
            status()
            env.setup_demo(now_ep_num=0, seed=record['scene_seed'], is_test=True, **config)
            report['verified'].append(verify_source(env, record, output))
            env.close_env(clear_cache=True)
            env = evaluator.class_decorator('blocks_ranking_rgb')
            status()
        env = None
        report.update(state='complete', note='Initial scenes only; no recovery rollouts collected or trained.')
        status()
    except BaseException as exc:
        report.update(state='failed', error=repr(exc))
        status()
        raise
    finally:
        if env is not None and hasattr(env, 'scene'):
            env.close_env(clear_cache=True)


if __name__ == '__main__':
    main()
