"""Read-only CPU audit of the archived RoboTwin GAWM experiment versus B/C.

Run from the repository root with the starVLA Python environment. Writes only
the requested evidence JSON; never changes datasets, checkpoints or live jobs.
This measures input contracts and records differences, not causal ablations.
"""
import argparse
import datetime
import hashlib
import json
import math
import re
import sys
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq
import yaml


ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / 'playground/Checkpoints/gawm_s_robotwin_continuous_next_49tasks_40k_20260905'
C = Path('/data/gaoxiang/ckpts/lila_starvla/gawm_official_C_20260919')
DATA = Path('/data/gaoxiang/RoboTwinGenerated/Clean/blocks_ranking_rgb')


def read_json(path):
    return json.loads(path.read_text())


def frame(path):
    with av.open(str(path)) as container:
        container.streams.video[0].codec_context.thread_count = 1
        return next(container.decode(video=0)).to_ndarray(format='rgb24').astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sources = [OLD / 'config.yaml', OLD / 'dataset_statistics.json',
               OLD / 'launch_verified.json', C / 'stage1.yaml',
               C / 'comparison_step117000.json']
    old = yaml.safe_load(sources[0].read_text())
    current = yaml.safe_load((C / 'stage1.yaml').read_text())
    paired = read_json(C / 'comparison_step117000.json')
    report = dict(created=datetime.datetime.now().astimezone().isoformat(),
                  scope='Archived 49-task 40k run; later Qwen/GAWM run separately identified',
                  live_jobs_modified=False,
                  BC_117k=dict(B=paired['B_successes'], C=paired['C_successes'],
                               trials=paired['trials'],
                               identical_outcomes=sum(x['B'] == x['C'] for x in paired['paired'])))
    colors = []
    for ep in (0, 20, 499):
        raw_path = Path('/data/gaoxiang/RoboTwinGenerated_raw/Clean/blocks_ranking_rgb/demo_clean/video') / f'episode{ep}.mp4'
        converted_path = DATA / f'videos/chunk-000/observation.images.cam_high/episode_{ep:06d}.mp4'
        raw, converted = frame(raw_path), frame(converted_path)
        colors.append(dict(episode=ep, raw=str(raw_path), converted=str(converted_path),
                           direct_mae=float(abs(raw-converted).mean()),
                           swapped_mae=float(abs(raw-converted[..., ::-1]).mean())))
    report['fresh_color_check'] = colors
    order = list(range(6)) + list(range(7, 13)) + [6, 13]
    stats = read_json(OLD / 'dataset_statistics.json')['aloha']['action']
    lo, hi = np.array(stats['q01']), np.array(stats['q99'])
    norm = lambda x: (x - lo) / (hi - lo) * 2 - 1
    rows = []
    for ep in np.linspace(0, 999, 20, dtype=int):
        path = DATA / f'data/chunk-000/episode_{ep:06d}.parquet'
        table = pq.read_table(path, columns=['observation.state', 'action'])
        state = np.asarray(table['observation.state'].to_pylist(), dtype=np.float32)
        action = np.asarray(table['action'].to_pylist(), dtype=np.float32)
        normalized = norm(action[:, order])
        inverse = (normalized + 1) / 2 * (hi-lo) + lo
        anchors = np.arange(len(action)-16)
        target = normalized[anchors[:, None]+np.arange(1,17)]
        copy_loss = abs(normalized[anchors,None]-target)
        rows.append(dict(episode=int(ep), frames=len(action),
                         state_action_max_abs=float(abs(state-action).max()),
                         q99_no_clip_roundtrip_max_abs=float(abs(inverse-action[:,order]).max()),
                         joint_targets_outside_unit_fraction=float((abs(normalized[:,:12])>1).mean()),
                         copy_state_next16_normalized_l1=float(copy_loss.mean()),
                         copy_state_next1_normalized_l1=float(copy_loss[:,0].mean())))
    report['fresh_state_and_action_check'] = rows
    # Exercise the real checkpoint deployment normalization, not only the
    # closed-form calculation above. The checkpoint weights are not loaded.
    sys.path.insert(0, str(ROOT))
    from deployment.model_server.policy_norm_processor import PolicyNormProcessor
    proc = PolicyNormProcessor(str(OLD / 'final_model/pytorch_model.pt'), unnorm_key='aloha')
    sample = action[::4, order]
    fields, cursor = {}, 0
    for key in proc.action_keys:
        width = proc._action_key_dims[key]
        fields[key] = sample[:, cursor:cursor+width].copy()
        cursor += width
    transformed = proc.transform.apply(fields)
    normalized = np.concatenate([transformed[k].numpy() for k in proc.action_keys], axis=-1)
    restored = proc.unapply_actions(normalized)
    report['actual_deployment_normalization_roundtrip'] = dict(
        episode=int(ep), samples=len(sample), action_keys=proc.action_keys,
        max_abs_error=float(abs(restored-sample).max()),
        note='Current deployment transform for archived checkpoint; no forced precision conversion.')
    report['normalization_note'] = ('q99 without input clipping and tanh_linear_tail output in continuous-next run; '
                                    'outside [-1,1] is not proof of clipping. Grippers map [0,1] to [-1,1].')
    report['training'] = dict(old=old, C=current,
                             old_new_robotwin_anchor_draws=40000*192,
                             BC_117k_anchor_draws=117000*128,
                             BC_316k_anchor_draws=316000*128,
                             note='Anchor draws exclude old 160k pretraining; not unique frames.')
    log = OLD / 'robotwin_eval_logs/gawm_s_continuous_next_49tasks_40k_20260907/evaluation.log'
    result = re.findall(r'blocks_ranking_rgb: Success rate:\s*(\d+)/(\d+)', log.read_text())
    report['old_logged_rgb_result'] = dict(successes=int(result[-1][0]), trials=int(result[-1][1]),
                                          source=str(log), final_100_episode_result=False)
    lr_checks = []
    for line in (OLD / 'metrics.jsonl').read_text().splitlines():
        row = json.loads(line)
        if row['step'] not in (1000, 10000, 20000, 30000, 40000):
            continue
        expected = 3e-6 + (3e-5-3e-6) * (1+math.cos(math.pi*(row['step']-1000)/39000))/2
        lr_checks.append(dict(step=row['step'], logged=row['learning_rate/base'],
                              expected=expected, abs_error=abs(row['learning_rate/base']-expected)))
    report['old_logged_scheduler_checks'] = lr_checks
    for name in ['rgb_scene_and_color_20260907.json', 'rgb_v2_final_failures_20260907/failure_stage_audit.json']:
        path = ROOT / 'examples/Robotwin/audits' / name
        sources.append(path)
        data = read_json(path)
        if 'episodes_detail' in data:
            data = {k:v for k,v in data.items() if k not in ('matches','episodes_detail')}
        report[name] = data
    later = ROOT / 'playground/Checkpoints/qwen_gawm_ranking_20260910/development_comparison.json'
    sources.append(later)
    report['later_development_comparison'] = read_json(later)
    report['source_sha256'] = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(args.output), BC=report['BC_117k'], colors=colors,
                         sampled_frames=sum(x['frames'] for x in rows),
                         state_action_max_abs=max(x['state_action_max_abs'] for x in rows),
                         roundtrip_max_abs=max(x['q99_no_clip_roundtrip_max_abs'] for x in rows),
                         old_result=report['old_logged_rgb_result']), indent=2))


if __name__ == '__main__':
    main()
