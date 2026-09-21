"""Prepare the full-trajectory Qwen/GAWM comparison without starting training."""
import copy
import hashlib
import json
from pathlib import Path

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT/'playground/Checkpoints/qwen_gawm_ranking_20260910'
SHARED = Path('/data/gaoxiang/ckpts/qwen_gawm_ranking_20260910')
OLD = ROOT/'playground/Checkpoints/gawm_grasp_precision_training_20260909'
QWEN = Path('/data/gaoxiang/ckpts/Qwen3-VL-OFT-RoboTwin2-All/checkpoints/steps_140000_pytorch_model.pt')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    base = OmegaConf.to_container(OmegaConf.load(OLD/'joint_train1000.yaml'), resolve=True)
    data = base['datasets']['vla_data']
    for key in ('training_anchor_manifest', 'dataset_options'):
        data.pop(key, None)
    data.update(data_mix='robotwin_ranking_rgb_continuous_next_wm', num_workers=2,
                per_device_batch_size=2, preserve_native_images=True)
    base['run_root_dir'] = str(SHARED)
    trainer = base['trainer']
    trainer.update(gradient_accumulation_steps=16, logging_frequency=20,
                   validation_batch_size=2, validation_samples_per_task=64,
                   is_resume=False, save_interval=5000, eval_interval=500)
    trainer['learning_rate'] = dict(base=1e-5, action_models=1e-4,
                                    backbone=dict(encoder=1e-5), spatial_focus=1e-4)
    for key in ('loss_latent_weight', 'latent_cosine_weight', 'visual_token_diversity_weight',
                'visual_token_variance_weight'):
        base['framework']['world_model'][key] = 0.
    for key in ('heatmap_loss_weight', 'coordinate_loss_weight'):
        base['framework']['spatial_focus'][key] = 0.
    rows = []
    for variant in ('gawm', 'qwen_mlp', 'qwen_act'):
        cfg = copy.deepcopy(base)
        if variant != 'gawm':
            cfg['framework'] = dict(name='QwenGAWM',
                qwenvl=dict(base_vlm='/data/gaoxiang/ckpts/Qwen3-VL-4B-Instruct', attn_implementation='sdpa'),
                action_model=dict(action_model_type='ACT' if variant=='qwen_act' else 'MLP',
                    action_horizon=16, action_dim=14, action_hidden_dim=384,
                    act_num_heads=8, act_num_layers=3, act_dim_feedforward=2048,
                    act_mlp_hidden_dim=512, act_dropout=.1, output_activation='tanh_linear_tail',
                    embodiment_heads={'aloha': dict(action_dim=14, action_horizon=16, state_dim=0,
                        action_spec_id='aloha_dual_joint_contgrip_next_recorded_14', gripper_indices=[12,13])}),
                qwen_training=dict(train_last_n_layers=0))
            cfg['datasets']['vla_data'].update(include_state=False, future_obs_frames=False,
                                               preserve_native_images=False)
            cfg['datasets']['vla_data'].pop('spatial_supervision_dir', None)
            cfg['trainer']['learning_rate'] = dict(base=1e-5, action_model=1e-4, qwen_vl_interface=1e-5)
            cfg['trainer']['pretrained_checkpoint'] = str(QWEN)
            cfg['trainer']['reload_modules'] = 'qwen_vl_interface'
        for stage, steps in [('warmup',500), ('joint',4500), ('smoke',20)]:
            current = copy.deepcopy(cfg)
            name = f'{variant}_{stage}'
            current['run_id'] = name
            t = current['trainer']
            t.update(max_train_steps=steps, num_warmup_steps=2 if stage=='smoke' else 50 if stage=='warmup' else 200)
            if stage=='joint':
                t['pretrained_checkpoint'] = str(SHARED/f'{variant}_warmup/final_model/pytorch_model.pt')
                t['reload_modules'] = None
            if variant=='gawm':
                frozen = ['action_models.franka','action_models.oxe_bridge']
                if stage=='warmup':
                    frozen += ['backbone','world_model','visual_token_pooler','task_embedding',
                               'embodiment_embedding','spatial_focus']
                t['freeze_modules'] = ','.join(frozen)
            else:
                # Four text layers fit a 40 GiB GPU with Adam; both heads use the same policy.
                current['framework']['qwen_training']['train_last_n_layers'] = 0 if stage=='warmup' else 4
                t['freeze_modules'] = ''
            if stage=='smoke':
                t.update(eval_interval=20, validation_samples_per_task=8, logging_frequency=1)
            path = OUT/f'{name}.yaml'
            OmegaConf.save(OmegaConf.create(current), path)
            rows.append(dict(name=name, path=str(path), sha256=digest(path), steps=steps))
    protocol_path = ROOT/'examples/Robotwin/audits/grasp_lift_protocol_20260909.json'
    protocol = json.loads(protocol_path.read_text())
    seeds = set(map(int, Path(protocol['training_seed_file']).read_text().split()))
    assert not seeds.intersection(r['seed'] for r in protocol['records'])
    save(OUT/'protocol.json', dict(task='blocks_ranking_rgb', mode='demo_clean',
        records=protocol['records'], scoring='Unmodified task.check_success and official step limit',
        execute_horizon=16, checkpoint_selection='Final joint-stage checkpoint; no selection on final test',
        development_scenes=20, test_scenes=100, no_scene_replacement=True,
        training_seed_sha256=digest(protocol['training_seed_file']),
        upstream_scene_protocol_sha256=digest(protocol_path),
        pretraining_overlap='Public OFT historical training scenes cannot be independently excluded'))
    save(OUT/'preparation.json', dict(state='prepared', configs=rows,
        source_split=str(OLD/'split.json'), split_sha256=digest(OLD/'split.json'),
        qwen_checkpoint=str(QWEN), normalization_sha256=digest(data['normalization_statistics_path']),
        effective_batch=32, train_seed=42, qwen_heads_random=True,
        phases=dict(warmup=500,joint=4500), qwen_joint_train_last_text_layers=4,
        full_sorting_trajectories=True, correction_mixture=False, final_test_used=False))
    print(OUT)


if __name__ == '__main__':
    main()
