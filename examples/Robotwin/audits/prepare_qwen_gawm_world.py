"""Prepare Qwen -> GAWM residual predictor -> ACT, matched to OFT ACT16."""
import json
from pathlib import Path
from omegaconf import OmegaConf
from examples.Robotwin.audits.prepare_qwen_gawm_ranking import ROOT,OUT as BASE,QWEN,digest,save

OUT=ROOT/'playground/Checkpoints/qwen_gawm_world_20260910'
SHARED=Path('/data/gaoxiang/ckpts/qwen_gawm_world_20260910')
WM=dict(n_future=2,future_recorded_offsets=[6,12],residual_predictor_dim=384,
        residual_predictor_depth=4,residual_predictor_heads=6,residual_predictor_ffn=1024,
        latent_stats_momentum=.9,loss_latent_weight=0.,latent_cosine_weight=0.)


def validate_config(cfg,stage):
    assert cfg.framework.name=='QwenGAWMWorld'
    assert OmegaConf.to_container(cfg.framework.world_model,resolve=True)==WM
    assert cfg.datasets.vla_data.future_obs_frames is True
    assert not cfg.trainer.is_resume
    expected=str(SHARED/'qwen_world_warmup/final_model/pytorch_model.pt') if stage=='joint' else str(QWEN)
    assert cfg.trainer.pretrained_checkpoint==expected
    assert cfg.trainer.reload_modules == (None if stage=='joint' else 'qwen_vl_interface')
    actual=OmegaConf.to_container(cfg,resolve=True)
    reference=OmegaConf.to_container(OmegaConf.load(BASE/f'qwen_act_{stage}.yaml'),resolve=True)
    actual['framework'].pop('world_model');actual['framework']['name']=reference['framework']['name']
    actual['datasets']['vla_data']['future_obs_frames']=reference['datasets']['vla_data']['future_obs_frames']
    for key in ('run_id','run_root_dir'):actual[key]=reference[key]
    actual['trainer']['pretrained_checkpoint']=reference['trainer']['pretrained_checkpoint']
    assert actual==reference,'Unexpected difference from ACT16 reference'


def main():
    OUT.mkdir(parents=True,exist_ok=False);rows=[]
    for stage in ('smoke','warmup','joint'):
        cfg=OmegaConf.load(BASE/f'qwen_act_{stage}.yaml')
        cfg.framework.name='QwenGAWMWorld';cfg.framework.world_model=WM
        cfg.datasets.vla_data.future_obs_frames=True
        cfg.run_id=f'qwen_world_{stage}';cfg.run_root_dir=str(SHARED)
        if stage=='joint':cfg.trainer.pretrained_checkpoint=str(SHARED/'qwen_world_warmup/final_model/pytorch_model.pt')
        validate_config(cfg,stage);path=OUT/f'{stage}.yaml';OmegaConf.save(cfg,path)
        rows.append(dict(stage=stage,path=str(path),sha256=digest(path),steps=int(cfg.trainer.max_train_steps)))
    protocol=json.loads((BASE/'protocol.json').read_text())
    protocol.update(action_horizon=16,execute_horizon=16,
        comparison='OFT Qwen + GAWM residual predictor + ACT versus OFT Qwen + ACT; same 16/16 control.',
        limitation='Adds prediction module, two memory frames and future-label residual-scale statistics; L1-only does not establish accurate future prediction. Action slots are not spatial patches. No DINO spatial-focus branch or state is added.')
    save(OUT/'protocol.json',protocol)
    save(OUT/'preparation.json',dict(state='prepared',configs=rows,qwen_checkpoint=str(QWEN),
        qwen_checkpoint_sha256=digest(QWEN),source_experiment=str(BASE),effective_batch=32,seed=42,
        stages=dict(warmup=500,joint=4500),new_world_model_and_act_random=True,
        future_recorded_offsets=[6,12],physical_time_offsets_known=False,
        loss='masked_action_L1_only',wm_diagnostics_and_scale_use_detached_future=True))


if __name__=='__main__':main()
