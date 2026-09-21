"""Add a 50-step ACT arm without modifying any existing campaign artifact."""
import json
from pathlib import Path

from omegaconf import OmegaConf

from examples.Robotwin.audits.prepare_qwen_gawm_ranking import ROOT, OUT as BASE, digest, save

OUT=ROOT/'playground/Checkpoints/qwen_gawm_chunk50_20260910'
SHARED=Path('/data/gaoxiang/ckpts/qwen_gawm_chunk50_20260910')


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    rows=[]
    for stage in ('smoke','warmup','joint'):
        cfg=OmegaConf.load(BASE/f'qwen_act_{stage}.yaml')
        cfg.framework.name='QwenGAWMChunk'
        cfg.framework.action_model.action_horizon=50
        cfg.framework.action_model.embodiment_heads.aloha.action_horizon=50
        cfg.datasets.vla_data.data_mix='robotwin_ranking_rgb_continuous_next50'
        cfg.run_root_dir=str(SHARED)
        cfg.run_id=f'qwen_act50_{stage}'
        if stage=='joint':
            cfg.trainer.pretrained_checkpoint=str(SHARED/'qwen_act50_warmup/final_model/pytorch_model.pt')
        path=OUT/f'{stage}.yaml';OmegaConf.save(cfg,path)
        rows.append(dict(stage=stage,path=str(path),sha256=digest(path),steps=int(cfg.trainer.max_train_steps)))
    protocol=json.loads((BASE/'protocol.json').read_text())
    protocol.update(action_horizon=50,execute_horizon=50,
        comparison='50 prediction/50 execution versus existing 16/16; both training horizon and feedback interval change.',
        original_oft_alignment='Only prediction/execution chunk size; ACT, continuous gripper normalization, data and partial finetuning remain this experiment contract.')
    save(OUT/'protocol.json',protocol)
    save(OUT/'preparation.json',dict(state='prepared',configs=rows,source_experiment=str(BASE),
        source_preparation_sha256=digest(BASE/'preparation.json'),seed=42,effective_batch=32,
        stages=dict(warmup=500,joint=4500),action_horizon=50,execute_horizon=50,
        data_change='Same observation anchors; next 1..50 recorded targets with invalid tail masking.',
        formal_initialization='Same published OFT Qwen; random ACT, never resume the 16-step trained head.'))


if __name__=='__main__':main()
