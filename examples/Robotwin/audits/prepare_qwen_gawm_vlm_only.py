"""Isolate initialization: original Qwen3-VL-Instruct versus RoboTwin OFT Qwen."""
import json
from pathlib import Path

from omegaconf import OmegaConf

from examples.Robotwin.audits.prepare_qwen_gawm_ranking import ROOT,OUT as BASE,digest,save

OUT=ROOT/'playground/Checkpoints/qwen_gawm_vlm_only_20260910'
SHARED=Path('/data/gaoxiang/ckpts/qwen_gawm_vlm_only_20260910')
BASE_VLM=Path('/data/gaoxiang/ckpts/Qwen3-VL-4B-Instruct')


def validate_config(cfg,stage):
    assert cfg.framework.name=='QwenGAWM'
    assert Path(cfg.framework.qwenvl.base_vlm).resolve()==BASE_VLM.resolve()
    assert cfg.framework.action_model.action_horizon==16
    assert not cfg.trainer.is_resume and cfg.trainer.get('reload_modules') is None
    expected=str(SHARED/'qwen_act_vlm_only_warmup/final_model/pytorch_model.pt') if stage=='joint' else None
    assert cfg.trainer.pretrained_checkpoint==expected,'Unexpected action-pretrained checkpoint'
    reference=OmegaConf.to_container(OmegaConf.load(BASE/f'qwen_act_{stage}.yaml'),resolve=True)
    actual=OmegaConf.to_container(cfg,resolve=True)
    for key in ('run_id','run_root_dir'):actual[key]=reference[key]
    for key in ('pretrained_checkpoint','reload_modules'):actual['trainer'][key]=reference['trainer'].get(key)
    assert actual==reference,'Initialization comparison changed additional settings'


def main():
    OUT.mkdir(parents=True,exist_ok=False);rows=[]
    for stage in ('smoke','warmup','joint'):
        cfg=OmegaConf.load(BASE/f'qwen_act_{stage}.yaml')
        cfg.run_root_dir=str(SHARED);cfg.run_id=f'qwen_act_vlm_only_{stage}'
        cfg.trainer.pretrained_checkpoint=str(SHARED/'qwen_act_vlm_only_warmup/final_model/pytorch_model.pt') if stage=='joint' else None
        cfg.trainer.reload_modules=None
        validate_config(cfg,stage)
        path=OUT/f'{stage}.yaml';OmegaConf.save(cfg,path)
        rows.append(dict(stage=stage,path=str(path),sha256=digest(path),steps=int(cfg.trainer.max_train_steps)))
    protocol=json.loads((BASE/'protocol.json').read_text())
    protocol.update(action_horizon=16,execute_horizon=16,
        comparison='Original Qwen3-VL-4B-Instruct + random ACT vs RoboTwin OFT Qwen + random ACT; same 16/16 contract.',
        limitation='Tests initialization under the same limited-data, last-four-text-layer finetuning budget; does not establish the best achievable raw-VLM performance.')
    save(OUT/'protocol.json',protocol)
    index=json.loads((BASE_VLM/'model.safetensors.index.json').read_text())
    files=sorted(set(index['weight_map'].values()))
    files+=['config.json','model.safetensors.index.json','tokenizer.json','tokenizer_config.json','preprocessor_config.json','chat_template.json']
    save(OUT/'preparation.json',dict(state='prepared',configs=rows,source_experiment=str(BASE),
        source_preparation_sha256=digest(BASE/'preparation.json'),base_vlm=str(BASE_VLM),
        base_vlm_file_sha256={name:digest(BASE_VLM/name) for name in files},
        seed=42,effective_batch=32,stages=dict(warmup=500,joint=4500),
        action_horizon=16,execute_horizon=16,oft_checkpoint_loaded=False,
        formal_initialization='Original VLM weights only; random ACT; formal warmup starts fresh after smoke.'))


if __name__=='__main__':main()
