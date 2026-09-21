"""Matched visual/half-language ablations; development-only 20-scene campaign."""
import json
from pathlib import Path
from omegaconf import OmegaConf
from examples.Robotwin.audits.prepare_qwen_gawm_ranking import ROOT,OUT as BASE,QWEN,digest,save
OUT=ROOT/'playground/Checkpoints/qwen_gawm_depth_20260911'
SHARED=Path('/data/gaoxiang/ckpts/qwen_gawm_depth_20260911')
VARIANTS=('vit','merger','half')


def validate_config(cfg,stage,variant):
    assert variant in VARIANTS and stage in ('smoke','warmup','joint')
    assert cfg.framework.name=='QwenGAWMDepth' and cfg.framework.depth_variant==variant
    assert not cfg.trainer.is_resume
    expected=str(SHARED/f'qwen_depth_{variant}_warmup/final_model/pytorch_model.pt') if stage=='joint' else str(QWEN)
    assert cfg.trainer.pretrained_checkpoint==expected
    expected_layers=4 if variant=='half' and stage!='warmup' else 0
    assert cfg.framework.qwen_training.train_last_n_layers==expected_layers
    actual=OmegaConf.to_container(cfg,resolve=True)
    reference=OmegaConf.to_container(OmegaConf.load(BASE/f'qwen_act_{stage}.yaml'),resolve=True)
    actual['framework'].pop('depth_variant');actual['framework']['name']=reference['framework']['name']
    actual['framework']['qwen_training']=reference['framework']['qwen_training']
    for key in ('run_id','run_root_dir'):actual[key]=reference[key]
    actual['trainer']['pretrained_checkpoint']=reference['trainer']['pretrained_checkpoint']
    assert actual==reference,'Unexpected change from ACT16 training protocol'


def validate_protocol(protocol):
    assert protocol['test_scenes']==0 and protocol['development_scenes']==20
    assert len(protocol['records'])==20
    assert [r['seed'] for r in protocol['records']]==list(range(41000000,41000020))
    assert all(r['split']=='development' for r in protocol['records'])


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    protocol=json.loads((BASE/'protocol.json').read_text())
    protocol['records']=[r for r in protocol['records'] if r['split']=='development']
    protocol.update(test_scenes=0,action_horizon=16,stop_after_development=True,
        comparison='OFT ViT-only, ViT-merger and first-18-language-layer ACT vs existing full-36-layer ACT',
        limitation='One task, one training seed, 20 fixed development scenes; not a final held-out test or training-seed confidence interval.')
    validate_protocol(protocol);save(OUT/'protocol.json',protocol)
    rows=[]
    for variant in VARIANTS:
        dest=OUT/variant;dest.mkdir();save(dest/'protocol.json',protocol)
        configs=[]
        for stage in ('smoke','warmup','joint'):
            cfg=OmegaConf.load(BASE/f'qwen_act_{stage}.yaml')
            cfg.framework.name='QwenGAWMDepth';cfg.framework.depth_variant=variant
            if variant!='half':cfg.framework.qwen_training.train_last_n_layers=0
            cfg.run_root_dir=str(SHARED);cfg.run_id=f'qwen_depth_{variant}_{stage}'
            if stage=='joint':cfg.trainer.pretrained_checkpoint=str(SHARED/f'qwen_depth_{variant}_warmup/final_model/pytorch_model.pt')
            validate_config(cfg,stage,variant);path=dest/f'{stage}.yaml';OmegaConf.save(cfg,path)
            configs.append(dict(stage=stage,path=str(path),sha256=digest(path),steps=int(cfg.trainer.max_train_steps)))
        row=dict(state='prepared',variant=variant,configs=configs,source_checkpoint=str(QWEN),
            reference=str(BASE),effective_batch=32,seed=42,stages=dict(warmup=500,joint=4500),
            evaluation_scenes=20,evaluation_split='development',test_scenes=0)
        save(dest/'preparation.json',row);rows.append(row)
    save(OUT/'preparation.json',dict(state='prepared',variants=rows,qwen_checkpoint_sha256=digest(QWEN),
        protocol_sha256=digest(OUT/'protocol.json'),original_campaigns_remain_stopped=True))


if __name__=='__main__':main()
