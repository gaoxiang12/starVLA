"""Summarize exactly 20 paired development scenes; never wait for final-test data."""
import argparse,json
from examples.Robotwin.audits.prepare_qwen_gawm_depth import OUT,BASE,VARIANTS,save,validate_protocol
from examples.Robotwin.audits.summarize_qwen_gawm_ranking import paired_statistics,wilson


def summarize(variant):
    dest=OUT/variant;protocol=json.loads((dest/'protocol.json').read_text());validate_protocol(protocol)
    a=[];b=[]
    for record in protocol['records']:
        name=f"scene_{record['scene_id']:03d}/result.json"
        old=json.loads((BASE/'qwen_act_campaign_eval/development'/name).read_text())
        new=json.loads((dest/'eval/development'/name).read_text())
        assert old['state']==new['state']=='complete' and old['seed']==new['seed']==record['seed']
        assert old['initial']==new['initial'] and not old['training_scene_matches'] and not new['training_scene_matches']
        assert old['step_limit']==new['step_limit']==new['action_budget']==1200
        assert old['execute_horizon']==new['execute_horizon']==new['action_horizon']==16
        assert not new['engineering_smoke']
        a.append(old['success']);b.append(new['success'])
    stats=paired_statistics(a,b)
    stats['gained_seeds']=[r['seed'] for i,r in enumerate(protocol['records']) if b[i] and not a[i]]
    stats['lost_seeds']=[r['seed'] for i,r in enumerate(protocol['records']) if a[i] and not b[i]]
    result=dict(state='complete',variant=variant,split='development',scene_identity_verified=True,
        successes=sum(b),scenes=20,success_rate=sum(b)/20,wilson_95_ci=wilson(sum(b),20),
        paired=stats,performance=json.loads((dest/'performance.json').read_text()),
        limitation=protocol['limitation'],pretraining_overlap=protocol['pretraining_overlap'],test_scenes=0)
    reference=OUT/'reference_performance.json'
    if reference.exists():result['reference_performance']=json.loads(reference.read_text())
    save(dest/'development_comparison.json',result)
    files=[OUT/v/'development_comparison.json' for v in VARIANTS]
    if all(p.exists() for p in files):
        save(OUT/'development_comparison.json',dict(state='complete',split='development',test_scenes=0,
            reference_successes=19,reference_scenes=20,variants={v:json.loads(p.read_text()) for v,p in zip(VARIANTS,files)}))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--variant',choices=VARIANTS,required=True)
    print(json.dumps(summarize(p.parse_args().variant),indent=2))
