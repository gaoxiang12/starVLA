"""Pair 50/50 with the existing 16/16 ACT group, keeping the protocol distinction."""
import argparse
import json

from examples.Robotwin.audits.prepare_qwen_gawm_chunk50 import BASE,OUT,save
from examples.Robotwin.audits.summarize_qwen_gawm_ranking import paired_statistics,wilson


def summarize(split):
    protocol=json.loads((OUT/'protocol.json').read_text())
    records=[r for r in protocol['records'] if r['split']==split]
    old=[];new=[]
    for record in records:
        name=f"scene_{record['scene_id']:03d}/result.json"
        a=json.loads((BASE/'qwen_act_campaign_eval'/split/name).read_text())
        b=json.loads((OUT/'eval'/split/name).read_text())
        assert a['state']==b['state']=='complete' and a['seed']==b['seed']==record['seed']
        assert a['initial']==b['initial'] and not a['training_scene_matches'] and not b['training_scene_matches']
        assert a['step_limit']==b['step_limit']==1200
        assert a['execute_horizon']==16 and b['execute_horizon']==50 and b['action_horizon']==50
        assert not b['engineering_smoke'] and b['action_budget']==1200
        old.append(a['success']);new.append(b['success'])
    stats=paired_statistics(old,new)
    stats['gained_seeds']=[r['seed'] for i,r in enumerate(records) if new[i] and not old[i]]
    stats['lost_seeds']=[r['seed'] for i,r in enumerate(records) if old[i] and not new[i]]
    result=dict(state='complete',split=split,comparison='Qwen ACT 50/50 vs 16/16',
        successes_50=sum(new),scenes=len(new),wilson_95_ci_50=wilson(sum(new),len(new)),
        paired=stats,limitation=protocol['comparison'],pretraining_overlap=protocol['pretraining_overlap'])
    save(OUT/f'{split}_comparison.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--split',choices=['development','test'],required=True)
    print(json.dumps(summarize(p.parse_args().split),indent=2))
