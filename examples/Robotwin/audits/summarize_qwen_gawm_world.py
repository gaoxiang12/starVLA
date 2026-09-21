"""Pair Qwen + world-model ACT with RoboTwin OFT-initialized ACT under the same 16/16 protocol."""
import argparse
import json

from examples.Robotwin.audits.prepare_qwen_gawm_world import BASE,OUT,save
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
        assert a['execute_horizon']==16 and b['execute_horizon']==16 and b['action_horizon']==16
        assert not b['engineering_smoke'] and b['action_budget']==1200
        old.append(a['success']);new.append(b['success'])
    stats=paired_statistics(old,new)
    stats['gained_seeds']=[r['seed'] for i,r in enumerate(records) if new[i] and not old[i]]
    stats['lost_seeds']=[r['seed'] for i,r in enumerate(records) if old[i] and not new[i]]
    result=dict(state='complete',split=split,comparison='OFT Qwen + world-model ACT16 vs RoboTwin OFT Qwen ACT16',
        successes_world=sum(new),scenes=len(new),wilson_95_ci_world=wilson(sum(new),len(new)),
        successes_oft_initialized=sum(old),paired=stats,limitation=protocol['limitation'],pretraining_overlap=protocol['pretraining_overlap'])
    save(OUT/f'{split}_comparison.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--split',choices=['development','test'],required=True)
    print(json.dumps(summarize(p.parse_args().split),indent=2))
