"""Pair full-sorting outcomes only after scene identity and coverage checks."""
import argparse
import json
import math

import numpy as np

from examples.Robotwin.audits.prepare_qwen_gawm_ranking import OUT, save


def paired_statistics(baseline, candidate):
    a=np.asarray(baseline,dtype=bool);b=np.asarray(candidate,dtype=bool)
    if a.ndim!=1 or a.shape!=b.shape or not len(a):
        raise ValueError('Nonempty paired outcomes required')
    diff=b.astype(float)-a.astype(float)
    rng=np.random.default_rng(20260910)
    bootstrap=diff[rng.integers(0,len(a),(20000,len(a)))].mean(1)
    gained=int((b&~a).sum());lost=int((a&~b).sum());discordant=gained+lost
    exact=min(1.,2*sum(math.comb(discordant,k) for k in range(min(gained,lost)+1))/2**discordant) if discordant else 1.
    return dict(scenes=len(a),baseline_successes=int(a.sum()),candidate_successes=int(b.sum()),
        difference_percentage_points=float(diff.mean()*100),
        paired_bootstrap_95_percentile_ci_pp=(np.quantile(bootstrap,[.025,.975])*100).tolist(),
        gained=gained,lost=lost,exact_mcnemar_p=exact,
        uncertainty_note='Scene sampling uncertainty for one training seed; not training-seed variability.')


def wilson(successes,total):
    z=1.959963984540054;p=successes/total;den=1+z*z/total
    center=(p+z*z/(2*total))/den
    half=z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/den
    return [center-half,center+half]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split',choices=['development','test'],required=True)
    args=parser.parse_args()
    protocol=json.loads((OUT/'protocol.json').read_text())
    records=[r for r in protocol['records'] if r['split']==args.split]
    results={}
    for variant in ('gawm','qwen_mlp','qwen_act'):
        results[variant]=[]
        for record in records:
            p=OUT/f'{variant}_campaign_eval'/args.split/f"scene_{record['scene_id']:03d}"/'result.json'
            value=json.loads(p.read_text())
            assert value['state']=='complete' and value['seed']==record['seed']
            assert value['step_limit']==1200 and value['execute_horizon']==16
            assert not value['training_scene_matches']
            results[variant].append(value)
    for index in range(len(records)):
        a=results['gawm'][index]['initial']
        for variant in ('qwen_mlp','qwen_act'):
            b=results[variant][index]['initial']
            assert a==b,f'Scene mismatch for {variant}, seed {records[index]["seed"]}'
    outcomes={v:[r['success'] for r in rows] for v,rows in results.items()}
    comparison={}
    for base in ('gawm','qwen_mlp'):
        stats=paired_statistics(outcomes[base],outcomes['qwen_act'])
        stats['gained_seeds']=[r['seed'] for i,r in enumerate(records) if outcomes['qwen_act'][i] and not outcomes[base][i]]
        stats['lost_seeds']=[r['seed'] for i,r in enumerate(records) if outcomes[base][i] and not outcomes['qwen_act'][i]]
        comparison[f'qwen_act_vs_{base}']=stats
    per_model={v:dict(successes=sum(outcomes[v]),total=len(records),success_rate=float(np.mean(outcomes[v])),
        wilson_95_ci=wilson(sum(outcomes[v]),len(records)),
        held_red_green_blue=np.asarray([r['held_red_green_blue'] for r in rows]).sum(0).tolist(),
        correct_final_color_order=sum(r['arrangement']['correct_color_order'] for r in rows),
        action_budget_failures=sum(r['termination']=='action_budget' for r in rows)) for v,rows in results.items()}
    save(OUT/f'{args.split}_comparison.json',dict(state='complete',split=args.split,
        scene_identity_verified=True,models=per_model,paired_comparisons=comparison,
        pretraining_overlap=protocol['pretraining_overlap']))
    print(json.dumps(per_model,indent=2))


if __name__=='__main__':main()
