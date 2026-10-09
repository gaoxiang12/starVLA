"""Record bounded-scale, fixed-teacher and action metrics; stop only on nonfinite values."""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import statistics
import time


def review(rows):
    keys=('dino_future_loss','dino_current_loss','dino_copy_loss','dino_to_copy_ratio',
          'l1_action_loss','predicted_latent_rms','visual_content_rms','latent_batch_std',
          'temporal_observed_delta_rms','temporal_smoothness_loss','temporal_dense_loss',
          'temporal_dense_weighted_loss','temporal_dense_valid_fraction')
    step=rows[-1]['step']
    recent=[r for r in rows if r['step']>step-2000]
    earlier=[r for r in rows if step-4000<r['step']<=step-2000]
    def means(data):
        return {k:statistics.mean(r[k] for r in data if k in r) for k in keys if any(k in r for r in data)}
    a,b=means(recent),means(earlier)
    finite=all(math.isfinite(v) for row in rows for v in row.values() if isinstance(v,(int,float)))
    warnings=[]
    if not finite: warnings.append('nonfinite_metrics')
    if abs(a.get('predicted_latent_rms',1.)-1.)>.02: warnings.append('prediction_scale_outside_expected_range')
    if step>=4000 and a.get('dino_to_copy_ratio',0)>1.: warnings.append('future_prediction_worse_than_copy_current')
    if b and a.get('dino_future_loss',0)>b.get('dino_future_loss',float('inf'))*1.2: warnings.append('fixed_teacher_error_up_over_20_percent')
    if b and a.get('l1_action_loss',0)>b.get('l1_action_loss',float('inf'))*1.2: warnings.append('action_error_up_over_20_percent')
    return dict(step=step,finite=finite,recent_2000=a,previous_2000=b,warnings=warnings,
                interpretation='Trend alerts are diagnostic; finite values do not establish convergence or rollout quality.')


def main():
    p=argparse.ArgumentParser();p.add_argument('--queue',type=Path,required=True);args=p.parse_args()
    q=args.queue.resolve();run=Path(json.loads((q/'plan.json').read_text())['run'])
    while True:
        result=dict(checked_at=time.time(),status='waiting_training',run=str(run))
        status=json.loads((q/'status.json').read_text()) if (q/'status.json').exists() else {}
        metrics=run/'metrics.jsonl'
        if metrics.exists():
            # Ignore only an in-progress trailing write; malformed complete rows are errors.
            data=metrics.read_text();lines=data.splitlines()
            if data and not data.endswith('\n'):lines=lines[:-1]
            rows=[json.loads(line) for line in lines if line]
            if rows:
                result.update(review(rows),status=status.get('status'),metrics_age_seconds=time.time()-metrics.stat().st_mtime)
                if not result['finite'] and status.get('status')=='training':
                    pid=status['pid'];cmdline=Path(f'/proc/{pid}/cmdline').read_text()
                    if str(q/'run_experiment.py') not in cmdline:raise RuntimeError('Controller identity mismatch')
                    os.kill(pid,signal.SIGTERM);result['stop_requested']='nonfinite_metrics'
        target=q/'latent_monitor.json';temp=target.with_suffix('.tmp')
        temp.write_text(json.dumps(result,indent=2)+'\n');temp.replace(target)
        if status.get('status') in ('complete','failed','stopped'):return
        time.sleep(30)

if __name__=='__main__':main()
