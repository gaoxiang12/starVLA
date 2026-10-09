"""Persist latent-scale health observations; never stop or modify training."""
import argparse
from collections import deque
import datetime
import fcntl
import json
import math
from pathlib import Path
import statistics
import time

FIELDS = ('latent_loss', 'l1_action_loss', 'delta_target_rms', 'delta_copy_mse',
          'delta_to_copy_ratio', 'delta_scale', 'visual_content_rms',
          'visual_tokens_rms', 'latent_mse_over_delta_scale_sq')


def assess(rows):
    alerts = []
    if not rows:
        return dict(health='waiting_for_metrics', alerts=[], window={})
    recent = rows[-50:]
    for row in recent:
        if any(isinstance(v, (int, float)) and not math.isfinite(v) for v in row.values()):
            alerts.append('nonfinite_metric')
            break
    if any(k not in row for row in recent for k in FIELDS):
        alerts.append('missing_scale_metrics')
    window = {}
    for key in FIELDS:
        values = [r[key] for r in recent if key in r and math.isfinite(r[key])]
        if values:
            window[key] = dict(mean=statistics.mean(values), median=statistics.median(values),
                               minimum=min(values), maximum=max(values))
    # A fixed, non-affine LN should keep content RMS very close to one.
    tail = rows[-3:]
    if len(tail) == 3 and all(not .95 <= r.get('visual_content_rms', 1.) <= 1.05 for r in tail):
        alerts.append('content_scale_outside_0.95_1.05')
    if len(recent) >= 20:
        if window.get('latent_loss', {}).get('median', 0) > 4:
            alerts.append('sustained_large_absolute_latent_mse')
        if window.get('delta_to_copy_ratio', {}).get('median', 0) > 1.5:
            alerts.append('sustained_prediction_worse_than_copy')
    if len(rows) >= 100:
        previous = rows[-100:-50]
        previous_loss = statistics.median([r['latent_loss'] for r in previous if math.isfinite(r['latent_loss'])]) if all(math.isfinite(r.get('latent_loss', math.nan)) for r in previous) else None
        previous_relative = statistics.median([r['latent_mse_over_delta_scale_sq'] for r in previous if math.isfinite(r['latent_mse_over_delta_scale_sq'])]) if all(math.isfinite(r.get('latent_mse_over_delta_scale_sq', math.nan)) for r in previous) else None
        if (previous_loss is not None and previous_relative is not None
                and "latent_loss" in window and "latent_mse_over_delta_scale_sq" in window):
            if (window['latent_loss']['median'] > max(1., 4 * previous_loss)
                    and window['latent_mse_over_delta_scale_sq']['median'] > max(1., 2 * previous_relative)):
                alerts.append('absolute_and_scale_normalized_loss_rising')
    return dict(health='warning' if alerts else 'healthy', alerts=alerts, window=window,
                window_records=len(recent), window_first_step=recent[0]['step'], last_step=rows[-1]['step'])


def write_json(path, value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def main(args):
    run=Path(args.run).resolve();out=run/'latent_monitor';out.mkdir(exist_ok=True)
    lock=(out/'monitor.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    rows=deque(maxlen=300);offset=0;records=0;nonfinite_steps=[];last_update=time.time();last_record_step=-1
    while True:
        path=run/'metrics.jsonl'
        if path.exists():
            if path.stat().st_size < offset:raise RuntimeError('Metrics file was truncated')
            with path.open() as stream:
                stream.seek(offset)
                while True:
                    pos=stream.tell();line=stream.readline()
                    if not line or not line.endswith('\n'):
                        offset=pos;break
                    row=json.loads(line)
                    if row['step']<=last_record_step:raise RuntimeError('Non-increasing optimizer steps')
                    last_record_step=row['step'];rows.append(row);records+=1;last_update=time.time()
                    if any(isinstance(v,(int,float)) and not math.isfinite(v) for v in row.values()):nonfinite_steps.append(row['step'])
        cluster_path=run/'cluster_status.json'
        cluster=json.loads(cluster_path.read_text()) if cluster_path.exists() else {'status':'starting'}
        result=assess(list(rows))
        if nonfinite_steps and 'nonfinite_metric' not in result['alerts']:result['alerts'].append('historical_nonfinite_metric')
        if cluster['status'] in ('starting','running') and time.time()-last_update>900:result['alerts'].append('no_new_metrics_for_15_minutes')
        if cluster['status']=='failed':result['alerts'].append('training_failed')
        if result['alerts']:result['health']='warning'
        result.update(run=str(run),training_status=cluster['status'],checked_at=datetime.datetime.now().isoformat(),
                      records_seen=records,nonfinite_steps=nonfinite_steps,latest=dict(rows[-1]) if rows else None,
                      automatic_training_control=False,thresholds_note='Heuristic warnings; healthy does not prove convergence or rollout quality.')
        write_json(out/'status.json',result)
        with (out/'history.jsonl').open('a') as stream:stream.write(json.dumps({k:v for k,v in result.items() if k!='latest'})+'\n')
        print(json.dumps({k:result.get(k) for k in ('checked_at','training_status','health','last_step','alerts')}),flush=True)
        if cluster['status'] in ('complete','stopped','failed'):break
        if args.once:break
        time.sleep(args.poll_seconds)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);p.add_argument('--poll-seconds',type=int,default=60);p.add_argument('--once',action='store_true');args=p.parse_args()
    if not 1<=args.poll_seconds<=60:p.error('poll-seconds must be in [1,60]')
    main(args)
