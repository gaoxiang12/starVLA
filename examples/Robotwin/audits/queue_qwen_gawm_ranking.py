"""Gate the three formal runs on smoke completion, then produce paired reports."""
import json
import os
import subprocess
import time

import psutil

from examples.Robotwin.audits.prepare_qwen_gawm_ranking import ROOT, OUT, digest, save

PYTHON=ROOT.parent/'.venvs/starVLA/bin/python'
VARIANTS=('qwen_act','qwen_mlp','gawm')


def main():
    path=OUT/'queue_status.json'
    assert not path.exists()
    report=dict(state='waiting_for_smokes',pid=os.getpid(),birth=psutil.Process().create_time(),runs={})
    pinned={str(p):digest(p) for p in (ROOT/'examples/Robotwin/audits/run_qwen_gawm_ranking.py',
        ROOT/'examples/Robotwin/audits/summarize_qwen_gawm_ranking.py',OUT/'preparation.json',OUT/'protocol.json')}
    def status():save(path,dict(report,time=time.strftime('%Y-%m-%d %H:%M:%S'),source_sha256=pinned))
    def check():
        for p,sha in pinned.items():assert digest(p)==sha,f'Queued source changed: {p}'
    children={}
    try:
        while True:
            states={v:json.loads((OUT/f'{v}_smoke_status.json').read_text()) for v in VARIANTS}
            report['smokes']={v:r['state'] for v,r in states.items()}
            if any(r['state']=='failed' for r in states.values()):raise RuntimeError('A smoke failed; formal runs were not started')
            if all(r['state']=='complete' for r in states.values()):break
            status();time.sleep(10)
        # Smoke scenes must already be identical across all three models.
        initial=[]
        for variant in VARIANTS:
            result=json.loads((OUT/f'{variant}_smoke_eval/development/scene_000/result.json').read_text())
            assert result['state']=='complete' and result['seed']==41000000
            initial.append(result['initial'])
        assert initial[0]==initial[1]==initial[2]
        check()
        for variant,gpu,port in [('qwen_act','1',29910),('qwen_mlp','2',29920),('gawm','3',29930)]:
            command=[str(PYTHON),'-u','-m','examples.Robotwin.audits.run_qwen_gawm_ranking',
                     '--variant',variant,'--gpu',gpu,'--train-port',str(port),'--eval-port',str(6920+int(gpu))]
            with (OUT/f'{variant}_campaign_supervisor.log').open('x') as log:
                proc=subprocess.Popen(command,cwd=ROOT,env=dict(os.environ,PYTHONPATH=str(ROOT)),
                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            children[variant]=proc
            report['runs'][variant]=dict(pid=proc.pid,birth=psutil.Process(proc.pid).create_time(),gpu=gpu,command=command)
            save(OUT/f'{variant}_campaign_launch.json',report['runs'][variant])
        report['state']='running';status()
        summarized=set()
        while True:
            check()
            for variant,proc in children.items():
                state_path=OUT/f'{variant}_campaign_status.json'
                state=json.loads(state_path.read_text()) if state_path.exists() else dict(state='starting')
                report['runs'][variant]['state']=state['state']
                if state['state']=='failed' or (proc.poll() is not None and proc.returncode!=0):
                    raise RuntimeError(f'{variant} failed; unaffected independent runs keep their own supervision')
            for split,count in [('development',20),('test',100)]:
                if split in summarized:continue
                if all(len(list((OUT/f'{v}_campaign_eval'/split).glob('scene_*/result.json')))==count for v in VARIANTS):
                    command=[str(PYTHON),'-m','examples.Robotwin.audits.summarize_qwen_gawm_ranking','--split',split]
                    with (OUT/f'{split}_comparison.log').open('x') as log:
                        subprocess.run(command,cwd=ROOT,check=True,stdout=log,stderr=subprocess.STDOUT)
                    summarized.add(split)
            report['summarized_splits']=sorted(summarized)
            if all(proc.poll()==0 for proc in children.values()) and summarized=={'development','test'}:break
            status();time.sleep(10)
        report['state']='complete'
    except BaseException as error:
        report.update(state='failed',error=repr(error));raise
    finally:status()


if __name__=='__main__':main()
