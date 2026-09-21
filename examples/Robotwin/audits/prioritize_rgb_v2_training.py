"""Temporarily pause our color sweep, restore it after v2 training exits."""
import json
import os
from pathlib import Path
import signal
import time

import psutil

ROOT = Path(__file__).resolve().parents[3]
CAMP = ROOT / 'playground/Checkpoints/gawm_rgb_focus_20260907'
PAIR = ROOT / 'playground/Checkpoints/gawm_rgb_color_pair10_20260907'
V2 = ROOT / 'playground/Checkpoints/gawm_rgb_focus_local_v2_5k_20260907'
STATE = CAMP / 'color_pause_during_v2.json'


def main():
    if STATE.exists():
        raise RuntimeError('Pause lease already exists; inspect before repeating')
    color = json.loads((PAIR/'manifest.json').read_text())
    train = json.loads((V2/'supervisor_status.json').read_text())
    assert train['stage'] == 'train' and train['state'] == 'running'
    parent = psutil.Process(color['supervisor_pid'])
    training = psutil.Process(train['pid'])
    assert str(ROOT/'examples/Robotwin/audits/run_rgb_color_diagnostic.py') in parent.cmdline()
    assert str(PAIR) in parent.cmdline()
    assert str(ROOT/'examples/Robotwin/train_files/starvla_gawm_rgb_focus_local_v2.yaml') in training.cmdline()
    paused = []

    def save(state):
        payload = dict(state=state, watcher_pid=os.getpid(), training_pid=training.pid,
                       training_birth=training.create_time(), paused=paused,
                       time=time.strftime('%Y-%m-%d %H:%M:%S'))
        temporary = STATE.with_suffix('.tmp')
        temporary.write_text(json.dumps(payload, indent=2)+'\n')
        temporary.replace(STATE)

    def pause(process):
        if process.status() == psutil.STATUS_STOPPED:
            return
        item = dict(pid=process.pid, birth=process.create_time(), cmd=process.cmdline())
        os.kill(process.pid, signal.SIGSTOP)
        paused.append(item)
        save('pausing')

    def stop(signum, frame):
        raise RuntimeError(f'Stopping pause watcher on signal {signum}')

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop)
    try:
        pause(parent)
        for _ in range(3):
            for process in parent.children(recursive=True):
                try:
                    pause(process)
                except psutil.NoSuchProcess:
                    pass
        save('paused_for_training')
        while training.is_running() and training.status() != psutil.STATUS_ZOMBIE:
            time.sleep(15)
    finally:
        # Only resume identities this watcher paused, descendants before parent.
        for item in reversed(paused):
            try:
                process = psutil.Process(item['pid'])
                if process.create_time() == item['birth']:
                    os.kill(process.pid, signal.SIGCONT)
            except psutil.NoSuchProcess:
                pass
        save('resumed')


if __name__ == '__main__':
    main()
