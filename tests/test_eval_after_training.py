import json
from pathlib import Path

import pytest

from scripts.eval_libero_after_training import free_gpus, training_checkpoint


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_running_or_failed_training_never_triggers_eval(tmp_path):
    assert training_checkpoint(tmp_path, 100) is None
    write(tmp_path / 'cluster_status.json', {'status': 'running'})
    assert training_checkpoint(tmp_path, 100) is None
    for status in ('failed', 'stopped'):
        write(tmp_path / 'cluster_status.json', {'status': status})
        with pytest.raises(RuntimeError, match=status):
            training_checkpoint(tmp_path, 100)


def test_success_requires_exact_final_step_and_all_rng_files(tmp_path):
    write(tmp_path / 'cluster_status.json', {'status': 'complete', 'world_size': 2})
    write(tmp_path / 'training_complete.json', {'completed_steps': 100})
    state = tmp_path / 'checkpoints/steps_100_training_state'
    write(state / 'complete.json', {'step': 100, 'world_size': 2})
    for path in (tmp_path / 'checkpoints/steps_100_pytorch_model.pt', state / 'optimizer.bin',
                 state / 'scheduler.bin', tmp_path / 'config.full.yaml',
                 tmp_path / 'dataset_statistics.json', state / 'random_states_0.pkl'):
        path.write_bytes(b'fixture')
    with pytest.raises(RuntimeError, match='random_states_1'):
        training_checkpoint(tmp_path, 100)
    (state / 'random_states_1.pkl').write_bytes(b'fixture')
    assert training_checkpoint(tmp_path, 100).name == 'steps_100_pytorch_model.pt'
    write(tmp_path / 'training_complete.json', {'completed_steps': 99})
    with pytest.raises(RuntimeError, match='step'):
        training_checkpoint(tmp_path, 100)


def test_occupied_gpus_are_not_taken(monkeypatch):
    monkeypatch.setattr('scripts.eval_libero_after_training.subprocess.check_output',
                        lambda *a, **k: '0, 6779\n1, 0\n2, 1900\n3, 2\n')
    assert free_gpus([0, 1, 2, 3]) == [1, 3]
