import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from examples.Robotwin.eval_files.run_robotwin_eval_retry import run_with_retry


@pytest.mark.parametrize('failure,expected_calls', [('vk::Device::waitForFences: ErrorDeviceLost', 2), ('ValueError: invalid action', 1)])
def test_only_vulkan_failure_retries_and_partial_metrics_stay_separate(tmp_path, failure, expected_calls):
    calls = []

    def runner(command, env):
        calls.append(command)
        metrics = Path(env['ROBOTWIN_RANKING_METRICS_PATH'])
        metrics.write_text(json.dumps({'attempt': len(calls), 'success': False}) + '\n')
        if len(calls) == 1:
            (Path(env['ROBOTWIN_LOG_ROOT']) / 'task_eval.log').write_text(failure)
            return SimpleNamespace(returncode=134)
        return SimpleNamespace(returncode=0)

    metrics, logs = tmp_path/'metrics.jsonl', tmp_path/'logs'
    command = ['evaluation', '--seed', '0']
    result = run_with_retry(command, {}, metrics, logs, runner=runner)
    assert len(calls) == expected_calls and all(c == command for c in calls)
    assert result == (0 if expected_calls == 2 else 134)
    assert json.loads(metrics.read_text())['attempt'] == expected_calls
    assert json.loads((tmp_path/'metrics.attempt_1.jsonl').read_text())['attempt'] == 1


def test_zero_success_rate_is_completed_evaluation_and_is_not_retried(tmp_path):
    calls = []

    def runner(command, env):
        calls.append(command)
        Path(env['ROBOTWIN_RANKING_METRICS_PATH']).write_text('{"success": false}\n')
        return SimpleNamespace(returncode=0)

    assert run_with_retry(['evaluation'], {}, tmp_path/'metrics.jsonl', tmp_path/'logs', runner=runner) == 0
    assert len(calls) == 1
