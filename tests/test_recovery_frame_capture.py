import gc
import weakref

import pytest

from examples.Robotwin.audits.recovery_frame_capture import record_frame_phases


class Environment:
    FRAME_IDX = 0

    def _take_picture(self):
        self.FRAME_IDX += 1


@pytest.mark.parametrize('fail', [False, True])
def test_recorder_restores_method_and_releases_environment(fail):
    env = Environment()
    reference = weakref.ref(env)
    phases, counts = [], []
    phase = 'handoff'
    try:
        with record_frame_phases(env, phases, lambda: phase, counts.append):
            env._take_picture()
            phase = 'expert_rgb_replan'
            env._take_picture()
            if fail:
                raise RuntimeError('expert failure')
    except RuntimeError:
        assert fail
    assert phases == ['handoff', 'expert_rgb_replan'] and counts == [1, 2]
    assert '_take_picture' not in env.__dict__
    env._take_picture()
    assert env.FRAME_IDX == 3 and counts == [1, 2]
    env = None
    gc.collect()
    assert reference() is None


def test_existing_instance_hook_is_restored():
    env = Environment()
    calls = []
    def original():
        env.FRAME_IDX += 1
        calls.append('original')
    env._take_picture = original
    with record_frame_phases(env, [], lambda: 'phase', lambda count: None):
        env._take_picture()
    assert env._take_picture is original and calls == ['original']


def test_incomplete_capture_restores_hook_on_assertion():
    env = Environment()
    original = lambda: None
    env._take_picture = original
    with pytest.raises(AssertionError):
        with record_frame_phases(env, [], lambda: 'phase', lambda count: None):
            env._take_picture()
    assert env._take_picture is original
