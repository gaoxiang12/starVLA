from unittest.mock import patch

import numpy as np
import pytest

from examples.Robotwin.eval_files.model2robotwin_interface import ModelClient


class FakePolicy:
    def __init__(self, *args):
        self.calls = 0

    def get_server_metadata(self):
        return dict(action_chunk_size=16)

    def predict_action(self, payload):
        self.calls += 1
        actions = np.repeat((100*(self.calls-1)+np.arange(16))[:, None], 14, axis=1).astype(np.float32)
        return dict(data=dict(actions=actions[None]))


def example():
    return dict(lang='blocks ranking rgb', image=[np.zeros((24, 32, 3), np.uint8)]*3,
                state=np.zeros(14, np.float32))


@patch('examples.Robotwin.eval_files.model2robotwin_interface.WebsocketClientPolicy', FakePolicy)
def test_short_execution_starts_each_new_prediction_at_zero():
    model = ModelClient('unused', execute_horizon=4)
    values = [model.step(example(), step=i)[0] for i in range(7)]
    assert values == [0, 1, 2, 3, 100, 101, 102]
    assert model.client.calls == 2
    assert model.action_chunk_size == 16


@patch('examples.Robotwin.eval_files.model2robotwin_interface.WebsocketClientPolicy', FakePolicy)
def test_default_executes_the_full_model_chunk():
    model = ModelClient('unused')
    values = [model.step(example(), step=i)[0] for i in range(17)]
    assert values == list(range(16)) + [100]
    assert model.execute_horizon == 16


@pytest.mark.parametrize('horizon', [0, -1, 17])
@patch('examples.Robotwin.eval_files.model2robotwin_interface.WebsocketClientPolicy', FakePolicy)
def test_execution_horizon_must_fit_prediction(horizon):
    with pytest.raises(ValueError, match='execute_horizon'):
        ModelClient('unused', execute_horizon=horizon)
