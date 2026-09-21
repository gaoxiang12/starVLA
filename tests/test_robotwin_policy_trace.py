import json
from types import SimpleNamespace

import numpy as np
import pytest

from examples.Robotwin.eval_files.policy_input_trace import trace_policy_call


def test_trace_preserves_exact_call_and_pickle_free_replay(tmp_path):
    native = np.full((3, 24, 32, 3), [231, 52, 11], dtype=np.uint8)
    request = dict(examples=[dict(image=native[:, :22, :22], native_images=native,
                                  image_history=native[None], state=np.arange(14, dtype=np.float32),
                                  lang='blocks_ranking_rgb', episode_start=True)],
                   unnorm_key='aloha', do_sample=False)
    response = dict(data=dict(actions=np.arange(224, dtype=np.float32).reshape(1, 16, 14)))
    calls = []

    def predict(payload):
        assert payload is request
        calls.append(payload)
        return response

    client = SimpleNamespace(predict_action=predict)
    path = tmp_path / 'trace.npz'
    with trace_policy_call(client, path):
        assert client.predict_action(request) is response
    assert client.predict_action is predict
    assert len(calls) == 1
    with np.load(path, allow_pickle=False) as saved:
        for key in ('image', 'native_images', 'image_history', 'state'):
            np.testing.assert_array_equal(saved[key], request['examples'][0][key])
        np.testing.assert_array_equal(saved['actions'], response['data']['actions'])
        assert json.loads(saved['metadata'].item())['episode_start'] is True
    np.testing.assert_array_equal(native[0, 0, 0], [231, 52, 11])


def test_trace_restores_client_when_policy_fails(tmp_path):
    def predict(payload):
        raise RuntimeError('policy failed')

    client = SimpleNamespace(predict_action=predict)
    with pytest.raises(RuntimeError, match='policy failed'):
        with trace_policy_call(client, tmp_path / 'trace.npz'):
            client.predict_action({})
    assert client.predict_action is predict
    assert not (tmp_path / 'trace.npz').exists()
