"""Contract checks preventing plausible but invalid OFT/GAWM comparisons."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from examples.Robotwin.audits.oft_reference_adapter import INSTRUCTION, UNNORM_KEY, adapt_client
from examples.Robotwin.audits.serve_oft_reference import audit_normalization, float_head_input, CheckedPolicy
from examples.Robotwin.eval_files import model2robotwin_interface as interface


@pytest.mark.parametrize('horizon', [16, 50])
def test_real_client_omits_state_and_preserves_rgb_action_order(monkeypatch, horizon):
    calls = []
    action = np.arange(50*14, dtype=np.float32).reshape(50, 14)

    class Client:
        def __init__(self, *args):
            pass

        def get_server_metadata(self):
            return dict(action_chunk_size=50, task_language_mode='metadata')

        def predict_action(self, payload):
            calls.append(payload)
            return dict(data=dict(actions=action[None]))

    monkeypatch.setattr(interface, 'WebsocketClientPolicy', Client)
    client = adapt_client(interface.ModelClient('unused', unnorm_key=UNNORM_KEY, execute_horizon=horizon))
    images = [np.full((240, 320, 3), color, np.uint8) for color in ([255, 0, 0], [0, 255, 0], [0, 0, 255])]
    source = dict(image=images, lang='blocks ranking rgb', state=np.arange(14), object_xyz='must not pass')
    first = client.step(source, step=0)
    last = client.step(source, step=horizon-1)
    next_chunk = client.step(source, step=horizon)
    assert len(calls) == 2
    assert calls[0]['examples'][0]['episode_start'] is True
    assert calls[1]['examples'][0]['episode_start'] is False
    for call in calls:
        example = call['examples'][0]
        assert 'state' not in example and 'object_xyz' not in example
        assert example['lang'] == INSTRUCTION and call['unnorm_key'] == UNNORM_KEY
        for im, color in zip(example['image'], ([255, 0, 0], [0, 255, 0], [0, 0, 255])):
            assert im.shape == (224, 224, 3)
            np.testing.assert_array_equal(im[0, 0], color)
    np.testing.assert_array_equal(first, action[0, interface.MODEL_TO_ROBOTWIN_JOINT_ORDER])
    np.testing.assert_array_equal(last, action[horizon-1, interface.MODEL_TO_ROBOTWIN_JOINT_ORDER])
    np.testing.assert_array_equal(next_chunk, first)
    assert 'state' in source and source['lang'] == 'blocks ranking rgb'


def test_fp32_head_cast_matches_explicit_reference():
    from starVLA.model.modules.action_model.MLP_ActionHeader import MLPResNet
    head = MLPResNet(2, 8, 16, 14).float().eval()
    inputs = torch.randn(50, 8).bfloat16()
    expected = head(inputs.float())
    head.register_forward_pre_hook(float_head_input)
    actual = head(inputs)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert actual.dtype == torch.float32 and np.isfinite(actual.detach().numpy()).all()


def test_released_statistics_with_training_normalizer(tmp_path):
    from deployment.model_server.policy_norm_processor import PolicyNormProcessor
    source = Path('/data/gaoxiang/ckpts/Qwen3-VL-OFT-RoboTwin2-All')
    assert (source/'dataset_statistics.json').is_file()
    for name in ('config.yaml', 'dataset_statistics.json'):
        (tmp_path/name).write_bytes((source/name).read_bytes())
    checkpoint = tmp_path/'checkpoints/metadata_only.pt'
    checkpoint.parent.mkdir()
    checkpoint.touch()  # Only metadata is read; no checkpoint deserialization.
    proc = PolicyNormProcessor(str(checkpoint), unnorm_key=UNNORM_KEY)
    wrapper = SimpleNamespace(_framework=SimpleNamespace(norm_stats=json.loads(
        (source/'dataset_statistics.json').read_text())), _get_processor=lambda key: proc)
    assert audit_normalization(wrapper)['maximum_difference'] <= 1e-6


def test_server_rejects_state_and_logs_actual_valid_payload(tmp_path):
    received = []

    class Wrapper:
        def predict_action(self, examples, unnorm_key):
            received.extend(examples)
            return dict(actions=np.zeros((1, 50, 14), np.float32))

    policy = CheckedPolicy(Wrapper(), tmp_path/'queries.jsonl')
    example = dict(image=[np.zeros((224, 224, 3), np.uint8)]*3, lang=INSTRUCTION, episode_start=True)
    with pytest.raises(AssertionError):
        policy.predict_action([dict(example, state=np.zeros(14))], unnorm_key=UNNORM_KEY)
    assert not received
    policy.predict_action([example], unnorm_key=UNNORM_KEY)
    assert set(received[0]) == {'image', 'lang'}
    row = json.loads((tmp_path/'queries.jsonl').read_text())
    assert row['episode_index'] == row['query_index'] == 0 and row['state_present'] is False
