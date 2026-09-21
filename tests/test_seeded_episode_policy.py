import json

import numpy as np
import pytest
import torch

from deployment.model_server.tools.seeded_episode_policy import SeededEpisodePolicy


class RandomPolicy:
    def __init__(self):
        self._framework = torch.nn.Linear(1, 1)
        self.fail = False

    def predict_action(self, examples, **kwargs):
        value = torch.randn(1, 16, 14).numpy()
        if self.fail:
            raise RuntimeError('temporary failure')
        return {'actions': value}


def test_episode_reset_repeats_noise_and_preserves_external_rng(tmp_path):
    policy = SeededEpisodePolicy(RandomPolicy(), 42, tmp_path/'rng.jsonl')
    original = torch.random.get_rng_state().clone()
    first = [policy.predict_action([{'episode_start': start}])['actions'] for start in [True, False]]
    torch.testing.assert_close(torch.random.get_rng_state(), original, atol=0, rtol=0)
    assert not np.array_equal(first[0], first[1])
    repeated = [policy.predict_action([{'episode_start': start}])['actions'] for start in [True, False]]
    for a, b in zip(first, repeated):
        np.testing.assert_array_equal(a, b)
    rows = [json.loads(line) for line in (tmp_path/'rng.jsonl').read_text().splitlines()]
    assert [r['query_index'] for r in rows] == [0, 1, 0, 1]
    assert [r['episode_index'] for r in rows] == [0, 0, 1, 1]
    assert rows[0]['action_sha256'] == rows[2]['action_sha256']


def test_bad_episode_contract_and_failure_do_not_advance_query():
    raw = RandomPolicy()
    policy = SeededEpisodePolicy(raw, 42)
    for examples in [[{}], [{'episode_start': False}], [{'episode_start': True}]*2]:
        with pytest.raises(ValueError):
            policy.predict_action(examples)
    original = torch.random.get_rng_state().clone()
    raw.fail = True
    with pytest.raises(RuntimeError):
        policy.predict_action([{'episode_start': True}])
    torch.testing.assert_close(torch.random.get_rng_state(), original, atol=0, rtol=0)
    assert policy.query_index == 0
    raw.fail = False
    retried = policy.predict_action([{'episode_start': False}])['actions']
    other = SeededEpisodePolicy(RandomPolicy(), 42)
    np.testing.assert_array_equal(retried, other.predict_action([{'episode_start': True}])['actions'])
