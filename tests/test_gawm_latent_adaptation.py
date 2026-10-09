"""Guard fixed-target adaptation against leakage and misleading masked losses."""

from types import SimpleNamespace
from unittest.mock import patch
import json

import torch
import pytest
from omegaconf import OmegaConf

from scripts.experiment_gawm_latent import checkpoint_hash, episode_partition, fit
from starVLA.model.framework.WM4A.GAWM import GAWM
from starVLA.model.modules.world_model.visual_token_delta_world_model import VisualTokenLatentWorldModel


def small_world_model():
    return VisualTokenLatentWorldModel(latent_dim=8, goal_dim=8, n_future=2,
                                      num_tokens=4, dim=8, depth=1, num_heads=2, ffn_dim=16)


def test_padding_does_not_add_cosine_loss_or_change_scale():
    model = small_world_model().train()
    latent = torch.randn(2, 3, 4, 8)
    mask = torch.zeros(2, 3, 4, dtype=torch.bool)
    old_scale = model.delta_scale.clone()
    result = model(latent, ctx_len=1, loss_mask=mask)
    assert result["latent_loss"].item() == 0
    assert result["latent_cosine_loss"].item() == 0
    assert model._delta_scale_ready.item() == 0
    torch.testing.assert_close(old_scale, model.delta_scale)
    (result["latent_loss"] + result["latent_cosine_loss"]).backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_masked_horizon_does_not_dilute_direction_accuracy():
    model = small_world_model().eval()
    with torch.no_grad():
        model.residual_predictor.out.bias.fill_(1.0)
    latent = torch.zeros(1, 3, 4, 8)
    latent[:, 1] = 1
    latent[:, 2] = -100
    mask = torch.ones(1, 3, 4, dtype=torch.bool)
    mask[:, 2] = False
    result = model(latent, ctx_len=1, loss_mask=mask)
    torch.testing.assert_close(result["latent_cosine_loss"], torch.tensor(0.), atol=1e-6, rtol=0)
    torch.testing.assert_close(result["delta_direction_cosine"], torch.tensor(1.))
    assert result["latent_loss"].item() == 0


def test_residual_correction_uses_same_units_and_path_at_inference():
    model = small_world_model().eval()
    model.delta_scale.fill_(0.5)
    latent = torch.zeros(2, 3, 4, 8)
    latent[:, 1:] = 2.0
    correction = torch.ones(2, 2, 4, 8, requires_grad=True)
    result = model(latent, ctx_len=1, residual_correction=correction)
    torch.testing.assert_close(result["latent_loss"], torch.tensor(2.25))
    torch.testing.assert_close(result["pred_future_latent"],
                               model.regress_future(latent[:, :1], residual_correction=correction))
    result["latent_loss"].backward()
    assert correction.grad is not None and correction.grad.abs().sum() > 0
    with pytest.raises(ValueError, match="residual_correction"):
        model.regress_future(latent[:, :1], residual_correction=torch.zeros(2, 1, 4, 8))


def test_episode_split_is_disjoint_repeatable_and_excludes_corruption():
    steps = [(episode, frame) for episode in range(20) for frame in range(10)]
    partitions = episode_partition(steps, [3], 17, (40, 10, 10))
    assert partitions == episode_partition(steps, [3], 17, (40, 10, 10))
    seen = set()
    for partition in partitions.values():
        episodes = set(partition["episodes"])
        assert not (episodes & seen) and 3 not in episodes
        seen.update(episodes)
        assert len(partition["indices"]) == len(set(partition["indices"]))
        assert all(steps[i][0] in episodes for i in partition["indices"])


def test_state_adapter_preserves_warm_start_and_receives_gradients():
    config = OmegaConf.load("examples/LIBERO/train_files/starvla_gawm_full_12epochs.yaml")
    config.framework.world_model.condition_world_model_on_state = True
    config.framework.world_model.freeze_visual_token_pooler = True
    config.framework.world_model.train_encoder = False
    backbone = torch.nn.Linear(1, 1)
    backbone.model = SimpleNamespace(config=SimpleNamespace(hidden_size=32))
    backbone.patch_feature_dim = 32
    backbone.train_encoder = False
    with patch("starVLA.model.framework.WM4A.GAWM.get_world_model", return_value=backbone):
        model = GAWM(config).train()
    assert not model.backbone.training and not model.visual_token_pooler.training
    assert all(not p.requires_grad for p in model.visual_token_pooler.parameters())
    goal = torch.randn(2, model.task_emb_dim)
    state = torch.randn(2, 8)
    condition = model.condition_world_model(goal, state, "franka")
    torch.testing.assert_close(condition, goal, atol=0, rtol=0)
    condition.square().mean().backward()
    adapter = model.world_model_state_encoders["franka"]
    assert adapter[-1].weight.grad.abs().sum() > 0
    with torch.no_grad():
        adapter[-1].weight.add_(adapter[-1].weight.grad, alpha=-0.1)
    assert not torch.equal(model.condition_world_model(goal, state, "franka"),
                           model.condition_world_model(goal, state + 1, "franka"))


@pytest.mark.parametrize("conditioning", ["goal", "residual"])
def test_adaptation_exports_reloadable_model_and_json_metrics(tmp_path, conditioning):
    config = OmegaConf.load("examples/LIBERO/train_files/starvla_gawm_full_12epochs.yaml")
    config.framework.world_model.update({
        "train_encoder": False, "freeze_visual_token_pooler": True,
        "condition_world_model_on_state": False, "visual_token_dim": 8,
        "visual_tokens_per_view": 4, "residual_predictor_dim": 8,
        "residual_predictor_depth": 1, "residual_predictor_heads": 2,
        "residual_predictor_ffn": 16, "state_cond_hidden_dim": 8,
    })
    config.framework.lang_cond.update({"embed_dim": 8, "text_hidden_dim": 8,
                                      "text_depth": 1, "text_heads": 2, "text_ffn_dim": 16})
    config.framework.action_model.update({"action_hidden_dim": 8, "act_num_heads": 2,
                                         "act_num_layers": 1, "act_dim_feedforward": 16,
                                         "act_mlp_hidden_dim": 16})

    def fake_backbone(**kwargs):
        backbone = torch.nn.Linear(1, 1)
        backbone.model = SimpleNamespace(config=SimpleNamespace(hidden_size=8))
        backbone.patch_feature_dim = 8
        backbone.train_encoder = False
        return backbone

    with patch("starVLA.model.framework.WM4A.GAWM.get_world_model", side_effect=fake_backbone):
        model = GAWM(config)
        OmegaConf.save(model.config, tmp_path / "config.yaml")
        torch.save(model.state_dict(), tmp_path / "source.pt")
        (tmp_path / "dataset_statistics.json").write_text('{}')
        cache = tmp_path / "cache"
        cache.mkdir()
        (cache / "manifest.json").write_text(json.dumps({
            "checkpoint_sha256": checkpoint_hash(tmp_path / "source.pt"),
            "config": OmegaConf.to_container(model.config, resolve=True),
            "datasets": {"tiny": {}}, "limitation": "synthetic regression test",
        }))
        for split in ("train", "val", "test"):
            torch.save({"latent": torch.randn(3, 3, 12, 8), "goal": torch.randn(3, 8),
                        "state": torch.randn(3, 8), "mask": torch.ones(3, 3, 12, dtype=torch.bool),
                        "action": torch.randn(3, 8, 7), "action_mask": torch.ones(3, 8, dtype=torch.bool)},
                       cache / f"tiny.{split}.pt")
        output = tmp_path / "result"
        fit(SimpleNamespace(output=str(output), cache=str(cache), config=str(tmp_path / "config.yaml"),
                            checkpoint=str(tmp_path / "source.pt"), device="cpu", seed=7, variant="state",
                            batch_size=2, lr=1e-4, epochs=1, cosine_weight=0.1, action_weight=1., action_tolerance=.02,
                            state_conditioning=conditioning, adapter_only=True))
        result = json.loads((output / "result.json").read_text())
        assert result["best_epoch"] in (0, 1)
        assert result["test"]["pooled"]["latent_loss"] > 0
        restored = GAWM(OmegaConf.load(output / "config.yaml"))
        restored.load_state_dict(torch.load(output / "final_model/pytorch_model.pt", weights_only=True), strict=True)
        latent = torch.randn(2, 3, 12, 8)
        task, state = torch.randn(2, 8), torch.randn(2, 8)
        goal = restored.condition_world_model(task, state, "franka")
        correction = restored.world_model_correction(latent[:, :1], task, state, "franka")
        restored.eval()
        supervised = restored.world_model(latent, ctx_len=1, goal=goal, update_stats=False,
                                          residual_correction=correction)["pred_future_latent"]
        deployed = restored.world_model.regress_future(latent[:, :1], goal=goal, residual_correction=correction)
        torch.testing.assert_close(supervised, deployed)
