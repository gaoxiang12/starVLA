import hashlib
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from starVLA.model.framework.WM4A.GAWM import (
    CompositionalTextEncoder,
    GAWM,
    VisualTokenPooler,
)
from starVLA.model.framework.share_tools import apply_config_compat
from starVLA.model.modules.world_model.visual_token_delta_world_model import (
    VisualTokenLatentWorldModel,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_CONFIG = REPO_ROOT / (
    "playground/Checkpoints/"
    "starvla_lewm_unified_taskfilter_from40k_160k_20260820/config.yaml"
)
RECIPE_CONFIG = REPO_ROOT / (
    "examples/UnifiedPretrain/train_files/starvla_gawm_unified_pretrain.yaml"
)


class FakeEncoder(nn.Module):
    def __init__(self, hidden_size: int = 768):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.anchor = nn.Parameter(torch.zeros(()))


def build_model(config_path: Path) -> GAWM:
    config = apply_config_compat(OmegaConf.load(config_path))
    with patch(
        "starVLA.model.modules.world_model.dinov3_loader.build_dinov3",
        return_value=(FakeEncoder(), None, 0),
    ):
        return GAWM(config=config)


class GAWMCheckpointCompatibilityTest(unittest.TestCase):
    def test_saved_config_and_recipe_have_identical_architecture(self):
        saved = build_model(CHECKPOINT_CONFIG)
        recipe = build_model(RECIPE_CONFIG)
        saved_shapes = {
            name: tuple(tensor.shape) for name, tensor in saved.state_dict().items()
        }
        recipe_shapes = {
            name: tuple(tensor.shape) for name, tensor in recipe.state_dict().items()
        }

        self.assertEqual(saved_shapes, recipe_shapes)
        self.assertEqual(len(saved_shapes), 330)
        self.assertEqual(sum(p.numel() for p in saved.parameters()), 34_899_801)
        self.assertEqual(list(saved.action_models), ["aloha", "franka", "oxe_bridge"])
        manifest = "\n".join(
            f"{name}:{shape}" for name, shape in sorted(saved_shapes.items())
        )
        self.assertEqual(
            hashlib.sha256(manifest.encode()).hexdigest(),
            "bd8cd1d9710022ae243ff77fe6f283c95bde49d396366c3f8fdfe0023001276b",
        )

    def test_checkpoint_state_dict_loads_strictly(self):
        source = build_model(CHECKPOINT_CONFIG)
        target = build_model(RECIPE_CONFIG)
        result = target.load_state_dict(source.state_dict(), strict=True)
        self.assertEqual(result.missing_keys, [])
        self.assertEqual(result.unexpected_keys, [])

    def test_only_supported_path_is_accepted(self):
        config = OmegaConf.load(RECIPE_CONFIG)
        config.framework.action_model.action_model_type = "MLP"
        with self.assertRaisesRegex(ValueError, "only ACT"):
            GAWM(config=config)


class GAWMComponentsTest(unittest.TestCase):
    def test_spatial_pooler_preserves_shapes_and_masks_views(self):
        pooler = VisualTokenPooler(32, 24, num_views=3, tokens_per_view=4)
        patches = torch.randn(2, 3, 3, 16, 32, requires_grad=True)
        mask = torch.tensor([[True, True, False], [True, True, True]])
        tokens, content = pooler(
            patches, return_content=True, view_valid_mask=mask
        )
        self.assertEqual(tokens.shape, (2, 3, 12, 24))
        self.assertEqual(content.shape, tokens.shape)
        self.assertTrue(torch.equal(tokens[0, :, 8:], torch.zeros_like(tokens[0, :, 8:])))
        tokens.square().mean().backward()
        self.assertTrue(torch.isfinite(patches.grad).all())

    def test_text_encoder_is_compositional_and_trainable(self):
        encoder = CompositionalTextEncoder(
            output_dim=24,
            hidden_dim=32,
            depth=1,
            num_heads=4,
            ffn_dim=64,
            max_length=32,
            dropout=0.0,
        )
        first, first_mask = encoder.tokenize(
            [" Pick  up   the cup "], device=torch.device("cpu")
        )
        normalized, normalized_mask = encoder.tokenize(
            ["pick up the cup"], device=torch.device("cpu")
        )
        self.assertTrue(torch.equal(first, normalized))
        self.assertTrue(torch.equal(first_mask, normalized_mask))
        output = encoder(["pick up the cup", "拿起杯子"], device=torch.device("cpu"))
        self.assertEqual(output.shape, (2, 24))
        output.square().mean().backward()
        self.assertIsNotNone(encoder.token_embedding.weight.grad)

    def test_latent_predictor_updates_and_preserves_fp32_statistics(self):
        model = VisualTokenLatentWorldModel(
            latent_dim=24,
            goal_dim=16,
            n_future=2,
            num_tokens=12,
            dim=24,
            depth=1,
            num_heads=4,
            ffn_dim=48,
            stats_momentum=0.9,
            detach_input=True,
        ).train()
        latent = torch.randn(2, 3, 12, 24, requires_grad=True)
        goal = torch.randn(2, 16)
        output = model(latent, ctx_len=1, goal=goal)
        self.assertEqual(output["pred_future_latent"].shape, (2, 2, 12, 24))
        self.assertEqual(model.delta_scale.dtype, torch.float32)
        self.assertEqual(float(model._delta_scale_ready), 1.0)
        (output["latent_loss"] + output["latent_cosine_loss"]).backward()
        self.assertIsNone(latent.grad)
        self.assertIsNotNone(model.residual_predictor.anchor_proj.weight.grad)


if __name__ == "__main__":
    unittest.main()
