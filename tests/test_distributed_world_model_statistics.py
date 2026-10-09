"""The GAWM inference scale must match the pooled batch on every rank."""

from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from starVLA.model.modules.world_model.visual_token_delta_world_model import (
    VisualTokenLatentWorldModel,
)


def _check_statistics(rank, rendezvous):
    dist.init_process_group(
        "gloo", init_method=rendezvous, rank=rank, world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        model = VisualTokenLatentWorldModel(
            latent_dim=4, goal_dim=None, n_future=2, num_tokens=2,
            dim=4, depth=1, num_heads=1, ffn_dim=8, stats_momentum=0.5,
        )
        # Production GAWM enables this from world_model.sync_latent_stats.
        model.sync_stats = True
        # 2 valid values of 1 on rank 0; 4 valid values of 3 on rank 1.
        # Averaging per-rank RMS (2) would be wrong: pooled RMS=sqrt(38/6).
        residual = torch.full((1, 1, 2, 2), 1.0 if rank == 0 else 3.0)
        mask = torch.tensor([[[[1.0], [float(rank)]]]])
        model._update_delta_scale(residual, mask)
        expected = torch.tensor([(2.0 + 36.0) / 6.0]).sqrt()
        torch.testing.assert_close(model.delta_scale, expected)
        # An all-masked rank must contribute zero count, not a clamped one.
        model._update_delta_scale(torch.full_like(residual, 4), torch.full_like(mask, rank))
        expected = expected * 0.5 + 2.0
        torch.testing.assert_close(model.delta_scale, expected)
        # Also cover unmasked batches with different local batch sizes.
        residual = torch.full((rank + 1, 1, 2, 2), float(2 * rank + 1))
        model._update_delta_scale(residual)
        expected = expected * 0.5 + torch.tensor([(4.0 + 72.0) / 12.0]).sqrt() * 0.5
        torch.testing.assert_close(model.delta_scale, expected)
        assert model.delta_scale.dtype == torch.float32
        assert model._delta_scale_ready.item() == 1.0
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo required")
def test_gawm_statistics_match_pooled_batch_across_ranks(tmp_path: Path):
    mp.spawn(_check_statistics, args=((tmp_path / "rendezvous").as_uri(),), nprocs=2, join=True)
