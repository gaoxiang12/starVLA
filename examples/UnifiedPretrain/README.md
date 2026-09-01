# Unified GAWM reproduction

This directory now contains one supported recipe: the three-embodiment GAWM
run saved in
`playground/Checkpoints/starvla_lewm_unified_taskfilter_from40k_160k_20260820`.
It shares the DINOv3 visual trunk, compositional text encoder, spatial latent
predictor, and routes homogeneous batches to native ACT heads:

| Tag | Action shape | State dim |
| --- | --- | --- |
| `aloha` | `16 x 14` | 14 |
| `franka` | `8 x 7` | 8 |
| `oxe_bridge` | `3 x 7` | 8 |

The saved run warm-started from step 40k of an earlier checkpoint. On another
machine, pass its local path explicitly:

```bash
PRETRAINED_CHECKPOINT=/path/to/steps_40000_pytorch_model.pt \
  examples/UnifiedPretrain/train_files/run_unified_pretrain.sh
```

The default launch is 160k optimizer steps on all GPUs in `CUDA_DEVS`. A short
smoke run can be started with:

```bash
CUDA_DEVS=0 BATCH=2 STEPS=100 \
  examples/UnifiedPretrain/train_files/run_unified_pretrain.sh
```

At serving time, `unnorm_key` must be one of `aloha`, `franka`, or
`oxe_bridge` so normalization statistics and the ACT head are selected
together.
