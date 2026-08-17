# Unified pretraining

This recipe trains one shared visual/language/latent-world-model trunk on
augmented LIBERO, RoboTwin, and Bridge while routing each homogeneous batch to
the matching native ACT head.

| Embodiment tag | Action space | Shape | Control / targets |
| --- | --- | --- | --- |
| `franka` | Cartesian EEF delta + gripper | `8 x 7` | 20 Hz; WM at 0/.2/.4 s |
| `oxe_bridge` | Cartesian EEF delta + gripper | `3 x 7` | 5 Hz; WM at 0/.2/.4 s |
| `aloha` | dual-arm joints + grippers | `16 x 14` | 30 Hz; WM at 0/.2/.4 s |

The data root is `/home/gaoxiang/data/gaoxiang`. LIBERO's two real views are
padded to the shared three-view layout and accompanied by `view_valid_mask`;
Bridge uses images 0/1/2 and RoboTwin uses its three native cameras.

The Bridge LeRobot v2.0 export does not include GR00T modality metadata. Copy
the unified three-view declaration into the dataset before training:

```bash
cp examples/UnifiedPretrain/train_files/bridge_modality.json \
  /home/gaoxiang/data/gaoxiang/datasets/IPEC-COMMUNITY/bridge_orig_lerobot_git/meta/modality.json
```

Language conditioning is routed per embodiment: LIBERO metadata receives
stable Unicode/case normalization, RoboTwin uses its 50 task directory names,
and Bridge uses conservative lexical aliases (for example, `put`/`place`/`move`
with the same objects and target). Raw metadata is never rewritten. Generate a
reviewable Bridge sidecar audit with:

```bash
.venv/bin/python examples/UnifiedPretrain/data_tools/canonicalize_bridge_tasks.py \
  /home/gaoxiang/data/gaoxiang/datasets/IPEC-COMMUNITY/bridge_orig_lerobot_git
```

Run a short smoke job first:

```bash
CUDA_DEVS=0 BATCH=2 STEPS=100 \
  examples/UnifiedPretrain/train_files/run_unified_pretrain.sh
```

Then launch the full recipe by omitting `STEPS` and choosing the desired GPU
set. `unnorm_key` must be one of `franka`, `oxe_bridge`, or `aloha` at serving
time so the server selects both the correct normalization statistics and ACT
head.
