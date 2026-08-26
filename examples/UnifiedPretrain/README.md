# Unified pretraining

This recipe trains one shared visual/language/latent-world-model trunk on
augmented LIBERO, RoboTwin, Bridge, DROID, success-filtered KUKA, and a curated
community SO-family subset while routing each homogeneous batch to the matching
native ACT head.

| Embodiment tag | Action space | Shape | Control / targets |
| --- | --- | --- | --- |
| `franka` | Cartesian EEF delta + gripper | `8 x 7` | 20 Hz; WM at 0/.2/.4 s |
| `oxe_bridge` | Cartesian EEF delta + gripper | `3 x 7` | 5 Hz; WM at 0/.2/.4 s |
| `oxe_droid` | Cartesian EEF delta + gripper | `16 x 7` | 15 Hz; WM at 0/.2/.4 s |
| `kuka` | Cartesian EEF delta RPY + absolute open-gripper | `8 x 7` | 10 Hz; WM at 0/.2/.4 s |
| `aloha` | dual-arm joints + grippers | `16 x 14` | 30 Hz; WM at 0/.2/.4 s |
| `so100` | absolute single-arm joints | `16 x 6` | 30 Hz; WM at 0/.2/.4 s |
| `so101` | absolute single-arm joints | `16 x 6` | 30 Hz; WM at 0/.2/.4 s |
| `so_follower` | calibrated follower joints | `16 x 6` | 30 Hz; WM at 0/.2/.4 s |

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

Bridge training also requires three real MP4 camera files per episode. Scan
containers and generate the configured non-destructive episode blacklist with:

```bash
.venv/bin/python examples/UnifiedPretrain/data_tools/scan_bridge_videos.py \
  /home/gaoxiang/data/gaoxiang/datasets/IPEC-COMMUNITY/bridge_orig_lerobot_git
```

The loader reads `meta/video_health/bad_episodes.jsonl`; source videos and
episode metadata are never deleted or rewritten.

Build the SO-family manifest after downloading or updating
`community_dataset_v3`. The builder selects compatible SO100/SO101 main and
follower 6-D joint schemas at 30 Hz, requires at least one usable view and
complete local parquet/video files, and rejects only low-quality task text.
Directory names such as `test` are not treated as quality labels. Exact
action/state trajectory copies are recorded as episode exclusions in the
manifest; source metadata and media remain unchanged:

```bash
.venv/bin/python \
  examples/UnifiedPretrain/data_tools/build_community_so100_manifest.py \
  /home/gaoxiang/data/gaoxiang/community_dataset_v3 \
  --source-revision 19933f69e5f4d0a979953e4b5fcfa35656dcbc8f
```

The generated `community_so_family_manifest.json` is the only new file under
the dataset root. Single-view roots are padded to the common view count and
identified by `view_valid_mask`.

Prepare KUKA non-destructively. The audit verifies every Parquet and video
container, fully decodes a deterministic video sample, detects exact
state/action/video copies, and computes q01/q99 from all retained frames:

```bash
.venv/bin/python \
  examples/UnifiedPretrain/data_tools/prepare_kuka_pretrain.py \
  /home/gaoxiang/data/gaoxiang/kuka_lerobot
cp examples/UnifiedPretrain/train_files/kuka_modality.json \
  /home/gaoxiang/data/gaoxiang/kuka_lerobot/meta/modality.json
```

The source files remain unchanged. Exclusions are recorded in
`meta/pretrain_audit/excluded_episodes.jsonl`; the loader pads KUKA's single
real view to three and marks the other two invalid.

Run a short smoke job first:

```bash
CUDA_DEVS=0 BATCH=2 STEPS=100 \
  examples/UnifiedPretrain/train_files/run_unified_pretrain.sh
```

Then launch the full recipe by omitting `STEPS` and choosing the desired GPU
set. `unnorm_key` must be one of `franka`, `oxe_bridge`, or `aloha` at serving
time so the server selects both the correct normalization statistics and ACT
head.

## Follow-up world-model capacity ablation

After the `starvla_lewm_unified_pretrain_200k_20260818` baseline completes,
run a medium-capacity LEWM-OFT comparison with:

```yaml
framework:
  world_model:
    visual_tokens_per_view: 25
    residual_predictor_dim: 512
    residual_predictor_depth: 6
    residual_predictor_heads: 8
    residual_predictor_ffn: 2048
```

Keep the ACT heads and all data/training settings unchanged so the comparison
isolates world-model capacity. If compute permits an extra ablation, first run
`25` tokens per view with the baseline `384`-dimensional, 4-layer predictor to
separate spatial-token gains from transformer-capacity gains. Review
`delta_to_copy_ratio` and `delta_direction_cosine` at the baseline 10k
checkpoint before scheduling the comparison.
