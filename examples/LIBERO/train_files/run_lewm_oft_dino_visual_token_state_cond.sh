#!/bin/bash
# Warm-start from the best 20k SWA policy and train only a zero-initialized
# proprio residual. At step zero the conditioned policy is exactly the SWA
# policy; only the new state encoder is trainable.

set -e

export RUN_ID=${RUN_ID:-lewm_oft_libero_dinov2b_spatial4x4_statecond_from_swa20k}
export PRETRAINED_CKPT=${PRETRAINED_CKPT:-playground/Checkpoints/lewm_oft_libero_dinov2b_spatial4x4_nosigreg_noactionflow_2k/checkpoints/swa_16000_18000_20000_pytorch_model.pt}
export STEPS=${STEPS:-10000}
export SAVE_INTERVAL=${SAVE_INTERVAL:-2000}
export WARMUP=${WARMUP:-500}
export LOGGING_FREQUENCY=${LOGGING_FREQUENCY:-100}
export BASE_LR=${BASE_LR:-1e-4}
export ACTION_LR=${ACTION_LR:-1e-4}
export USE_STATE_COND=true
export STATE_COND_DIM=8
export STATE_COND_HIDDEN_DIM=${STATE_COND_HIDDEN_DIM:-256}
export STATE_COND_DROPOUT=${STATE_COND_DROPOUT:-0.1}
export STATE_COND_ONLY=true
export INCLUDE_STATE=true
export DELTA_SIGREG_WEIGHT=0
export ACTION_WEIGHT=0
export IS_RESUME=false
export CUDA_DEVS=${CUDA_DEVS:-4,5,6,7}
export NUM_PROCESSES=${NUM_PROCESSES:-4}
export MAIN_PORT=${MAIN_PORT:-29594}
export ACCELERATE_BIN=${ACCELERATE_BIN:-.venv/bin/accelerate}
export WANDB_MODE=${WANDB_MODE:-offline}
export FREEZE_MODULES=${FREEZE_MODULES:-backbone,world_model,visual_token_pooler,task_embedding,action_model,action_query_proj,future_action_context_proj}

exec bash examples/LIBERO/train_files/run_lewm_oft_dino_visual_token_train.sh
