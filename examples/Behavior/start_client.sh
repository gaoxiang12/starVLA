#!/bin/bash

# Debug: print current python environment
echo "Using Python: $(which python)"

### MANUALLY SET THESE ###
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# set necessary environment variables
export star_vla_python="${star_vla_python:-$(command -v python)}"
export sim_python="${sim_python:-python}"
export TASKS_JSONL_PATH="${TASKS_JSONL_PATH:-${SCRIPT_DIR}/tasks.jsonl}"
export BEHAVIOR_ASSET_PATH="${BEHAVIOR_ASSET_PATH:-${REPO_ROOT}/BEHAVIOR-1K/datasets}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# set model path and port
MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/playground/Checkpoints/BEHAVIOR-QwenDual-Pretrained-224/checkpoints/steps_300000_pytorch_model.pt}"
PORT=10197
WRAPPERS="RGBLowResWrapper" # DefaultWrapper, RGBLowResWrapper or RichObservationWrapper
USE_STATE=True  

# set task name
TASK_NAME="turning_on_radio" 
EVAL_INSTANCE_IDS="0"
### END OF MANUALLY SETUP ###


# Force Vulkan to use only the NVIDIA ICD to avoid duplicate ICDs seen by the loader
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
# Prefer NVIDIA GLX vendor when any GL deps are touched
export __GLX_VENDOR_LIBRARY_NAME=nvidia

# run single task
export DEBUG=true
echo "▶️ Running task '${TASK_NAME}'..."
CUDA_VISIBLE_DEVICES=5 ${sim_python} examples/Behavior/start_behavior_env.py \
    --ckpt-path ${MODEL_PATH} \
    --eval-instance-ids \"${EVAL_INSTANCE_IDS}\"  \
    --eval-on-train-instances True \
    --port ${PORT} \
    --task-name ${TASK_NAME} \
    --behavior-tasks-jsonl-path ${TASKS_JSONL_PATH} \
    --behavior-asset-path ${BEHAVIOR_ASSET_PATH} \
    --wrappers ${WRAPPERS} \
    --use-state ${USE_STATE}
    

# stop server
echo "⏹️ Stopping server (PID: ${SERVER_PID})..."
kill ${SERVER_PID}
wait ${SERVER_PID} 2>/dev/null
echo "✅ Server stopped"