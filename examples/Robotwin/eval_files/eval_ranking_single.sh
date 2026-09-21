#!/usr/bin/env bash
set -euo pipefail
scripts=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
root=$(cd "${scripts}/../../.." && pwd)
cd "${root}"
run_id=$1
task=$2
gpu=$3
port=$4
run="${root}/playground/Checkpoints/${run_id}"
checkpoint="${run}/final_model/pytorch_model.pt"
test -s "${checkpoint}"
export CUDA_VISIBLE_DEVICES="${gpu}"
export ROBOTWIN_PATH="${root}/../RoboTwin"
export STARVLA_PYTHON="${root}/../.venvs/starVLA/bin/python"
export ROBOTWIN_PYTHON="${root}/../.venvs/RoboTwin/bin/python"
export DEPLOY_POLICY_TEMPLATE_PATH="${scripts}/deploy_policy_gawm_aloha.yml"
export ROBOTWIN_EVAL_RUNNER_PATH="${scripts}/robotwin_ranking_eval_runner.py"
export ROBOTWIN_RANKING_METRICS_PATH="${run}/ranking_episode_metrics.jsonl"
export ROBOTWIN_LOG_ROOT="${run}/robotwin_eval_logs"
export ROBOTWIN_EVAL_VIDEO_LOG=1
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export ROBOTWIN_USE_BF16=0
if [[ "${task}" == blocks_ranking_rgb && "${run_id}" == gawm_rgb_focus_* ]]; then
    export ROBOTWIN_POLICY_TRACE_DIR="${run}/policy_traces"
    PYTHONPATH="${root}:${PYTHONPATH:-}" NO_ALBUMENTATIONS_UPDATE=1 \
    "${STARVLA_PYTHON}" "${scripts}/../audits/analyze_rgb_focus_usage.py" \
        --run "${run}" --checkpoint "${checkpoint}" --samples 128 \
        --split-manifest "${scripts}/../audits/rgb_scene_safe_validation_20260907.json" \
        --output "${run}/scene_safe_validation.json" \
        > "${run}/scene_safe_validation.log" 2>&1
fi
"${STARVLA_PYTHON}" "${scripts}/run_robotwin_eval_retry.py" \
    --metrics "${ROBOTWIN_RANKING_METRICS_PATH}" --log-root "${ROBOTWIN_LOG_ROOT}" --attempts 2 -- \
    bash "${scripts}/start_eval.sh" --mode demo_clean --name "${run_id}" \
    --ckpt "${checkpoint}" --seed 0 --episodes 100 --jobs-per-gpu 1 \
    --base-port "${port}" "${task}"
"${STARVLA_PYTHON}" "${scripts}/summarize_robotwin_eval.py" \
    --result-root "${ROBOTWIN_PATH}/eval_result" --setting "${run_id}" \
    --modes demo_clean --tasks "${task}" --episodes 100 \
    --output-dir "${run}/eval_summary" --strict
