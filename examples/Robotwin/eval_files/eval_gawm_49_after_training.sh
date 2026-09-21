#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "${script_dir}/../../.." && pwd)
cd "${repo_root}"

run_dir=${TRAIN_RUN_DIR:-playground/Checkpoints/gawm_s_base_robotwin_clean1000_49tasks_40k}
train_pid_file=${TRAIN_PID_FILE:-${run_dir}/train.pid}
final_checkpoint=${FINAL_CHECKPOINT:-${run_dir}/final_model/pytorch_model.pt}
task_file=${ROBOTWIN_TASK_FILE:-${script_dir}/robotwin_49_tasks.txt}
deploy_template=${DEPLOY_POLICY_TEMPLATE_PATH:-${script_dir}/deploy_policy_gawm_aloha.yml}
robotwin_root=${ROBOTWIN_PATH:-${repo_root}/../RoboTwin}
starvla_python=${STARVLA_PYTHON:-${repo_root}/../.venvs/starVLA/bin/python}
robotwin_python=${ROBOTWIN_PYTHON:-${repo_root}/../.venvs/RoboTwin/bin/python}
episodes=${EPISODES:-100}
seed=${ROBOTWIN_SEED:-0}
gpu_devices=${CUDA_VISIBLE_DEVICES:-4,5,6}
jobs_per_gpu=${ROBOTWIN_JOBS_PER_GPU:-1}
base_port=${ROBOTWIN_BASE_PORT:-5694}
poll_seconds=${TRAIN_POLL_SECONDS:-30}
status_file=${run_dir}/robotwin_eval_watcher.status

if [[ ! -f "${train_pid_file}" ]]; then
  echo "Training PID file not found: ${train_pid_file}" >&2
  exit 1
fi
if [[ ! -f "${task_file}" ]]; then
  echo "RoboTwin task list not found: ${task_file}" >&2
  exit 1
fi
if [[ ! -f "${deploy_template}" ]]; then
  echo "GAWM deploy template not found: ${deploy_template}" >&2
  exit 1
fi
if [[ ! -x "${starvla_python}" || ! -x "${robotwin_python}" ]]; then
  echo "Evaluation Python environments are missing." >&2
  echo "StarVLA: ${starvla_python}" >&2
  echo "RoboTwin: ${robotwin_python}" >&2
  exit 1
fi
if [[ ! -f "${robotwin_root}/script/eval_policy.py" ]]; then
  echo "RoboTwin evaluator not found under: ${robotwin_root}" >&2
  exit 1
fi
if [[ $(grep -cvE '^[[:space:]]*(#|$)' "${task_file}") -ne 49 ]]; then
  echo "Expected exactly 49 evaluation tasks in ${task_file}" >&2
  exit 1
fi

train_pid=$(<"${train_pid_file}")
if [[ ! "${train_pid}" =~ ^[0-9]+$ ]]; then
  echo "Invalid training PID: ${train_pid}" >&2
  exit 1
fi

printf 'state=waiting_for_training pid=%s time=%s\n' \
  "${train_pid}" "$(date --iso-8601=seconds)" >"${status_file}"
echo "[WATCHER] Waiting for training PID ${train_pid} to finish."
while kill -0 "${train_pid}" 2>/dev/null; do
  sleep "${poll_seconds}"
done

if [[ ! -s "${final_checkpoint}" ]]; then
  printf 'state=blocked reason=missing_final_checkpoint time=%s\n' \
    "$(date --iso-8601=seconds)" >"${status_file}"
  echo "Training exited without a final checkpoint: ${final_checkpoint}" >&2
  exit 1
fi
# Rich wraps the remainder of this message across lines in redirected logs.
# Match its completion prefix; the final checkpoint is checked separately above.
if ! grep -Fq 'Training complete.' "${run_dir}/train.log"; then
  printf 'state=blocked reason=training_not_marked_complete time=%s\n' \
    "$(date --iso-8601=seconds)" >"${status_file}"
  echo "Final checkpoint exists, but the training log has no completion marker." >&2
  exit 1
fi

# Check the client environment before launching all tasks and policy servers.
if ! PYTHONNOUSERSITE=1 PYTHONPATH="${repo_root}:${PYTHONPATH:-}" \
  "${robotwin_python}" -c 'from examples.Robotwin.eval_files import model2robotwin_interface'; then
  printf 'state=blocked reason=missing_eval_dependencies time=%s\n' \
    "$(date --iso-8601=seconds)" >"${status_file}"
  echo "RoboTwin policy client import failed; check the evaluation environment." >&2
  exit 1
fi

sleep 10
timestamp=$(date +%Y%m%d_%H%M%S)
policy_name=${ROBOTWIN_POLICY_LABEL:-gawm_s_clean1000_49tasks_final_${timestamp}}
eval_root=${ROBOTWIN_LOG_ROOT:-${run_dir}/robotwin_eval_logs/${policy_name}}
mkdir -p "${eval_root}"

export ROBOTWIN_PATH="${robotwin_root}"
export STARVLA_PYTHON="${starvla_python}"
export ROBOTWIN_PYTHON="${robotwin_python}"
export DEPLOY_POLICY_TEMPLATE_PATH="${deploy_template}"
export CUDA_VISIBLE_DEVICES="${gpu_devices}"
export ROBOTWIN_EVAL_VIDEO_LOG=${ROBOTWIN_EVAL_VIDEO_LOG:-0}
export PYTHONNOUSERSITE=1

printf 'state=evaluating policy=%s checkpoint=%s mode=demo_clean episodes=%s gpus=%s time=%s\n' \
  "${policy_name}" "${final_checkpoint}" "${episodes}" "${gpu_devices}" \
  "$(date --iso-8601=seconds)" >"${status_file}"

mode=demo_clean
mode_log_root=${eval_root}/${mode}
# Record failures from either evaluation or aggregation instead of leaving a
# stale state=evaluating marker after the watcher exits.
trap 'rc=$?; printf "state=failed exit_code=%s policy=%s logs=%s time=%s\n" "$rc" "$policy_name" "$eval_root" "$(date --iso-8601=seconds)" >"$status_file"; exit "$rc"' ERR
echo "[WATCHER] Starting ${mode}: 49 tasks x ${episodes} episodes on GPUs ${gpu_devices}."
ROBOTWIN_LOG_ROOT="${mode_log_root}" \
  bash "${script_dir}/start_eval.sh" \
    --mode "${mode}" \
    --name "${policy_name}" \
    --ckpt "${final_checkpoint}" \
    --seed "${seed}" \
    --episodes "${episodes}" \
    --jobs-per-gpu "${jobs_per_gpu}" \
    --base-port "${base_port}" \
    "${task_file}"

summary_dir=${eval_root}/summary
"${starvla_python}" "${script_dir}/summarize_robotwin_eval.py" \
  --result-root "${robotwin_root}/eval_result" \
  --setting "${policy_name}" \
  --modes demo_clean \
  --tasks "${task_file}" \
  --episodes "${episodes}" \
  --output-dir "${summary_dir}" \
  --strict

printf 'state=complete policy=%s summary=%s time=%s\n' \
  "${policy_name}" "${summary_dir}" "$(date --iso-8601=seconds)" >"${status_file}"
echo "[WATCHER] RoboTwin 49-task clean evaluation complete: ${summary_dir}"
