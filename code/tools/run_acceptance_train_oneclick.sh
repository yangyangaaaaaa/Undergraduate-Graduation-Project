#!/usr/bin/env bash
set -euo pipefail

# Server-side one-command runner for GeoExplorer acceptance training.
# Short remote command:
#   /root/geoexplorer/run_acceptance_train

MODE="${1:---smoke}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REMOTE_ROOT="${GEO_ROOT:-/root/geoexplorer}"
CODE_DIR="${GEOEXPLORER_CODE_DIR:-${REMOTE_ROOT}/GeoExplorer}"
ANALYSIS_DIR="${ACCEPTANCE_TRAIN_ANALYSIS_DIR:-${REMOTE_ROOT}/analysis}"
LATEST_LINK="${ANALYSIS_DIR}/acceptance_train_latest"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"
GPU_ID="${ACCEPTANCE_TRAIN_GPU:-1}"
RUN_ID="${ACCEPTANCE_TRAIN_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

if [ -d "${REMOTE_ROOT}/env/nvidia_535_288" ]; then
  DEFAULT_LD_LIBRARY_PATH="${REMOTE_ROOT}/env/nvidia_535_288:/usr/local/nvidia/lib64:/usr/local/cuda/lib64"
else
  DEFAULT_LD_LIBRARY_PATH="/usr/local/nvidia/lib64:/usr/local/cuda/lib64"
fi
SAFE_LD_LIBRARY_PATH="${ACCEPTANCE_LD_LIBRARY_PATH:-${DEFAULT_LD_LIBRARY_PATH}}"
PYTHONPATH_VALUE="${PYTHONPATH_VALUE:-${REMOTE_ROOT}/env/geoexplorer_site:${REMOTE_ROOT}:${CODE_DIR}:/root/src/compare_baselines_bundle_20260505_v2/compare_baselines_bundle/gomaa_geo_official}"

first_existing() {
  for path in "$@"; do
    if [ -f "$path" ]; then
      printf '%s\n' "$path"
      return 0
    fi
  done
  return 1
}

LLM_CHECKPOINT="${GEOEXPLORER_LLM_CHECKPOINT:-$(first_existing \
  "${REMOTE_ROOT}/results/checkpoint/env_modeling_fullrerun_20260407_111046/state_action.ckpt" \
  "${REMOTE_ROOT}/ab_experiments/official_retrain_20260517/geoexplorer_pristine_gomaa_seed42_480k/run_geoexplorer_official_pristine_seed42/checkpoint/official_pretrain_seed42_e50/state_action.ckpt" \
)}"

case "${MODE}" in
  --smoke|smoke)
    MODE_NAME="smoke"
    TIMESTEPS="${ACCEPTANCE_TRAIN_TIMESTEPS:-20}"
    BACKGROUND=0
    ;;
  --full|full)
    MODE_NAME="full"
    TIMESTEPS="${ACCEPTANCE_TRAIN_TIMESTEPS:-480000}"
    BACKGROUND=1
    ;;
  --status|status)
    if [ ! -L "${LATEST_LINK}" ]; then
      echo "No acceptance training run yet: ${LATEST_LINK}"
      exit 0
    fi
    RUN_ROOT="$(readlink -f "${LATEST_LINK}")"
    echo "LATEST: ${RUN_ROOT}"
    if [ -f "${RUN_ROOT}/train.pid" ]; then
      PID="$(cat "${RUN_ROOT}/train.pid")"
      if ps -p "${PID}" >/dev/null 2>&1; then
        echo "STATUS: running pid=${PID}"
      else
        echo "STATUS: not running pid=${PID}"
      fi
    fi
    if [ -f "${RUN_ROOT}/checkpoint/env_exploration/heartbeat.json" ]; then
      echo "HEARTBEAT:"
      cat "${RUN_ROOT}/checkpoint/env_exploration/heartbeat.json"
    fi
    if [ -f "${RUN_ROOT}/logs/train.log" ]; then
      echo
      echo "LOG TAIL:"
      tail -n 40 "${RUN_ROOT}/logs/train.log"
    fi
    exit 0
    ;;
  --tail|tail)
    RUN_ROOT="$(readlink -f "${LATEST_LINK}")"
    tail -n 120 -f "${RUN_ROOT}/logs/train.log"
    exit 0
    ;;
  *)
    echo "Usage: /root/geoexplorer/run_acceptance_train [--smoke|--full|--status|--tail]"
    echo "Default: --smoke"
    exit 2
    ;;
esac

RUN_ROOT="${ANALYSIS_DIR}/acceptance_train_${MODE_NAME}_${RUN_ID}"
LOG_DIR="${RUN_ROOT}/logs"
CKPT_ROOT="${RUN_ROOT}/checkpoint"
mkdir -p "${LOG_DIR}" "${CKPT_ROOT}"

echo "[1/4] training output: ${RUN_ROOT}"
echo "[2/4] checking training inputs"
test -d "${CODE_DIR}"
test -f "${CODE_DIR}/train.py"
test -f "${CODE_DIR}/data/swissview/swissview100_sat_patches.npy"
test -f "${LLM_CHECKPOINT}"

ln -sfn "${RUN_ROOT}" "${LATEST_LINK}"

cat > "${RUN_ROOT}/README.md" <<EOF
# GeoExplorer acceptance training run

Mode: ${MODE_NAME}
Generated: $(date -Is)
Output root: ${RUN_ROOT}
Log: ${RUN_ROOT}/logs/train.log
Checkpoint dir: ${RUN_ROOT}/checkpoint/env_exploration
Status command: /root/geoexplorer/run_acceptance_train --status
Tail command: /root/geoexplorer/run_acceptance_train --tail

This runner uses a short smoke training by default. Use --full only when a long 480k-timestep training run is intended.
EOF

export PYTHONPATH="${PYTHONPATH_VALUE}"
export LD_LIBRARY_PATH="${SAFE_LD_LIBRARY_PATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

export GEOEXPLORER_DATASET="${GEOEXPLORER_DATASET:-swissview}"
export GEOEXPLORER_PATCH_SIZE="${GEOEXPLORER_PATCH_SIZE:-5}"
export GEOEXPLORER_TRAIN_CKPT_ROOT="${CKPT_ROOT}"
export GEOEXPLORER_TRAIN_EXPT="${GEOEXPLORER_TRAIN_EXPT:-env_exploration}"
export GEOEXPLORER_TRAIN_NAME="${GEOEXPLORER_TRAIN_NAME:-geoexplorer.pt}"
export GEOEXPLORER_TRAIN_PREFIX="${GEOEXPLORER_TRAIN_PREFIX:-geoexplorer_}"
export GEOEXPLORER_TRAIN_LOG="${GEOEXPLORER_TRAIN_LOG:-expt_logs.txt}"
export GEOEXPLORER_LLM_CHECKPOINT="${LLM_CHECKPOINT}"
export GEOEXPLORER_DEVICE="${GEOEXPLORER_DEVICE:-cuda:0}"
export GEOEXPLORER_RANDOM_SEED="${GEOEXPLORER_RANDOM_SEED:-321}"
export GEOEXPLORER_MAX_TRAINING_TIMESTEPS="${TIMESTEPS}"
export GEOEXPLORER_UPDATE_TIMESTEP="${GEOEXPLORER_UPDATE_TIMESTEP:-20}"
export GEOEXPLORER_SAVE_MODEL_FREQ="${GEOEXPLORER_SAVE_MODEL_FREQ:-1000}"
export GEOEXPLORER_CHECKPOINT_EVERY_EPISODES="${GEOEXPLORER_CHECKPOINT_EVERY_EPISODES:-1}"
export GEOEXPLORER_VAL_EVERY_EPISODES="${GEOEXPLORER_VAL_EVERY_EPISODES:-1}"
export GEOEXPLORER_VAL_MAX_IMAGES="${GEOEXPLORER_VAL_MAX_IMAGES:-5}"
export GEOEXPLORER_VAL_DISTS="${GEOEXPLORER_VAL_DISTS:-4,5,6,7,8}"
export GEOEXPLORER_REWARD="${GEOEXPLORER_REWARD:-in}"
export GEOEXPLORER_FACTOR="${GEOEXPLORER_FACTOR:-1.0}"
export GEOEXPLORER_GATE_MODE="${GEOEXPLORER_GATE_MODE:-linear}"
export GEOEXPLORER_GATE_FLOOR="${GEOEXPLORER_GATE_FLOOR:-0.405}"
export GEOEXPLORER_PBRS_COEF="${GEOEXPLORER_PBRS_COEF:-0.10}"
export GEOEXPLORER_ROUTE_SAMPLE_EVERY_EPISODES="${GEOEXPLORER_ROUTE_SAMPLE_EVERY_EPISODES:-1}"
export GEOEXPLORER_ROUTE_SAMPLE_MAX_PER_EPISODE="${GEOEXPLORER_ROUTE_SAMPLE_MAX_PER_EPISODE:-4}"

{
  echo "MODE=${MODE_NAME}"
  echo "TIMESTEPS=${TIMESTEPS}"
  echo "GPU_ID=${GPU_ID}"
  echo "CODE_DIR=${CODE_DIR}"
  echo "LLM_CHECKPOINT=${LLM_CHECKPOINT}"
  echo "GEOEXPLORER_DATASET=${GEOEXPLORER_DATASET}"
  echo "GEOEXPLORER_GATE_MODE=${GEOEXPLORER_GATE_MODE}"
  echo "GEOEXPLORER_GATE_FLOOR=${GEOEXPLORER_GATE_FLOOR}"
  echo "GEOEXPLORER_PBRS_COEF=${GEOEXPLORER_PBRS_COEF}"
} > "${RUN_ROOT}/training_env_summary.txt"

echo "[3/4] starting ${MODE_NAME} training on GPU ${GPU_ID}"
if [ "${BACKGROUND}" -eq 1 ]; then
  (
    cd "${CODE_DIR}"
    exec "${PYTHON_BIN}" -u train.py
  ) > "${LOG_DIR}/train.log" 2>&1 &
  PID=$!
  echo "${PID}" > "${RUN_ROOT}/train.pid"
  echo "[4/4] launched background training pid=${PID}"
  echo "LOG: ${LOG_DIR}/train.log"
  echo "STATUS: /root/geoexplorer/run_acceptance_train --status"
  echo "TAIL: /root/geoexplorer/run_acceptance_train --tail"
else
  (
    cd "${CODE_DIR}"
    "${PYTHON_BIN}" -u train.py
  ) 2>&1 | tee "${LOG_DIR}/train.log"
  echo "[4/4] training finished"
  find "${RUN_ROOT}" -maxdepth 3 -type f | sort > "${RUN_ROOT}/file_inventory.txt"
  echo "DONE: ${RUN_ROOT}"
  echo "LATEST: ${LATEST_LINK}"
fi
