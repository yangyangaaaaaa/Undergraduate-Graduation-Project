#!/usr/bin/env bash
set -euo pipefail

# Server-side one-command runner for GeoExplorer acceptance testing and visuals.
# Intended remote command:
#   /usr/bin/env -u LD_LIBRARY_PATH /usr/bin/bash /root/geoexplorer/GeoExplorer/acceptance_demo/run_acceptance_demo_oneclick.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="${ACCEPTANCE_BASE_DIR:-${SCRIPT_DIR}}"
ASSET_BASE_DIR="${ACCEPTANCE_ASSET_BASE_DIR:-/root/geoexplorer/acceptance_demo_assets}"
REMOTE_ROOT="${GEO_ROOT:-/root/geoexplorer}"
UGP_ROOT="${UGP_ROOT:-${ASSET_BASE_DIR}/Undergraduate-Graduation-Project}"
MONITORING_DIR="${MONITORING_DIR:-${REMOTE_ROOT}/ab_experiments/visualization_20260518/anchor0624_swissviewmonuments_qualitative/monitoring}"
VIS_ROOT="${VIS_ROOT:-${REMOTE_ROOT}/analysis/pipeline_20260517_anchor0624_visualization}"
RUN_ID="${ACCEPTANCE_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${ACCEPTANCE_OUT_ROOT:-${REMOTE_ROOT}/analysis/acceptance_demo_oneclick_${RUN_ID}}"
GPU_ID="${ACCEPTANCE_GPU:-0}"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"
MODE="run"
ONE_IMAGE="${ACCEPTANCE_IMAGE_IDX:-}"
CUSTOM_IMAGE="${ACCEPTANCE_CUSTOM_IMAGE:-}"
CUSTOM_START="${ACCEPTANCE_CUSTOM_START:-}"
CUSTOM_GOAL="${ACCEPTANCE_CUSTOM_GOAL:-}"
CUSTOM_DISTANCE="${ACCEPTANCE_CUSTOM_DISTANCE:-}"
CUSTOM_WORK_DIR="${ACCEPTANCE_CUSTOM_WORK_DIR:-}"

PYTHONPATH_VALUE="${PYTHONPATH_VALUE:-/root/geoexplorer/env/geoexplorer_site:/root/geoexplorer:/root/geoexplorer/GeoExplorer:/root/src/compare_baselines_bundle_20260505_v2/compare_baselines_bundle/gomaa_geo_official:${MONITORING_DIR}}"
SAFE_LD_LIBRARY_PATH="${ACCEPTANCE_LD_LIBRARY_PATH:-/usr/local/nvidia/lib64:/usr/local/cuda/lib64}"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --visual-only)
      MODE="--visual-only"
      ;;
    --one-image|--image|--img|--img-idx)
      shift
      if [ "$#" -eq 0 ]; then
        echo "Missing image index after --one-image" >&2
        exit 2
      fi
      ONE_IMAGE="$1"
      ;;
    --custom-image|--custom-img|--custom)
      shift
      if [ "$#" -eq 0 ]; then
        echo "Missing image path after --custom-image" >&2
        exit 2
      fi
      CUSTOM_IMAGE="$1"
      ;;
    --start|--start-patch)
      shift
      if [ "$#" -eq 0 ]; then
        echo "Missing patch index after --start" >&2
        exit 2
      fi
      CUSTOM_START="$1"
      ;;
    --goal|--goal-patch)
      shift
      if [ "$#" -eq 0 ]; then
        echo "Missing patch index after --goal" >&2
        exit 2
      fi
      CUSTOM_GOAL="$1"
      ;;
    --distance|--dist)
      shift
      if [ "$#" -eq 0 ]; then
        echo "Missing distance after --distance" >&2
        exit 2
      fi
      CUSTOM_DISTANCE="$1"
      ;;
    --help|-h)
      echo "Usage: /root/geoexplorer/run_acceptance_demo [--visual-only] [--one-image IMG_IDX]"
      echo "       /root/geoexplorer/run_acceptance_demo --custom-image /path/to/image.png [--start PATCH --goal PATCH]"
      echo "Example: /root/geoexplorer/run_acceptance_demo --one-image 189"
      echo "Example: /root/geoexplorer/run_acceptance_demo --visual-only --one-image 189"
      echo "Example: /root/geoexplorer/run_acceptance_demo --custom-image /root/demo.png"
      echo "Example: /root/geoexplorer/run_acceptance_demo --custom-image /root/demo.png --start 20 --goal 4"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      echo "Usage: /root/geoexplorer/run_acceptance_demo [--visual-only] [--one-image IMG_IDX]" >&2
      exit 2
      ;;
  esac
  shift
done

if [ -n "${CUSTOM_IMAGE}" ] && [ -n "${ONE_IMAGE}" ]; then
  echo "Please use either --one-image for dataset images or --custom-image for a new image, not both." >&2
  exit 2
fi
if [ -n "${CUSTOM_IMAGE}" ] && [ "${MODE}" = "--visual-only" ]; then
  echo "--custom-image needs fresh inference because the custom embedding is generated on the fly; drop --visual-only." >&2
  exit 2
fi
if { [ -n "${CUSTOM_START}" ] && [ -z "${CUSTOM_GOAL}" ]; } || { [ -z "${CUSTOM_START}" ] && [ -n "${CUSTOM_GOAL}" ]; }; then
  echo "--start and --goal must be provided together." >&2
  exit 2
fi
if [ -n "${CUSTOM_IMAGE}" ]; then
  VIS_ROOT="${OUT_ROOT}/custom_visualization_source"
  CUSTOM_WORK_DIR="${CUSTOM_WORK_DIR:-${OUT_ROOT}/custom_image}"
fi

mkdir -p "${OUT_ROOT}" "${OUT_ROOT}/logs"

echo "[1/5] acceptance output: ${OUT_ROOT}"
echo "[2/5] checking server inputs"
test -d "${REMOTE_ROOT}/GeoExplorer"
test -f "${MONITORING_DIR}/qualitative_visualization_runner.py"
test -f "${MONITORING_DIR}/paper_baseline_evaluator.py"
test -f "${BASE_DIR}/build_acceptance_demo_visuals.py"
test -f "${BASE_DIR}/prepare_acceptance_custom_image.py"
test -f "${UGP_ROOT}/results/tables/main_benchmark/paper_baseline_compare_table.csv"
test -d "${UGP_ROOT}/results/figures/chapter2_dataset/manual_redraw_assets"
if [ -n "${CUSTOM_IMAGE}" ]; then
  test -f "${CUSTOM_IMAGE}"
elif [ ! -d "${VIS_ROOT}/asset_cache/aerial_view" ] && [ -d "${ASSET_BASE_DIR}/visualization_asset_cache" ]; then
  mkdir -p "${VIS_ROOT}"
  cp -a "${ASSET_BASE_DIR}/visualization_asset_cache" "${VIS_ROOT}/asset_cache"
fi
if [ -z "${CUSTOM_IMAGE}" ]; then
  test -d "${VIS_ROOT}/asset_cache/aerial_view"
fi

export PYTHONPATH="${PYTHONPATH_VALUE}"
export LD_LIBRARY_PATH="${SAFE_LD_LIBRARY_PATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export UGP_ROOT="${UGP_ROOT}"
export BISHE_ROOT="${BASE_DIR}"
export GEO_ROOT="${REMOTE_ROOT}"
export VIS_ROOT="${VIS_ROOT}"
export ACCEPTANCE_OUT_DIR="${OUT_ROOT}/figures"
export ACCEPTANCE_REPORT_PATH="${OUT_ROOT}/acceptance_demo_visuals_zh.md"
if [ -n "${ONE_IMAGE}" ]; then
  export ACCEPTANCE_IMAGE_IDX="${ONE_IMAGE}"
  export ACCEPTANCE_INFER_IMAGE="${ONE_IMAGE}"
  echo "single-image mode: img_idx=${ONE_IMAGE}"
fi
if [ -n "${CUSTOM_IMAGE}" ]; then
  export ACCEPTANCE_CUSTOM_IMAGE="${CUSTOM_IMAGE}"
  export ACCEPTANCE_IMAGE_IDX="0"
  export ACCEPTANCE_INFER_IMAGE="0"
  export ACCEPTANCE_TEST_PATH="${CUSTOM_WORK_DIR}/custom_sat_patches.npy"
  export ACCEPTANCE_FIXED_GOAL_MODE="none"
  if [ -n "${CUSTOM_START}" ]; then
    export ACCEPTANCE_CUSTOM_START="${CUSTOM_START}"
    export ACCEPTANCE_CUSTOM_GOAL="${CUSTOM_GOAL}"
  fi
  if [ -n "${CUSTOM_DISTANCE}" ]; then
    export ACCEPTANCE_DISTANCES="${CUSTOM_DISTANCE}"
  fi
  echo "custom-image mode: ${CUSTOM_IMAGE}"
fi
if [ -f "${BASE_DIR}/fonts/simsun.ttc" ]; then
  export ACCEPTANCE_CJK_FONT="${BASE_DIR}/fonts/simsun.ttc"
elif [ -f "${ASSET_BASE_DIR}/fonts/simsun.ttc" ]; then
  export ACCEPTANCE_CJK_FONT="${ASSET_BASE_DIR}/fonts/simsun.ttc"
fi

if [ "${MODE}" != "--visual-only" ]; then
  if [ -n "${CUSTOM_IMAGE}" ]; then
    echo "[3/5] preparing custom image embeddings"
    "${PYTHON_BIN}" -u "${BASE_DIR}/prepare_acceptance_custom_image.py" \
      --image "${CUSTOM_IMAGE}" \
      --output-dir "${CUSTOM_WORK_DIR}" \
      --asset-cache-dir "${VIS_ROOT}/asset_cache/aerial_view" \
      2>&1 | tee "${OUT_ROOT}/logs/00_custom_image_prepare.log"
  fi
  echo "[3/5] running fresh qualitative inference on GPU ${GPU_ID}"
  (
    cd "${MONITORING_DIR}"
    CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" -u qualitative_visualization_runner.py
  ) 2>&1 | tee "${OUT_ROOT}/logs/01_inference.log"
else
  echo "[3/5] visual-only mode: reuse latest ${VIS_ROOT}"
fi

echo "[4/5] building acceptance GIF/PNG visual package"
"${PYTHON_BIN}" -u "${BASE_DIR}/build_acceptance_demo_visuals.py" 2>&1 | tee "${OUT_ROOT}/logs/02_visual_generation.log"

echo "[5/5] writing inventory"
{
  echo "# GeoExplorer acceptance demo one-click result"
  echo
  echo "Generated: $(date -Is)"
  echo "Output root: ${OUT_ROOT}"
  echo "Inference output root: ${VIS_ROOT}"
  echo "Visual package: ${OUT_ROOT}/figures"
  echo "Report: ${OUT_ROOT}/acceptance_demo_visuals_zh.md"
  echo
  echo "## Files"
  find "${OUT_ROOT}" -maxdepth 3 -type f | sort
} > "${OUT_ROOT}/README.md"

ln -sfn "${OUT_ROOT}" "${REMOTE_ROOT}/analysis/acceptance_demo_latest"

find "${OUT_ROOT}" -maxdepth 2 -type f \( -name '*.gif' -o -name '*.png' -o -name '*.json' -o -name '*.md' \) | sort
echo
echo "DONE: ${OUT_ROOT}"
echo "LATEST: ${REMOTE_ROOT}/analysis/acceptance_demo_latest"
