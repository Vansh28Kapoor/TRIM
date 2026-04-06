#!/bin/bash
# ============================================================
# SLURM job: TRIM-Thr evaluation on NCSA Delta (gpuA40x4, 2× A40)
#
# GPU memory layout (see launch_servers.sh for details):
#   GPU 0 — target 8B  (exclusive, 90%)
#   GPU 1 — draft 1.5B (35%) + PRM 7B (50%)
#
# Submit with defaults:
#   sbatch scripts/run_TRIM_Thr_eval.sh
#
# Override params with flags:
#   sbatch scripts/run_TRIM_Thr_eval.sh \
#     --target-model-name Qwen/Qwen3-8B --target-disable-thinking true \
#     --datasets math500,aime,gsm8k \
#     --eval-split test \
#     --thresholds 0,0.1,0.3,0.5,0.7,0.9,1 \
#     --output-dir ./thr_results
#
# Pass-through of extra TRIM_Thr.py flags is allowed, e.g.:
#   sbatch scripts/run_TRIM_Thr_eval.sh --batch_size 16
# ============================================================
#SBATCH --job-name=trim_thr_eval
#SBATCH --partition=gpuA100x4
#SBATCH --account=bfow-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=2
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/thr_eval_%j.out
#SBATCH --error=logs/thr_eval_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail
mkdir -p logs

# ---- Environment --------------------------------------------
CONDA_BASE="$(conda info --base 2>/dev/null || echo "${HOME}/miniconda3")"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate trim

echo "=== Job ${SLURM_JOB_ID} on $(hostname) ==="
nvidia-smi

# ---- Model configuration ------------------------------------
export TARGET_MODEL="${TARGET_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
export DRAFT_MODEL="${DRAFT_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
export PRM_MODEL="${PRM_MODEL:-Qwen/Qwen2.5-Math-PRM-7B}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"

# ---- Defaults -----------------------------------------------
EVAL_SPLIT="${EVAL_SPLIT:-test}"
OUTPUT_DIR="${OUTPUT_DIR:-./thr_results}"
TARGET_DISABLE_THINKING="${TARGET_DISABLE_THINKING:-false}"
THRESHOLDS="${THRESHOLDS:-0,0.1,0.3,0.5,0.7,0.9,1}"
DATASETS_CSV="${DATASETS_CSV:-math500,aime,gsm8k}"

# ---- Argument parsing ---------------------------------------
PASSTHROUGH_ARGS=()
while (($#)); do
    case "$1" in
        --draft_model_name|--draft-model-name)
            [[ $# -ge 2 ]] || { echo "[error] Missing value for $1" >&2; exit 2; }
            export DRAFT_MODEL="$2"; shift 2 ;;
        --draft_model_name=*|--draft-model-name=*)
            export DRAFT_MODEL="${1#*=}"; shift ;;
        --target_model_name|--target-model-name)
            [[ $# -ge 2 ]] || { echo "[error] Missing value for $1" >&2; exit 2; }
            export TARGET_MODEL="$2"; shift 2 ;;
        --target_model_name=*|--target-model-name=*)
            export TARGET_MODEL="${1#*=}"; shift ;;
        --prm_model_name|--prm-model-name)
            [[ $# -ge 2 ]] || { echo "[error] Missing value for $1" >&2; exit 2; }
            export PRM_MODEL="$2"; shift 2 ;;
        --prm_model_name=*|--prm-model-name=*)
            export PRM_MODEL="${1#*=}"; shift ;;
        --datasets)
            [[ $# -ge 2 ]] || { echo "[error] Missing value for $1" >&2; exit 2; }
            DATASETS_CSV="$2"; shift 2 ;;
        --datasets=*)
            DATASETS_CSV="${1#*=}"; shift ;;
        --eval_split|--eval-split)
            [[ $# -ge 2 ]] || { echo "[error] Missing value for $1" >&2; exit 2; }
            EVAL_SPLIT="$2"; shift 2 ;;
        --eval_split=*|--eval-split=*)
            EVAL_SPLIT="${1#*=}"; shift ;;
        --output_dir|--output-dir)
            [[ $# -ge 2 ]] || { echo "[error] Missing value for $1" >&2; exit 2; }
            OUTPUT_DIR="$2"; shift 2 ;;
        --output_dir=*|--output-dir=*)
            OUTPUT_DIR="${1#*=}"; shift ;;
        --target_disable_thinking|--target-disable-thinking)
            [[ $# -ge 2 ]] || { echo "[error] Missing value for $1" >&2; exit 2; }
            TARGET_DISABLE_THINKING="$2"; shift 2 ;;
        --target_disable_thinking=*|--target-disable-thinking=*)
            TARGET_DISABLE_THINKING="${1#*=}"; shift ;;
        --thresholds)
            [[ $# -ge 2 ]] || { echo "[error] Missing value for $1" >&2; exit 2; }
            THRESHOLDS="$2"; shift 2 ;;
        --thresholds=*)
            THRESHOLDS="${1#*=}"; shift ;;
        *)
            PASSTHROUGH_ARGS+=("$1"); shift ;;
    esac
done

# ---- Launch vLLM servers ------------------------------------
source scripts/launch_servers.sh

# ---- Run evaluation -----------------------------------------
IFS=',' read -r -a DATASETS <<< "${DATASETS_CSV}"

echo "[trim-thr] Effective models: TARGET_MODEL=${TARGET_MODEL}, DRAFT_MODEL=${DRAFT_MODEL}, PRM_MODEL=${PRM_MODEL}"
echo "[trim-thr] Effective settings: datasets=${DATASETS_CSV}, eval_split=${EVAL_SPLIT}, thresholds=${THRESHOLDS}, output_dir=${OUTPUT_DIR}"

for DATASET in "${DATASETS[@]}"; do
    DATASET_TRIMMED="${DATASET//[[:space:]]/}"
    [[ -z "${DATASET_TRIMMED}" ]] && continue

    DATASET_OUTPUT_DIR="${OUTPUT_DIR}/${DATASET_TRIMMED}"

    TRIM_THR_ARGS=(
        --draft_model_name      "${DRAFT_MODEL}"
        --target_model_name     "${TARGET_MODEL}"
        --prm_model_name        "${PRM_MODEL}"
        --eval_dataset_name     "${DATASET_TRIMMED}"
        --eval_split            "${EVAL_SPLIT}"
        --output_dir            "${DATASET_OUTPUT_DIR}"
        --target_disable_thinking "${TARGET_DISABLE_THINKING}"
        --thresholds            "${THRESHOLDS}"
    )
    TRIM_THR_ARGS+=("${PASSTHROUGH_ARGS[@]}")

    echo "=== Evaluating on ${DATASET_TRIMMED} ==="
    printf '[trim-thr] Python args:'
    printf ' %q' "${TRIM_THR_ARGS[@]}"
    printf '\n'
    python TRIM_Thr.py "${TRIM_THR_ARGS[@]}"
done

echo "=== TRIM-Thr evaluation complete ==="
