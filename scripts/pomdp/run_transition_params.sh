#!/bin/bash
# ============================================================
# run_transition_params.sh — Build POMDP transition function params
#
# Launches vLLM servers, then runs:
#   pomdp_params/get_transition_function.py
#
# Submit with defaults:
#   sbatch scripts/pomdp/run_transition_params.sh
#
# Override models:
#   sbatch scripts/pomdp/run_transition_params.sh --target-model-name Qwen/Qwen3-8B
#
# Override transition params with flags:
#   sbatch scripts/pomdp/run_transition_params.sh \
#     --transition-benchmark aime \
#     --transition-split train \
#     --thr 0.35 \
#     --max-steps 30 \
#     --terminal-predictor-dir pomdp_data \
#     --rerun
# ============================================================
#SBATCH --job-name=trim_pomdp_transition_params
#SBATCH --partition=gpuA40x4
#SBATCH --account=bfow-delta-gpu         # ← replace with your allocation
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=2
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/trim_pomdp_transition_params_%j.out
#SBATCH --error=logs/trim_pomdp_transition_params_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail
mkdir -p logs

# ---- Environment --------------------------------------------
CONDA_BASE="$(conda info --base 2>/dev/null || echo "${HOME}/miniconda3")"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate trim-pomdp

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

echo "=== Job ${SLURM_JOB_ID:-local} on $(hostname) ==="
nvidia-smi

TRANSITION_BENCHMARK="${TRANSITION_BENCHMARK:-math}"
TRANSITION_SPLIT="${TRANSITION_SPLIT:-train}"
THR="${THR:-0.35}"
MAX_STEPS="${MAX_STEPS:-30}"
TERMINAL_PREDICTOR_DIR="${TERMINAL_PREDICTOR_DIR:-pomdp_data}"
RERUN="${RERUN:-0}"

# Single source of truth for server/client model identity.
export TARGET_MODEL="${TARGET_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
export DRAFT_MODEL="${DRAFT_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
export PRM_MODEL="${PRM_MODEL:-Qwen/Qwen2.5-Math-PRM-7B}"

while (($#)); do
    case "$1" in
        --draft_model_name|--draft-model-name)
            [[ $# -ge 2 ]] || { echo "[error] Missing value for $1" >&2; exit 2; }
            export DRAFT_MODEL="$2"
            shift 2
            ;;
        --draft_model_name=*|--draft-model-name=*)
            export DRAFT_MODEL="${1#*=}"
            shift
            ;;
        --target_model_name|--target-model-name)
            [[ $# -ge 2 ]] || { echo "[error] Missing value for $1" >&2; exit 2; }
            export TARGET_MODEL="$2"
            shift 2
            ;;
        --target_model_name=*|--target-model-name=*)
            export TARGET_MODEL="${1#*=}"
            shift
            ;;
        --prm_model_name|--prm-model-name)
            [[ $# -ge 2 ]] || { echo "[error] Missing value for $1" >&2; exit 2; }
            export PRM_MODEL="$2"
            shift 2
            ;;
        --prm_model_name=*|--prm-model-name=*)
            export PRM_MODEL="${1#*=}"
            shift
            ;;
        --transition-benchmark|--train-benchmark)
            [[ $# -ge 2 ]] || { echo "[error] Missing value for $1" >&2; exit 2; }
            TRANSITION_BENCHMARK="$2"
            shift 2
            ;;
        --transition-benchmark=*|--train-benchmark=*)
            TRANSITION_BENCHMARK="${1#*=}"
            shift
            ;;
        --transition-split|--train-split)
            [[ $# -ge 2 ]] || { echo "[error] Missing value for $1" >&2; exit 2; }
            TRANSITION_SPLIT="$2"
            shift 2
            ;;
        --transition-split=*|--train-split=*)
            TRANSITION_SPLIT="${1#*=}"
            shift
            ;;
        --thr)
            [[ $# -ge 2 ]] || { echo "[error] Missing value for $1" >&2; exit 2; }
            THR="$2"
            shift 2
            ;;
        --thr=*)
            THR="${1#*=}"
            shift
            ;;
        --max-steps)
            [[ $# -ge 2 ]] || { echo "[error] Missing value for $1" >&2; exit 2; }
            MAX_STEPS="$2"
            shift 2
            ;;
        --max-steps=*)
            MAX_STEPS="${1#*=}"
            shift
            ;;
        --terminal-predictor-dir)
            [[ $# -ge 2 ]] || { echo "[error] Missing value for $1" >&2; exit 2; }
            TERMINAL_PREDICTOR_DIR="$2"
            shift 2
            ;;
        --terminal-predictor-dir=*)
            TERMINAL_PREDICTOR_DIR="${1#*=}"
            shift
            ;;
        --rerun)
            RERUN="1"
            shift
            ;;
        --no-rerun)
            RERUN="0"
            shift
            ;;
        *)
            echo "[error] Unknown argument for run_transition_params.sh: $1" >&2
            exit 2
            ;;
    esac
done

RERUN_FLAG=""
if [[ "${RERUN}" == "1" ]]; then
    RERUN_FLAG="--rerun"
fi

echo "[pomdp-transition] Launching vLLM servers..."
source scripts/launch_servers.sh

echo "[pomdp-transition] Building transition params (benchmark=${TRANSITION_BENCHMARK}, split=${TRANSITION_SPLIT})"
TRANSITION_ARGS=(
    --train-benchmark "${TRANSITION_BENCHMARK}"
    --train-split "${TRANSITION_SPLIT}"
    --draft-model-name "${DRAFT_MODEL}"
    --target-model-name "${TARGET_MODEL}"
    --prm-model-name "${PRM_MODEL}"
    --thr "${THR}"
    --max-steps "${MAX_STEPS}"
    --draft-server-url "http://localhost:${DRAFT_PORT}/v1"
    --target-server-url "http://localhost:${TARGET_PORT}/v1"
    --prm-server-url "http://localhost:${PRM_PORT}"
    --terminal-predictor-dir "${TERMINAL_PREDICTOR_DIR}"
)
if [[ -n "${RERUN_FLAG}" ]]; then
    TRANSITION_ARGS+=("${RERUN_FLAG}")
fi

echo "[pomdp-transition] Effective settings: benchmark=${TRANSITION_BENCHMARK}, split=${TRANSITION_SPLIT}, thr=${THR}, max_steps=${MAX_STEPS}, terminal_predictor_dir=${TERMINAL_PREDICTOR_DIR}, rerun=${RERUN}"
echo "[pomdp-transition] Effective models: TARGET_MODEL=${TARGET_MODEL}, DRAFT_MODEL=${DRAFT_MODEL}, PRM_MODEL=${PRM_MODEL}"
printf '[pomdp-transition] Python args:'
printf ' %q' "${TRANSITION_ARGS[@]}"
printf '\n'

python pomdp_params/get_transition_function.py "${TRANSITION_ARGS[@]}"

echo "[pomdp-transition] Done."
