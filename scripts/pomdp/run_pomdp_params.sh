#!/bin/bash
# ============================================================
# run_pomdp_params.sh — Build POMDP observation/transition params
#
# Runs, in order:
#   1) pomdp_params/get_observation_function.py
#   2) pomdp_params/get_transition_function.py
#
# Submit with defaults:
#   sbatch scripts/pomdp/run_pomdp_params.sh
#
# Override models:
#   sbatch scripts/pomdp/run_pomdp_params.sh --target-model-name Qwen/Qwen3-8B
#
# Override params with flags:
#   sbatch scripts/pomdp/run_pomdp_params.sh \
#     --obs-benchmark omnimath \
#     --bin-size 0.05 \
#     --transition-benchmark aime \
#     --transition-split train \
#     --thr 0.35 \
#     --max-steps 30 \
#     --terminal-predictor-dir pomdp_data \
#     --rerun
# ============================================================
#SBATCH --job-name=trim_pomdp_params
#SBATCH --partition=gpuA40x4
#SBATCH --account=bfow-delta-gpu         # ← replace with your allocation
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=2
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/trim_pomdp_params_%j.out
#SBATCH --error=logs/trim_pomdp_params_%j.err
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

OBS_BENCHMARK="${OBS_BENCHMARK:-omnimath}"
BIN_SIZE="${BIN_SIZE:-0.05}"
TRANSITION_BENCHMARK="${TRANSITION_BENCHMARK:-aime}"
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
        --obs-benchmark)
            [[ $# -ge 2 ]] || { echo "[error] Missing value for $1" >&2; exit 2; }
            OBS_BENCHMARK="$2"
            shift 2
            ;;
        --obs-benchmark=*)
            OBS_BENCHMARK="${1#*=}"
            shift
            ;;
        --bin-size)
            [[ $# -ge 2 ]] || { echo "[error] Missing value for $1" >&2; exit 2; }
            BIN_SIZE="$2"
            shift 2
            ;;
        --bin-size=*)
            BIN_SIZE="${1#*=}"
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
            echo "[error] Unknown argument for run_pomdp_params.sh: $1" >&2
            exit 2
            ;;
    esac
done

RERUN_FLAG=""
if [[ "${RERUN}" == "1" ]]; then
    RERUN_FLAG="--rerun"
fi

echo "[pomdp] Launching vLLM servers..."
source scripts/launch_servers.sh

echo "[pomdp] Building observation function params (benchmark=${OBS_BENCHMARK}, bin_size=${BIN_SIZE})"
OBS_ARGS=(
    --benchmark "${OBS_BENCHMARK}"
    --bin-size "${BIN_SIZE}"
    --prm-model-name "${PRM_MODEL}"
    --prm-server-url "http://localhost:${PRM_PORT}"
)
if [[ -n "${RERUN_FLAG}" ]]; then
    OBS_ARGS+=("${RERUN_FLAG}")
fi

echo "[pomdp] Effective models: TARGET_MODEL=${TARGET_MODEL}, DRAFT_MODEL=${DRAFT_MODEL}, PRM_MODEL=${PRM_MODEL}"
echo "[pomdp] Effective observation settings: benchmark=${OBS_BENCHMARK}, bin_size=${BIN_SIZE}, rerun=${RERUN}"
printf '[pomdp] Observation Python args:'
printf ' %q' "${OBS_ARGS[@]}"
printf '\n'

python pomdp_params/get_observation_function.py "${OBS_ARGS[@]}"

echo "[pomdp] Building transition function params (benchmark=${TRANSITION_BENCHMARK}, split=${TRANSITION_SPLIT})"
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

echo "[pomdp] Effective transition settings: benchmark=${TRANSITION_BENCHMARK}, split=${TRANSITION_SPLIT}, thr=${THR}, max_steps=${MAX_STEPS}, terminal_predictor_dir=${TERMINAL_PREDICTOR_DIR}, rerun=${RERUN}"
printf '[pomdp] Transition Python args:'
printf ' %q' "${TRANSITION_ARGS[@]}"
printf '\n'

python pomdp_params/get_transition_function.py "${TRANSITION_ARGS[@]}"

echo "[pomdp] Done."
