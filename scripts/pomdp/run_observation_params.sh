#!/bin/bash
# ============================================================
# run_observation_params.sh — Build POMDP observation function params
#
# Launches vLLM servers, then runs:
#   pomdp_params/get_observation_function.py
#
# Submit with defaults:
#   sbatch scripts/pomdp/run_observation_params.sh
#
# Override params with flags:
#   sbatch scripts/pomdp/run_observation_params.sh \
#     --prm-model-name Qwen/Qwen2.5-Math-PRM-7B \
#     --obs-benchmark omnimath \
#     --bin-size 0.05 \
#     --rerun
# ============================================================
#SBATCH --job-name=trim_pomdp_observation_params
#SBATCH --partition=gpuA40x4
#SBATCH --account=bfow-delta-gpu         # ← replace with your allocation
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/trim_pomdp_observation_params_%j.out
#SBATCH --error=logs/trim_pomdp_observation_params_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail
mkdir -p logs

# ---- Environment --------------------------------------------
CONDA_BASE="$(conda info --base 2>/dev/null || echo "${HOME}/miniconda3")"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate trim-pomdp

# Use SLURM_SUBMIT_DIR (set by sbatch) so cd targets the repo root
# regardless of where SLURM copies the script internally.
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

echo "=== Job ${SLURM_JOB_ID:-local} on $(hostname) ==="
nvidia-smi

OBS_BENCHMARK="${OBS_BENCHMARK:-omnimath}"
BIN_SIZE="${BIN_SIZE:-0.05}"
RERUN="${RERUN:-0}"

# Single source of truth for server/client model identity.
export PRM_MODEL="${PRM_MODEL:-Qwen/Qwen2.5-Math-PRM-7B}"

while (($#)); do
    case "$1" in
        --prm_model_name|--prm-model-name)
            [[ $# -ge 2 ]] || { echo "[error] Missing value for $1" >&2; exit 2; }
            export PRM_MODEL="$2"
            shift 2
            ;;
        --prm_model_name=*|--prm-model-name=*)
            export PRM_MODEL="${1#*=}"
            shift
            ;;
        --obs-benchmark|--benchmark)
            [[ $# -ge 2 ]] || { echo "[error] Missing value for $1" >&2; exit 2; }
            OBS_BENCHMARK="$2"
            shift 2
            ;;
        --obs-benchmark=*|--benchmark=*)
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
        --rerun)
            RERUN="1"
            shift
            ;;
        --no-rerun)
            RERUN="0"
            shift
            ;;
        *)
            echo "[error] Unknown argument for run_observation_params.sh: $1" >&2
            exit 2
            ;;
    esac
done

RERUN_FLAG=""
if [[ "${RERUN}" == "1" ]]; then
    RERUN_FLAG="--rerun"
fi

# ---- Launch PRM model (GPU 0, 90% utilization) ----
PRM_PORT="${PRM_PORT:-30002}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
PRM_MEM_UTIL="${PRM_MEM_UTIL:-0.90}"

echo "[pomdp-observation] Launching PRM model on GPU 0..."
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARNING}"
VLLM_LOG_DIR="${VLLM_LOG_DIR:-logs}"
mkdir -p "${VLLM_LOG_DIR}"

# Helper: wait for vLLM server to become ready
wait_for_server() {
    local port=$1
    local name=$2
    local max_wait=${3:-600}
    local elapsed=0
    echo "[pomdp-observation] Waiting for ${name} on port ${port} ..."
    while true; do
        if curl -sf "http://localhost:${port}/health" > /dev/null 2>&1; then
            echo "[pomdp-observation] ${name} is ready (port ${port}, ${elapsed}s)."
            return 0
        fi
        if (( elapsed >= max_wait )); then
            echo "[pomdp-observation] ERROR: ${name} did not start within ${max_wait}s." >&2
            return 1
        fi
        sleep 10
        elapsed=$((elapsed + 10))
    done
}

# Launch PRM server
CUDA_VISIBLE_DEVICES=0 vllm serve "${PRM_MODEL}" \
    --dtype auto \
    --trust-remote-code \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${PRM_MEM_UTIL}" \
    --enable-prefix-caching \
    --disable-log-requests \
    --disable-log-stats \
    --port "${PRM_PORT}" \
    > "${VLLM_LOG_DIR}/vllm_prm.log" 2>&1 &
VLLM_PRM_PID=$!
wait_for_server "${PRM_PORT}" "PRM (${PRM_MODEL})"

# Cleanup on exit
kill_servers() {
    echo "[pomdp-observation] Shutting down PRM server ..."
    [ -n "${VLLM_PRM_PID:-}" ] && kill "${VLLM_PRM_PID}" 2>/dev/null && wait "${VLLM_PRM_PID}" 2>/dev/null || true
    echo "[pomdp-observation] PRM server stopped."
}
trap kill_servers EXIT

nvidia-smi

echo "[pomdp-observation] Building observation params (benchmark=${OBS_BENCHMARK}, bin_size=${BIN_SIZE})"
OBS_ARGS=(
    --benchmark "${OBS_BENCHMARK}"
    --bin-size "${BIN_SIZE}"
    --prm-model-name "${PRM_MODEL}"
    --prm-server-url "http://localhost:${PRM_PORT}"
)
if [[ -n "${RERUN_FLAG}" ]]; then
    OBS_ARGS+=("${RERUN_FLAG}")
fi

echo "[pomdp-observation] Effective settings: benchmark=${OBS_BENCHMARK}, bin_size=${BIN_SIZE}, rerun=${RERUN}, prm_model=${PRM_MODEL}, prm_server_url=http://localhost:${PRM_PORT}"
printf '[pomdp-observation] Python args:'
printf ' %q' "${OBS_ARGS[@]}"
printf '\n'

python pomdp_params/get_observation_function.py "${OBS_ARGS[@]}"

echo "[pomdp-observation] Done."
