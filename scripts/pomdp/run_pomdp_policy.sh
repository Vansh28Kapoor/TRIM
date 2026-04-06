#!/bin/bash
# ============================================================
# run_pomdp_policy.sh — Precompute POMDP action lookup table
#
# Runs get_pomdp_policy.py on a CPU-only node using a
# multiprocessing pool of SARSOP solvers (via PyJulia).
#
# Submit with defaults:
#   sbatch scripts/pomdp/run_pomdp_policy.sh
#
# Override any param with flags, e.g.:
#   sbatch scripts/pomdp/run_pomdp_policy.sh \
#     --workers 64 \
#     --p-slm 0.866 --p-llm 0.9704 \
#     --cost-per-token 0.25 --task-reward 100 \
#     --closeness-thr 0.5 --belief-step 0.025 \
#     --avg-token-count 60 --max-steps 30 \
#     --sarsop-timeout 20 \
#     --terminal-benchmark math --terminal-split train \
#     --obs-benchmark math-train --obs-bin-size 0.05 \
#     --output pomdp_data/my_table.pkl
#
# Pass-through: any unrecognised flag is forwarded to get_pomdp_policy.py.
# ============================================================
#SBATCH --job-name=trim_pomdp_policy
#SBATCH --partition=cpu
#SBATCH --account=bfow-delta-cpu         # ← replace with your allocation
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=240G
#SBATCH --time=30:00
#SBATCH --output=logs/trim_pomdp_policy_%j.out
#SBATCH --error=logs/trim_pomdp_policy_%j.err

set -euo pipefail
mkdir -p logs

# ---- Environment --------------------------------------------
CONDA_BASE="$(conda info --base 2>/dev/null || echo "${HOME}/miniconda3")"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate trim-pomdp

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

echo "=== Job ${SLURM_JOB_ID:-local} on $(hostname) ==="
echo "CPUs available: $(nproc)"

# ---- Defaults (mirror _DEFAULTS in get_pomdp_policy.py) -----
WORKERS="${WORKERS:-64}"
CHUNKSIZE="${CHUNKSIZE:-1}"
P_SLM="${P_SLM:-0.866}"
P_LLM="${P_LLM:-0.9704}"
AVG_TOKEN_COUNT="${AVG_TOKEN_COUNT:-60}"
MAX_STEPS="${MAX_STEPS:-30}"
SARSOP_TIMEOUT="${SARSOP_TIMEOUT:-20.0}"
TOKEN_COUNTS="${TOKEN_COUNTS:-5,10,20,30,40,50,75,100,125,150,175,200,250,300,350,400,450,500}"
BELIEF_STEP="${BELIEF_STEP:-0.025}"
CLOSENESS_THR="${CLOSENESS_THR:-0.5}"
COST_PER_TOKEN="${COST_PER_TOKEN:-0.25}"
TASK_REWARD="${TASK_REWARD:-100}"
TERMINAL_BENCHMARK="${TERMINAL_BENCHMARK:-math}"
TERMINAL_SPLIT="${TERMINAL_SPLIT:-train}"
OBS_BENCHMARK="${OBS_BENCHMARK:-math-train}"
OBS_BIN_SIZE="${OBS_BIN_SIZE:-0.05}"
OUTPUT="${OUTPUT:-}"  # leave empty → auto-derived name in get_pomdp_policy.py

# ---- Argument parsing ---------------------------------------
PASSTHROUGH_ARGS=()
while (($#)); do
    case "$1" in
        --workers)              [[ $# -ge 2 ]] || { echo "[error] Missing value for $1" >&2; exit 2; }
                                WORKERS="$2";              shift 2 ;;
        --workers=*)            WORKERS="${1#*=}";          shift ;;
        --chunksize)            [[ $# -ge 2 ]] || { echo "[error] Missing value for $1" >&2; exit 2; }
                                CHUNKSIZE="$2";            shift 2 ;;
        --chunksize=*)          CHUNKSIZE="${1#*=}";        shift ;;
        --p-slm)                [[ $# -ge 2 ]] || { echo "[error] Missing value for $1" >&2; exit 2; }
                                P_SLM="$2";                shift 2 ;;
        --p-slm=*)              P_SLM="${1#*=}";            shift ;;
        --p-llm)                [[ $# -ge 2 ]] || { echo "[error] Missing value for $1" >&2; exit 2; }
                                P_LLM="$2";                shift 2 ;;
        --p-llm=*)              P_LLM="${1#*=}";            shift ;;
        --avg-token-count)      [[ $# -ge 2 ]] || { echo "[error] Missing value for $1" >&2; exit 2; }
                                AVG_TOKEN_COUNT="$2";      shift 2 ;;
        --avg-token-count=*)    AVG_TOKEN_COUNT="${1#*=}";  shift ;;
        --max-steps)            [[ $# -ge 2 ]] || { echo "[error] Missing value for $1" >&2; exit 2; }
                                MAX_STEPS="$2";            shift 2 ;;
        --max-steps=*)          MAX_STEPS="${1#*=}";        shift ;;
        --sarsop-timeout)       [[ $# -ge 2 ]] || { echo "[error] Missing value for $1" >&2; exit 2; }
                                SARSOP_TIMEOUT="$2";       shift 2 ;;
        --sarsop-timeout=*)     SARSOP_TIMEOUT="${1#*=}";   shift ;;
        --token-counts)         [[ $# -ge 2 ]] || { echo "[error] Missing value for $1" >&2; exit 2; }
                                TOKEN_COUNTS="$2";         shift 2 ;;
        --token-counts=*)       TOKEN_COUNTS="${1#*=}";     shift ;;
        --belief-step)          [[ $# -ge 2 ]] || { echo "[error] Missing value for $1" >&2; exit 2; }
                                BELIEF_STEP="$2";          shift 2 ;;
        --belief-step=*)        BELIEF_STEP="${1#*=}";      shift ;;
        --closeness-thr)        [[ $# -ge 2 ]] || { echo "[error] Missing value for $1" >&2; exit 2; }
                                CLOSENESS_THR="$2";        shift 2 ;;
        --closeness-thr=*)      CLOSENESS_THR="${1#*=}";   shift ;;
        --cost-per-token)       [[ $# -ge 2 ]] || { echo "[error] Missing value for $1" >&2; exit 2; }
                                COST_PER_TOKEN="$2";       shift 2 ;;
        --cost-per-token=*)     COST_PER_TOKEN="${1#*=}";   shift ;;
        --task-reward)          [[ $# -ge 2 ]] || { echo "[error] Missing value for $1" >&2; exit 2; }
                                TASK_REWARD="$2";          shift 2 ;;
        --task-reward=*)        TASK_REWARD="${1#*=}";      shift ;;
        --terminal-benchmark)   [[ $# -ge 2 ]] || { echo "[error] Missing value for $1" >&2; exit 2; }
                                TERMINAL_BENCHMARK="$2";   shift 2 ;;
        --terminal-benchmark=*) TERMINAL_BENCHMARK="${1#*=}"; shift ;;
        --terminal-split)       [[ $# -ge 2 ]] || { echo "[error] Missing value for $1" >&2; exit 2; }
                                TERMINAL_SPLIT="$2";       shift 2 ;;
        --terminal-split=*)     TERMINAL_SPLIT="${1#*=}";   shift ;;
        --obs-benchmark)        [[ $# -ge 2 ]] || { echo "[error] Missing value for $1" >&2; exit 2; }
                                OBS_BENCHMARK="$2";        shift 2 ;;
        --obs-benchmark=*)      OBS_BENCHMARK="${1#*=}";    shift ;;
        --obs-bin-size)         [[ $# -ge 2 ]] || { echo "[error] Missing value for $1" >&2; exit 2; }
                                OBS_BIN_SIZE="$2";         shift 2 ;;
        --obs-bin-size=*)       OBS_BIN_SIZE="${1#*=}";     shift ;;
        --output)               [[ $# -ge 2 ]] || { echo "[error] Missing value for $1" >&2; exit 2; }
                                OUTPUT="$2";               shift 2 ;;
        --output=*)             OUTPUT="${1#*=}";           shift ;;
        *)
            PASSTHROUGH_ARGS+=("$1")
            shift
            ;;
    esac
done

# ---- Build Python args --------------------------------------
POLICY_ARGS=(
    --workers          "${WORKERS}"
    --chunksize        "${CHUNKSIZE}"
    --p-slm            "${P_SLM}"
    --p-llm            "${P_LLM}"
    --avg-token-count  "${AVG_TOKEN_COUNT}"
    --max-steps        "${MAX_STEPS}"
    --sarsop-timeout   "${SARSOP_TIMEOUT}"
    --token-counts     "${TOKEN_COUNTS}"
    --belief-step      "${BELIEF_STEP}"
    --closeness-thr    "${CLOSENESS_THR}"
    --cost-per-token   "${COST_PER_TOKEN}"
    --task-reward      "${TASK_REWARD}"
    --terminal-benchmark "${TERMINAL_BENCHMARK}"
    --terminal-split   "${TERMINAL_SPLIT}"
    --obs-benchmark    "${OBS_BENCHMARK}"
    --obs-bin-size     "${OBS_BIN_SIZE}"
)
[[ -n "${OUTPUT}" ]] && POLICY_ARGS+=(--output "${OUTPUT}")
POLICY_ARGS+=("${PASSTHROUGH_ARGS[@]}")

echo "[pomdp-policy] p_slm=${P_SLM}, p_llm=${P_LLM}, cost_per_token=${COST_PER_TOKEN}, task_reward=${TASK_REWARD}, closeness_thr=${CLOSENESS_THR}, belief_step=${BELIEF_STEP}, avg_token_count=${AVG_TOKEN_COUNT}, max_steps=${MAX_STEPS}, sarsop_timeout=${SARSOP_TIMEOUT}"
echo "[pomdp-policy] terminal=${TERMINAL_BENCHMARK}/${TERMINAL_SPLIT}, obs=${OBS_BENCHMARK} (bin=${OBS_BIN_SIZE}), workers=${WORKERS}"
printf '[pomdp-policy] Python args:'
printf ' %q' "${POLICY_ARGS[@]}"
printf '\n'

python get_pomdp_policy.py "${POLICY_ARGS[@]}"

echo "[pomdp-policy] Done."
