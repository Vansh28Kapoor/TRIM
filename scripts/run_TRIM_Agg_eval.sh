#!/bin/bash
# ============================================================
# SLURM job: TRIM-Agg evaluation on NCSA Delta (gpuA40x4, 2× A40)
#
# GPU memory layout (see launch_servers.sh for details):
#   GPU 0 — target 8B  (exclusive, 90%)
#   GPU 1 — draft 1.5B (35%) + PRM 7B (50%)
#
# Submit:  sbatch scripts/run_TRIM_Agg_eval.sh
#
# Override evaluation params with flags:
#   sbatch scripts/run_TRIM_Agg_eval.sh \
#     --target-model-name Qwen/Qwen3-8B --target-disable-thinking true \
#     --datasets math500,aime,gsm8k \
#     --eval-split test \
#     --checkpoint ./checkpoints/8e-4/policy_best.pt
#
# Pass-through of extra TRIM_Agg.py flags is allowed, e.g.:
#   sbatch scripts/run_TRIM_Agg_eval.sh --datasets math500 --use_wandb false
# ============================================================
#SBATCH --job-name=trim_agg_eval
#SBATCH --partition=gpuA40x4
#SBATCH --account=bfow-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=2
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err

set -euo pipefail
mkdir -p logs

# ---- Environment --------------------------------------------
CONDA_BASE="$(conda info --base 2>/dev/null || echo "${HOME}/miniconda3")"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate trim

# export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
# export HF_HUB_OFFLINE=1
# export HF_DATASETS_OFFLINE=1
# export TRANSFORMERS_OFFLINE=1

echo "=== Job ${SLURM_JOB_ID} on $(hostname) ==="
nvidia-smi

# ---- Model configuration ------------------------------------
# Defaults can be overridden via env vars or via this script's model flags.
export TARGET_MODEL="${TARGET_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
export DRAFT_MODEL="${DRAFT_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
export PRM_MODEL="${PRM_MODEL:-Qwen/Qwen2.5-Math-PRM-7B}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"

# ---- Checkpoint to evaluate ---------------------------------
# Default resolves after flag parsing so --cost-per-token is applied first.
CHECKPOINT=""
CHECKPOINT_SET=false
EVAL_SPLIT="${EVAL_SPLIT:-test}"
OUTPUT_DIR="${OUTPUT_DIR:-./rlpolicy_results}"
TARGET_DISABLE_THINKING="${TARGET_DISABLE_THINKING:-false}"
DATASETS_CSV="${DATASETS_CSV:-math500,aime,gsm8k}"
COST_PER_TOKEN="${COST_PER_TOKEN:-8e-4}"

# Parse known script flags; forward unknown flags to TRIM_Agg.py.
PASSTHROUGH_ARGS=()
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
        --checkpoint)
            [[ $# -ge 2 ]] || { echo "[error] Missing value for $1" >&2; exit 2; }
            CHECKPOINT="$2"
            CHECKPOINT_SET=true
            shift 2
            ;;
        --checkpoint=*)
            CHECKPOINT="${1#*=}"
            CHECKPOINT_SET=true
            shift
            ;;
        --datasets)
            [[ $# -ge 2 ]] || { echo "[error] Missing value for $1" >&2; exit 2; }
            DATASETS_CSV="$2"
            shift 2
            ;;
        --datasets=*)
            DATASETS_CSV="${1#*=}"
            shift
            ;;
        --eval_split|--eval-split)
            [[ $# -ge 2 ]] || { echo "[error] Missing value for $1" >&2; exit 2; }
            EVAL_SPLIT="$2"
            shift 2
            ;;
        --eval_split=*|--eval-split=*)
            EVAL_SPLIT="${1#*=}"
            shift
            ;;
        --output_dir|--output-dir)
            [[ $# -ge 2 ]] || { echo "[error] Missing value for $1" >&2; exit 2; }
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --output_dir=*|--output-dir=*)
            OUTPUT_DIR="${1#*=}"
            shift
            ;;
        --target_disable_thinking|--target-disable-thinking)
            [[ $# -ge 2 ]] || { echo "[error] Missing value for $1" >&2; exit 2; }
            TARGET_DISABLE_THINKING="$2"
            shift 2
            ;;
        --target_disable_thinking=*|--target-disable-thinking=*)
            TARGET_DISABLE_THINKING="${1#*=}"
            shift
            ;;
        --cost_per_token|--cost-per-token)
            [[ $# -ge 2 ]] || { echo "[error] Missing value for $1" >&2; exit 2; }
            COST_PER_TOKEN="$2"
            shift 2
            ;;
        --cost_per_token=*|--cost-per-token=*)
            COST_PER_TOKEN="${1#*=}"
            shift
            ;;
        *)
            if [[ "$1" != --* && "${CHECKPOINT_SET}" == false ]]; then
                CHECKPOINT="$1"
                CHECKPOINT_SET=true
            else
                PASSTHROUGH_ARGS+=("$1")
            fi
            shift
            ;;
    esac
done

# ---- Resolve default checkpoint from cost_per_token -----------
if [[ "${CHECKPOINT_SET}" == false ]]; then
    CHECKPOINT="./rlpolicy_checkpoints/${COST_PER_TOKEN}/policy_best.pt"
fi

# ---- Datasets to evaluate -----------------------------------
IFS=',' read -r -a DATASETS <<< "${DATASETS_CSV}"

# ---- Launch vLLM servers ------------------------------------
source scripts/launch_servers.sh

# ---- Run evaluation -----------------------------------------
echo "[trim-eval] Effective models: TARGET_MODEL=${TARGET_MODEL}, DRAFT_MODEL=${DRAFT_MODEL}, PRM_MODEL=${PRM_MODEL}"
echo "[trim-eval] Effective eval settings: datasets=${DATASETS_CSV}, eval_split=${EVAL_SPLIT}, checkpoint=${CHECKPOINT}, output_dir=${OUTPUT_DIR}, target_disable_thinking=${TARGET_DISABLE_THINKING}, cost_per_token=${COST_PER_TOKEN}"

for DATASET in "${DATASETS[@]}"; do
    DATASET_TRIMMED="${DATASET//[[:space:]]/}"
    if [[ -z "${DATASET_TRIMMED}" ]]; then
        continue
    fi

    TRIM_EVAL_ARGS=(
        --mode eval
        --draft_model_name "${DRAFT_MODEL}"
        --target_model_name "${TARGET_MODEL}"
        --prm_model_name "${PRM_MODEL}"
        --eval_dataset_name "${DATASET_TRIMMED}"
        --eval_split "${EVAL_SPLIT}"
        --checkpoint "${CHECKPOINT}"
        --output_dir "${OUTPUT_DIR}"
        --target_disable_thinking "${TARGET_DISABLE_THINKING}"
        --cost_per_token "${COST_PER_TOKEN}"
    )
    TRIM_EVAL_ARGS+=("${PASSTHROUGH_ARGS[@]}")

    echo "=== Evaluating on ${DATASET} ==="
    printf '[trim-eval] Python args:'
    printf ' %q' "${TRIM_EVAL_ARGS[@]}"
    printf '\n'
    python TRIM_Agg.py "${TRIM_EVAL_ARGS[@]}"
done

echo "=== Evaluation complete ==="
