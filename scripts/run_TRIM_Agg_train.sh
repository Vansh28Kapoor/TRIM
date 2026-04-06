#!/bin/bash
# ============================================================
# SLURM job: TRIM-Agg training on NCSA Delta (gpuA40x4, 2× A40)
#
# GPU memory layout (see launch_servers.sh for details):
#   GPU 0 — target 8B  (exclusive, 90%)
#   GPU 1 — draft 1.5B (30%) + PRM 7B (40%)
#
# Submit with defaults (Qwen2.5-7B-Instruct target):
#   sbatch scripts/run_TRIM_Agg_train.sh
#
# Use Qwen3-8B target instead (disable thinking):
#   TARGET_MODEL=Qwen/Qwen3-8B TARGET_DISABLE_THINKING=true sbatch scripts/run_TRIM_Agg_train.sh
#   sbatch scripts/run_TRIM_Agg_train.sh --target_model_name Qwen/Qwen3-8B --target_disable_thinking true
#
# Override training params with flags:
#   sbatch scripts/run_TRIM_Agg_train.sh \
#     --num-epochs 5 --batch-size 16 --eval-every 10 \
#     --train-dataset-name math --train-split train \
#     --eval-dataset-name math500 --eval-split test
#
# Pass-through of extra TRIM_Agg.py flags is allowed, e.g.:
#   sbatch scripts/run_TRIM_Agg_train.sh --use_wandb false --wandb_project trim
#
# IMPORTANT: model-name flags are consumed here and exported as env vars
# before launch_servers.sh runs, so server and TRIM_Agg.py stay consistent.
# ============================================================
#SBATCH --job-name=trim_agg_train
#SBATCH --partition=gpuA40x4
#SBATCH --account=bfow-delta-gpu         # ← replace with your allocation
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=2
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail
mkdir -p logs

# ---- Environment --------------------------------------------
# Source conda shell functions (needed in non-interactive SLURM jobs).
CONDA_BASE="$(conda info --base 2>/dev/null || echo "${HOME}/miniconda3")"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate trim

# # Offline mode (models pre-downloaded to $HF_HOME)
# export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
# Uncomment the lines below if running without internet access:
# export HF_HUB_OFFLINE=1
# export HF_DATASETS_OFFLINE=1
# export TRANSFORMERS_OFFLINE=1

echo "=== Job ${SLURM_JOB_ID} on $(hostname) ==="
nvidia-smi

# ---- Model configuration ------------------------------------
export TARGET_MODEL="${TARGET_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
export DRAFT_MODEL="${DRAFT_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
export PRM_MODEL="${PRM_MODEL:-Qwen/Qwen2.5-Math-PRM-7B}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"

TRAIN_DATASET_NAME="${TRAIN_DATASET_NAME:-math}"
TRAIN_SPLIT="${TRAIN_SPLIT:-train}"
EVAL_DATASET_NAME="${EVAL_DATASET_NAME:-math500}"
EVAL_SPLIT="${EVAL_SPLIT:-test}"
NUM_EPOCHS="${NUM_EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-64}"
VAL_FRACTION="${VAL_FRACTION:-0.1}"
VAL_EVERY="${VAL_EVERY:-50}"
EVAL_EVERY="${EVAL_EVERY:-50}"
SAVE_DIR="${SAVE_DIR:-./rlpolicy_checkpoints}"
OUTPUT_DIR="${OUTPUT_DIR:-./rlpolicy_results}"
SEED="${SEED:-10}"
TARGET_DISABLE_THINKING="${TARGET_DISABLE_THINKING:-false}"
COST_PER_TOKEN="${COST_PER_TOKEN:-8e-4}"
RESUME="${RESUME:-false}"

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
        --train_dataset_name|--train-dataset-name)
            [[ $# -ge 2 ]] || { echo "[error] Missing value for $1" >&2; exit 2; }
            TRAIN_DATASET_NAME="$2"
            shift 2
            ;;
        --train_dataset_name=*|--train-dataset-name=*)
            TRAIN_DATASET_NAME="${1#*=}"
            shift
            ;;
        --train_split|--train-split)
            [[ $# -ge 2 ]] || { echo "[error] Missing value for $1" >&2; exit 2; }
            TRAIN_SPLIT="$2"
            shift 2
            ;;
        --train_split=*|--train-split=*)
            TRAIN_SPLIT="${1#*=}"
            shift
            ;;
        --eval_dataset_name|--eval-dataset-name)
            [[ $# -ge 2 ]] || { echo "[error] Missing value for $1" >&2; exit 2; }
            EVAL_DATASET_NAME="$2"
            shift 2
            ;;
        --eval_dataset_name=*|--eval-dataset-name=*)
            EVAL_DATASET_NAME="${1#*=}"
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
        --num_epochs|--num-epochs)
            [[ $# -ge 2 ]] || { echo "[error] Missing value for $1" >&2; exit 2; }
            NUM_EPOCHS="$2"
            shift 2
            ;;
        --num_epochs=*|--num-epochs=*)
            NUM_EPOCHS="${1#*=}"
            shift
            ;;
        --batch_size|--batch-size)
            [[ $# -ge 2 ]] || { echo "[error] Missing value for $1" >&2; exit 2; }
            BATCH_SIZE="$2"
            shift 2
            ;;
        --batch_size=*|--batch-size=*)
            BATCH_SIZE="${1#*=}"
            shift
            ;;
        --eval_every|--eval-every)
            [[ $# -ge 2 ]] || { echo "[error] Missing value for $1" >&2; exit 2; }
            EVAL_EVERY="$2"
            shift 2
            ;;
        --eval_every=*|--eval-every=*)
            EVAL_EVERY="${1#*=}"
            shift
            ;;
        --val_fraction|--val-fraction)
            [[ $# -ge 2 ]] || { echo "[error] Missing value for $1" >&2; exit 2; }
            VAL_FRACTION="$2"
            shift 2
            ;;
        --val_fraction=*|--val-fraction=*)
            VAL_FRACTION="${1#*=}"
            shift
            ;;
        --val_every|--val-every)
            [[ $# -ge 2 ]] || { echo "[error] Missing value for $1" >&2; exit 2; }
            VAL_EVERY="$2"
            shift 2
            ;;
        --val_every=*|--val-every=*)
            VAL_EVERY="${1#*=}"
            shift
            ;;
        --save_dir|--save-dir)
            [[ $# -ge 2 ]] || { echo "[error] Missing value for $1" >&2; exit 2; }
            SAVE_DIR="$2"
            shift 2
            ;;
        --save_dir=*|--save-dir=*)
            SAVE_DIR="${1#*=}"
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
        --seed)
            [[ $# -ge 2 ]] || { echo "[error] Missing value for $1" >&2; exit 2; }
            SEED="$2"
            shift 2
            ;;
        --seed=*)
            SEED="${1#*=}"
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
        --resume)
            [[ $# -ge 2 ]] || { echo "[error] Missing value for $1" >&2; exit 2; }
            RESUME="$2"
            shift 2
            ;;
        --resume=*)
            RESUME="${1#*=}"
            shift
            ;;
        *)
            PASSTHROUGH_ARGS+=("$1")
            shift
            ;;
    esac
done

# ---- Launch vLLM servers ------------------------------------
source scripts/launch_servers.sh

# ---- Run training -------------------------------------------
# Build the exact Python args and print them for reproducibility.
TRIM_TRAIN_ARGS=(
    --mode train
    --draft_model_name "${DRAFT_MODEL}"
    --target_model_name "${TARGET_MODEL}"
    --prm_model_name "${PRM_MODEL}"
    --train_dataset_name "${TRAIN_DATASET_NAME}"
    --train_split "${TRAIN_SPLIT}"
    --eval_dataset_name "${EVAL_DATASET_NAME}"
    --eval_split "${EVAL_SPLIT}"
    --num_epochs "${NUM_EPOCHS}"
    --batch_size "${BATCH_SIZE}"
    --val_fraction "${VAL_FRACTION}"
    --val_every "${VAL_EVERY}"
    --eval_every "${EVAL_EVERY}"
    --save_dir "${SAVE_DIR}"
    --output_dir "${OUTPUT_DIR}"
    --seed "${SEED}"
    --target_disable_thinking "${TARGET_DISABLE_THINKING}"
    --cost_per_token "${COST_PER_TOKEN}"
    --resume "${RESUME}"
)
TRIM_TRAIN_ARGS+=("${PASSTHROUGH_ARGS[@]}")

echo "[trim-train] Effective models: TARGET_MODEL=${TARGET_MODEL}, DRAFT_MODEL=${DRAFT_MODEL}, PRM_MODEL=${PRM_MODEL}"
echo "[trim-train] Effective training settings: train_dataset=${TRAIN_DATASET_NAME}/${TRAIN_SPLIT}, eval_dataset=${EVAL_DATASET_NAME}/${EVAL_SPLIT}, num_epochs=${NUM_EPOCHS}, batch_size=${BATCH_SIZE}, val_fraction=${VAL_FRACTION}, val_every=${VAL_EVERY}, eval_every=${EVAL_EVERY}, save_dir=${SAVE_DIR}, output_dir=${OUTPUT_DIR}, seed=${SEED}, target_disable_thinking=${TARGET_DISABLE_THINKING}, cost_per_token=${COST_PER_TOKEN}, resume=${RESUME}"
printf '[trim-train] Python args:'
printf ' %q' "${TRIM_TRAIN_ARGS[@]}"
printf '\n'

python TRIM_Agg.py "${TRIM_TRAIN_ARGS[@]}"

echo "=== Training complete ==="
