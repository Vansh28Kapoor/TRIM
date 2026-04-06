#!/usr/bin/env bash
# ============================================================
# launch_servers.sh — Start vLLM servers for TRIM experiments
#
# GPU layout (2× A40-48 GB):
#
#   GPU 0: target 8B  (exclusive, 90%)   — ~16 GiB weights (bf16)
#   GPU 1: draft 1.5B (35%)              — ~3  GiB weights
#          PRM   7B   (50%)              — ~14 GiB weights
#
# Draft launches first on GPU 1 (tiny footprint), then PRM.
# 35% + 50% = 85% of 46 GB ≈ 39 GB, within budget.
# Target gets GPU 0 exclusively.
#
# Supported targets (override via TARGET_MODEL env var):
#   Qwen/Qwen2.5-7B-Instruct   (default)
#   Qwen/Qwen3-8B              (use with target_disable_thinking=true)
#
# Usage:
#   source scripts/launch_servers.sh   # starts servers in bg
#   kill_servers                        # tears them down
# ============================================================
set -euo pipefail

# ---- Model names (override via env vars) --------------------
TARGET_MODEL="${TARGET_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
DRAFT_MODEL="${DRAFT_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
PRM_MODEL="${PRM_MODEL:-Qwen/Qwen2.5-Math-PRM-7B}"

# ---- Ports ---------------------------------------------------
TARGET_PORT="${TARGET_PORT:-30000}"
DRAFT_PORT="${DRAFT_PORT:-30001}"
PRM_PORT="${PRM_PORT:-30002}"

# ---- Sequence length -----------------------------------------
# PRM has max_position_embeddings=4096 (binding constraint).
# Qwen3-8B has 40960, Qwen2.5-1.5B-Instruct has 32768.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"

# ---- GPU memory utilization ----------------------------------
TARGET_MEM_UTIL="${TARGET_MEM_UTIL:-0.90}"   # GPU 0 (exclusive)
DRAFT_MEM_UTIL="${DRAFT_MEM_UTIL:-0.35}"     # GPU 1 (shared with PRM; 1.5B only needs ~3 GiB)
PRM_MEM_UTIL="${PRM_MEM_UTIL:-0.50}"         # GPU 1 (shared with draft; 7B needs ~14 GiB)

# ---- Helper: wait for a vLLM server to become ready ---------
wait_for_server() {
    local port=$1
    local name=$2
    local max_wait=${3:-900}
    local elapsed=0
    echo "[launch] Waiting for ${name} on port ${port} ..."
    while true; do
        if curl -sf "http://localhost:${port}/health" > /dev/null 2>&1; then
            echo "[launch] ${name} is ready (port ${port}, ${elapsed}s)."
            return 0
        fi
        if (( elapsed >= max_wait )); then
            echo "[launch] ERROR: ${name} did not start within ${max_wait}s." >&2
            return 1
        fi
        sleep 10
        elapsed=$((elapsed + 10))
    done
}

# ---- Suppress verbose per-request logging from vLLM servers --
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARNING}"
VLLM_LOG_FLAGS=(--disable-log-requests --disable-log-stats)

# Redirect all vLLM server output to separate log files so the
# SLURM .out/.err only contain the Python training script output.
VLLM_LOG_DIR="${VLLM_LOG_DIR:-logs/vllm_${SLURM_JOB_ID:-local}}"
mkdir -p "${VLLM_LOG_DIR}"

# ---- Launch target model (GPU 0, exclusive) -----------------
# For Qwen3 targets: thinking mode is disabled client-side via the chat template
# (enable_thinking=False in apply_chat_template), not at the server level.
echo "[launch] Starting target model: ${TARGET_MODEL}"
CUDA_VISIBLE_DEVICES=0 vllm serve "${TARGET_MODEL}" \
    --dtype auto \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${TARGET_MEM_UTIL}" \
    --enable-prefix-caching \
    "${VLLM_LOG_FLAGS[@]}" \
    --port "${TARGET_PORT}" \
    > "${VLLM_LOG_DIR}/vllm_target.log" 2>&1 &
VLLM_TARGET_PID=$!
wait_for_server "${TARGET_PORT}" "target (${TARGET_MODEL})"

# ---- Launch draft model (GPU 1, shared with PRM) ------------
echo "[launch] Starting draft model: ${DRAFT_MODEL}"
CUDA_VISIBLE_DEVICES=1 vllm serve "${DRAFT_MODEL}" \
    --dtype auto \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${DRAFT_MEM_UTIL}" \
    --enable-prefix-caching \
    "${VLLM_LOG_FLAGS[@]}" \
    --port "${DRAFT_PORT}" \
    > "${VLLM_LOG_DIR}/vllm_draft.log" 2>&1 &
VLLM_DRAFT_PID=$!
wait_for_server "${DRAFT_PORT}" "draft (${DRAFT_MODEL})"

# ---- Launch PRM (GPU 1, shared with draft) ------------------
# Qwen2ForProcessRewardModel uses STEP pooling (one score per <extra_0>
# token) by default.  Do NOT pass --task classify — that overrides the
# model's built-in pooler to LAST, returning only one score.
# The model is V0-only in vLLM 0.8.x, so /pooling lives at the root URL.
echo "[launch] Starting PRM: ${PRM_MODEL}"
CUDA_VISIBLE_DEVICES=1 vllm serve "${PRM_MODEL}" \
    --dtype auto \
    --trust-remote-code \
    --gpu-memory-utilization "${PRM_MEM_UTIL}" \
    --enable-prefix-caching \
    "${VLLM_LOG_FLAGS[@]}" \
    --port "${PRM_PORT}" \
    > "${VLLM_LOG_DIR}/vllm_prm.log" 2>&1 &
VLLM_PRM_PID=$!
wait_for_server "${PRM_PORT}" "PRM (${PRM_MODEL})"

nvidia-smi
echo "[launch] All servers ready."
echo "[launch]   target  PID=${VLLM_TARGET_PID}  port=${TARGET_PORT}  (GPU 0, ${TARGET_MEM_UTIL})"
echo "[launch]   draft   PID=${VLLM_DRAFT_PID}   port=${DRAFT_PORT}  (GPU 1, ${DRAFT_MEM_UTIL})"
echo "[launch]   PRM     PID=${VLLM_PRM_PID}     port=${PRM_PORT}  (GPU 1, ${PRM_MEM_UTIL})"

# ---- Convenience cleanup function ---------------------------
kill_servers() {
    echo "[launch] Shutting down vLLM servers ..."
    for pid in ${VLLM_TARGET_PID:-} ${VLLM_DRAFT_PID:-} ${VLLM_PRM_PID:-}; do
        [ -n "$pid" ] && kill "$pid" 2>/dev/null && wait "$pid" 2>/dev/null || true
    done
    echo "[launch] All servers stopped."
}
trap kill_servers EXIT
