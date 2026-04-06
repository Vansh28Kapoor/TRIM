#!/usr/bin/env bash
# ============================================================
# setup_env.sh — One-time environment setup for TRIM
#
# Run on a Delta login node (or in an interactive session):
#   bash scripts/setup_env.sh
#
# This script:
#   1. Creates the 'trim' conda environment
#   2. Installs all Python dependencies
#   3. Pre-downloads model weights to $HF_HOME
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== TRIM environment setup ==="
echo "Repository: ${REPO_DIR}"

# ---- Initialize conda ----------------------------------------
# conda is provided by the user's miniconda3 installation.
# Source the conda shell functions in case this script is run
# in a non-interactive shell where they are not yet available.
CONDA_BASE="$(conda info --base 2>/dev/null || echo "${HOME}/miniconda3")"
source "${CONDA_BASE}/etc/profile.d/conda.sh"

# ---- Create conda environment --------------------------------
if conda env list | grep -q "^trim "; then
    echo "[setup] Conda env 'trim' already exists. Updating ..."
    conda env update -f "${REPO_DIR}/environment.yml" --prune
else
    echo "[setup] Creating conda env 'trim' ..."
    conda env create -f "${REPO_DIR}/environment.yml"
fi

conda activate trim

# ---- Verify key packages ------------------------------------
echo "[setup] Verifying installation ..."
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')"
python -c "import vllm; print(f'vLLM {vllm.__version__}')"
python -c "import transformers; print(f'Transformers {transformers.__version__}')"

# ---- Pre-download models (optional) -------------------------

if [ -t 0 ]; then
    read -rp "Download model weights now? [y/N] " download
else
    download="n"
fi
if [[ "${download}" =~ ^[Yy]$ ]]; then
    echo "[setup] Downloading model weights ..."
    python -c "
from huggingface_hub import snapshot_download
models = [
    'Qwen/Qwen2.5-Math-7B-Instruct',
    'Qwen/Qwen2.5-Math-1.5B-Instruct',
    'Qwen/Qwen2.5-Math-PRM-7B',
]
for m in models:
    print(f'Downloading {m} ...')
    snapshot_download(m)
    print(f'  Done: {m}')
"
    echo "[setup] All models downloaded."
fi

# ---- Export locked environment for reproducibility -----------
conda env export --no-builds > "${REPO_DIR}/environment.lock.yml"
echo "[setup] Locked environment written to environment.lock.yml"

echo "=== Setup complete ==="
echo "Activate with: conda activate trim"
