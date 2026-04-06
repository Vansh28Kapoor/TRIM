#!/usr/bin/env bash
# ============================================================
# setup_tpu_pomdp_env.sh — Minimal environment for get_pomdp_policy.py
#
# Intended for TPU machines: no CUDA / PyTorch / vLLM installed.
#
# Julia and its packages are installed under NFS_HOME (default:
# /home/vansh/nfs_vansh) so they persist across TPU reimaging.
#
# Usage (run once):
#   bash scripts/setup_tpu_pomdp_env.sh
#
# Creates a 'trim-pomdp-tpu' conda env that runs:
#   python get_pomdp_policy.py --workers 128 --closeness-thr 0.4
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
ENV_NAME="trim-pomdp-tpu"
JULIA_VERSION="1.10.7"

# NFS home — Julia lives here so it persists across TPU reimaging
NFS_HOME="${NFS_HOME:-/home/vansh/nfs_vansh}"
JULIA_INSTALL_DIR="${NFS_HOME}/julia-${JULIA_VERSION}"
export JULIA_DEPOT_PATH="${NFS_HOME}/.julia"

echo "=== TPU POMDP environment setup ==="
echo "Repository  : ${REPO_DIR}"
echo "Env name    : ${ENV_NAME}"
echo "Julia       : ${JULIA_VERSION}"
echo "Julia dir   : ${JULIA_INSTALL_DIR}"
echo "Julia depot : ${JULIA_DEPOT_PATH}"
echo ""

# ===========================================================================
# PART 1 — Conda + Python
# ===========================================================================

CONDA_BASE="$(conda info --base 2>/dev/null || echo "${HOME}/miniconda3")"
source "${CONDA_BASE}/etc/profile.d/conda.sh"

# ---- 1. Create conda env with Python 3.11 -----------------------------------
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "[1/6] Conda env '${ENV_NAME}' already exists."
else
    echo "[1/6] Creating conda env '${ENV_NAME}' (Python 3.11) …"
    conda create -n "${ENV_NAME}" python=3.11 pip -y
fi

conda activate "${ENV_NAME}"

# ---- 2. Install Python packages via pip --------------------------------------
echo "[2/6] Installing Python packages …"
pip install -r "${REPO_DIR}/requirements_tpu_pomdp.txt"

# ===========================================================================
# PART 2 — Julia + POMDP packages
# ===========================================================================

# ---- 3. Install Julia tarball directly into NFS (no juliaup) ----------------
echo "[3/6] Installing Julia ${JULIA_VERSION} into ${JULIA_INSTALL_DIR} …"

if [ -x "${JULIA_INSTALL_DIR}/bin/julia" ]; then
    echo "  Julia already installed."
else
    ARCH="$(uname -m)"   # x86_64 or aarch64
    case "${ARCH}" in
        x86_64)  JULIA_ARCH="x86_64";  JULIA_URL_ARCH="x64" ;;
        aarch64) JULIA_ARCH="aarch64"; JULIA_URL_ARCH="aarch64" ;;
        *) echo "Unsupported arch: ${ARCH}"; exit 1 ;;
    esac
    JULIA_MINOR="${JULIA_VERSION%.*}"   # e.g. 1.10
    TARBALL="julia-${JULIA_VERSION}-linux-${JULIA_ARCH}.tar.gz"
    URL="https://julialang-s3.julialang.org/bin/linux/${JULIA_URL_ARCH}/${JULIA_MINOR}/${TARBALL}"
    echo "  Downloading ${URL} …"
    curl -fL "${URL}" -o "/tmp/${TARBALL}"
    mkdir -p "${JULIA_INSTALL_DIR}"
    tar -xzf "/tmp/${TARBALL}" -C "${JULIA_INSTALL_DIR}" --strip-components=1
    rm "/tmp/${TARBALL}"
fi

export PATH="${JULIA_INSTALL_DIR}/bin:${PATH}"
echo "  $(julia --version)"

# ---- 4. Install Julia POMDP packages ----------------------------------------
echo "[4/6] Installing Julia POMDP packages …"
julia --project="${REPO_DIR}/julia" -e '
    using Pkg
    if !any(r -> r.name == "General", values(Pkg.Registry.reachable_registries()))
        Pkg.Registry.add("General")
    end
    try
        Pkg.Registry.add(Pkg.RegistrySpec(url="https://github.com/JuliaPOMDP/Registry"))
    catch e
        @info "JuliaPOMDP registry may already be added" exception=e
    end
    Pkg.instantiate()
    Pkg.precompile()
    println("  Julia packages ready")
'

# ---- 5. Configure PyJulia bridge --------------------------------------------
echo "[5/7] Configuring PyJulia bridge …"
export JULIA_PROJECT="${REPO_DIR}/julia"
# Rebuild PyCall so it is linked to the current conda Python
PYTHON_BIN="$(which python)"
julia --project="${REPO_DIR}/julia" -e "
    ENV[\"PYTHON\"] = \"${PYTHON_BIN}\"
    import Pkg
    Pkg.build(\"PyCall\")
    println(\"  PyCall rebuilt for \", ENV[\"PYTHON\"])
"
# Install the PyJulia bridge (compiled_modules=False for statically-linked Python)
python -c "
from julia.api import Julia
Julia(compiled_modules=False)
print('  PyJulia bridge configured (compiled_modules=False)')
"

# ---- 6. Build Julia sysimage -----------------------------------------------
# compiled_modules=False forces JIT-compilation per worker (~60s each).
# A sysimage bakes all POMDP packages into a single .so, cutting startup to ~2s.
SYSIMAGE="${NFS_HOME}/pomdp_sysimage.so"
echo "[6/7] Building Julia sysimage at ${SYSIMAGE} …"
if [ -f "${SYSIMAGE}" ]; then
    echo "  Sysimage already exists. Delete it to rebuild."
else
    julia --project="${REPO_DIR}/julia" -e '
        import Pkg
        Pkg.add("PackageCompiler")
        using PackageCompiler
        create_sysimage(
            [:POMDPs, :POMDPTools, :NativeSARSOP, :QuickPOMDPs, :PyCall];
            sysimage_path = ARGS[1],
            project = ARGS[2],
        )
        println("  Sysimage built successfully")
    ' "${SYSIMAGE}" "${REPO_DIR}/julia"
fi

# ---- 7. Write conda activation hooks ----------------------------------------
echo "[7/7] Writing conda activation hooks …"
ACTIVATE_DIR="${CONDA_PREFIX}/etc/conda/activate.d"
DEACTIVATE_DIR="${CONDA_PREFIX}/etc/conda/deactivate.d"
mkdir -p "${ACTIVATE_DIR}" "${DEACTIVATE_DIR}"

cat > "${ACTIVATE_DIR}/trim_pomdp_tpu.sh" << ACTEOF
#!/usr/bin/env bash
export JULIA_DEPOT_PATH="${NFS_HOME}/.julia"
export JULIA_PROJECT="${REPO_DIR}/julia"
export JULIA_NUM_THREADS=\${JULIA_NUM_THREADS:-auto}
export JULIA_SYSIMAGE="${SYSIMAGE}"
export PYTHON="${CONDA_PREFIX}/bin/python"
export PATH="${JULIA_INSTALL_DIR}/bin:\${CONDA_PREFIX}/bin:\${PATH}"
ACTEOF

cat > "${DEACTIVATE_DIR}/trim_pomdp_tpu.sh" << DEACTEOF
#!/usr/bin/env bash
unset JULIA_DEPOT_PATH
unset JULIA_PROJECT
unset JULIA_NUM_THREADS
DEACTEOF

chmod +x "${ACTIVATE_DIR}/trim_pomdp_tpu.sh" "${DEACTIVATE_DIR}/trim_pomdp_tpu.sh"

# ---- Verify Python packages now that julia is in PATH -----------------------
echo ""
echo "Verifying Python packages …"
python -c "import numpy; print(f'  NumPy          {numpy.__version__}')"
python -c "import julia; print(f'  PyJulia        {julia.__version__}')"
python -c "import scipy; print(f'  SciPy          {scipy.__version__}')"
python -c "from scipy.stats import gaussian_kde; print('  gaussian_kde   OK')"
python -c "import datasets; print(f'  datasets       {datasets.__version__}')"
# quickpomdps hits Julia at import time; must init with compiled_modules=False first
python -c "
from julia.api import Julia
Julia(compiled_modules=False)
import quickpomdps
print('  quickpomdps    OK')
"

echo ""
echo "=== Setup complete ==="
echo ""
echo "Activate:  conda activate ${ENV_NAME}"
echo ""
echo "Run POMDP precomputation:"
echo "  python get_pomdp_policy.py --workers 128 --closeness-thr 0.4"
echo ""
