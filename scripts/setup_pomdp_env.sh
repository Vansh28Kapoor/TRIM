#!/usr/bin/env bash
# ============================================================
# setup_pomdp_env.sh — Reproducible environment for TRIM + POMDP
#
# Usage (run once from a Delta login node or interactive session):
#   bash scripts/setup_pomdp_env.sh
#
# Creates a 'trim-pomdp' conda env that can run BOTH:
#   - TRIM_Agg.py  (PPO training / inference, torch+vLLM)
#   - get_pomdp_policy.py  (POMDP precomputation, Julia+SARSOP)
#
# Reproducibility guarantees
# --------------------------
# Python : requirements_pomdp.txt pins major.minor ranges.
#          Exact resolved versions exported to:
#            environment_pomdp.lock.yml, requirements.lock.txt
# Julia  : julia/Project.toml + julia/Manifest.toml lock the
#          full Julia dependency graph.  Pkg.instantiate()
#          restores exact versions.
# juliaup: pins a specific Julia release (JULIA_VERSION below).
#
# To reproduce on a fresh machine
# --------------------------------
#   git clone <repo> && cd TRIM_code
#   bash scripts/setup_pomdp_env.sh
#   conda activate trim-pomdp
#   python pomdp_action_eval.py --workers 128
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
ENV_NAME="trim-pomdp"
JULIA_VERSION="1.10.7"

echo "=== TRIM + POMDP environment setup ==="
echo "Repository : ${REPO_DIR}"
echo "Env name   : ${ENV_NAME}"
echo "Julia      : ${JULIA_VERSION}"
echo ""

# ===========================================================================
# PART 1 — Conda + Python
# ===========================================================================

CONDA_BASE="$(conda info --base 2>/dev/null || echo "${HOME}/miniconda3")"
source "${CONDA_BASE}/etc/profile.d/conda.sh"

# ---- 1. Create conda env with Python 3.11 -----------------------------------
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "[1/9] Conda env '${ENV_NAME}' already exists."
else
    echo "[1/9] Creating conda env '${ENV_NAME}' (Python 3.11) …"
    conda create -n "${ENV_NAME}" python=3.11 pip -y
fi

conda activate "${ENV_NAME}"

# ---- 2. Install Python packages via pip --------------------------------------
echo "[2/9] Installing Python packages …"
pip install -r "${REPO_DIR}/requirements_pomdp.txt"

# ---- 3. Verify core Python packages -----------------------------------------
echo "[3/9] Verifying Python installation …"
python -c "import numpy;  print(f'  NumPy          {numpy.__version__}')"
python -c "import julia;  print(f'  PyJulia        {julia.__version__}')"
python -c "import scipy;  print(f'  SciPy          {scipy.__version__}')"
python -c "import sklearn; print(f'  scikit-learn   {sklearn.__version__}')"
# torch/vllm verification (may fail if no GPU on login node — that's OK)
python -c "import torch; print(f'  PyTorch        {torch.__version__}')" 2>/dev/null \
    || echo "  PyTorch        (skipped — no GPU)"
python -c "import vllm;  print(f'  vLLM           {vllm.__version__}')" 2>/dev/null \
    || echo "  WARNING: vLLM not importable — re-run setup_pomdp_env.sh from a GPU node before submitting jobs"

# ===========================================================================
# PART 2 — Julia + POMDP packages
# ===========================================================================

# ---- 4. Install Julia via juliaup -------------------------------------------
echo "[4/9] Installing Julia ${JULIA_VERSION} …"
if command -v juliaup &>/dev/null; then
    echo "  juliaup already installed."
else
    curl -fsSL https://install.julialang.org | sh -s -- --yes
fi
export PATH="${HOME}/.juliaup/bin:${PATH}"

juliaup add "${JULIA_VERSION}"
juliaup default "${JULIA_VERSION}"
echo "  $(julia --version)"

# ---- 5. Install Julia POMDP packages ----------------------------------------
echo "[5/9] Installing Julia POMDP packages …"
julia --project="${REPO_DIR}/julia" -e '
    using Pkg
    # Ensure both General and JuliaPOMDP registries are available
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
echo "  Manifest.toml written — commit for exact reproducibility."

# ---- 6. Configure PyJulia bridge --------------------------------------------
echo "[6/9] Configuring PyJulia bridge …"
export JULIA_PROJECT="${REPO_DIR}/julia"
PYTHON_BIN="$(which python)"
julia --project="${REPO_DIR}/julia" -e "
    ENV[\"PYTHON\"] = \"${PYTHON_BIN}\"
    import Pkg
    Pkg.build(\"PyCall\")
    println(\"  PyCall rebuilt for \", ENV[\"PYTHON\"])
"
python -c "
from julia.api import Julia
Julia(compiled_modules=False)
print('  PyJulia bridge configured (compiled_modules=False)')
"

# ---- 7. Build Julia sysimage ------------------------------------------------
# Without a sysimage, each worker pays ~60s JIT on first Julia call.
# IMPORTANT: Do NOT include PyCall — conda Python is statically linked to
# libpython, causing 'free(): invalid pointer' crashes.
SYSIMAGE="${HOME}/pomdp_sysimage.so"
echo "[7/9] Building Julia sysimage at ${SYSIMAGE} …"
if [ -f "${SYSIMAGE}" ]; then
    echo "  Sysimage already exists. Delete it to rebuild."
else
    julia --project="${REPO_DIR}/julia" -e '
        import Pkg
        Pkg.add("PackageCompiler")
        using PackageCompiler
        create_sysimage(
            [:POMDPs, :POMDPTools, :NativeSARSOP, :QuickPOMDPs];
            sysimage_path = ARGS[1],
            project = ARGS[2],
        )
        println("  Sysimage built successfully")
    ' "${SYSIMAGE}" "${REPO_DIR}/julia"
fi

# ---- 8. Set conda activate/deactivate hooks ---------------------------------
echo "[8/9] Writing conda activation hooks …"
ACTIVATE_DIR="${CONDA_PREFIX}/etc/conda/activate.d"
DEACTIVATE_DIR="${CONDA_PREFIX}/etc/conda/deactivate.d"
mkdir -p "${ACTIVATE_DIR}" "${DEACTIVATE_DIR}"

cat > "${ACTIVATE_DIR}/trim_pomdp.sh" << ACTEOF
#!/usr/bin/env bash
export JULIA_PROJECT="${REPO_DIR}/julia"
export JULIA_NUM_THREADS=1
export JULIA_SYSIMAGE="${SYSIMAGE}"
export PATH="\${HOME}/.juliaup/bin:\${CONDA_PREFIX}/bin:\${PATH}"
ACTEOF

cat > "${DEACTIVATE_DIR}/trim_pomdp.sh" << DEACTEOF
#!/usr/bin/env bash
unset JULIA_PROJECT
unset JULIA_NUM_THREADS
unset JULIA_SYSIMAGE
DEACTEOF

chmod +x "${ACTIVATE_DIR}/trim_pomdp.sh" "${DEACTIVATE_DIR}/trim_pomdp.sh"

# ---- 9. Export locked environment snapshots ----------------------------------
echo "[9/9] Exporting lock files …"
conda env export --no-builds > "${REPO_DIR}/environment_pomdp.lock.yml"
pip freeze > "${REPO_DIR}/requirements.lock.txt"
echo "  environment_pomdp.lock.yml  (conda)"
echo "  requirements.lock.txt       (pip)"

echo ""
echo "=== Setup complete ==="
echo ""
echo "Activate:  conda activate ${ENV_NAME}"
echo ""
echo "Run POMDP precomputation:"
echo "  python get_pomdp_policy.py --workers <N> --cost-per-token <COST>"
echo ""
echo "Run inference (same as 'trim' env):"
echo "  python TRIM_Agg.py --mode eval ..."
