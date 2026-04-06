# TRIM: Targeted Stepwise Routing for Multi-Step Reasoning

<p align="center">
  <a href="https://arxiv.org/abs/2601.10245"><img src="https://img.shields.io/badge/arXiv-2601.10245-b31b1b.svg" alt="arXiv"></a>
  <a href="https://vansh28kapoor.github.io/trim.github.io/"><img src="https://img.shields.io/badge/Project-Website-blue" alt="Website"></a>
</p>

Official implementation of **[TRIM: Hybrid Inference via Targeted Stepwise Routing in Multi-Step Reasoning Tasks](https://arxiv.org/abs/2601.10245)** (ICLR 2026).

**Authors:** Vansh Kapoor, Aman Gupta, Hao Chen, Anurag Beniwal, Jing Huang, Aviral Kumar

> **TL;DR:** Instead of routing entire queries to a single model, TRIM routes individual *reasoning steps* to the appropriate model — a cheap draft model handles routine steps while an expensive target model is called only for critical, error-prone steps. This achieves up to **5–6x cost efficiency** over query-level routing on MATH-500 and AIME benchmarks.

---

## Overview

Multi-step mathematical reasoning is vulnerable to *cascading failures*: a single erroneous step can propagate through an entire solution. TRIM addresses this by performing **step-level routing** using a Process Reward Model (PRM) to identify uncertain or erroneous reasoning steps and selectively escalate only those to a stronger model.

TRIM implements four routing strategies with increasing sophistication:

| Strategy | Description | Key Property |
|----------|-------------|--------------|
| **TRIM-Thr** | Fixed threshold on PRM score | Simple baseline; no training required |
| **TRIM-Agg** | PPO-trained policy using aggregate PRM features | Learned cost–accuracy trade-off |
| **TRIM-POMDP** | Bayesian belief over correctness states + precomputed POMDP policy | Uncertainty-aware; accounts for noisy PRM signals |

### Default Model Configuration

| Role | Model | Description |
|------|-------|-------------|
| Draft (M_w) | `Qwen/Qwen2.5-1.5B-Instruct` | Cheap model for routine steps |
| Target (M_s) | `Qwen/Qwen2.5-7B-Instruct` | Expensive model for critical steps |
| PRM | `Qwen/Qwen2.5-Math-PRM-7B` | Process Reward Model for step scoring |

> **Using Qwen3-8B as target:** Pass `--target_model_name Qwen/Qwen3-8B --target_disable_thinking true` to suppress its internal thinking block (since the draft model is non-thinking).

---

## Installation

### Prerequisites

- **Hardware:** 2x NVIDIA GPUs with >= 40 GB VRAM each (tested on A40-48GB, A100-80GB)
- **Software:** Linux, CUDA 12.x, conda (Miniconda or Anaconda)

### Quick Setup (TRIM-Thr and TRIM-Agg)

```bash
git clone https://github.com/Vansh28Kapoor/TRIM_code.git
cd TRIM_code

# Create conda environment and install dependencies
bash scripts/setup_env.sh

conda activate trim
```

<details>
<summary><b>Manual installation</b></summary>

```bash
conda create -n trim python=3.11 pip -y
conda activate trim
pip install -r requirements.txt
```

</details>

### Full Setup (includes TRIM-POMDP)

TRIM-POMDP requires Julia for the SARSOP POMDP solver:

```bash
bash scripts/setup_pomdp_env.sh

conda activate trim-pomdp
```

This script:
1. Creates a `trim-pomdp` conda environment with Python 3.11
2. Installs Python dependencies (`requirements_pomdp.txt`)
3. Installs Julia 1.10.7 via `juliaup`
4. Instantiates Julia POMDP packages from `julia/Project.toml`
5. Configures the PyJulia bridge
6. Builds a Julia sysimage for fast worker startup
7. Sets conda activation hooks for Julia environment variables

### Download Model Weights

```bash
# Download all required models (runs during setup or manually)
python -c "
from huggingface_hub import snapshot_download
for model in [
    'Qwen/Qwen2.5-7B-Instruct',
    'Qwen/Qwen2.5-1.5B-Instruct',
    'Qwen/Qwen2.5-Math-PRM-7B',
]:
    print(f'Downloading {model} ...')
    snapshot_download(model)
"
```

---

## Quick Start

### 1. Launch vLLM Servers

All TRIM methods require three vLLM servers (draft, target, PRM). The launcher script handles GPU placement automatically:

```bash
# GPU 0: target model (exclusive, 90% VRAM)
# GPU 1: draft model (35% VRAM) + PRM (50% VRAM)
source scripts/launch_servers.sh
```

The servers expose OpenAI-compatible endpoints:
- Target: `http://localhost:30000/v1`
- Draft: `http://localhost:30001/v1`
- PRM: `http://localhost:30002` (Pooling API)

### 2. Run TRIM-Thr Evaluation (No Training Required)

```bash
python TRIM_Thr.py \
    --eval_dataset_name math500 \
    --eval_split test \
    --thresholds 0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1
```

Special threshold values: `0` = draft-only baseline, `1` = target-only baseline.

### 3. Train TRIM-Agg Policy

```bash
python TRIM_Agg.py \
    --mode train \
    --train_dataset_name math \
    --train_split train \
    --eval_dataset_name math500 \
    --eval_split test \
    --num_epochs 10 \
    --batch_size 64 \
    --cost_per_token 8e-4
```

### 4. Evaluate TRIM-Agg

```bash
python TRIM_Agg.py \
    --mode eval \
    --eval_dataset_name math500 \
    --eval_split test \
    --checkpoint ./rlpolicy_checkpoints/8e-4/policy_final.pt
```

---

## SLURM Submission

All scripts support SLURM submission with configurable parameters:

```bash
# TRIM-Agg training
sbatch scripts/run_TRIM_Agg_train.sh

# TRIM-Agg evaluation (loops over multiple datasets)
sbatch scripts/run_TRIM_Agg_eval.sh --datasets math500,aime,gsm8k

# TRIM-Thr evaluation
sbatch scripts/run_TRIM_Thr_eval.sh --datasets math500,aime,gsm8k

# TRIM-POMDP evaluation
sbatch scripts/run_TRIM_POMDP_eval.sh --datasets math500,aime
```

Override any parameter via flags or environment variables:

```bash
# Via flags
sbatch scripts/run_TRIM_Agg_train.sh \
    --target-model-name Qwen/Qwen3-8B \
    --target-disable-thinking true \
    --num-epochs 5 --batch-size 32

# Via environment variables
TARGET_MODEL=Qwen/Qwen3-8B TARGET_DISABLE_THINKING=true \
    sbatch scripts/run_TRIM_Agg_train.sh
```

> Edit `--account` in the SLURM scripts to match your cluster allocation before submitting.

---

## TRIM-POMDP Pipeline

TRIM-POMDP requires precomputed POMDP parameters and action tables. The pipeline has three offline stages (Steps 1–3, run once) followed by evaluation (Step 4).

### Step 1: Build Observation Model

Fits a reflected-KDE observation model from [ProcessBench](https://huggingface.co/datasets/Qwen/ProcessBench) data. The model maps PRM score observations to likelihoods under each POMDP state (S0 = correct, S1 = irrecoverably incorrect, S2 = recoverable error at current step).

```bash
sbatch scripts/pomdp/run_observation_params.sh
# or directly (requires PRM vLLM server):
python pomdp_params/get_observation_function.py \
    --benchmark math-train --bin-size 0.05
```

**Outputs** (saved to `pomdp_data/`):
- `math-train_reflected_kde_obs_model.pkl` — the KDE observation model used at inference by `TRIM_POMDP.py`
- `math-train_obs_distributions_bin_size_0.05.pkl` — discretized observation distributions

### Step 2: Build Transition Parameters

Generates full solutions with draft and target models, scores each with the PRM, locates first-error steps via thresholding, and fits KDE-based terminal-step predictors (hazard probability of termination at each step).

```bash
sbatch scripts/pomdp/run_transition_params.sh
# or directly (requires all three vLLM servers):
python pomdp_params/get_transition_function.py \
    --train-benchmark math --train-split train --thr 0.35
```

**Outputs** (saved to `pomdp_data/`):
- `math_train_terminal_predictor_slm.pkl` — draft model terminal-step predictor
- `math_train_terminal_predictor_llm.pkl` — target model terminal-step predictor
- `math_train_terminal_predictor_slm_llm.pkl` — combined terminal-step predictor
- `transition_params/math_train_<model>_transition_params.pkl` — per-model correctness probabilities (p_slm, p_llm) used by `get_pomdp_policy.py`

> Steps 1 and 2 can be run together: `sbatch scripts/pomdp/run_pomdp_params.sh`

### Step 3: Precompute POMDP Action Table

For every reachable `(token_count, step_no, belief)` state on a discretized belief simplex, solves an independent SARSOP POMDP and records the optimal action (0 = keep draft, 1 = use target). The full lookup table is saved to disk so inference-time queries are pure NumPy lookups.

**This step is CPU-only (no GPU needed)** and is compute-intensive — it parallelizes across a multiprocessing pool of Julia SARSOP solvers via PyJulia. We ran this on Google Cloud TPU v4 host machines (240 CPU cores, 400 GB RAM) with 120 workers.

```bash
# On a SLURM CPU node:
sbatch scripts/pomdp/run_pomdp_policy.sh --cost-per-token 0.25 --workers 64

# Or directly:
python get_pomdp_policy.py \
    --workers 120 \
    --cost-per-token 0.25 \
    --task-reward 100 \
    --closeness-thr 0.5 \
    --belief-step 0.025
```

**Output:** `pomdp_data/pomdp_action_table_cost<ratio>_cthr<thr>.pkl` (one per cost value, ~19 MB each). Checkpoints auto-save every 5,000 tasks; re-running the same command resumes from the last checkpoint.

<details>
<summary><b>Environment setup for POMDP action table precomputation</b></summary>

This step requires a Julia + Python environment with the SARSOP solver. Two setup paths are provided:

**On SLURM clusters (GPU or CPU nodes):**
```bash
bash scripts/setup_pomdp_env.sh
conda activate trim-pomdp
```

**On TPU host machines (or large CPU-only machines):**
```bash
bash scripts/setup_tpu_pomdp_env.sh
conda activate trim-pomdp-tpu
```

The TPU setup installs Julia from a tarball (no `juliaup`) to a persistent NFS path, builds a Julia sysimage excluding PyCall (required for statically-linked conda Python), and sets `JULIA_NUM_THREADS=1` via conda activation hooks to prevent thread explosion on high-core-count machines.

**Key environment notes:**
- **Max workers:** Scale to ~50% of available cores. On a 240-core machine, 120 workers is safe (~4 GB RAM per worker); 128+ can cause OOM.
- **`JULIA_NUM_THREADS` must be `1`:** With `auto`, each worker spawns threads equal to total cores (e.g., 240 cores x 120 workers = 28,800 threads → SIGABRT).
- **Julia sysimage must exclude PyCall** when conda Python is statically linked to libpython (causes `free(): invalid pointer`). The sysimage is built with only `[:POMDPs, :POMDPTools, :NativeSARSOP, :QuickPOMDPs]`.
- **Orphan cleanup:** If workers crash (OOM/SIGABRT), orphan Julia processes may hold file locks. Run `pkill -9 julia; pkill -9 -f "multiprocessing.spawn"` before restarting.

</details>

### Step 4: Evaluate TRIM-POMDP

```bash
sbatch scripts/run_TRIM_POMDP_eval.sh --datasets math500 \
    --cost-per-tokens 0.15,0.25,0.35,0.45,0.55,0.65,0.75
```

---

## Project Structure

```
TRIM_code/
├── TRIM_Agg.py                    # TRIM-Agg: PPO-based learned routing
├── TRIM_Thr.py                    # TRIM-Thr: threshold-based routing
├── TRIM_POMDP.py                  # TRIM-POMDP: POMDP-based routing
├── utils.py                       # Shared utilities (generation, PRM client, prompts)
├── get_pomdp_policy.py            # POMDP action table precomputation (Julia/SARSOP)
│
├── math_eval/                     # Math evaluation toolkit
│   ├── parser.py                  #   Answer extraction and normalization
│   ├── math_equal.py              #   Symbolic math equivalence checking
│   └── data/                      #   Benchmark datasets (JSONL format)
│       ├── math500/               #     MATH-500
│       ├── aime/                  #     AIME
│       ├── gsm8k/                 #     GSM8K
│       ├── olympiadbench/         #     OlympiadBench
│       └── ...                    #     (+ math, aime24, amc23, cmimc, minerva_math)
│
├── pomdp_params/                  # POMDP parameter estimation scripts
│   ├── get_observation_function.py    #   KDE observation model from ProcessBench
│   └── get_transition_function.py     #   Transition params from model solutions
│
├── pomdp_data/                    # Precomputed POMDP artifacts
│   ├── pomdp_action_table_*.pkl   #   Action lookup tables (one per cost value)
│   ├── *_reflected_kde_obs_model.pkl  #   KDE observation models
│   └── transition_params/         #   Model outputs + transition pickles
│
├── scripts/
│   ├── launch_servers.sh          # Start vLLM servers (draft + target + PRM)
│   ├── run_TRIM_Agg_train.sh     # SLURM: TRIM-Agg training
│   ├── run_TRIM_Agg_eval.sh      # SLURM: TRIM-Agg evaluation
│   ├── run_TRIM_Thr_eval.sh      # SLURM: TRIM-Thr evaluation
│   ├── run_TRIM_POMDP_eval.sh    # SLURM: TRIM-POMDP evaluation
│   ├── setup_env.sh              # Environment setup (base)
│   ├── setup_pomdp_env.sh        # Environment setup (with Julia/POMDP)
│   ├── prepare_aime_dataset.py   # Download and split AIME dataset
│   └── pomdp/                    # POMDP precomputation scripts
│       ├── run_observation_params.sh
│       ├── run_transition_params.sh
│       ├── run_pomdp_params.sh
│       └── run_pomdp_policy.sh
│
├── julia/                         # Julia POMDP package configuration
│   ├── Project.toml               #   Package dependencies
│   └── Manifest.toml              #   Locked dependency graph
│
├── requirements.txt               # Python dependencies (base)
├── requirements_pomdp.txt         # Python dependencies (+ Julia/POMDP)
├── environment.yml                # Conda environment (base)
└── environment_pomdp.yml          # Conda environment (+ Julia/POMDP)
```

---

## Benchmarks

Evaluation datasets are stored as JSONL files in `math_eval/data/`. The math evaluation toolkit (`math_eval/parser.py`, `math_eval/math_equal.py`) is adapted from [Qwen2.5-Math](https://github.com/QwenLM/Qwen2.5-Math).

| Dataset | Split | Description |
|---------|-------|-------------|
| `math500` | test | MATH-500 (primary evaluation) |
| `aime` | train, test | AIME competition problems (1983–2024) |
| `gsm8k` | test | Grade school math |
| `olympiadbench` | test | International Olympiad problems |
| `minerva_math` | test | Minerva Math |
| `math` | train | MATH training set (for TRIM-Agg training and POMDP param estimation) |

### Preparing the AIME Dataset

The AIME train/test split is generated from [di-zhang-fdu/AIME_1983_2024](https://huggingface.co/datasets/di-zhang-fdu/AIME_1983_2024) on Hugging Face:

```bash
python scripts/prepare_aime_dataset.py
```

**Split strategy:**
- **Pre-2000:** Alternating years — even offset from 1983 → test, odd offset → train
- **Post-2000:** Part I → train, Part II → test

This ensures no year-level leakage between splits. Output files are written to `math_eval/data/aime/{train,test}.jsonl`.

---

## GPU Memory Layout

The default configuration targets **2x A40-48GB** GPUs:

```
GPU 0:  Target model (Qwen2.5-7B-Instruct)    — 90% VRAM (~14 GiB weights)
GPU 1:  Draft model  (Qwen2.5-1.5B-Instruct)  — 35% VRAM (~3  GiB weights)
        PRM          (Qwen2.5-Math-PRM-7B)     — 50% VRAM (~14 GiB weights)
```

Override via environment variables:

```bash
export TARGET_MEM_UTIL=0.85   # target model GPU memory fraction
export DRAFT_MEM_UTIL=0.30    # draft model GPU memory fraction
export PRM_MEM_UTIL=0.45      # PRM GPU memory fraction
export MAX_MODEL_LEN=4096     # max sequence length (all models)
```

---

## Configuration Reference

### TRIM-Agg (`TRIM_Agg.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--mode` | — | `train` or `eval` |
| `--target_model_name` | `Qwen/Qwen2.5-7B-Instruct` | Target (expensive) model |
| `--draft_model_name` | `Qwen/Qwen2.5-1.5B-Instruct` | Draft (cheap) model |
| `--prm_model_name` | `Qwen/Qwen2.5-Math-PRM-7B` | Process Reward Model |
| `--target_disable_thinking` | `false` | Set `true` for Qwen3 targets |
| `--train_dataset_name` | `math` | Training dataset |
| `--eval_dataset_name` | `math500` | Evaluation dataset |
| `--cost_per_token` | `8e-4` | Token cost for reward shaping |
| `--num_epochs` | `10` | Training epochs |
| `--batch_size` | `64` | Batch size |
| `--checkpoint` | — | Path to policy checkpoint (eval mode) |

### TRIM-Thr (`TRIM_Thr.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--thresholds` | `0,0.1,0.3,0.5,0.7,0.9,1` | Comma-separated PRM thresholds |
| `--eval_dataset_name` | `math500` | Evaluation dataset |
| `--batch_size` | `32` | Batch size |

### TRIM-POMDP (`TRIM_POMDP.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--closeness_thresholds` | `0.4` | Belief closeness thresholds |
| `--cost_per_tokens` | `0.15,...,0.75` | Cost values (comma-separated) |
| `--task_reward` | `100` | Task reward for POMDP |
| `--action_table_dir` | `pomdp_data` | Directory with precomputed tables |

---

## Implementation

- **Python dependencies:** Exact versions in `requirements.lock.txt` and `environment_pomdp.lock.yml`
- **Julia dependencies:** Locked via `julia/Manifest.toml`
- **Random seeds:** All scripts default to `--seed 10`

To implement with exact package versions:

```bash
conda create -n trim python=3.11 pip -y
conda activate trim
pip install -r requirements.lock.txt
```

---

## Citation

```bibtex
@article{kapoor2026trim,
  title={TRIM: Hybrid Inference via Targeted Stepwise Routing in Multi-Step Reasoning Tasks},
  author={Kapoor, Vansh and Gupta, Aman and Chen, Hao and Beniwal, Anurag and Huang, Jing and Kumar, Aviral},
  journal={arXiv preprint arXiv:2601.10245},
  year={2026}
}
```

---

## Acknowledgements

- [Qwen2.5-Math](https://github.com/QwenLM/Qwen2.5-Math) by the Qwen team — math evaluation toolkit adapted from this repository

## License

This project is released under the [MIT License](LICENSE).
