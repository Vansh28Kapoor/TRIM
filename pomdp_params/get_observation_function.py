"""Build reflected-KDE observation models for the POMDP from ProcessBench data.

Pipeline
--------
1. Load Qwen/ProcessBench, segregate samples into POMDP states (S0, S1, S2).
2. Score every sample with a Process Reward Model (PRM) served via vLLM.
3. Fit a reflected-KDE observation model per state and precompute bin probs.

The PRM evaluation uses ``ServerPRM`` from ``utils.py`` (shared with ``TRIM_Agg.py``) — it queries
a vLLM Pooling API (``Qwen/Qwen2.5-Math-PRM-7B`` with STEP pooler) so the same
server can be shared with the training loop.


Intended benchmarks
-------------------
This script is primarily used with two benchmarks:

* ``omnimath``  — Uses the ``omnimath`` split of ProcessBench directly.

* ``math-train`` — Uses the ``math`` split of ProcessBench, but **excludes
  any questions that appear in MATH-500** (``math_eval/data/math500/test.jsonl``).
  This ensures no overlap between the KDE training data and the MATH-500
  evaluation set, so the POMDP parameters are built only from the training
  portion of the MATH dataset.

Usage
-----
First start the PRM vLLM server::

    CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen2.5-Math-PRM-7B \\
        --dtype auto --trust-remote-code \\
        --gpu-memory-utilization 0.90 \\
        --enable-prefix-caching --port 30002

Then run::

    python pomdp_params/get_observation_function.py \\
        --benchmark omnimath --bin-size 0.05

    python pomdp_params/get_observation_function.py \\
        --benchmark math-train --bin-size 0.05

Add ``--rerun`` to regenerate cached artefacts even if they already exist.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import logging
from math import ceil
from pathlib import Path
import sys
from typing import Any

import json
import random
import cloudpickle as cp
import numpy as np
from datasets import Dataset, load_dataset, load_from_disk
from scipy.integrate import dblquad
from scipy.stats import gaussian_kde
from transformers import AutoTokenizer

# Allow imports from the repo root when running as
# `python pomdp_params/get_observation_function.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import ServerPRM, seed_everything

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_DIR / "pomdp_data"

PROCESS_BENCH_PRM_SCORED_PATH = DATA_DIR / "ProcessBench_PRM_Scored"
PROCESS_BENCH_STATE_CLASSIFIED_PATH = DATA_DIR / "ProcessBenchData_state_classified"

# math500 exclusion set path (used when --benchmark math-train)
MATH500_PATH = REPO_DIR / "math_eval" / "data" / "math500" / "test.jsonl"


def load_math500_problems() -> set[str]:
    """Return the set of problem strings in the MATH-500 evaluation set.

    Used to exclude MATH-500 questions when building the KDE model for the
    ``math-train`` benchmark, so that there is no overlap between the POMDP
    observation-model training data and the evaluation set.
    """
    if not MATH500_PATH.exists():
        raise FileNotFoundError(
            f"MATH-500 data not found at {MATH500_PATH}. "
            "Expected math_eval/data/math500/test.jsonl in the repo root."
        )
    problems: set[str] = set()
    with open(MATH500_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                problems.add(json.loads(line)["problem"])
    logger.info("Loaded %d MATH-500 problems for exclusion.", len(problems))
    return problems


# ===================================================================
# 1.  Data segregation — assign POMDP states to ProcessBench samples
# ===================================================================

def get_divided_data_eff(
    subset_data: Any,
    indices: list[int],
    num_parts: int = 4,
) -> list[dict]:
    """Segregate ProcessBench samples into POMDP state classes.

    Improved sample-efficiency variant, but sample segregration not independent:

    * **Correct samples** (``label == -1``): chopped at *every* partition
      boundary, each yielding an independent State-0 example.
    * **Incorrect samples** (``label != -1``):
      - Steps up to and including the first error → State 2.
      - If there are steps *beyond* the first error, each partition
        boundary past ``transition_step`` yields a State-1 example.
      This means a single incorrect sample can contribute to *both*
      State 1 and State 2.

    States
    ------
    * State 0 — all selected steps are correct (no error has occurred).
    * State 1 — an error has occurred but the solution continues
    * State 2 — the most recently added step is the first erroneous one.
    """
    output_list: list[dict] = []

    for index in indices:
        data = subset_data[index]
        num_steps = len(data["steps"])
        label = data["label"]
        raw_rewards = data.get("raw_rewards")

        benchmark = data.get("benchmark")

        if label == -1:
            # All steps are correct → State 0.
            # Chop at every partition boundary to create multiple examples.
            part_size = max(1, ceil(num_steps / num_parts))
            for k in range(1, num_parts + 1):
                sel = min(k * part_size, num_steps)
                output_list.append({
                    "problem": data["problem"],
                    "selected_steps": data["steps"][:sel],
                    "state": 0,
                    "label": label,
                    **({"benchmark": benchmark} if benchmark is not None else {}),
                    **({"PRM Rewards": raw_rewards[:sel]} if raw_rewards is not None else {}),
                })
                if sel >= num_steps:
                    break
        else:
            transition_step = label  # 0-indexed first-error step

            # --- State 2: include up to and including the first error ---
            sel2 = transition_step + 1
            output_list.append({
                "problem": data["problem"],
                "selected_steps": data["steps"][:sel2],
                "state": 2,
                "label": label,
                **({"benchmark": benchmark} if benchmark is not None else {}),
                **({"PRM Rewards": raw_rewards[:sel2]} if raw_rewards is not None else {}),
            })

            # --- State 1: steps beyond the first error ---
            num_remaining = num_steps - (transition_step + 1)
            if num_remaining > 0:
                part_size = max(1, ceil(num_remaining / num_parts))
                k = random.randint(1, num_parts)
                sel = min(
                    transition_step + 1 + k * part_size,
                    num_steps,
                )
                output_list.append({
                    "problem": data["problem"],
                    "selected_steps": data["steps"][:sel],
                    "state": 1,
                    "label": label,
                    **({"benchmark": benchmark} if benchmark is not None else {}),
                    **({"PRM Rewards": raw_rewards[:sel]} if raw_rewards is not None else {}),
                })

    return output_list


def get_divided_data_indp(
    subset_data: Any,
    indices: list[int],
    num_parts: int = 4,
) -> list[dict]:
    """Independent segregation logic (one sample → one example)."""
    output_list: list[dict] = []
    switch = 0

    for i, index in enumerate(indices):
        data = subset_data[index]
        num_steps = len(data["steps"])
        raw_rewards = data.get("raw_rewards")
        benchmark = data.get("benchmark")

        if data["label"] == -1:
            selected_num_steps = min(
                num_steps,
                (num_parts - (i % num_parts)) * ceil(num_steps / num_parts),
            )
            output_list.append({
                "problem": data["problem"],
                "selected_steps": data["steps"][:selected_num_steps],
                "state": 0,
                "label": data["label"],
                **({"benchmark": benchmark} if benchmark is not None else {}),
                **({"PRM Rewards": raw_rewards[:selected_num_steps]} if raw_rewards is not None else {}),
            })
        else:
            switch += 1
            transition_step = data["label"]
            if switch % 2 == 0:
                sel = transition_step + 1
                output_list.append({
                    "problem": data["problem"],
                    "selected_steps": data["steps"][:sel],
                    "state": 2,
                    "label": data["label"],
                    **({"benchmark": benchmark} if benchmark is not None else {}),
                    **({"PRM Rewards": raw_rewards[:sel]} if raw_rewards is not None else {}),
                })
            else:
                sel = transition_step + 1
                if transition_step == num_steps - 1:
                    output_list.append({
                        "problem": data["problem"],
                        "selected_steps": data["steps"][:sel],
                        "state": 2,
                        "label": data["label"],
                        **({"benchmark": benchmark} if benchmark is not None else {}),
                        **({"PRM Rewards": raw_rewards[:sel]} if raw_rewards is not None else {}),
                    })
                    continue
                num_false_steps = num_steps - transition_step
                selected_num_steps = min(
                    num_steps,
                    transition_step + 1
                    + (num_parts - (i % num_parts)) * ceil(num_false_steps / num_parts),
                )
                output_list.append({
                    "problem": data["problem"],
                    "selected_steps": data["steps"][:selected_num_steps],
                    "state": 1,
                    "label": data["label"],
                    **({"benchmark": benchmark} if benchmark is not None else {}),
                    **({"PRM Rewards": raw_rewards[:selected_num_steps]} if raw_rewards is not None else {}),
                })

    return output_list


# ===================================================================
# 2.  PRM scoring via vLLM Pooling API for entire ProcessBench (before any splitting)  
# ===================================================================

def score_raw_process_bench(
    *,
    prm_server_url: str,
    prm_model_name: str,
    rerun: bool = False,
) -> Dataset:
    """Score every raw ProcessBench sample (full step list) with the PRM and cache.

    Each sample's *complete* ``steps`` list is scored once.  The resulting
    per-step reward sequence is stored as ``raw_rewards`` so that any
    step-prefix can be featurised later without hitting the PRM server again.

    Saves to ``PROCESS_BENCH_PRM_SCORED_PATH``.
    """
    if not rerun and PROCESS_BENCH_PRM_SCORED_PATH.exists():
        logger.info(
            "PRM-scored ProcessBench already cached at %s — skipping.",
            PROCESS_BENCH_PRM_SCORED_PATH,
        )
        return load_from_disk(str(PROCESS_BENCH_PRM_SCORED_PATH))

    logger.info("Loading Qwen/ProcessBench from HuggingFace Hub …")
    pb = load_dataset("Qwen/ProcessBench")

    tokenizer = AutoTokenizer.from_pretrained(prm_model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    prm = ServerPRM(
        server_url=prm_server_url,
        model_name=prm_model_name,
        tokenizer=tokenizer,
    )

    # Collect all (benchmark, row) pairs in order so we can fan out in parallel
    all_pairs: list[tuple[str, dict]] = []
    for benchmark in pb.keys():
        subset = pb[benchmark]
        logger.info("Queuing benchmark=%s (%d samples) …", benchmark, len(subset))
        for i in range(len(subset)):
            all_pairs.append((benchmark, subset[i]))

    logger.info("Scoring %d samples via PRM at %s …", len(all_pairs), prm_server_url)

    def _score(idx: int) -> tuple[int, list[float]]:
        _, row = all_pairs[idx]
        return idx, prm.score(row["problem"], row["steps"])

    results: dict[int, list[float]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=prm.max_workers) as pool:
        futs = {pool.submit(_score, i): i for i in range(len(all_pairs))}
        for fut in concurrent.futures.as_completed(futs):
            idx, rewards = fut.result()
            results[idx] = rewards

    flat_data: list[dict] = [
        {
            "problem": all_pairs[i][1]["problem"],
            "steps":   all_pairs[i][1]["steps"],
            "label":   all_pairs[i][1]["label"],
            "benchmark": all_pairs[i][0],
            "raw_rewards": results[i],
        }
        for i in range(len(all_pairs))
    ]

    raw_scored = Dataset.from_list(flat_data)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    raw_scored.save_to_disk(str(PROCESS_BENCH_PRM_SCORED_PATH))
    logger.info(
        "Saved PRM-scored ProcessBench → %s  (%d rows)",
        PROCESS_BENCH_PRM_SCORED_PATH,
        len(raw_scored),
    )
    return raw_scored

def extract_prm_features(rewards: list[float]) -> tuple[float, float, float]:
    """Compute (min_score, current_step_prob, geometric_mean) from PRM rewards.

    Matches the state features used in ``TRIM_Agg.py``:
    * min of all scores except the last
    * the last score (current step)
    * geometric mean of all scores except the last
    """
    x = np.array(rewards, dtype=np.float64)
    if x.size > 1:
        prev = x[:-1]
        min_score = float(prev.min())
        safe = np.clip(prev, 1e-9, None)
        geom = float(np.exp(np.mean(np.log(safe))))
    else:
        min_score = 1.0
        geom = 1.0
    current = float(x[-1])
    return min_score, current, geom


def build_split_dataset(
    raw_scored: Dataset,
    *,
    split_mode: str = "efficient",
    rerun: bool = False,
) -> Dataset:
    """Apply ``split_fn`` to the raw-scored ProcessBench data and extract PRM features.

    For every split row the per-step PRM rewards are derived by slicing the
    cached ``raw_rewards`` to ``len(selected_steps)``, so no additional server
    calls are needed.  Features (Min Score, Current Step Prob, Geometric Mean)
    are then computed from the partial reward sequence.

    Saves to ``PROCESS_BENCH_STATE_CLASSIFIED_PATH``.
    """
    if not rerun and PROCESS_BENCH_STATE_CLASSIFIED_PATH.exists():
        logger.info(
            "Split dataset already cached at %s — skipping.", PROCESS_BENCH_STATE_CLASSIFIED_PATH
        )
        return load_from_disk(str(PROCESS_BENCH_STATE_CLASSIFIED_PATH))

    split_fn = (
        get_divided_data_eff if split_mode == "efficient"
        else get_divided_data_indp
    )

    split_rows = split_fn(raw_scored, list(range(len(raw_scored))))
    flat_data: list[dict] = []
    for split_row in split_rows:
        m, c, g = extract_prm_features(split_row["PRM Rewards"])
        flat_data.append({
            **split_row,
            "Min Score": m,
            "Current Step Prob": c,
            "Geometric Mean": g,
        })

    final_dataset = Dataset.from_list(flat_data)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    final_dataset.save_to_disk(str(PROCESS_BENCH_STATE_CLASSIFIED_PATH))
    logger.info(
        "Saved split dataset → %s  (%d rows)", PROCESS_BENCH_STATE_CLASSIFIED_PATH, len(final_dataset)
    )
    return final_dataset


# ===================================================================
# 3.  Reflected KDE observation model
# ===================================================================

class ReflectedKDEObservationModel:
    """Kernel density estimator on [0,1]^2 using boundary reflection.

    For each POMDP state a 2-D KDE is fitted to (Min Score, Current Step Prob)
    observations.  The *reflection* trick mirrors each kernel across all
    boundaries of the unit square so that probability mass does not leak
    outside [0, 1]^2 — this is important because PRM scores are bounded.

    The 9-fold reflection evaluates the original density at::

        (x, y), (2-x, y), (-x, y),
        (x, 2-y), (2-x, 2-y), (-x, 2-y),
        (x, -y), (2-x, -y), (-x, -y)

    and sums the contributions.
    """

    def __init__(self, states: tuple[int, ...] = (0, 1, 2), bw_method: str = "scott"):
        self.states = list(states)
        self.kde_models: dict[int, gaussian_kde] = {}
        self.bw_method = bw_method

    def train(self, train_data: list[tuple[int, list[float]]]) -> None:
        state_data: dict[int, list[list[float]]] = {s: [] for s in self.states}
        for state, obs in train_data:
            if state in state_data:
                state_data[state].append(obs)

        for state, observations in state_data.items():
            data = np.array(observations, dtype=float)
            if data.ndim != 2 or data.shape[1] != 2:
                raise ValueError(f"State {state}: Expected (N,2) array, got {data.shape}")
            self.kde_models[state] = gaussian_kde(data.T, bw_method=self.bw_method)

    def pdf(self, state: int, point: tuple[float, float]) -> float:
        if state not in self.kde_models:
            raise KeyError(f"No KDE model for state {state}")

        x, y = point
        total = 0.0
        # Evaluate density at 9 reflection points
        for x_ref in [x, 2 - x, -x]:
            for y_ref in [y, 2 - y, -y]:
                total += self.kde_models[state](np.array([[x_ref], [y_ref]]))[0] ## Correct evaluation for 2D KDE: pass point as column vector
        return total

    def precompute_obs_probs(
        self,
        bin_size: float = 0.05,
        epsabs: float = 1e-6,
        epsrel: float = 1e-6,
    ) -> dict[int, tuple[list[tuple[float, float]], list[float]]]:
        """Integrate reflected density over bins and normalise per state."""
        bins = np.arange(0, 1.0001, bin_size)
        obs_probs: dict[int, tuple[list, list]] = {}

        for state in self.states:
            bin_centers: list[tuple[float, float]] = []
            bin_probs: list[float] = []
            total_mass = 0.0

            for i in range(len(bins) - 1):
                for j in range(len(bins) - 1):
                    x_low, x_high = bins[i], bins[i + 1]
                    y_low, y_high = bins[j], bins[j + 1]
                    # Integrate reflected density over bin
                    mass, _ = dblquad(
                        lambda y, x: self.pdf(state, (x, y)),
                        x_low, x_high,
                        lambda x: y_low,
                        lambda x: y_high,
                        epsabs=epsabs,
                        epsrel=epsrel,
                    )
                
                    center_x = round((x_low + x_high) / 2, 4)
                    center_y = round((y_low + y_high) / 2, 4)
                    bin_centers.append((center_x, center_y))
                    bin_probs.append(mass)
                    total_mass += mass

            logger.info("Total mass for state %d: %.6f", state, total_mass)
            if total_mass > 0:
                bin_probs = [p / total_mass for p in bin_probs]
            obs_probs[state] = (bin_centers, bin_probs)

        return obs_probs

    def sample(self, state: int, n: int = 1) -> np.ndarray:
        """Generate *n* samples from the reflected KDE in [0,1]^2."""
        if state not in self.kde_models:
            raise KeyError(f"No KDE model for state {state}")

        raw_samples = self.kde_models[state].resample(n)
        # Fold samples into [0,1]x[0,1] using reflection
        samples = np.empty_like(raw_samples)
        for dim_idx in range(2): # Process x and y dimensions separately
            d = raw_samples[dim_idx, :] % 2 # Apply modulo 2 to bring into [0,2) range
            d = np.where(d < 0, d + 2, d) # Handle negative values (Python modulo handles this, but ensure [0,2))
            d = np.where(d > 1, 2 - d, d) # Reflect values >1: x → 2 - x
            samples[dim_idx, :] = d
        return samples.T


def fit_and_save_kde(
    scored_dataset: Dataset,
    *,
    benchmark: str,
    bin_size: float,
    rerun: bool = False,
) -> None:
    """Fit a reflected-KDE observation model for *benchmark* and persist.

    For ``benchmark == 'math-train'``: the ProcessBench ``math`` split is used,
    but questions that appear in MATH-500 (``math_eval/data/math500/test.jsonl``)
    are excluded.  This keeps the KDE training data disjoint from the evaluation
    set.  For ``benchmark == 'omnimath'`` no such filtering is applied.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    model_path = DATA_DIR / f"{benchmark}_reflected_kde_obs_model.pkl"
    dist_path = DATA_DIR / f"{benchmark}_obs_distributions_bin_size_{bin_size}.pkl"

    if not rerun and model_path.exists() and dist_path.exists():
        logger.info("KDE artefacts already exist — skipping. Use --rerun to regenerate.")
        return

    # For math-train: use the ProcessBench "math" split, then drop any question
    # that is present in the MATH-500 evaluation set to avoid train/eval overlap.
    pb_benchmark = "math" if benchmark == "math-train" else benchmark
    filtered = scored_dataset.filter(lambda x: x["benchmark"] == pb_benchmark)

    if benchmark == "math-train":
        math500_problems = load_math500_problems()
        before = len(filtered)
        filtered = filtered.filter(lambda x: x["problem"] not in math500_problems)
        logger.info(
            "math-train: excluded %d MATH-500 questions, %d remain.",
            before - len(filtered),
            len(filtered),
        )

    if len(filtered) == 0:
        logger.warning("No samples for benchmark=%s — skipping KDE fit.", benchmark)
        return

    observations_list = [
        [filtered[i]["Min Score"], filtered[i]["Current Step Prob"]]
        for i in range(len(filtered))
    ]
    states_list = [filtered[i]["state"] for i in range(len(filtered))]
    training_data = list(zip(states_list, observations_list))

    kde_model = ReflectedKDEObservationModel()
    kde_model.train(training_data)

    with open(model_path, "wb") as f:
        cp.dump(kde_model, f)
    logger.info("Saved KDE model → %s", model_path)

    obs_distributions = kde_model.precompute_obs_probs(bin_size=bin_size)
    with open(dist_path, "wb") as f:
        cp.dump(obs_distributions, f)
    logger.info("Saved obs distributions → %s", dist_path)


# ===================================================================
# 4.  CLI entry point
# ===================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build reflected-KDE observation models for the POMDP.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--split-mode", choices=["efficient", "legacy"], default="efficient",
        help="Sample segregation strategy for ProcessBench.",
    )
    p.add_argument(
        "--seed", type=int, default=10,
        help="Random seed for reproducibility.",
    )
    p.add_argument(
        "--benchmark", type=str, default="omnimath",
        help="Benchmark for KDE fit: 'omnimath' (ProcessBench omnimath split) or "
             "'math-train' (ProcessBench math split, with MATH-500 questions removed "
             "to prevent train/eval overlap — see math_eval/data/math500/test.jsonl).",
    )
    p.add_argument(
        "--bin-size", type=float, default=0.05,
        help="Bin size for precomputed observation probabilities.",
    )
    p.add_argument(
        "--prm-server-url", type=str, default="http://localhost:30002",
        help="Base URL of the vLLM PRM server (Pooling API).",
    )
    p.add_argument(
        "--prm-model-name", type=str, default="Qwen/Qwen2.5-Math-PRM-7B",
        help="HuggingFace model name for the PRM.",
    )
    p.add_argument(
        "--rerun", action="store_true",
        help="Regenerate all cached artefacts even if they exist on disk.",
    )
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    args = parse_args()
    
    # Seed all RNGs for reproducibility
    seed_everything(args.seed)
    logger.info("Random seed set to %d", args.seed)

    # Step 1: Score the entire raw ProcessBench (full steps) with PRM and cache
    raw_scored = score_raw_process_bench(
        prm_server_url=args.prm_server_url,
        prm_model_name=args.prm_model_name,
        rerun=args.rerun,
    )

    # Step 2: Apply split_fn and derive per-row PRM features from cached rewards
    split_dataset = build_split_dataset(
        raw_scored,
        split_mode=args.split_mode,
        rerun=args.rerun,
    )

    # Step 3: Fit reflected KDE and precompute observation probabilities
    fit_and_save_kde(
        split_dataset,
        benchmark=args.benchmark,
        bin_size=args.bin_size,
        rerun=args.rerun,
    )

    logger.info("All done.")


if __name__ == "__main__":
    main()
