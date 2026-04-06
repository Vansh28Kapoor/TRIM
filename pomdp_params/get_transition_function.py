"""Compute POMDP transition-function parameters from model-generated solutions.

Pipeline
--------
1. Load a math benchmark (default: AIME).
2. Generate full solutions with the draft and target model if outputs do not
   already exist on disk.  Existing outputs are re-used automatically; pass
   ``--rerun`` to force regeneration.
3. Score each incorrect solution with the PRM (via vLLM Pooling API) to
   locate the first erroneous step using a configurable threshold.
4. Persist transition-parameter pickles plus KDE-based terminal predictors
    (SLM, LLM, and combined) and probability plots for steps 2..(max_steps-1).

Usage
-----
Start the vLLM servers for draft, target, and PRM models, then::

    python pomdp_params/get_transition_function.py \\
        --train-benchmark math \\
        --train-split train \\
        --thr 0.35

To regenerate cached model outputs::

    python pomdp_params/get_transition_function.py \\
        --train-benchmark math \\
        --train-split train \\
        --rerun
"""

from __future__ import annotations

import argparse
import concurrent.futures
import logging
import os
import pickle
import sys
import warnings
from pathlib import Path
from typing import Optional

# Suppress SyntaxWarning from Jinja2 template compilation inside tokenizers
# (e.g. Qwen chat templates trigger "'tuple' object is not callable" at <string>:1)
warnings.filterwarnings("ignore", category=SyntaxWarning, message="'tuple' object is not callable")

import numpy as np
from scipy.stats import gaussian_kde
from datasets import load_dataset
from openai import OpenAI
from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
# Path setup — allow imports from the repo root (utils, math_eval) whether
# this script is invoked as  ``python pomdp_params/get_transition_function.py``
# or via  ``python -m pomdp_params.get_transition_function``.
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from math_eval.math_equal import math_equal
from math_eval.parser import (
    STRIP_EXCEPTIONS,
    extract_answer,
    parse_ground_truth,
    strip_string,
)
from utils import ServerPRM, _is_degenerate, generate_full_solutions, seed_everything

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _model_short_name(model_name: str) -> str:
    """Derive a filesystem-safe short name from a HuggingFace model path.

    Examples
    --------
    >>> _model_short_name("Qwen/Qwen2.5-1.5B-Instruct")
    'qwen2.5-1.5b-instruct'
    >>> _model_short_name("Qwen/Qwen3-8B")
    'qwen3-8b'
    """
    return model_name.split("/")[-1].lower()


def thresholding_index(scores: list[float], thr: float) -> int:
    """Return the index of the first step scoring below *thr*.

    If no step falls below the threshold, returns the index of the minimum
    score (i.e. the most-likely erroneous step).

    Parameters
    ----------
    scores : list[float]
        Per-step PRM positive-class probabilities.
    thr : float
        Score threshold below which a step is considered erroneous.

    Returns
    -------
    int
        0-based index of the first erroneous step (or argmin as fallback).
    """
    a = np.asarray(scores)
    idxs = np.nonzero(a < thr)[0]
    return int(idxs[0]) if idxs.size else int(a.argmin())


class TerminalPredictor:
    """Kernel-density terminal-step predictor used by POMDP transitions."""

    def __init__(self, max_steps: int = 30):
        self.max_steps = max_steps
        self.pmf: Optional[np.ndarray] = None
        self.ccdf: Optional[np.ndarray] = None
        self.probs_dict: dict[int, float] = {}

    def train(self, train_data: list[int] | np.ndarray) -> None:
        """Fit KDE over terminal step counts and derive hazard probabilities."""
        data = np.asarray(train_data, dtype=float).reshape(-1)
        data = np.clip(data, 2, self.max_steps)

        # Match prior implementation: KDE + discretised PMF over steps [2, max_steps]
        kde = gaussian_kde(data, bw_method=0.1)
        pmf = np.array(
            [
                kde.integrate_box_1d(k - 0.5, k + 0.5)
                for k in range(2, self.max_steps + 1)
            ],
            dtype=float,
        )
        pmf = np.maximum(pmf, 0.0)
        pmf_sum = float(pmf.sum())
        if pmf_sum <= 0:
            raise ValueError("KDE integration produced zero PMF mass.")

        self.pmf = pmf / pmf_sum
        self.ccdf = np.array([self.pmf[i:].sum() for i in range(len(self.pmf))], dtype=float)

        probs: dict[int, float] = {0: 0.0, 1: 0.0}
        for step in range(2, self.max_steps):
            idx = step - 2
            denom = float(self.ccdf[idx])
            if denom <= 0:
                probs[step] = 1.0
            else:
                probs[step] = float((self.ccdf[idx] - self.ccdf[idx + 1]) / denom)
        probs[self.max_steps] = 1.0
        self.probs_dict = probs

    def predict(self, step_no: int) -> float:
        if step_no not in self.probs_dict:
            raise KeyError(f"Step {step_no} missing from terminal predictor table.")
        return self.probs_dict[step_no]


def _save_terminal_plot(
    predictors: dict[str, TerminalPredictor],
    *,
    max_steps: int,
    out_path: Path,
    title: str,
) -> None:
    """Save a combined terminal-probability plot for steps [2, max_steps-1]."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available; skipping plot %s", out_path)
        return False

    try:
        xs = list(range(2, max_steps))
        plt.figure(figsize=(8, 4.5))
        for name, predictor in predictors.items():
            ys = [predictor.predict(step) for step in xs]
            plt.plot(xs, ys, marker="o", linewidth=1.5, label=name)
        plt.xlabel("Step number")
        plt.ylabel("Terminal probability")
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.legend()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()
        return True
    except Exception as exc:
        logger.warning("Failed to save terminal predictor plot %s: %s", out_path, exc)
        return False


def save_terminal_predictors(
    *,
    benchmark: str,
    split: str,
    predictor_dir: str,
    slm_steps: np.ndarray,
    llm_steps: np.ndarray,
    max_steps: int,
) -> None:
    """Train and persist SLM/LLM/combined terminal predictors and plots."""
    out_dir = Path(predictor_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    slm_predictor = TerminalPredictor(max_steps=max_steps)
    slm_predictor.train(slm_steps.astype(int).tolist())

    llm_predictor = TerminalPredictor(max_steps=max_steps)
    llm_predictor.train(llm_steps.astype(int).tolist())

    combo_predictor = TerminalPredictor(max_steps=max_steps)
    combo_predictor.train(np.concatenate([slm_steps, llm_steps]).astype(int).tolist())

    predictors = {
        "slm": slm_predictor,
        "llm": llm_predictor,
        "slm_llm": combo_predictor,
    }

    for name, predictor in predictors.items():
        pkl_path = out_dir / f"{benchmark}_{split}_terminal_predictor_{name}.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(predictor, f)
        logger.info("Saved terminal predictor → %s", pkl_path)

    plot_path = out_dir / f"{benchmark}_{split}_terminal_predictors_probs.png"
    saved = _save_terminal_plot(
        predictors,
        max_steps=max_steps,
        out_path=plot_path,
        title=f"{benchmark}/{split} terminal probs (slm, llm, slm_llm)",
    )
    if saved:
        logger.info("Saved terminal predictor plot → %s", plot_path)


# ---------------------------------------------------------------------------
# Output generation
# ---------------------------------------------------------------------------

def _outputs_path(output_dir: str, benchmark: str, split: str, model_name: str) -> Path:
    """Return the canonical path for a model-output pickle file."""
    short = _model_short_name(model_name)
    return Path(output_dir) / f"{benchmark}_{split}_{short}_outputs.pkl"


def load_or_generate_outputs(
    *,
    output_dir: str,
    benchmark: str,
    split: str,
    model_name: str,
    server_url: str,
    tokenizer: AutoTokenizer,
    questions: list[str],
    ground_truths: list[str],
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: Optional[int],
    rerun: bool,
) -> dict:
    """Load cached model outputs from disk, or generate and cache them.

    vLLM provides dynamic server-side batching, so all questions are submitted
    in a single API call — no client-side batching loop is required.

    The output dict stored on disk has the schema::

        {
            "output": list[str],           # raw model completions
            "questions": list[str],         # questions (self-documenting)
            "ground_truths": list[str],     # reference answers
        }

    Parameters
    ----------
    output_dir : str
        Directory in which to read/write output pickle files.
    benchmark : str
        Benchmark name (used in the filename).
    split : str
        Dataset split, e.g. ``"train"`` or ``"test"``.
    model_name : str
        Full HuggingFace model name (used to derive a short filename suffix).
    server_url : str
        vLLM server base URL for generation (must expose ``/v1``).
    tokenizer : AutoTokenizer
        Tokenizer for the model.
    questions : list[str]
        Questions to generate solutions for.
    ground_truths : list[str]
        Reference answers (stored alongside outputs for convenience).
    max_tokens : int
        Maximum completion tokens per solution.
    temperature : float
        Sampling temperature.
    top_p : float
        Nucleus sampling probability.
    top_k : int or None
        Top-k sampling parameter.
    rerun : bool
        If ``True``, regenerate even if a cached file already exists.

    Returns
    -------
    dict
        Loaded or freshly generated output dict.
    """
    pkl_path = _outputs_path(output_dir, benchmark, split, model_name)

    if not rerun and pkl_path.exists():
        logger.info("Loading cached outputs from %s", pkl_path)
        with open(pkl_path, "rb") as f:
            return pickle.load(f)

    logger.info(
        "Generating full solutions for %d questions using %s …",
        len(questions),
        model_name,
    )
    client = OpenAI(base_url=server_url, api_key="EMPTY", timeout=1800.0)

    # vLLM provides dynamic server-side batching; send all prompts at once.
    outputs = generate_full_solutions(
        client=client,
        model_name=model_name,
        tokenizer=tokenizer,
        questions=questions,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        disable_thinking=False,
    )
    logger.info("Generation complete: %d solutions received.", len(outputs))

    data = {
        "output": outputs,
        "questions": questions,
        "ground_truths": ground_truths,
    }

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(pkl_path, "wb") as f:
        pickle.dump(data, f)
    logger.info("Saved outputs → %s", pkl_path)

    return data


# ---------------------------------------------------------------------------
# PRM scoring
# ---------------------------------------------------------------------------

def extract_incorrect_steps(
    model_outputs: list[list[str]],
    questions: list[str],
    incorrect_indices: list[int],
    prm: ServerPRM,
    thr: float,
) -> list[int]:
    """Identify the first erroneous step for each incorrect solution.

    Scores each incorrect solution with the PRM server and applies
    ``thresholding_index`` to locate the step where the solution first goes
    wrong.

    Parameters
    ----------
    model_outputs : list[list[str]]
        Step-split solutions for the incorrect problems (same length and order
        as *incorrect_indices*).
    questions : list[str]
        Full list of questions (indexed by *incorrect_indices*).
    incorrect_indices : list[int]
        Dataset indices of the incorrectly solved problems.
    prm : ServerPRM
        Initialised PRM client (vLLM Pooling API).
    thr : float
        PRM score threshold for the first-error detection.

    Returns
    -------
    list[int]
        0-based index of the first erroneous step for each incorrect problem.
    """
    def _score(i: int) -> tuple[int, list[float]]:
        return i, prm.score(questions[incorrect_indices[i]], model_outputs[i])

    results: dict[int, list[float]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=prm.max_workers) as pool:
        futs = {pool.submit(_score, i): i for i in range(len(incorrect_indices))}
        for fut in concurrent.futures.as_completed(futs):
            i, rewards = fut.result()
            results[i] = rewards

    return [
        thresholding_index(results[i], thr) if results[i] else 0
        for i in range(len(incorrect_indices))
    ]


# ---------------------------------------------------------------------------
# Transition parameter computation
# ---------------------------------------------------------------------------

def compute_transition_params(
    *,
    benchmark: str,
    split: str,
    model_name: str,
    server_url: str,
    output_dir: str,
    questions: list[str],
    ground_truths: list[str],
    prm: ServerPRM,
    tokenizer: AutoTokenizer,
    thr: float,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: Optional[int],
    max_steps: int,
    rerun: bool,
) -> np.ndarray:
    """Generate outputs, score incorrect solutions, and save transition params.

    This is the main per-model, per-split processing step.  Results are saved
    to ``{output_dir}/{benchmark}_{split}_{short_model_name}_transition_params.pkl``.

    The saved dict has the schema::

        {
            "correct_steps": np.ndarray,    # step counts for correct solutions
            "incorrect_steps": np.ndarray,  # first-error step indices (+1, 1-based)
            "accuracy": float,              # next-step accuracy
        }

    Next-step accuracy is defined as the fraction of total steps (across all
    problems) that are correct::

        accuracy = (sum(correct_steps) + sum(incorrect_steps) - len(incorrect_steps))
                   / (sum(correct_steps) + sum(incorrect_steps))

    Parameters
    ----------
    benchmark : str
        Benchmark name (e.g. ``"aime"``).
    split : str
        Dataset split, e.g. ``"train"`` or ``"test"``.
    model_name : str
        Full HuggingFace model name.
    server_url : str
        vLLM server URL for generation.
    output_dir : str
        Directory for reading/writing outputs and params.
    questions : list[str]
        Dataset questions.
    ground_truths : list[str]
        Reference answers.
    prm : ServerPRM
        Initialised PRM client.
    tokenizer : AutoTokenizer
        Tokenizer for the model.
    thr : float
        PRM threshold for first-error detection.
    max_tokens : int
        Maximum generation tokens per solution.
    temperature : float
        Sampling temperature.
    top_p : float
        Nucleus sampling probability.
    top_k : int or None
        Top-k sampling.
    max_steps : int
        Maximum reasoning steps used for step-count clipping.
    rerun : bool
        Force regeneration of cached outputs.
    Returns
    -------
    np.ndarray
        Per-problem generated step counts clipped to ``max_steps``.
    """
    short = _model_short_name(model_name)
    params_path = Path(output_dir) / f"{benchmark}_{split}_{short}_transition_params.pkl"

    # ── Step 1: load or generate model outputs ──────────────────────────────
    data = load_or_generate_outputs(
        output_dir=output_dir,
        benchmark=benchmark,
        split=split,
        model_name=model_name,
        server_url=server_url,
        tokenizer=tokenizer,
        questions=questions,
        ground_truths=ground_truths,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        rerun=rerun,
    )

    batch_results: list[str] = data["output"]

    # ── Step 2: evaluate predictions ────────────────────────────────────────
    predictions = [
        strip_string(
            extract_answer(ans, benchmark),
            skip_unit=benchmark in STRIP_EXCEPTIONS,
        )
        for ans in batch_results
    ]

    incorrect_index_set = {
        i
        for i, (pred, gt) in enumerate(zip(predictions, ground_truths))
        if not math_equal(pred, gt)
    }

    # Split and strip degenerate steps before any counting or PRM scoring
    all_steps: list[list[str]] = [
        [step for step in sol.split("\n\n") if not _is_degenerate(step)]
        for sol in batch_results
    ]
    steps_data = np.array([min(max_steps, len(steps)) for steps in all_steps])
    correct_index = [i for i in range(len(steps_data)) if i not in incorrect_index_set]
    incorrect_index = list(incorrect_index_set)

    incorrect_outputs: list[list[str]] = [all_steps[i] for i in incorrect_index]

    # ── Step 3: PRM scoring ──────────────────────────────────────────────────
    logger.info(
        "[%s | %s | %s] Scoring %d incorrect solutions with PRM …",
        benchmark, split, short, len(incorrect_outputs),
    )
    incorrect_steps_raw = extract_incorrect_steps(
        model_outputs=incorrect_outputs,
        questions=questions,
        incorrect_indices=incorrect_index,
        prm=prm,
        thr=thr,
    )

    # ── Step 4: aggregate and save ───────────────────────────────────────────
    # +1 converts from 0-based step index to 1-based step count
    incorrect_steps = np.array(incorrect_steps_raw, dtype=int) + 1
    correct_steps = np.array([len(all_steps[i]) for i in correct_index], dtype=int)

    # Next-step accuracy: fraction of total generated steps that are correct.
    # For a correct solution all steps count; for an incorrect solution only
    # steps up to (but not including) the first error count as correct, so
    # the correct contribution is  (incorrect_steps[j] - 1)  per problem.
    # Summing: correct = sum(correct_steps) + sum(incorrect_steps - 1)
    #                  = sum(correct_steps) + sum(incorrect_steps) - len(incorrect_steps)
    total_steps = int(correct_steps.sum()) + int(incorrect_steps.sum())
    correct_step_total = total_steps - len(incorrect_steps)
    accuracy = correct_step_total / total_steps if total_steps > 0 else float("nan")

    logger.info(
        "[%s | %s | %s] Next-step accuracy: %.4f",
        benchmark, split, short, accuracy,
    )
    print(f"{short} [{split}] Accuracy: {accuracy:.4f}")

    # Average token count per step across all solutions
    all_token_counts = [
        len(tokenizer.encode(step))
        for steps in all_steps
        for step in steps
    ]
    avg_token_count_per_step = float(np.mean(all_token_counts)) if all_token_counts else 0.0
    logger.info(
        "[%s | %s | %s] Avg tokens/step: %.1f",
        benchmark, split, short, avg_token_count_per_step,
    )

    params = {
        "correct_steps": correct_steps,
        "incorrect_steps": incorrect_steps,
        "accuracy": accuracy,
        "avg_token_count_per_step": avg_token_count_per_step,
    }

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(params_path, "wb") as f:
        pickle.dump(params, f)
    logger.info("Saved transition params → %s", params_path)
    return steps_data


def load_questions_and_ground_truths(
    *,
    data_dir: str,
    benchmark: str,
    split: str,
) -> tuple[list[str], list[str]]:
    """Load benchmark questions and parsed ground truths for a split."""
    data_path = f"{data_dir}/{benchmark}/{split}.jsonl"
    logger.info("Loading dataset: %s", data_path)
    dataset = load_dataset("json", data_files=data_path, split="train")

    q_key = "question" if "question" in dataset.column_names else "problem"
    questions: list[str] = [dataset[i][q_key] for i in range(len(dataset))]
    ground_truths: list[str] = [
        parse_ground_truth(sample, benchmark)[1]
        for sample in dataset
    ]
    return questions, ground_truths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute POMDP transition-function parameters from model-generated solutions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--train-benchmark", type=str, default="aime",
        help="Training benchmark name matching math_eval/data/{benchmark}.",
    )
    p.add_argument(
        "--train-split", type=str, default="train",
        help="Split to use for the training benchmark.",
    )
    p.add_argument(
        "--data-dir", type=str, default="math_eval/data",
        help="Path to benchmark jsonl files.",
    )
    p.add_argument(
        "--output-dir", type=str, default="pomdp_data/transition_params",
        help="Directory for saving output pickles and transition params.",
    )
    p.add_argument(
        "--draft-model-name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct",
        help="HuggingFace model name for the draft model.",
    )
    p.add_argument(
        "--draft-server-url", type=str, default="http://localhost:30001/v1",
        help="vLLM server URL for the draft model (OpenAI-compatible).",
    )
    p.add_argument(
        "--target-model-name", type=str, default="Qwen/Qwen2.5-7B-Instruct",
        help="HuggingFace model name for the target model.",
    )
    p.add_argument(
        "--target-server-url", type=str, default="http://localhost:30000/v1",
        help="vLLM server URL for the target model (OpenAI-compatible).",
    )
    p.add_argument(
        "--prm-model-name", type=str, default="Qwen/Qwen2.5-Math-PRM-7B",
        help="HuggingFace model name for the PRM.",
    )
    p.add_argument(
        "--prm-server-url", type=str, default="http://localhost:30002",
        help="vLLM server URL for the PRM (Pooling API — do NOT append /v1).",
    )
    p.add_argument(
        "--thr", type=float, default=0.35,
        help="PRM score threshold for identifying the first erroneous step.",
    )
    p.add_argument(
        "--max-tokens", type=int, default=4096,
        help="Maximum tokens per full solution.",
    )
    p.add_argument(
        "--temperature", type=float, default=0.7,
        help="Sampling temperature for generation.",
    )
    p.add_argument(
        "--top-p", type=float, default=0.8,
        help="Nucleus sampling probability for generation.",
    )
    p.add_argument(
        "--top-k", type=int, default=20,
        help="Top-k sampling parameter for generation.",
    )
    p.add_argument(
        "--max-steps", type=int, default=30,
        help="Maximum reasoning steps (step counts are clipped at this value).",
    )
    p.add_argument(
        "--seed", type=int, default=10,
        help="Random seed for reproducibility.",
    )
    p.add_argument(
        "--rerun", action="store_true",
        help="Regenerate cached outputs even if they already exist on disk.",
    )
    p.add_argument(
        "--gpus", type=str, default="",
        help="If set, assign CUDA_VISIBLE_DEVICES (for any local operations).",
    )
    p.add_argument(
        "--terminal-predictor-dir", type=str, default="pomdp_data",
        help="Directory to save terminal predictor pickles/plots.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the full transition-parameter computation pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    args = _parse_args()

    # Seed all RNGs for reproducibility
    seed_everything(args.seed)
    logger.info("Random seed set to %d", args.seed)

    if args.gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        logger.info("Set CUDA_VISIBLE_DEVICES=%s", args.gpus)

    # ── PRM tokenizer (shared across models and splits) ─────────────────────
    logger.info("Loading PRM tokenizer: %s", args.prm_model_name)
    prm_tokenizer = AutoTokenizer.from_pretrained(
        args.prm_model_name, trust_remote_code=True,
    )
    if prm_tokenizer.pad_token_id is None:
        prm_tokenizer.pad_token = prm_tokenizer.eos_token
    prm_tokenizer.padding_side = "right"

    prm = ServerPRM(
        server_url=args.prm_server_url,
        model_name=args.prm_model_name,
        tokenizer=prm_tokenizer,
    )

    # ── Tokenizers for generation models ────────────────────────────────────
    logger.info("Loading draft tokenizer: %s", args.draft_model_name)
    draft_tokenizer = AutoTokenizer.from_pretrained(
        args.draft_model_name, trust_remote_code=True,
    )

    logger.info("Loading target tokenizer: %s", args.target_model_name)
    target_tokenizer = AutoTokenizer.from_pretrained(
        args.target_model_name, trust_remote_code=True,
    )

    benchmark = args.train_benchmark
    split = args.train_split
    questions, ground_truths = load_questions_and_ground_truths(
        data_dir=args.data_dir,
        benchmark=benchmark,
        split=split,
    )

    logger.info(
        "Benchmark='%s', split='%s', problems=%d",
        benchmark,
        split,
        len(questions),
    )

    # ── Draft model ──────────────────────────────────────────────────────
    logger.info(
        "=== Draft model: %s | split: %s ===", args.draft_model_name, split
    )
    draft_steps = compute_transition_params(
        benchmark=benchmark,
        split=split,
        model_name=args.draft_model_name,
        server_url=args.draft_server_url,
        output_dir=args.output_dir,
        questions=questions,
        ground_truths=ground_truths,
        prm=prm,
        tokenizer=draft_tokenizer,
        thr=args.thr,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_steps=args.max_steps,
        rerun=args.rerun,
    )

    # ── Target model ─────────────────────────────────────────────────────
    logger.info(
        "=== Target model: %s | split: %s ===", args.target_model_name, split
    )
    target_steps = compute_transition_params(
        benchmark=benchmark,
        split=split,
        model_name=args.target_model_name,
        server_url=args.target_server_url,
        output_dir=args.output_dir,
        questions=questions,
        ground_truths=ground_truths,
        prm=prm,
        tokenizer=target_tokenizer,
        thr=args.thr,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_steps=args.max_steps,
        rerun=args.rerun,
    )

    save_terminal_predictors(
        benchmark=benchmark,
        split=split,
        predictor_dir=args.terminal_predictor_dir,
        slm_steps=draft_steps,
        llm_steps=target_steps,
        max_steps=args.max_steps,
    )

    logger.info("All splits complete.")


if __name__ == "__main__":
    main()
