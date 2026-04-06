"""
TRIM-POMDP: POMDP-based Routing for TRIM.

At each reasoning step the draft model M_w generates a step.  PRM scores are
used to compute a 2-D observation (min_score, current_step_prob), which is
fed through the reflected-KDE observation model to obtain a Bayesian belief
over the three POMDP correctness states (correct so far, irrecoverably incorrect due to an earlier error, and recoverable with only the latest step incorrect).
When the belief is close enough to uncertain (controlled by ``closeness_thr``),
the precomputed POMDP action table is consulted; otherwise the draft step is
kept (action=0).

Evaluation is run for every closeness threshold in ``--closeness_thresholds``
and results are saved as individual JSONL files plus a combined summary.

Prerequisites
-------------
Start vLLM servers for draft, target, and PRM::

    source scripts/launch_servers.sh

Usage
-----
    python TRIM_POMDP.py \\
        --eval_dataset_name math500 \\
        --eval_split test \\
        --closeness_thresholds 0.35,0.4,0.5 \\
        --action_table_dir pomdp_data
"""

from __future__ import annotations

import gc
import json
import logging
import math
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from datasets import load_dataset
from openai import OpenAI
from transformers import AutoTokenizer

from math_eval.math_equal import math_equal
from math_eval.parser import parse_ground_truth, strip_string, extract_answer, STRIP_EXCEPTIONS
from utils import (
    seed_everything,
    _is_degenerate,
    generate_steps,
    ServerPRM,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Belief snapping
# ---------------------------------------------------------------------------

def get_closest_belief(belief, grid_step, tol=1e-9):
    """Snap a belief triple to the nearest point on the discretised simplex."""
    p1, p2, p3 = belief

    if abs((p1 + p2 + p3) - 1.0) > tol:
        raise ValueError("belief must sum to 1")
    if min(p1, p2, p3) < -tol:
        raise ValueError("belief entries must be nonnegative")

    N = round(1.0 / grid_step)
    if abs(N * grid_step - 1.0) > tol:
        raise ValueError("1 must be divisible by grid_step")

    def snap_prob(p):
        k = p * N
        k_round = round(k)
        on_grid = abs(k - k_round) < tol
        k_floor = math.floor(k + tol)
        k_ceil = math.ceil(k - tol)
        return on_grid, k_round, k_floor, k_ceil

    p1_on_grid, p1_round, p1_floor_idx, p1_ceil_idx = snap_prob(p1)
    p2_on_grid, p2_round, p2_floor_idx, p2_ceil_idx = snap_prob(p2)
    p3_on_grid, _, _, _ = snap_prob(p3)

    if p1_on_grid or p2_on_grid or p3_on_grid:
        p1_closest_idx = p1_round
        p2_closest_idx = p2_round
        p3_closest_idx = N - p1_closest_idx - p2_closest_idx
    else:
        p1_delta_floor = p1 - p1_floor_idx * grid_step
        p1_delta_ceil  = p1_ceil_idx * grid_step - p1
        p2_delta_floor = p2 - p2_floor_idx * grid_step
        p2_delta_ceil  = p2_ceil_idx * grid_step - p2

        if min(p1_delta_floor, p2_delta_floor) + 2 * max(p1_delta_floor, p2_delta_floor) <= grid_step + tol:
            p1_closest_idx = p1_floor_idx
            p2_closest_idx = p2_floor_idx
        elif min(p1_delta_ceil, p2_delta_ceil) + 2 * max(p1_delta_ceil, p2_delta_ceil) <= grid_step + tol:
            p1_closest_idx = p1_ceil_idx
            p2_closest_idx = p2_ceil_idx
        else:
            if p1_delta_floor <= p2_delta_floor + tol:
                p1_closest_idx = p1_floor_idx
                p2_closest_idx = p2_ceil_idx
            else:
                p1_closest_idx = p1_ceil_idx
                p2_closest_idx = p2_floor_idx

        p3_closest_idx = N - p1_closest_idx - p2_closest_idx

    if p3_closest_idx < 0 or p3_closest_idx > N:
        raise RuntimeError("computed belief is outside simplex grid")

    result = (
        p1_closest_idx / N,
        p2_closest_idx / N,
        p3_closest_idx / N,
    )

    assert abs(sum(result) - 1.0) < tol
    return result


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

POMDP_DATA_DIR = Path(__file__).resolve().parent / "pomdp_data"


@dataclass
class Config:
    # --- LLM server endpoints ---
    draft_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    draft_server_url: str = "http://localhost:30001/v1"
    target_model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    target_server_url: str = "http://localhost:30000/v1"

    # --- PRM ---
    prm_model_name: str = "Qwen/Qwen2.5-Math-PRM-7B"
    prm_server_url: str = "http://localhost:30002"

    # --- Dataset ---
    eval_dataset_name: str = "math500"
    data_dir: str = "math_eval/data"
    eval_split: str = "test"

    # --- Generation ---
    step_separator: str = "\n\n"
    max_tokens_per_step: int = 2048
    max_steps: int = 30
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: Optional[int] = 20
    target_disable_thinking: bool = False  # Set True for thinking targets (Qwen3-8B) to avoid generating <think> tokens when M_w is non-thinking.
    max_workers: int = 32

    # --- POMDP routing ---
    closeness_thresholds: str = "0.4"
    cost_per_tokens: str = "0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75"      # comma-separated list, e.g. "0.25,0.75"
    task_reward: float = 1e2
    table_closeness_thr: float = 0.5   # cthr used when training the action table
    action_table_dir: str = "pomdp_data"
    obs_model_name: str = "math-train"

    # --- I/O ---
    batch_size: int = 32
    output_dir: str = "./pomdp_results"
    seed: int = 10


def _str2bool(v: str) -> bool:
    return v.lower() in ("true", "1", "yes")


def _parse_thresholds(s: str) -> list[float]:
    return sorted(float(x.strip()) for x in s.split(","))


def _parse_costs(s: str) -> list[float]:
    return sorted(float(x.strip()) for x in s.split(","))


def parse_args() -> Config:
    import argparse
    parser = argparse.ArgumentParser(description="TRIM-POMDP: POMDP-based routing evaluation")
    cfg = Config()
    for fld in cfg.__dataclass_fields__.values():
        name = f"--{fld.name}"
        if isinstance(fld.default, bool):
            parser.add_argument(name, type=_str2bool, default=fld.default)
        elif fld.default is None:
            parser.add_argument(name, type=str, default=fld.default)
        else:
            parser.add_argument(name, type=type(fld.default), default=fld.default)
    args = parser.parse_args()
    for key in cfg.__dataclass_fields__:
        setattr(cfg, key, getattr(args, key))
    return cfg


# ---------------------------------------------------------------------------
# Dataset loading + answer evaluation
# ---------------------------------------------------------------------------

def load_math_dataset(cfg: Config, dataset_name: str, split: str) -> tuple[list[str], list[str]]:
    data_path = os.path.join(cfg.data_dir, dataset_name, f"{split}.jsonl")
    if os.path.exists(data_path):
        dataset = load_dataset("json", data_files=data_path, split="train")
    else:
        raise FileNotFoundError(f"No data at {data_path}.")
    q_key = "question" if "question" in dataset.column_names else "problem"
    questions = list(dataset[q_key])
    ground_truths = [parse_ground_truth(sample, dataset_name)[1] for sample in dataset]
    return questions, ground_truths


def extract_prediction(text: str, benchmark_name: str) -> str:
    return strip_string(
        extract_answer(text, benchmark_name),
        skip_unit=benchmark_name in STRIP_EXCEPTIONS,
    )


def update_partial_answers(
    ans_list: list[tuple[int, str]],
    gen_outputs: list[str],
    finished_flags: torch.Tensor,
) -> tuple[list[tuple[int, str]], list[tuple[int, str]]]:
    active, completed = [], []
    for (idx, partial), gen, done in zip(ans_list, gen_outputs, finished_flags):
        full = (partial + "\n\n" + gen) if partial else gen
        if done:
            completed.append((idx, full))
        else:
            active.append((idx, full))
    return active, completed


# ---------------------------------------------------------------------------
# Observation model + action table loading
# ---------------------------------------------------------------------------

def load_observation_model(obs_model_name: str):
    """Load the reflected-KDE observation model from disk."""
    path = POMDP_DATA_DIR / f"{obs_model_name}_reflected_kde_obs_model.pkl"
    with open(path, "rb") as f:
        model = pickle.load(f)
    logger.info("Loaded observation model from %s", path)
    return model


def load_action_table(path: str | Path) -> dict:
    """Load a precomputed POMDP action table from disk."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    logger.info("Loaded action table from %s (%d entries)", path, len(data["table"]))
    return data


def find_action_table(
    action_table_dir: str,
    table_closeness_thr: float,
    cost_per_token: float,
    task_reward: float,
) -> Path:
    """Locate the precomputed action table.

    The filename convention (set by ``get_pomdp_policy.py``) is::

        pomdp_action_table_cost{ratio}_cthr{thr}.pkl

    where ``ratio = round(cost_per_token / task_reward, 6)`` and
    ``thr`` is the closeness threshold used *during training*.  A table
    trained with ``table_closeness_thr`` is valid for any eval threshold
    ``<= table_closeness_thr``.
    """
    cost_ratio = round(cost_per_token / task_reward, 6)
    fname = f"pomdp_action_table_cost{cost_ratio}_cthr{table_closeness_thr}.pkl"
    path = Path(action_table_dir) / fname
    if not path.exists():
        raise FileNotFoundError(
            f"Action table not found: {path}. "
            f"Run get_pomdp_policy.py --cost-per-token {cost_per_token} "
            f"--task-reward {task_reward} --closeness-thr {table_closeness_thr} first."
        )
    return path


# ---------------------------------------------------------------------------
# Belief computation from PRM observations
# ---------------------------------------------------------------------------

def compute_belief_from_prm(
    prm_scores: torch.Tensor,
    observation_model,
) -> np.ndarray:
    """Compute belief over POMDP states from PRM scores via observation model.

    For each sample, extracts PRM observation features (min_score, current_step_prob)
    and computes the posterior belief P(state | observation) assuming a uniform prior.

    Parameters
    ----------
    prm_scores : Tensor, shape (B, num_steps)
        PRM scores for each sample across all steps so far.
    observation_model : ReflectedKDEObservationModel
        Observation model providing ``pdf(state, point)`` for states {0, 1, 2}.

    Returns
    -------
    belief : ndarray, shape (B, 3)
        Posterior belief over states [correct, error-propagated, first-error].
    """
    scores = prm_scores.cpu().numpy()
    B = scores.shape[0]
    belief = np.zeros((B, 3), dtype=np.float64)

    for i in range(B):
        row = scores[i]
        num_steps = row.shape[0]

        # Extract observation features: (min_prev_scores, current_score)
        if num_steps > 1:
            min_score = float(row[:-1].min())
        else:
            min_score = 1.0
        current_score = float(row[-1])
        obs = (min_score, current_score)

        # P(obs | state) for each state
        likelihoods = np.array([
            observation_model.pdf(state, obs) for state in range(3)
        ])

        total = likelihoods.sum()
        if total > 0:
            belief[i] = likelihoods / total
        else:
            belief[i] = np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])

    return belief


def lookup_pomdp_actions(
    belief: np.ndarray,
    table_data: dict,
    token_counts_batch: list[int],
    step_no: int,
    closeness_thr: float,
) -> np.ndarray:
    """Look up POMDP actions from the precomputed table.

    Parameters
    ----------
    belief : ndarray, shape (B, 3)
    table_data : dict from load_action_table
    token_counts_batch : list of int, length B
    step_no : int
    closeness_thr : float

    Returns
    -------
    actions : ndarray of int, shape (B,)
        0 = accept draft, 1 = use target.
    """
    table = table_data["table"]
    cfg = table_data["config"]
    token_bins = np.array(cfg["token_counts"])
    belief_step = cfg["belief_step"]
    max_steps = cfg["max_steps"]

    B = belief.shape[0]
    actions = np.zeros(B, dtype=int)

    for i in range(B):
        p0, p1, p2 = belief[i]
        max_b = max(p0, p1, p2)

        # Closeness check: only consult table only if probability of being in S2 is greater than max_b - closeness_thr
        if max_b - p2 >= closeness_thr:
            actions[i] = 0  # keep draft
            continue

        if step_no >= max_steps:
            actions[i] = 0
            continue

        # Snap belief to grid
        try:
            sp0, sp1, sp2 = get_closest_belief((p0, p1, p2), belief_step)
        except (ValueError, RuntimeError):
            actions[i] = 0
            continue

        # Find closest token count bin
        tc = int(token_bins[np.argmin(np.abs(token_bins - token_counts_batch[i]))])

        key = (tc, step_no, sp0, sp1, sp2)
        actions[i] = table.get(key, 0)

    return actions


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------

def run_rollout(
    cfg: Config,
    closeness_thr: float,
    draft_client: OpenAI,
    target_client: OpenAI,
    draft_tokenizer: AutoTokenizer,
    target_tokenizer: AutoTokenizer,
    prm: ServerPRM,
    observation_model,
    table_data: dict,
    questions: list[str],
    question_indices: list[int],
) -> dict:
    """POMDP-based rollout for a batch of questions.

    Routing rule (per step, per sample):
        1. Draft model generates a reasoning step.
        2. PRM scores the step; observation features are extracted.
        3. Belief is computed via the observation model (Bayesian update).
        4. If max(belief) - belief[2] <= closeness_thr, consult the
           precomputed action table; otherwise keep draft (action=0).
        5. If action=1, target model regenerates the step.
    """
    B = len(question_indices)
    batch_map = {qidx: i for i, qidx in enumerate(question_indices)}
    step_histories: dict[int, list[str]] = {qidx: [] for qidx in question_indices}

    draft_tokens           = [0] * B
    target_tokens          = [0] * B
    discarded_draft_tokens = [0] * B
    target_calls           = [0] * B

    ans_list: list[tuple[int, str]] = [(qidx, "") for qidx in question_indices]
    completed_list: list[tuple[int, str]] = []

    prm.reset(question_indices)

    gen_kwargs = dict(
        max_tokens=cfg.max_tokens_per_step,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        top_k=cfg.top_k,
        stop=[cfg.step_separator],
    )

    for step in range(cfg.max_steps):
        if not ans_list:
            break

        active_qidx = [idx for idx, _ in ans_list]
        batch_idx   = [batch_map[idx] for idx in active_qidx]

        # ── 1. Draft model generates one step ──────────────────────────
        draft_texts, draft_finished, draft_tok_counts = generate_steps(
            draft_client, cfg.draft_model_name, draft_tokenizer,
            questions, ans_list, **gen_kwargs,
        )

        # Handle degenerate draft outputs
        degen_draft = [
            j for j, t in enumerate(draft_texts)
            if _is_degenerate(t) and not draft_finished[j]
        ]
        if degen_draft:
            cont_ans = [
                (ans_list[j][0], (ans_list[j][1] + "\n\n" + draft_texts[j]).strip())
                for j in degen_draft
            ]
            dc_texts, dc_fin, dc_tc = generate_steps(
                draft_client, cfg.draft_model_name, draft_tokenizer,
                questions, cont_ans, **gen_kwargs,
            )
            for k, j in enumerate(degen_draft):
                degen_tc = draft_tok_counts[j]
                draft_texts[j] = dc_texts[k]
                draft_finished[j] = dc_fin[k]
                draft_tok_counts[j] = degen_tc + dc_tc[k]

        for (qidx, _), text in zip(ans_list, draft_texts):
            step_histories[qidx].append(text)

        # ── 2. Determine routing action ────────────────────────────────
        # PRM scoring
        _, prm_tensor = prm.batch_score(questions, step_histories, active_qidx)

        # Compute belief from PRM observations
        belief = compute_belief_from_prm(prm_tensor, observation_model)

        # Look up POMDP actions
        actions = lookup_pomdp_actions(
            belief, table_data, draft_tok_counts, step, closeness_thr,
        )
        use_target = [bool(a) for a in actions]

        # ── 3. Replace rejected draft steps with target model output ───
        target_ans_list = [ans_list[i] for i in range(len(ans_list)) if use_target[i]]
        if target_ans_list:
            tgt_texts, tgt_finished, tgt_tok_counts = generate_steps(
                target_client, cfg.target_model_name, target_tokenizer,
                questions, target_ans_list, **gen_kwargs,
                disable_thinking=cfg.target_disable_thinking,
            )

            # Handle degenerate target outputs
            degen_tgt = [
                j for j, t in enumerate(tgt_texts)
                if _is_degenerate(t) and not tgt_finished[j]
            ]
            if degen_tgt:
                cont_ans = [
                    (target_ans_list[j][0],
                     (target_ans_list[j][1] + "\n\n" + tgt_texts[j]).strip())
                    for j in degen_tgt
                ]
                c_texts, c_fin, c_tc = generate_steps(
                    target_client, cfg.target_model_name, target_tokenizer,
                    questions, cont_ans, **gen_kwargs,
                    disable_thinking=cfg.target_disable_thinking,
                )
                for k, j in enumerate(degen_tgt):
                    degen_tc = tgt_tok_counts[j]
                    tgt_texts[j] = c_texts[k]
                    tgt_finished[j] = c_fin[k]
                    tgt_tok_counts[j] = degen_tc + c_tc[k]

            tgt_iter = iter(zip(tgt_texts, tgt_finished, tgt_tok_counts))
            for i in range(len(ans_list)):
                if use_target[i]:
                    tgt_text, tgt_fin, tgt_tc = next(tgt_iter)
                    qidx = active_qidx[i]
                    bi   = batch_idx[i]

                    discarded_draft_tokens[bi] += draft_tok_counts[i]
                    target_tokens[bi]          += tgt_tc
                    target_calls[bi]           += 1

                    step_histories[qidx][-1] = tgt_text
                    prm.reset([qidx])

                    draft_texts[i]      = tgt_text
                    draft_finished[i]   = tgt_fin
                    draft_tok_counts[i] = tgt_tc

        # Update draft token counts for accepted steps
        for i in range(len(ans_list)):
            if not use_target[i]:
                draft_tokens[batch_idx[i]] += draft_tok_counts[i]

        # ── 4. Advance answers, check termination ──────────────────────
        ans_list, newly_completed = update_partial_answers(
            ans_list, draft_texts, draft_finished,
        )
        completed_list.extend(newly_completed)

        prm.reset([idx for idx, _ in newly_completed])

    completed_list.extend(ans_list)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "completed":               completed_list,
        "draft_tokens":            draft_tokens,
        "target_tokens":           target_tokens,
        "discarded_draft_tokens":  discarded_draft_tokens,
        "target_calls":            target_calls,
    }


# ---------------------------------------------------------------------------
# Evaluation for a single closeness threshold
# ---------------------------------------------------------------------------

def evaluate(
    cfg: Config,
    closeness_thr: float,
    cost_per_token: float,
    draft_client: OpenAI,
    target_client: OpenAI,
    draft_tokenizer: AutoTokenizer,
    target_tokenizer: AutoTokenizer,
    prm: ServerPRM,
    observation_model,
    table_data: dict,
    questions: list[str],
    ground_truths: list[str],
) -> dict:
    all_correct     = 0
    total_draft     = 0
    total_target    = 0
    total_discarded = 0
    total_tgt_calls = 0
    n = len(questions)

    for batch_start in range(0, n, cfg.batch_size):
        batch_end  = min(batch_start + cfg.batch_size, n)
        batch_qidx = list(range(batch_start, batch_end))

        rollout = run_rollout(
            cfg, closeness_thr,
            draft_client, target_client,
            draft_tokenizer, target_tokenizer,
            prm, observation_model, table_data,
            questions, batch_qidx,
        )

        for qidx, answer_text in rollout["completed"]:
            pred = extract_prediction(answer_text, cfg.eval_dataset_name)
            if math_equal(pred, ground_truths[qidx]):
                all_correct += 1

        total_draft     += sum(rollout["draft_tokens"])
        total_target    += sum(rollout["target_tokens"])
        total_discarded += sum(rollout["discarded_draft_tokens"])
        total_tgt_calls += sum(rollout["target_calls"])

    total_tokens      = total_draft + total_target + total_discarded
    avg_target_tokens = total_target / max(n, 1)
    accuracy          = all_correct / max(n, 1)

    return {
        "closeness_thr":                           closeness_thr,
        "cost_per_token":                          cost_per_token,
        "accuracy":                                accuracy,
        "num_correct":                             all_correct,
        "avg_total_tokens_per_question":           total_tokens    / max(n, 1),
        "avg_draft_tokens_per_question":           total_draft     / max(n, 1),
        "avg_target_tokens_per_question":          avg_target_tokens,
        "avg_discarded_draft_tokens_per_question": total_discarded / max(n, 1),
        "avg_target_calls_per_question":           total_tgt_calls / max(n, 1),
        "draft_token_frac":  (total_draft + total_discarded) / max(total_tokens, 1),
        "target_token_frac": total_target                    / max(total_tokens, 1),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    cfg = parse_args()
    seed_everything(cfg.seed)

    closeness_thresholds = _parse_thresholds(cfg.closeness_thresholds)
    cost_per_tokens      = _parse_costs(cfg.cost_per_tokens)
    logger.info("Closeness thresholds to evaluate: %s", closeness_thresholds)
    logger.info("Cost-per-token values to evaluate: %s", cost_per_tokens)

    draft_client  = OpenAI(api_key="EMPTY", base_url=cfg.draft_server_url)
    target_client = OpenAI(api_key="EMPTY", base_url=cfg.target_server_url)

    draft_tokenizer  = AutoTokenizer.from_pretrained(cfg.draft_model_name,  trust_remote_code=True)
    target_tokenizer = AutoTokenizer.from_pretrained(cfg.target_model_name, trust_remote_code=True)

    prm_tokenizer = AutoTokenizer.from_pretrained(cfg.prm_model_name, trust_remote_code=True)
    if prm_tokenizer.pad_token_id is None:
        prm_tokenizer.pad_token = prm_tokenizer.eos_token
    prm_tokenizer.padding_side = "right"
    prm = ServerPRM(
        server_url=cfg.prm_server_url,
        model_name=cfg.prm_model_name,
        tokenizer=prm_tokenizer,
        max_workers=cfg.max_workers,
    )
    observation_model = load_observation_model(cfg.obs_model_name)

    questions, ground_truths = load_math_dataset(cfg, cfg.eval_dataset_name, cfg.eval_split)

    os.makedirs(cfg.output_dir, exist_ok=True)
    summary_path = Path(cfg.output_dir) / "summary.jsonl"

    with open(summary_path, "w") as summary_f:
        for cost in cost_per_tokens:
            cost_tag    = f"{cost:g}".replace(".", "_")
            cost_outdir = Path(cfg.output_dir) / f"cost_{cost_tag}"
            os.makedirs(cost_outdir, exist_ok=True)

            # Load action table once per cost; reused across all closeness thresholds.
            table_path = find_action_table(
                cfg.action_table_dir, cfg.table_closeness_thr,
                cost, cfg.task_reward,
            )
            table_data = load_action_table(table_path)

            for cthr in closeness_thresholds:
                logger.info(
                    "=== Evaluating cost_per_token=%.4g  closeness_thr=%.3f ===",
                    cost, cthr,
                )

                metrics = evaluate(
                    cfg, cthr, cost,
                    draft_client, target_client,
                    draft_tokenizer, target_tokenizer,
                    prm, observation_model,
                    table_data,
                    questions, ground_truths,
                )

                cthr_tag = f"{cthr:.2f}".replace(".", "_")
                out_path = cost_outdir / f"cthr_{cthr_tag}_metrics.jsonl"
                with open(out_path, "w") as f:
                    f.write(json.dumps(metrics) + "\n")

                summary_f.write(json.dumps(metrics) + "\n")
                summary_f.flush()

                logger.info(
                    "cost=%.4g  cthr=%.3f  acc=%.4f  avg_target_tokens/q=%.1f  "
                    "draft_frac=%.3f  target_frac=%.3f  tgt_calls/q=%.2f",
                    cost, cthr,
                    metrics["accuracy"],
                    metrics["avg_target_tokens_per_question"],
                    metrics["draft_token_frac"],
                    metrics["target_token_frac"],
                    metrics["avg_target_calls_per_question"],
                )

    logger.info("Summary saved → %s", summary_path)


if __name__ == "__main__":
    main()
