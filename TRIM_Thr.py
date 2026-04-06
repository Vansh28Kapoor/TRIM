"""
TRIM-Thr: Threshold-based Routing for TRIM.

At each reasoning step the draft model M_w generates a step.  If the PRM
score for that step falls *below* a fixed threshold the step is discarded and
the target model M_s regenerates it; otherwise the draft step is kept.

Special cases (no PRM evaluation required):
  * threshold = 0  → always accept draft  (draft-only baseline)
  * threshold = 1  → always reject draft  (target-only baseline)

Evaluation is run for every threshold in ``--thresholds`` and results are
saved as individual JSONL files plus a combined summary.

Prerequisites
-------------
Start vLLM servers for draft, target, and (when 0 < threshold < 1) PRM::

    source scripts/launch_servers.sh

Usage
-----
    python TRIM_Thr.py \\
        --eval_dataset_name math500 \\
        --eval_split test \\
        --thresholds 0,0.1,0.3,0.5,0.7,0.9,1
"""

from __future__ import annotations

import gc
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

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
# Configuration
# ---------------------------------------------------------------------------

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

    # --- Thresholds ---
    thresholds: str = "0,0.1,0.3,0.5,0.7,0.9,1"

    # --- I/O ---
    batch_size: int = 32
    output_dir: str = "./thr_results"
    seed: int = 10


def _str2bool(v: str) -> bool:
    return v.lower() in ("true", "1", "yes")


def _parse_thresholds(s: str) -> list[float]:
    return sorted(float(x.strip()) for x in s.split(","))


def parse_args() -> Config:
    import argparse
    parser = argparse.ArgumentParser(description="TRIM-Thr: threshold-based routing evaluation")
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
# Rollout
# ---------------------------------------------------------------------------

def run_rollout(
    cfg: Config,
    threshold: float,
    draft_client: OpenAI,
    target_client: OpenAI,
    draft_tokenizer: AutoTokenizer,
    target_tokenizer: AutoTokenizer,
    prm: Optional[ServerPRM],
    questions: list[str],
    question_indices: list[int],
) -> dict:
    """Threshold-based rollout for a batch of questions.

    Routing rule (per step, per sample):
        PRM score of current draft step < threshold  →  use target model
        threshold == 0  →  always accept draft  (prm is None)
        threshold == 1  →  always reject draft  (prm is None)
    """
    always_draft  = threshold == 0.0
    always_target = threshold == 1.0

    B = len(question_indices)
    batch_map = {qidx: i for i, qidx in enumerate(question_indices)}
    step_histories: dict[int, list[str]] = {qidx: [] for qidx in question_indices}

    draft_tokens           = [0] * B
    target_tokens          = [0] * B
    discarded_draft_tokens = [0] * B
    target_calls           = [0] * B

    ans_list: list[tuple[int, str]] = [(qidx, "") for qidx in question_indices]
    completed_list: list[tuple[int, str]] = []

    if prm is not None:
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

        # ── 1. Draft model generates one step (skipped for always_target) ──
        if not always_target:
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
                    # logger.info(
                    #     "Step %d: degenerate draft for q%d, continued (%d tokens).",
                    #     step, ans_list[j][0], draft_tok_counts[j],
                    # )

            for (qidx, _), text in zip(ans_list, draft_texts):
                step_histories[qidx].append(text)
        else:
            # Placeholder entries; overwritten by target output below
            draft_texts      = [""] * len(ans_list)
            draft_finished   = torch.zeros(len(ans_list), dtype=torch.bool)
            draft_tok_counts = [0] * len(ans_list)
            for qidx, _ in ans_list:
                step_histories[qidx].append("")

        # ── 2. PRM scoring (skipped for threshold 0 or 1) ─────────────────
        if not always_draft and not always_target and prm is not None:
            _, prm_tensor = prm.batch_score(questions, step_histories, active_qidx)
            current_scores = prm_tensor[:, -1].cpu().tolist()
            use_target = [score < threshold for score in current_scores]
        elif always_target:
            use_target = [True] * len(ans_list)
        else:
            use_target = [False] * len(ans_list)

        # ── 3. Replace rejected draft steps with target model output ───────
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
                    # logger.info(
                    #     "Step %d: degenerate target for q%d, continued (%d tokens).",
                    #     step, target_ans_list[j][0], tgt_tok_counts[j],
                    # )

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
                    if prm is not None:
                        prm.reset([qidx])

                    draft_texts[i]      = tgt_text
                    draft_finished[i]   = tgt_fin
                    draft_tok_counts[i] = tgt_tc

        # Update draft token counts for accepted steps
        for i in range(len(ans_list)):
            if not use_target[i]:
                draft_tokens[batch_idx[i]] += draft_tok_counts[i]

        # ── 4. Advance answers, check termination ──────────────────────────
        ans_list, newly_completed = update_partial_answers(
            ans_list, draft_texts, draft_finished,
        )
        completed_list.extend(newly_completed)

        if prm is not None:
            prm.reset([idx for idx, _ in newly_completed])

        # logger.info(
        #     "thr=%.2f | step %d: active=%d, completed=%d/%d, target_calls=%d",
        #     threshold, step, len(ans_list), len(completed_list), B, sum(use_target),
        # )

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
# Evaluation for a single threshold
# ---------------------------------------------------------------------------

def evaluate(
    cfg: Config,
    threshold: float,
    draft_client: OpenAI,
    target_client: OpenAI,
    draft_tokenizer: AutoTokenizer,
    target_tokenizer: AutoTokenizer,
    prm: Optional[ServerPRM],
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
            cfg, threshold,
            draft_client, target_client,
            draft_tokenizer, target_tokenizer,
            prm, questions, batch_qidx,
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
        "threshold":                               threshold,
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

    thresholds = _parse_thresholds(cfg.thresholds)
    logger.info("Thresholds to evaluate: %s", thresholds)

    needs_prm = any(0.0 < thr < 1.0 for thr in thresholds)

    draft_client  = OpenAI(api_key="EMPTY", base_url=cfg.draft_server_url)
    target_client = OpenAI(api_key="EMPTY", base_url=cfg.target_server_url)

    draft_tokenizer  = AutoTokenizer.from_pretrained(cfg.draft_model_name,  trust_remote_code=True)
    target_tokenizer = AutoTokenizer.from_pretrained(cfg.target_model_name, trust_remote_code=True)

    prm: Optional[ServerPRM] = None
    if needs_prm:
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

    questions, ground_truths = load_math_dataset(cfg, cfg.eval_dataset_name, cfg.eval_split)

    os.makedirs(cfg.output_dir, exist_ok=True)
    summary_path = Path(cfg.output_dir) / "summary.jsonl"

    with open(summary_path, "w") as summary_f:
        for thr in thresholds:
            logger.info("=== Evaluating threshold=%.2f ===", thr)
            metrics = evaluate(
                cfg, thr,
                draft_client, target_client,
                draft_tokenizer, target_tokenizer,
                prm, questions, ground_truths,
            )

            thr_tag  = f"{thr:.2f}".replace(".", "_")
            out_path = Path(cfg.output_dir) / f"thr_{thr_tag}_metrics.jsonl"
            with open(out_path, "w") as f:
                f.write(json.dumps(metrics) + "\n")

            summary_f.write(json.dumps(metrics) + "\n")
            summary_f.flush()

            logger.info(
                "thr=%.2f  acc=%.4f  avg_target_tokens/q=%.1f  "
                "draft_frac=%.3f  target_frac=%.3f  tgt_calls/q=%.2f",
                thr,
                metrics["accuracy"],
                metrics["avg_target_tokens_per_question"],
                metrics["draft_token_frac"],
                metrics["target_token_frac"],
                metrics["avg_target_calls_per_question"],
            )

    logger.info("Summary saved → %s", summary_path)


if __name__ == "__main__":
    main()
