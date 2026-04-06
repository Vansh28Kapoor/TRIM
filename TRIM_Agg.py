"""
TRIM-Agg: Learned Routing via PPO for TRIM.

A lightweight PPO policy learns *when* to accept a draft (cheap) model M_w
reasoning step vs. falling back to a target (expensive) model M_s, using PRM
(Process Reward Model) signals as part of the state representation.

Both draft and target LLMs are accessed via vLLM-served OpenAI-compatible
endpoints (batched ``client.completions.create`` with chat-templated prompts).
The PRM (Qwen2.5-Math-PRM-7B) is also served via vLLM with automatic prefix
caching (APC), so only newly appended step tokens are processed per call.

Supported model pairs
---------------------
- Draft:  ``Qwen/Qwen2.5-1.5B-Instruct``
- Target: ``Qwen/Qwen2.5-7B-Instruct`` (default) **or**
          ``Qwen/Qwen3-8B`` (use with ``--target_disable_thinking true``)
- PRM:    ``Qwen/Qwen2.5-Math-PRM-7B``

Prerequisites
-------------
Start vLLM servers for draft, target, and PRM models.  On 2× A40-48 GB
the recommended memory layout is::

    # GPU 0: target 7B  (exclusive, 90%)  — ~14 GiB weights (bf16)
    # GPU 1: draft 1.5B (35%)             — ~3  GiB weights
    #         PRM  7B   (50%)             — ~14 GiB weights

    # Target (Qwen2.5-7B-Instruct) — GPU 0 exclusive
    CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen2.5-7B-Instruct \\
        --dtype auto --max-model-len 4096 \\
        --gpu-memory-utilization 0.90 --enable-prefix-caching --port 30000

    # Draft (Qwen2.5-1.5B-Instruct) — GPU 1, shared with PRM
    CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen2.5-1.5B-Instruct \\
        --dtype auto --max-model-len 4096 \\
        --gpu-memory-utilization 0.35 --enable-prefix-caching --port 30001

    # PRM — GPU 1, shared with draft
    CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen2.5-Math-PRM-7B \\
        --dtype auto --trust-remote-code \\
        --gpu-memory-utilization 0.50 --enable-prefix-caching --port 30002

Or use the convenience script::

    source scripts/launch_servers.sh

Usage
-----
Training (Qwen2.5-7B-Instruct target, default)::

    python TRIM_Agg.py \\
        --mode train \\
        --train_dataset_name math \\
        --eval_dataset_name math500 \\
        --eval_split test \\
        --num_epochs 10 \\
        --save_dir ./checkpoints

Training (Qwen3-8B target, override — disable thinking)::

    python TRIM_Agg.py \\
        --mode train \\
        --target_model_name Qwen/Qwen3-8B \\
        --target_disable_thinking true \\
        --train_dataset_name math \\
        --eval_dataset_name math500 \\
        --eval_split test \\
        --num_epochs 10 \\
        --save_dir ./checkpoints

Evaluation::

    python TRIM_Agg.py \\
        --mode eval \\
        --eval_dataset_name math500 \\
        --eval_split test \\
        --checkpoint ./checkpoints/policy_best.pt \\
        --output_dir ./results
"""

from __future__ import annotations

import argparse
import concurrent.futures
import gc
import json
import logging
import os
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.nn.utils.rnn import pad_sequence

from datasets import load_dataset
import httpx
from openai import OpenAI
from transformers import AutoTokenizer

from math_eval.math_equal import math_equal
from math_eval.parser import parse_ground_truth, strip_string, extract_answer, STRIP_EXCEPTIONS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Reproducibility + shared utilities (imported from utils.py)
# ---------------------------------------------------------------------------

from utils import (
    seed_everything,
    SYSTEM_PROMPT,
    format_prompt,
    _DEGENERATE_RE,
    _is_degenerate,
    _build_prompt,
    generate_steps,
    ServerPRM,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class Config:
    """Central configuration for the speculative PPO pipeline."""

    # --- LLM server endpoints (vLLM OpenAI-compatible) ---
    # Draft model = cheap model M_w, Target model = expensive model M_s.
    draft_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    draft_server_url: str = "http://localhost:30001/v1"
    target_model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    target_server_url: str = "http://localhost:30000/v1"

    # --- PRM (vLLM server with automatic prefix caching) ---
    # NOTE: PRM uses V0 engine in vLLM 0.8.x, which exposes /pooling
    # at the root, not under /v1/.  Do NOT append /v1 here.
    prm_model_name: str = "Qwen/Qwen2.5-Math-PRM-7B"
    prm_server_url: str = "http://localhost:30002"

    # --- Dataset ---
    train_dataset_name: str = "math"
    eval_dataset_name: str = "math500"
    data_dir: str = "math_eval/data"
    train_split: str = "train"
    eval_split: str = "test"

    # --- Generation ---
    step_separator: str = "\n\n"
    max_tokens_per_step: int = 2048
    max_steps: int = 30
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: Optional[int] = 20
    target_disable_thinking: bool = False  # Set True for thinking targets (Qwen3-8B) to avoid generating <think> tokens when M_w is non-thinking.
    max_workers: int = 32  # concurrent HTTP requests to vLLM servers

    # --- PPO hyper-parameters ---
    lr: float = 3e-4
    gamma: float = 1.0
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    entropy_coef_final: float = 0.001  # linearly anneal entropy bonus
    max_grad_norm: float = 0.5
    ppo_epochs: int = 3
    normalize_advantages: bool = True

    # --- Reward shaping ---
    task_reward: float = 1.0
    cost_per_token: float = 8e-4

    # --- Policy network ---
    state_dim: int = 5
    hidden_dim: int = 64
    num_actions: int = 2  # 0 = accept draft, 1 = use target
    dropout: float = 0.1

    # --- Training schedule ---
    num_epochs: int = 10
    batch_size: int = 64
    mini_batch_size: int = 64
    val_fraction: float = 0.1  # fraction of train data held out for validation
    val_every: int = 50  # validate every N batches
    eval_every: int = 50  # test-set evaluation every N batches (monitoring only)

    # --- I/O ---
    save_dir: str = "./rlpolicy_checkpoints"
    output_dir: str = "./rlpolicy_results"
    checkpoint: Optional[str] = None
    seed: int = 10

    # --- Device ---
    policy_device: str = "cuda:0"

    # --- WandB ---
    use_wandb: bool = True
    wandb_project: str = "speculative-ppo"
    wandb_run_name: Optional[str] = None

    # --- Resume ---
    resume: bool = False  # auto-detect and resume from latest checkpoint in save_dir

    # --- Resume ---
    resume: bool = False  # auto-detect and resume from latest training state in save_dir


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="TRIM-Agg routing with PPO")
    parser.add_argument("--mode", choices=["train", "eval"], default="train")

    def _str2bool(v: str) -> bool:
        return v.lower() in ("true", "1", "yes")

    # Populate parser from Config fields.
    # NOTE: ``from __future__ import annotations`` makes fld.type a string.
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
    # Transfer parsed values into Config
    for key in cfg.__dataclass_fields__:
        setattr(cfg, key, getattr(args, key))
    cfg.mode = args.mode
    return cfg


# ---------------------------------------------------------------------------
# Prompt formatting  (PPO-specific helper — shared symbols live in utils.py)
# ---------------------------------------------------------------------------


def format_cost_tag(cost: float) -> str:
    """Format cost in compact scientific notation (e.g. 8e-4)."""
    mantissa, exp = f"{cost:.15e}".split("e")
    mantissa = mantissa.rstrip("0").rstrip(".")
    return f"{mantissa}e{int(exp)}"


# ---------------------------------------------------------------------------
# Policy network
# ---------------------------------------------------------------------------

class RoutingPolicy(nn.Module):
    """Lightweight actor-critic for step-level routing decisions.

    State space (dim=5):
        0. Product of PRM probabilities up to step t-1
        1. PRM probability at step t
        2. Current token length / normaliser
        3. Step number t / max_steps
        4. Minimum PRM probability up to step t-1

    Actions:
        0 = accept draft step from M_w
        1 = reject draft and use target model M_s
    """

    def __init__(self, state_dim: int = 5, hidden_dim: int = 64, num_actions: int = 2, dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
        )
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_actions),
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self._init_weights()

    def _init_weights(self):
        """Orthogonal initialization (PPO best practice from CleanRL/OpenAI)."""
        for module in self.encoder:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.zeros_(module.bias)
        for module in self.actor:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=0.01)
                nn.init.zeros_(module.bias)
        for module in self.critic:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                nn.init.zeros_(module.bias)

    def forward(self, state: torch.Tensor, action: Optional[torch.Tensor] = None):
        h = self.encoder(state)
        logits = self.actor(h)
        dist = Categorical(logits=logits)

        if action is None:
            action = dist.sample()

        return (
            action,
            dist.log_prob(action),
            dist.entropy(),
            self.critic(h).squeeze(-1),
            logits.argmax(dim=-1),
        )


# ---------------------------------------------------------------------------
# PPO trainer
# ---------------------------------------------------------------------------

class PPOTrainer:
    """PPO with clipped objective, GAE, and optional value-function clipping."""

    def __init__(self, policy: RoutingPolicy, cfg: Config, total_batches: int):
        self.policy = policy
        self.cfg = cfg
        self.device = next(policy.parameters()).device
        self.optimizer = optim.Adam(
            policy.parameters(), lr=cfg.lr, eps=1e-5, weight_decay=1e-5
        )
        self.lr_scheduler = optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=1.0,
            end_factor=0.0,
            total_iters=total_batches,
        )
        self.total_batches = total_batches
        self._update_count = 0

    # ----- GAE -----
    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        T, B = rewards.shape
        advantages = torch.zeros_like(rewards)
        values_ext = F.pad(values, (0, 0, 0, 1))  # (T+1, B)
        dones = ~F.pad(mask, (0, 0, 0, 1), value=False)[1:]
        gae = torch.zeros(B, device=rewards.device)
        for t in reversed(range(T)):
            not_done = (~dones[t]).to(rewards.dtype)
            delta = (
                rewards[t]
                + self.cfg.gamma * values_ext[t + 1] * not_done
                - values_ext[t]
            )
            gae = delta + self.cfg.gamma * self.cfg.gae_lambda * not_done * gae
            advantages[t] = gae
        returns = advantages + values_ext[:-1]
        return advantages, returns

    # ----- Vectorization -----
    def _vectorize(
        self,
        states: list[list[torch.Tensor]],
        actions: list[list[int]],
        logprobs: list[list[float]],
        values: list[list[float]],
        rewards: list[list[float]],
    ):
        num_turns = [len(s) for s in states]
        max_T = max(num_turns)
        B = len(states)

        mask = torch.ones(max_T, B, dtype=torch.bool, device=self.device)
        for i in range(B):
            mask[num_turns[i] :, i] = False

        padded_states = pad_sequence(
            [torch.stack(s) for s in states], batch_first=True
        )  # (B, max_T, D)
        padded_states = padded_states.transpose(0, 1).to(self.device)  # (max_T, B, D)

        def _pad(lst, pad_val):
            return [row + [pad_val] * (max_T - len(row)) for row in lst]

        actions_t = torch.tensor(list(zip(*_pad(actions, 0))), dtype=torch.long, device=self.device) # (T, B)
        logprobs_t = torch.tensor(list(zip(*_pad(logprobs, 0.0))), dtype=torch.float32, device=self.device)
        values_t = torch.tensor(list(zip(*_pad(values, 0.0))), dtype=torch.float32, device=self.device)
        rewards_t = torch.tensor(list(zip(*_pad(rewards, 0.0))), dtype=torch.float32, device=self.device)

        return mask, padded_states, actions_t, logprobs_t, values_t, rewards_t

    # ----- PPO update -----
    def update(
        self,
        states: list[list[torch.Tensor]],
        actions: list[list[int]],
        logprobs: list[list[float]],
        values: list[list[float]],
        rewards: list[list[float]],
    ) -> dict[str, float]:
        self.policy.train()
        mask, states_t, actions_t, logprobs_t, values_t, rewards_t = self._vectorize(
            states, actions, logprobs, values, rewards
        )
        advantages, returns = self.compute_gae(rewards_t, values_t, mask)
        T, B = actions_t.shape
        b_idx = np.arange(B)

        metrics = {"pg_loss": [], "value_loss": [], "entropy": [], "total_loss": [], "lr": [], "entropy_coef": []}

        # Linearly anneal entropy coefficient
        frac = min(self._update_count / max(self.total_batches, 1), 1.0)
        ent_coef = self.cfg.entropy_coef + frac * (self.cfg.entropy_coef_final - self.cfg.entropy_coef)

        # Normalize advantages across the entire batch (not per-minibatch)
        if self.cfg.normalize_advantages and mask.sum() > 1:
            adv_vals = advantages[mask]
            advantages[mask] = (adv_vals - adv_vals.mean()) / (adv_vals.std() + 1e-8)

        for _ in range(self.cfg.ppo_epochs):
            np.random.shuffle(b_idx)
            for mb_idx in np.array_split(b_idx, max(1, B // self.cfg.mini_batch_size)):
                # All time-steps, minibatch of examples
                mb_states = states_t[:, mb_idx]       # (T, mb, D)
                mb_actions = actions_t[:, mb_idx]      # (T, mb)
                mb_old_lp = logprobs_t[:, mb_idx]
                mb_old_val = values_t[:, mb_idx]
                mb_adv = advantages[:, mb_idx]
                mb_ret = returns[:, mb_idx]
                mb_mask = mask[:, mb_idx]

                # Flatten for a single forward pass
                flat_s = mb_states.reshape(-1, mb_states.shape[-1])
                flat_a = mb_actions.reshape(-1)
                _, new_lp, ent, new_val, _ = self.policy(flat_s, flat_a)
                new_lp = new_lp.view_as(mb_actions)
                ent = ent.view_as(mb_actions)
                new_val = new_val.view_as(mb_actions)

                ratio = torch.exp(new_lp - mb_old_lp)
                pg1 = -mb_adv * ratio
                pg2 = -mb_adv * torch.clamp(ratio, 1 - self.cfg.clip_coef, 1 + self.cfg.clip_coef)
                pg_loss = torch.max(pg1, pg2)[mb_mask].mean()

                # Clipped value loss
                v_unclipped = F.mse_loss(new_val, mb_ret, reduction="none")
                v_clipped_val = mb_old_val + torch.clamp(
                    new_val - mb_old_val,
                    -self.cfg.clip_coef * self.cfg.task_reward,
                    self.cfg.clip_coef * self.cfg.task_reward,
                )
                v_clipped = F.mse_loss(v_clipped_val, mb_ret, reduction="none")
                v_loss = 0.5 * torch.max(v_unclipped, v_clipped)[mb_mask].mean()

                ent_loss = ent[mb_mask].mean()
                loss = pg_loss + self.cfg.value_loss_coef * v_loss - ent_coef * ent_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()

                metrics["pg_loss"].append(pg_loss.item())
                metrics["value_loss"].append(v_loss.item())
                metrics["entropy"].append(ent_loss.item())
                metrics["total_loss"].append(loss.item())

        # Step LR scheduler and update count once per PPO update call
        self.lr_scheduler.step()
        self._update_count += 1
        metrics["lr"].append(self.optimizer.param_groups[0]["lr"])
        metrics["entropy_coef"].append(ent_coef)

        return {k: float(np.mean(v)) for k, v in metrics.items()}


# ---------------------------------------------------------------------------
# State observation builder  (mirrors right_ppo.py state space)
# ---------------------------------------------------------------------------

TOKEN_NORMALISER = 300


def build_observations(
    prm_scores: torch.Tensor,
    token_lengths: list[list[int]],
    batch_indices: list[int],
    step_num: int,
    max_steps: int,
) -> torch.Tensor:
    """Construct policy observation vectors.

    State features (per sample):
        0. min PRM score up to t-1
        1. product of PRM scores up to t-1
        2. PRM score at step t
        3. current step token length / TOKEN_NORMALISER
        4. step_num / max_steps
    """
    if prm_scores.size(1) > 1:
        prev = prm_scores[:, :-1]
        min_prev = prev.min(dim=1).values
        prod_prev = prev.prod(dim=1)
    else:
        n = prm_scores.size(0)
        min_prev = torch.ones(n, device=prm_scores.device)
        prod_prev = torch.ones(n, device=prm_scores.device)

    current_score = prm_scores[:, -1]
    tok_lens = torch.tensor(
        [token_lengths[i][-1] for i in batch_indices], dtype=torch.float32
    ) / TOKEN_NORMALISER
    step_frac = torch.full((len(batch_indices),), step_num / max_steps)

    obs = torch.stack([min_prev, prod_prev, current_score, tok_lens, step_frac], dim=-1)
    return obs


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_math_dataset(
    cfg: Config,
    dataset_name: str,
    split: str,
) -> tuple[list[str], list[str]]:
    """Load questions and ground-truth answers for a math benchmark.

    Returns (questions, ground_truths).
    """
    data_path = os.path.join(cfg.data_dir, dataset_name, f"{split}.jsonl")
    if os.path.exists(data_path):
        dataset = load_dataset("json", data_files=data_path, split="train")
    else:
        raise FileNotFoundError(
            f"No data at {data_path}. Please prepare the dataset according to the README instructions."
        )

    q_key = "question" if "question" in dataset.column_names else "problem"
    questions = list(dataset[q_key])

    ground_truths = [
        parse_ground_truth(sample, dataset_name)[1]
        for sample in dataset
    ]

    return questions, ground_truths


# ---------------------------------------------------------------------------
# Answer evaluation
# ---------------------------------------------------------------------------

def extract_prediction(text: str, benchmark_name: str) -> str:
    """Extract and normalise a prediction using math_eval utilities."""
    return strip_string(
        extract_answer(text, benchmark_name),
        skip_unit=benchmark_name in STRIP_EXCEPTIONS,
    )


# ---------------------------------------------------------------------------
# Update helpers (mirroring right_ppo.py logic)
# ---------------------------------------------------------------------------

def update_partial_answers(
    ans_list: list[tuple[int, str]],
    gen_outputs: list[str],
    finished_flags: torch.Tensor,
) -> tuple[list[tuple[int, str]], list[tuple[int, str]]]:
    """Split partial answers into still-active and completed."""
    active, completed = [], []
    for (idx, partial), gen, done in zip(ans_list, gen_outputs, finished_flags):
        full = (partial + "\n\n" + gen) if partial else gen
        if done:
            completed.append((idx, full))
        else:
            active.append((idx, full))
    return active, completed


# ---------------------------------------------------------------------------
# Main rollout logic
# ---------------------------------------------------------------------------

def run_rollout(
    cfg: Config,
    draft_client: OpenAI,
    target_client: OpenAI,
    draft_tokenizer: AutoTokenizer,
    target_tokenizer: AutoTokenizer,
    prm: ServerPRM,
    policy: RoutingPolicy,
    questions: list[str],
    question_indices: list[int],
    deterministic: bool = False,
) -> dict:
    """Run speculative reasoning rollout for a batch of questions.

    At each step:
            1. Draft model M_w generates a reasoning step (batched completions API).
      2. PRM scores all steps so far (vLLM server with prefix caching).
      3. Policy decides: accept draft step (action=0) or reject (action=1).
            4. For rejected steps, target model M_s generates a replacement step.
      5. Repeat until all prompts finish or budget is exhausted.

    Returns a dictionary with trajectories, rewards, and metrics.
    """
    policy.eval()  # disable dropout during trajectory collection
    device = next(policy.parameters()).device
    B = len(question_indices)

    # Per-sample bookkeeping
    batch_map = {qidx: i for i, qidx in enumerate(question_indices)}
    step_histories: dict[int, list[str]] = {qidx: [] for qidx in question_indices}
    token_lengths: list[list[int]] = [[] for _ in range(B)]

    # Trajectory buffers
    obs_buf: list[list[torch.Tensor]] = [[] for _ in range(B)]
    act_buf: list[list[int]] = [[] for _ in range(B)]
    lp_buf: list[list[float]] = [[] for _ in range(B)]
    val_buf: list[list[float]] = [[] for _ in range(B)]
    rew_buf: list[list[float]] = [[] for _ in range(B)]

    # Token accounting
    draft_tokens = [0] * B
    target_tokens = [0] * B
    discarded_draft_tokens = [0] * B
    target_calls = [0] * B

    ans_list = [(qidx, "") for qidx in question_indices] #list of partial answers, aligned with question_indices
    completed_list: list[tuple[int, str]] = []

    # Reset PRM caches for this batch
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
        batch_idx = [batch_map[idx] for idx in active_qidx]

        # --- 1. Draft model generates one step ---
        draft_texts, draft_finished, draft_tok_counts = generate_steps(
            draft_client, cfg.draft_model_name, draft_tokenizer,
            questions, ans_list, **gen_kwargs,
        )

        # Handle degenerate draft outputs (e.g. "---"):
        # Append the degenerate text to the partial answer so the draft
        # model sees full context, then ask it to continue.  The degenerate
        # step is spliced out afterward — only the continuation is kept.
        # Skip entries that are already finished — no continuation is possible.
        draft_degen_indices = [
            j for j, t in enumerate(draft_texts)
            if _is_degenerate(t) and not draft_finished[j]
        ]
        if draft_degen_indices:
            draft_cont_ans = [
                (ans_list[j][0],
                 (ans_list[j][1] + "\n\n" + draft_texts[j]).strip())
                for j in draft_degen_indices
            ]
            dc_texts, dc_fin, dc_tc = generate_steps(
                draft_client, cfg.draft_model_name, draft_tokenizer,
                questions, draft_cont_ans, **gen_kwargs,
            )
            for k, j in enumerate(draft_degen_indices):
                degen_draft_tc = draft_tok_counts[j]
                draft_texts[j] = dc_texts[k]
                draft_finished[j] = dc_fin[k]
                draft_tok_counts[j] = degen_draft_tc + dc_tc[k]
                # logger.info(
                #     "Step %d: degenerate draft output for q%d, "
                #     "continued → %d tokens (incl. %d degenerate).",
                #     step, ans_list[j][0], draft_tok_counts[j], degen_draft_tc,
                # )

        # Update step histories with draft outputs
        for (qidx, _), text in zip(ans_list, draft_texts):
            # Can also do: label = ("Solution: " + text) if not step_histories[qidx] else text
            label = text
            step_histories[qidx].append(label)

        for i, bi in enumerate(batch_idx):
            token_lengths[bi].append(draft_tok_counts[i])

        # --- 2. PRM scoring (KV-cached when enabled) ---
        _, prm_tensor = prm.batch_score(questions, step_histories, active_qidx)
        # for prnt_bindx, prnt_qindx in enumerate(active_qidx):
        #     print(f"Question: {questions[prnt_qindx]} \n Soln: {step_histories[prnt_qindx]} \n PRM Scores: {prm_tensor[prnt_bindx]} \n")

        # --- 3. Policy decision ---
        obs = build_observations(prm_tensor, token_lengths, batch_idx, step, cfg.max_steps)
        obs = obs.to(device)
        with torch.no_grad():
            action, log_prob, entropy, value, action_greedy = policy(obs)

        if deterministic:
            action = action_greedy

        # Record trajectory
        for i, bi in enumerate(batch_idx):
            obs_buf[bi].append(obs[i].cpu())
            act_buf[bi].append(action[i].item())
            lp_buf[bi].append(log_prob[i].item())
            val_buf[bi].append(value[i].item())

        use_target = action.cpu().bool()

        # --- 4. For rejected steps, replace with target model output ---
        target_ans_list = [ans_list[i] for i in range(len(ans_list)) if use_target[i]]
        if target_ans_list:
            tgt_texts, tgt_finished, tgt_tok_counts = generate_steps(
                target_client, cfg.target_model_name, target_tokenizer,
                questions, target_ans_list, **gen_kwargs,
                disable_thinking=cfg.target_disable_thinking,
            )

            # Handle degenerate target outputs (e.g. "---"):
            degen_indices = [
                j for j, t in enumerate(tgt_texts) if _is_degenerate(t) and not tgt_finished[j]
            ]
            if degen_indices:
                cont_ans = [
                    (target_ans_list[j][0],
                     (target_ans_list[j][1] + "\n\n" + tgt_texts[j]).strip())
                    for j in degen_indices
                ]
                c_texts, c_fin, c_tc = generate_steps(
                    target_client, cfg.target_model_name, target_tokenizer,
                    questions, cont_ans, **gen_kwargs,
                    disable_thinking=cfg.target_disable_thinking,
                )
                for k, j in enumerate(degen_indices):
                    # print(f"Continuation Target Text: {c_texts[k]}, Degenerate Text: {tgt_texts[j]}")
                    degen_tc = tgt_tok_counts[j]
                    tgt_texts[j] = c_texts[k]
                    tgt_finished[j] = c_fin[k]
                    tgt_tok_counts[j] = degen_tc + c_tc[k]
                    # logger.info(
                    #     "Step %d: degenerate target output for q%d, "
                    #     "continued → %d tokens (incl. %d degenerate).",
                    #     step, target_ans_list[j][0], tgt_tok_counts[j], degen_tc,
                    # )

            tgt_iter = iter(zip(tgt_texts, tgt_finished, tgt_tok_counts))
            for i in range(len(ans_list)):
                if use_target[i]:
                    tgt_text, tgt_fin, tgt_tc = next(tgt_iter)
                    qidx = active_qidx[i]
                    bi = batch_idx[i]

                    # Track discarded draft tokens
                    discarded_draft_tokens[bi] += draft_tok_counts[i]
                    target_tokens[bi] += tgt_tc

                    # Overwrite step history & token length
                    # label = ("Solution: " + tgt_text) if len(step_histories[qidx]) == 1 else tgt_text
                    label = tgt_text
                    step_histories[qidx][-1] = label
                    token_lengths[bi][-1] = tgt_tc

                    # Invalidate PRM cache for this problem (step was replaced)
                    prm.reset([qidx])

                    draft_texts[i] = tgt_text
                    draft_finished[i] = tgt_fin
                    # print(f"Target Model Text: {tgt_text}")

        # Update draft token counts for accepted steps
        for i in range(len(ans_list)):
            if not use_target[i]:
                draft_tokens[batch_idx[i]] += draft_tok_counts[i]
            else:
                target_calls[batch_idx[i]] += 1

        # --- 5. Cost reward for this step ---
        for i, bi in enumerate(batch_idx):
            if use_target[i]:
                rew_buf[bi].append(-cfg.cost_per_token * token_lengths[bi][-1])
            else:
                rew_buf[bi].append(0.0)

        # --- 6. Update answers and check termination ---
        ans_list, newly_completed = update_partial_answers(
            ans_list, draft_texts, draft_finished,
        )
        completed_list.extend(newly_completed)

        # Clean up PRM caches for completed problems
        prm.reset([idx for idx, _ in newly_completed])

        # logger.info(
        #     f"Step {step}: active={len(ans_list)}, completed={len(completed_list)}/{B}, "
        #     f"target_calls={int(use_target.sum())}"
        # )

        if not ans_list:
            break

    # Any remaining active prompts become completed with what they have
    completed_list.extend(ans_list)

    # One-time cleanup at the end of the rollout to reclaim GPU memory
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "completed": completed_list,
        "obs": obs_buf,
        "actions": act_buf,
        "logprobs": lp_buf,
        "values": val_buf,
        "rewards": rew_buf,
        "draft_tokens": draft_tokens,
        "target_tokens": target_tokens,
        "discarded_draft_tokens": discarded_draft_tokens,
        "target_calls": target_calls,
    }


# ---------------------------------------------------------------------------
# RNG fast-forward helper (for resuming from legacy policy-only checkpoints)
# ---------------------------------------------------------------------------

def _fastforward_numpy_rng(
    cfg: Config,
    n_train: int,
    global_batch: int,
) -> tuple[np.ndarray, int, int]:
    """Advance the numpy RNG past all calls made during training up to
    *global_batch*, returning the shuffled epoch indices for the current
    (in-progress) epoch, the ``batch_start`` to resume from within that epoch,
    and the epoch number.

    Caller is responsible for seeding numpy and consuming the train/val
    permutation BEFORE calling this function, so the RNG is already past
    that step.  This function replays only the epoch-level and PPO
    minibatch shuffles:

      Per completed epoch : 1× np.random.shuffle(epoch_arr)
                          + batches_per_epoch × ppo_epochs × np.random.shuffle(b_idx)
      Current epoch       : 1× np.random.shuffle(epoch_arr)
                          + completed_batches × ppo_epochs × np.random.shuffle(b_idx)
    """
    batches_per_epoch = max(1, (n_train + cfg.batch_size - 1) // cfg.batch_size)
    completed_epochs = global_batch // batches_per_epoch
    batches_in_current_epoch = global_batch % batches_per_epoch

    def _replay_epoch_shuffles(n_batches: int) -> np.ndarray:
        arr = np.arange(n_train)
        np.random.shuffle(arr)
        epoch_indices = arr.copy()
        for b in range(n_batches):
            B = min(cfg.batch_size, n_train - b * cfg.batch_size)
            b_idx = np.arange(B)
            for _ in range(cfg.ppo_epochs):
                np.random.shuffle(b_idx)
        return epoch_indices

    for _ in range(completed_epochs):
        _replay_epoch_shuffles(batches_per_epoch)

    epoch_indices = _replay_epoch_shuffles(batches_in_current_epoch)
    resume_batch_start = batches_in_current_epoch * cfg.batch_size

    return epoch_indices, resume_batch_start, completed_epochs


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(cfg: Config):
    logger.info("Initializing models ...")
    seed_everything(cfg.seed)

    if cfg.use_wandb:
        import wandb
        wandb.init(
            project=cfg.wandb_project,
            name=cfg.wandb_run_name or f"spec-ppo-{cfg.cost_per_token}",
            config=vars(cfg),
        )

    # OpenAI clients pointing at vLLM servers
    draft_client = OpenAI(api_key="EMPTY", base_url=cfg.draft_server_url, timeout=1800.0)
    target_client = OpenAI(api_key="EMPTY", base_url=cfg.target_server_url, timeout=1800.0)

    # Tokenizers (for prompt construction with chat templates)
    draft_tokenizer = AutoTokenizer.from_pretrained(cfg.draft_model_name, trust_remote_code=True)
    target_tokenizer = AutoTokenizer.from_pretrained(cfg.target_model_name, trust_remote_code=True)

    # PRM (vLLM server with automatic prefix caching)
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

    # Policy
    policy = RoutingPolicy(
        state_dim=cfg.state_dim,
        hidden_dim=cfg.hidden_dim,
        num_actions=cfg.num_actions,
        dropout=cfg.dropout,
    ).to(cfg.policy_device)

    if cfg.checkpoint and os.path.exists(cfg.checkpoint):
        ckpt = torch.load(cfg.checkpoint, map_location=cfg.policy_device)
        # Support both raw state-dicts and full training-state dicts
        policy.load_state_dict(ckpt.get("policy_state_dict", ckpt))
        logger.info(f"Loaded policy checkpoint from {cfg.checkpoint}")

    # Data
    all_train_questions, all_train_gt = load_math_dataset(
        cfg, cfg.train_dataset_name, cfg.train_split
    )
    eval_questions, eval_gt = load_math_dataset(
        cfg, cfg.eval_dataset_name, cfg.eval_split
    )

    # --- Validation split from training data ---
    n_total = len(all_train_questions)
    n_val = max(1, int(n_total * cfg.val_fraction))
    perm = np.random.permutation(n_total)
    val_indices = perm[:n_val].tolist()
    train_indices = perm[n_val:].tolist()

    train_questions = [all_train_questions[i] for i in train_indices]
    train_gt = [all_train_gt[i] for i in train_indices]
    val_questions = [all_train_questions[i] for i in val_indices]
    val_gt = [all_train_gt[i] for i in val_indices]
    logger.info(f"Train/val split: {len(train_questions)} train, {len(val_questions)} val (from {n_total} total)")

    # Compute total batches for LR scheduler
    batches_per_epoch = max(1, (len(train_questions) + cfg.batch_size - 1) // cfg.batch_size)
    total_batches = batches_per_epoch * cfg.num_epochs
    trainer = PPOTrainer(policy, cfg, total_batches=total_batches)

    run_save_dir = os.path.join(cfg.save_dir, format_cost_tag(cfg.cost_per_token))
    os.makedirs(run_save_dir, exist_ok=True)
    best_val_score = -float("inf")
    global_batch = 0
    start_epoch = 0
    resume_batch_start = 0
    resume_epoch_indices: Optional[np.ndarray] = None

    if cfg.resume:
        # Tier 1: full training state (saved by this code going forward)
        full_states = sorted(
            [p for p in Path(run_save_dir).glob("training_state_*.pt")],
            key=lambda p: int(re.search(r"training_state_(\d+)\.pt", p.name).group(1)),
        )
        # Tier 2: legacy policy-only checkpoints (e.g. "17.pt")
        legacy_ckpts = sorted(
            [p for p in Path(run_save_dir).glob("*.pt")
             if re.fullmatch(r"\d+\.pt", p.name)],
            key=lambda p: int(p.stem),
        )

        if full_states:
            resume_path = full_states[-1]
            logger.info(f"Resuming (full state) from {resume_path}")
            rs = torch.load(resume_path, map_location=cfg.policy_device)
            policy.load_state_dict(rs["policy_state_dict"])
            trainer.optimizer.load_state_dict(rs["optimizer_state_dict"])
            trainer.lr_scheduler.load_state_dict(rs["lr_scheduler_state_dict"])
            trainer._update_count = rs["update_count"]
            global_batch = rs["global_batch"]
            start_epoch = rs["epoch"]
            resume_batch_start = rs["next_batch_start"]
            resume_epoch_indices = np.array(rs["epoch_indices"])
            best_val_score = rs["best_val_score"]
            np.random.set_state(rs["numpy_rng_state"])
            # Restore torch/cuda/random RNG states (added after initial version)
            if "torch_rng_state" in rs:
                torch.random.set_rng_state(rs["torch_rng_state"])
            if "torch_cuda_rng_state" in rs and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rs["torch_cuda_rng_state"])
            if "random_rng_state" in rs:
                random.setstate(rs["random_rng_state"])
            logger.info(
                f"Resumed: epoch={start_epoch + 1}, global_batch={global_batch}, "
                f"next_batch_start={resume_batch_start}, "
                f"lr={trainer.optimizer.param_groups[0]['lr']:.2e}, "
                f"best_val_score={best_val_score:.4f}"
            )

        elif legacy_ckpts:
            resume_path = legacy_ckpts[-1]
            n_ckpt = int(resume_path.stem)           # e.g. 17
            global_batch = n_ckpt * cfg.eval_every
            logger.info(
                f"Resuming (legacy policy-only) from {resume_path} "
                f"(inferred global_batch={global_batch})"
            )
            ckpt = torch.load(resume_path, map_location=cfg.policy_device)
            policy.load_state_dict(ckpt.get("policy_state_dict", ckpt))
            # Advance numpy RNG past all calls already made during training.
            # seed_everything + np.random.permutation(n_total) were already called
            # above, so _fastforward_numpy_rng starts from that point.
            resume_epoch_indices, resume_batch_start, start_epoch = _fastforward_numpy_rng(
                cfg, len(train_questions), global_batch
            )
            # Fast-forward LR scheduler to match saved position
            trainer._update_count = global_batch
            for _ in range(global_batch):
                trainer.lr_scheduler.step()
            logger.info(
                f"Resumed: epoch={start_epoch + 1}, global_batch={global_batch}, "
                f"next_batch_start={resume_batch_start}, "
                f"lr={trainer.optimizer.param_groups[0]['lr']:.2e}"
            )

        else:
            logger.warning(
                f"--resume specified but no checkpoints found in {run_save_dir}. "
                "Starting from scratch."
            )

    for epoch in range(start_epoch, cfg.num_epochs):
        logger.info(f"=== Epoch {epoch + 1}/{cfg.num_epochs} ===")

        # Re-use saved epoch indices when resuming mid-epoch; shuffle fresh otherwise.
        if epoch == start_epoch and resume_epoch_indices is not None:
            indices = resume_epoch_indices
            epoch_batch_start = resume_batch_start
            resume_epoch_indices = None  # only apply once
        else:
            indices = np.arange(len(train_questions))
            np.random.shuffle(indices)
            epoch_batch_start = 0

        for batch_start in range(epoch_batch_start, len(train_questions), cfg.batch_size):
            batch_end = min(batch_start + cfg.batch_size, len(train_questions))
            batch_qidx = indices[batch_start:batch_end].tolist()
            global_batch += 1

            # --- Rollout ---
            rollout = run_rollout(
                cfg, draft_client, target_client,
                draft_tokenizer, target_tokenizer,
                prm, policy,
                train_questions, batch_qidx, deterministic=False,
            )

            # --- Task reward ---
            correct = 0
            for qidx, answer_text in rollout["completed"]:
                bi = {qi: i for i, qi in enumerate(batch_qidx)}[qidx]
                pred = extract_prediction(answer_text, cfg.train_dataset_name)
                if math_equal(pred, train_gt[qidx]):
                    rollout["rewards"][bi][-1] += cfg.task_reward
                    correct += 1

            train_acc = correct / len(batch_qidx)
            avg_reward = np.mean([sum(r) for r in rollout["rewards"]])
            avg_draft_tokens = float(np.mean(rollout["draft_tokens"]))
            avg_target_tokens = float(np.mean(rollout["target_tokens"]))
            avg_discarded_draft_tokens = float(np.mean(rollout["discarded_draft_tokens"]))
            avg_target_calls = float(np.mean(rollout["target_calls"]))

            # --- PPO update ---
            loss_metrics = trainer.update(
                rollout["obs"], rollout["actions"], rollout["logprobs"],
                rollout["values"], rollout["rewards"],
            )

            log_dict = {
                "train/accuracy": train_acc,
                "train/avg_reward": avg_reward,
                "train/avg_draft_tokens_per_question": avg_draft_tokens,
                "train/avg_target_tokens_per_question": avg_target_tokens,
                "train/avg_discarded_draft_tokens_per_question": avg_discarded_draft_tokens,
                "train/avg_target_calls_per_question": avg_target_calls,
                **{f"train/{k}": v for k, v in loss_metrics.items()},
            }
            logger.info(
                f"  Batch {global_batch}: acc={train_acc:.3f}, reward={avg_reward:.3f}, "
                f"pg_loss={loss_metrics['pg_loss']:.4f}, lr={loss_metrics['lr']:.2e}"
            )
            if cfg.use_wandb:
                wandb.log(log_dict, step=global_batch)

            # --- Periodic validation (for best model selection) ---
            if global_batch % cfg.val_every == 0:
                val_metrics = evaluate(
                    cfg, draft_client, target_client,
                    draft_tokenizer, target_tokenizer,
                    prm, policy,
                    val_questions, val_gt,
                    cfg.train_dataset_name,
                )
                val_score = (
                    val_metrics["accuracy"]
                    - cfg.cost_per_token * val_metrics["avg_target_tokens_per_question"]
                )
                logger.info(
                    f"  [Val] acc={val_metrics['accuracy']:.3f}, "
                    f"avg_target_tokens/q={val_metrics['avg_target_tokens_per_question']:.1f}, "
                    f"score={val_score:.4f}, "
                    f"draft_frac={val_metrics['draft_token_frac']:.3f}"
                )
                if cfg.use_wandb:
                    wandb.log(
                        {f"val/{k}": v for k, v in val_metrics.items()}
                        | {"val/score": val_score},
                        step=global_batch,
                    )

                if val_score > best_val_score:
                    best_val_score = val_score
                    save_path = os.path.join(run_save_dir, "policy_best.pt")
                    torch.save(policy.state_dict(), save_path)
                    logger.info(f"  New best model saved (val_score={best_val_score:.4f})")

            # --- Periodic test evaluation (monitoring only, not for model selection) ---
            if global_batch % cfg.eval_every == 0:
                ckpt_idx = global_batch // cfg.eval_every
                save_path = os.path.join(run_save_dir, f"{ckpt_idx}.pt")
                torch.save(policy.state_dict(), save_path)

                # Full training state for seamless resumption
                torch.save(
                    {
                        "policy_state_dict": policy.state_dict(),
                        "optimizer_state_dict": trainer.optimizer.state_dict(),
                        "lr_scheduler_state_dict": trainer.lr_scheduler.state_dict(),
                        "update_count": trainer._update_count,
                        "global_batch": global_batch,
                        "epoch": epoch,
                        # next_batch_start and epoch_indices let us resume mid-epoch
                        "next_batch_start": batch_start + cfg.batch_size,
                        "epoch_indices": indices.tolist(),
                        "best_val_score": best_val_score,
                        "numpy_rng_state": np.random.get_state(),
                        "torch_rng_state": torch.random.get_rng_state(),
                        "torch_cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                        "random_rng_state": random.getstate(),
                    },
                    os.path.join(run_save_dir, f"training_state_{ckpt_idx}.pt"),
                )

                eval_metrics = evaluate(
                    cfg, draft_client, target_client,
                    draft_tokenizer, target_tokenizer,
                    prm, policy,
                    eval_questions, eval_gt,
                    cfg.eval_dataset_name,
                )
                eval_score = (
                    eval_metrics["accuracy"]
                    - cfg.cost_per_token * eval_metrics["avg_target_tokens_per_question"]
                )
                logger.info(
                    f"  [Eval] acc={eval_metrics['accuracy']:.3f}, "
                    f"avg_target_tokens/q={eval_metrics['avg_target_tokens_per_question']:.1f}, "
                    f"score={eval_score:.4f}, "
                    f"draft_frac={eval_metrics['draft_token_frac']:.3f}"
                )
                if cfg.use_wandb:
                    wandb.log(
                        {f"eval/{k}": v for k, v in eval_metrics.items()}
                        | {"eval/score": eval_score},
                        step=global_batch,
                    )


    if cfg.use_wandb:
        wandb.finish()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    cfg: Config,
    draft_client: OpenAI,
    target_client: OpenAI,
    draft_tokenizer: AutoTokenizer,
    target_tokenizer: AutoTokenizer,
    prm: ServerPRM,
    policy: RoutingPolicy,
    questions: list[str],
    ground_truths: list[str],
    benchmark_name: str,
) -> dict[str, float]:
    """Evaluate the learned policy on a test set (greedy actions)."""
    policy.eval()
    all_correct = 0
    total_draft = 0
    total_target = 0
    total_discarded = 0
    total_target_calls = 0
    n = len(questions)

    for batch_start in range(0, n, cfg.batch_size):
        batch_end = min(batch_start + cfg.batch_size, n)
        batch_qidx = list(range(batch_start, batch_end))

        rollout = run_rollout(
            cfg, draft_client, target_client,
            draft_tokenizer, target_tokenizer,
            prm, policy,
            questions, batch_qidx, deterministic=True,
        )

        for qidx, answer_text in rollout["completed"]:
            pred = extract_prediction(answer_text, benchmark_name)
            if math_equal(pred, ground_truths[qidx]):
                all_correct += 1

        total_draft += sum(rollout["draft_tokens"])
        total_target += sum(rollout["target_tokens"])
        total_discarded += sum(rollout["discarded_draft_tokens"])
        total_target_calls += sum(rollout["target_calls"])

    total_tokens = total_draft + total_target + total_discarded
    return {
        "accuracy": all_correct / max(n, 1),
        "num_correct": all_correct,
        "avg_total_tokens_per_question": total_tokens / max(n, 1),
        "avg_draft_tokens_per_question": total_draft / max(n, 1),
        "avg_target_tokens_per_question": total_target / max(n, 1),
        "avg_discarded_draft_tokens_per_question": total_discarded / max(n, 1),
        "avg_target_calls_per_question": total_target_calls / max(n, 1),
        "draft_token_frac": (total_draft + total_discarded) / max(total_tokens, 1),
        "target_token_frac": total_target / max(total_tokens, 1),
    }


def run_eval_standalone(cfg: Config):
    """Entry point for standalone evaluation mode."""
    seed_everything(cfg.seed)
    logger.info("Initializing models for evaluation ...")

    draft_client = OpenAI(api_key="EMPTY", base_url=cfg.draft_server_url, timeout=1800.0)
    target_client = OpenAI(api_key="EMPTY", base_url=cfg.target_server_url, timeout=1800.0)

    draft_tokenizer = AutoTokenizer.from_pretrained(cfg.draft_model_name, trust_remote_code=True)
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

    policy = RoutingPolicy(
        state_dim=cfg.state_dim, hidden_dim=cfg.hidden_dim, num_actions=cfg.num_actions,
        dropout=cfg.dropout,
    ).to(cfg.policy_device)

    if cfg.checkpoint and os.path.exists(cfg.checkpoint):
        policy.load_state_dict(torch.load(cfg.checkpoint, map_location=cfg.policy_device))
        logger.info(f"Loaded checkpoint: {cfg.checkpoint}")
    else:
        raise FileNotFoundError(f"Checkpoint not found: {cfg.checkpoint}")

    questions, ground_truths = load_math_dataset(
        cfg, cfg.eval_dataset_name, cfg.eval_split
    )
    metrics = evaluate(
        cfg, draft_client, target_client,
        draft_tokenizer, target_tokenizer,
        prm, policy,
        questions, ground_truths,
        cfg.eval_dataset_name,
    )

    run_output_dir = os.path.join(cfg.output_dir, format_cost_tag(cfg.cost_per_token))
    os.makedirs(run_output_dir, exist_ok=True)
    out_path = os.path.join(run_output_dir, "eval_metrics.jsonl")
    with open(out_path, "w") as f:
        f.write(json.dumps(metrics) + "\n")

    logger.info(f"Evaluation results saved to {out_path}")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = parse_args()
    if cfg.mode == "train":
        train(cfg)
    else:
        run_eval_standalone(cfg)
