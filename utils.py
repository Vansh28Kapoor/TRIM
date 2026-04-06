"""Shared utilities for the TRIM-Agg pipeline.

This module provides building blocks that are used by both the main training
script (``TRIM_Agg.py``) and offline pre-processing scripts.

TRIM naming used throughout:
- ``M_w``: draft/cheap model
- ``M_s``: target/expensive model
(e.g. ``pomdp_params/get_transition_function.py``).  Keeping them here
avoids circular imports and makes each downstream script self-contained.

Public API
----------
- ``seed_everything``       — reproducible RNG seeding
- ``SYSTEM_PROMPT``         — shared math-problem system prompt
- ``format_prompt``         — wrap a question with the system prompt
- ``_DEGENERATE_RE``        — compiled regex for degenerate outputs
- ``_is_degenerate``        — predicate for degenerate generation
- ``_build_prompt``         — chat-template prompt builder
- ``generate_steps``        — batched single-step generation via vLLM
- ``generate_full_solutions``  — batched full-solution generation via vLLM
- ``ServerPRM``             — Qwen2.5-Math-PRM-7B via vLLM Pooling API
"""

from __future__ import annotations

import concurrent.futures
import logging
import os
import random
import re
import time
from typing import Any, Optional

import httpx
import numpy as np
import torch
from openai import OpenAI
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    """Seed all RNGs for exact reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "Solve the following math problem step by step, given in LaTeX format, "
    "clearly and concisely, and present the final answer as \\boxed{{X}}, "
    "where X is the fully simplified solution."
)


def format_prompt(question: str) -> str:
    """Wrap *question* with the shared math system prompt."""
    return f"{SYSTEM_PROMPT}\nQuestion: {question}"


# ---------------------------------------------------------------------------
# Degenerate-output detection
# ---------------------------------------------------------------------------

_DEGENERATE_RE = re.compile(r"^[\s\-–—=_*#`~|/\\<>:;,.!?(){}[\]]*$")


def _is_degenerate(text: str) -> bool:
    """Return True if *text* contains no meaningful reasoning content."""
    return len(text.strip()) == 0 or bool(_DEGENERATE_RE.match(text))


# ---------------------------------------------------------------------------
# Generation helpers  (batched completions via vLLM server)
# ---------------------------------------------------------------------------

def _build_prompt(
    tokenizer: AutoTokenizer,
    question: str,
    partial_answer: str,
    disable_thinking: bool = False,
    continue_solution: bool = False,
) -> str:
    """Build a raw prompt string for the completions API.

    Applies the chat template with ``continue_final_message=True`` so the
    model continues generating from the assistant's partial answer.  The
    server's automatic prefix caching (APC) ensures shared prefixes across
    steps are not re-encoded.

    For Qwen3 models, ``disable_thinking=True`` injects the
    ``<think>\\n\\n</think>\\n\\n`` prefix so the model skips its internal
    chain-of-thought and produces the answer directly.  For non-thinking
    models (e.g. Qwen2.5-7B-Instruct), leave at the default (``False``).

    Parameters
    ----------
    tokenizer : AutoTokenizer
        Tokenizer whose ``apply_chat_template`` will be called.
    question : str
        The math problem text.
    partial_answer : str
        Any already-generated partial answer to continue from.
    disable_thinking : bool
        ``False`` (default) for non-thinking models (e.g. Qwen2.5-7B-Instruct).
        ``True`` for Qwen3-family targets to suppress the thinking block.
    """
    if partial_answer:
        assistant_content = "Solution: " + partial_answer + "\n\n"
    elif continue_solution:
        assistant_content = "Solution: "
    else:
        assistant_content = "Solution:\n\n"

    messages = [
        {"role": "user", "content": format_prompt(question)},
        {"role": "assistant", "content": assistant_content},
    ]

    template_kwargs: dict[str, Any] = dict(
        tokenize=False,
        continue_final_message=True,
    )
    if disable_thinking:
        try:
            return tokenizer.apply_chat_template(
                messages, enable_thinking=False, **template_kwargs,
            )
        except TypeError:  # unexpected "enable_thinking" argument
            pass
    return tokenizer.apply_chat_template(messages, **template_kwargs)


def generate_steps(
    client: OpenAI,
    model_name: str,
    tokenizer: AutoTokenizer,
    questions: list[str],
    partial_answers: list[tuple[int, str]],
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: Optional[int],
    stop: list[str],
    disable_thinking: bool = False,
    max_model_len: int = 4096,
) -> tuple[list[str], torch.Tensor, list[int]]:
    """Generate one reasoning step per prompt via batched completions.

    Builds raw prompt strings using the chat template and sends them in a
    single ``client.completions.create(prompt=[...])`` call.  The vLLM server
    processes all prompts as a native batch for maximum throughput.

    Parameters
    ----------
    client : OpenAI
        OpenAI-compatible client pointed at a vLLM server.
    model_name : str
        Model identifier as registered with the vLLM server.
    tokenizer : AutoTokenizer
        Tokenizer for the model (used for chat-template formatting and token
        counting).
    questions : list[str]
        Full list of questions in the current episode batch.
    partial_answers : list[tuple[int, str]]
        ``(question_index, partial_answer)`` pairs — one entry per active
        problem.
    max_tokens : int
        Maximum completion tokens per step.
    temperature : float
        Sampling temperature.
    top_p : float
        Nucleus sampling probability.
    top_k : int or None
        Top-k sampling parameter (passed as ``extra_body``).
    stop : list[str]
        Stop strings (e.g. ``["\\n\\n"]`` to break at paragraph boundaries).
    disable_thinking : bool
        Passed to ``_build_prompt``; set ``True`` for Qwen3 targets to suppress the thinking block.

    Returns
    -------
    texts : list[str]
        Generated text for each prompt (one step only).
    finished : Tensor[bool]
        True if the step contains a final answer (``\\boxed{}``) or the model
        hit EOS without a stop token.
    token_counts : list[int]
        Number of completion tokens per generation.
    """
    prompts = [
        _build_prompt(tokenizer, questions[idx], partial, disable_thinking=disable_thinking)
        for idx, partial in partial_answers
    ]

    max_prompt_len = max(len(tokenizer.encode(p)) for p in prompts)
    max_tokens = max(1, min(max_tokens, max_model_len - max_prompt_len))

    response = client.completions.create(
        model=model_name,
        prompt=prompts,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stop=stop,
        extra_body={"top_k": top_k} if top_k is not None else {},
    )

    # Sort choices by index to align with input order
    sorted_choices = sorted(response.choices, key=lambda c: c.index)

    texts = [c.text for c in sorted_choices]
    finished = torch.tensor([
        any(tok in t for tok in ("\\boxed{", "boxed{"))
        or c.finish_reason not in ("stop",)  # EOS / max-tokens → done
        for c, t in zip(sorted_choices, texts)
    ])
    token_counts = [len(tokenizer.encode(t)) for t in texts]
    return texts, finished, token_counts


def generate_full_solutions(
    client: OpenAI,
    model_name: str,
    tokenizer: AutoTokenizer,
    questions: list[str],
    max_tokens: int = 4096,
    temperature: float = 0.7,
    top_p: float = 0.8,
    top_k: Optional[int] = 20,
    disable_thinking: bool = False,
    max_model_len: int = 4096,
    batch_size: int = 500,
) -> list[str]:
    """Generate a complete solution for each question in a single batch call.

    Unlike ``generate_steps``, no stop token is set so the model generates the
    full response in one pass.  Useful for offline pre-processing (e.g. computing
    POMDP transition parameters).

    Parameters
    ----------
    client : OpenAI
        OpenAI-compatible client pointed at a vLLM server.
    model_name : str
        Model identifier as registered with the vLLM server.
    tokenizer : AutoTokenizer
        Tokenizer for the model (used for chat-template prompt building).
    questions : list[str]
        Math questions to solve.
    max_tokens : int
        Maximum completion tokens per solution.
    temperature : float
        Sampling temperature.
    top_p : float
        Nucleus sampling probability.
    top_k : int or None
        Top-k sampling parameter (passed as ``extra_body``).
    disable_thinking : bool
        Passed to ``_build_prompt``; ``False`` (default) for non-thinking targets
        (e.g. Qwen2.5-7B-Instruct).  Set ``True`` for Qwen3 targets to suppress
        the internal thinking block.

    Returns
    -------
    list[str]
        Generated text for each question, in the same order as the input.
    """
    prompts = [
        _build_prompt(tokenizer, q, partial_answer="", disable_thinking=disable_thinking)
        for q in questions
    ]

    max_prompt_len = max(len(tokenizer.encode(p)) for p in prompts)
    max_tokens = max(1, min(max_tokens, max_model_len - max_prompt_len))

    results: list[str] = []
    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start : start + batch_size]
        response = client.completions.create(
            model=model_name,
            prompt=chunk,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            extra_body={"top_k": top_k} if top_k is not None else {},
        )
        sorted_choices = sorted(response.choices, key=lambda c: c.index)
        results.extend(c.text for c in sorted_choices)
    return results


# ---------------------------------------------------------------------------
# Process Reward Model  (Qwen2.5-Math-PRM-7B via vLLM Pooling API)
# ---------------------------------------------------------------------------

class ServerPRM:
    """Qwen2.5-Math-PRM-7B scoring via vLLM Pooling API.

    Qwen2.5-Math-PRM-7B is a ``Qwen2ForProcessRewardModel`` — a reward model,
    **not** a cross-encoder/reranker.  In vLLM the correct serving path is:

    * **no** ``--task`` flag (let the model use its built-in STEP pooler)
    * endpoint ``/pooling`` at the **root** URL (not ``/v1/pooling``)
    * STEP pooling returns one softmax'd 2-class output per ``<extra_0>``
      step marker

    We extract the positive-class (class-1) probability at every ``<extra_0>``
    step-separator position to obtain one PRM score per reasoning step.

    Server setup::

        CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen2.5-Math-PRM-7B \\
            --dtype auto --trust-remote-code \\
            --gpu-memory-utilization 0.40 \\
            --enable-prefix-caching --port 30002
    """

    STEP_TOKEN = "<extra_0>"

    def __init__(
        self,
        server_url: str,
        model_name: str,
        tokenizer: AutoTokenizer,
        max_workers: int = 32,
        timeout_s: float = 300.0,
    ):
        """Initialise the PRM client.

        Parameters
        ----------
        server_url : str
            Base URL of the vLLM server, e.g. ``"http://localhost:30002"``.
            Do **not** append ``/v1`` — the pooling endpoint is at the root.
        model_name : str
            HuggingFace model identifier, e.g. ``"Qwen/Qwen2.5-Math-PRM-7B"``.
        tokenizer : AutoTokenizer
            Pre-loaded tokenizer for the PRM model.
        max_workers : int
            Thread-pool size for concurrent scoring requests.
        timeout_s : float
            HTTP request timeout in seconds.
        """
        self.base_url = server_url.rstrip("/")
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.max_workers = max_workers
        self.client = httpx.Client(
            base_url=self.base_url, timeout=httpx.Timeout(timeout_s),
        )

        self.step_token_id: int = tokenizer.convert_tokens_to_ids(self.STEP_TOKEN)
        if self.step_token_id is None or self.step_token_id == tokenizer.unk_token_id:
            raise ValueError(
                f"Tokenizer could not resolve {self.STEP_TOKEN!r} to a valid token id."
            )

    def reset(self, indices: list[int] | None = None) -> None:
        """No-op — prefix caching is managed by the vLLM server."""

    # -- internal helpers --------------------------------------------------

    def _format_text(self, question: str, step_history: list[str]) -> str:
        """Build the PRM input with ``<extra_0>`` after each step."""
        assistant_content = "".join(
            f"{step}{self.STEP_TOKEN}" for step in step_history
        )
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
            {"role": "assistant", "content": assistant_content},
        ]
        return self.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False,
        )

    def _pool_token_scores(
        self,
        text: str,
        max_retries: int = 5,
        truncate_prompt_tokens: int = 4096,
    ) -> dict[str, Any]:
        """Call the vLLM Pooling API (STEP pooler returns per-step scores).

        ``truncate_prompt_tokens`` is passed to the server so that overlong
        sequences are silently truncated from the left (keeping the most
        recent steps), avoiding 400 errors on long solutions.

        Retries up to ``max_retries`` times with exponential backoff on
        transient connection errors (server disconnect, timeout).
        """
        delay = 2.0
        for attempt in range(max_retries):
            try:
                resp = self.client.post(
                    "/pooling",
                    json={
                        "model": self.model_name,
                        "input": text,
                        "truncate_prompt_tokens": truncate_prompt_tokens,
                    },
                )
                resp.raise_for_status()
                return resp.json()
            except (
                httpx.RemoteProtocolError,
                httpx.ReadTimeout,
                httpx.ConnectError,
                httpx.ReadError,
            ) as exc:
                if attempt == max_retries - 1:
                    raise
                logger.warning(
                    "PRM request failed (%s), retrying in %.1fs (attempt %d/%d).",
                    exc, delay, attempt + 1, max_retries,
                )
                time.sleep(delay)
                delay *= 2.0
                # Recreate the client to discard stale socket file descriptors
                # left over from a server crash.
                self.client = httpx.Client(
                    base_url=self.base_url,
                    timeout=self.client.timeout,
                )

    @staticmethod
    def _extract_first_item(payload: dict[str, Any]) -> dict[str, Any]:
        """Extract the first data item from a /pooling response payload."""
        data = payload.get("data")
        if not isinstance(data, list) or not data:
            raise RuntimeError(f"Unexpected /pooling response: {payload}")
        return data[0]

    @staticmethod
    def _normalize_token_outputs(item: dict[str, Any]) -> list[Any]:
        """Extract per-token outputs from the /pooling response item.

        vLLM versions may expose these under different keys; try the common
        ones.
        """
        if isinstance(item.get("data"), list):
            return item["data"]
        outputs = item.get("outputs")
        if isinstance(outputs, dict) and isinstance(outputs.get("data"), list):
            return outputs["data"]
        raise RuntimeError(
            f"Could not find token outputs in /pooling item: {item}"
        )

    @staticmethod
    def _positive_prob(tok_output: Any) -> float:
        """Get positive-class probability from one token-classify output.

        Expects a 2-class output where class 1 is the positive label.
        """
        if isinstance(tok_output, list) and len(tok_output) == 2:
            return float(tok_output[1])
        if isinstance(tok_output, dict):
            for key in ("data", "scores", "probabilities"):
                val = tok_output.get(key)
                if isinstance(val, list) and len(val) == 2:
                    return float(val[1])
        raise RuntimeError(f"Unsupported token output format: {tok_output}")

    # -- scoring -----------------------------------------------------------

    def _score_single(self, question: str, step_history: list[str]) -> list[float]:
        """Return per-step positive-class probabilities for one problem."""
        if not step_history:
            return []

        text = self._format_text(question, step_history)

        payload = self._pool_token_scores(text)
        item = self._extract_first_item(payload)
        step_outputs = self._normalize_token_outputs(item)

        # With STEP pooling the server returns one output per <extra_0> marker
        # that survived truncation.  If server-side truncation dropped leading
        # steps, pad those positions with 1.0 (optimistic default — they were
        # already accepted in a prior round).
        n_returned = len(step_outputs)
        n_expected = len(step_history)
        if n_returned > n_expected:
            raise RuntimeError(
                f"Step output length mismatch: got {n_returned} outputs "
                f"but expected at most {n_expected} (one per {self.STEP_TOKEN!r} marker)."
            )
        if n_returned < n_expected:
            logger.warning(
                "PRM returned %d scores for %d steps — %d leading step(s) were "
                "truncated server-side; padding with 1.0.",
                n_returned, n_expected, n_expected - n_returned,
            )

        n_padded = n_expected - n_returned
        return [1.0] * n_padded + [
            self._positive_prob(step_outputs[i])
            for i in range(n_returned)
        ]

    def score(self, question: str, step_history: list[str]) -> list[float]:
        """Score a single problem's step history.

        Single-sample convenience wrapper around ``_score_single``.  Used by
        ``get_observation_function.py`` and ``get_transition_function.py``.

        Parameters
        ----------
        question : str
            The math problem text.
        step_history : list[str]
            Reasoning steps generated so far.

        Returns
        -------
        list[float]
            Per-step positive-class PRM probabilities.
        """
        return self._score_single(question, step_history)

    # -- public API --------------------------------------------------------

    def batch_score(
        self,
        questions: list[str],
        step_histories: dict[int, list[str]],
        active_indices: list[int],
    ) -> tuple[dict[int, list[float]], torch.Tensor]:
        """Score all active problems via concurrent HTTP requests.

        Parameters
        ----------
        questions : list[str]
            Full question list for the batch (indexed by *active_indices*).
        step_histories : dict[int, list[str]]
            Mapping from question index to accumulated step history.
        active_indices : list[int]
            Subset of question indices to score in this call.

        Returns
        -------
        reward_dict : dict[int, list[float]]
            Per-step scores keyed by question index.
        reward_tensor : torch.Tensor
            Scores stacked into a 2-D tensor padded to
            ``(len(active_indices), max_steps_so_far)``.
        """
        results: dict[int, list[float]] = {}

        def _call(idx: int) -> tuple[int, list[float]]:
            return idx, self._score_single(questions[idx], step_histories[idx])

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers,
        ) as pool:
            futs = {pool.submit(_call, idx): idx for idx in active_indices}
            for fut in concurrent.futures.as_completed(futs):
                idx, scores = fut.result()
                results[idx] = scores

        all_rewards: list[list[float]] = []
        for idx in active_indices:
            all_rewards.append(results[idx])

        reward_tensor = torch.tensor(all_rewards)  # all rows equal length
        return results, reward_tensor
