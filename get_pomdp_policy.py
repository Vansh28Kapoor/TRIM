"""Precompute and store POMDP routing actions for all reachable belief states.

For every ``(token_count, step_no, p0, p1, p2)`` combination where the
belief triple lies on the discretised simplex and satisfies the closeness
constraint ``p2 > max(p0, p1, p2) - closeness_thr``, an independent SARSOP
POMDP is solved with that belief as the initial state, and the optimal
action is recorded.  The full lookup table is persisted to disk so that
inference-time queries are pure NumPy lookups — no Julia call needed.

Usage
-----
Precompute (run once, writes ``pomdp_action_table.pkl``):

    python get_pomdp_policy.py --workers 128 --closeness-thr 0.4

Load and query at inference time:

    from get_pomdp_policy import load_action_table, get_POMDP_action
    table = load_action_table("pomdp_action_table.pkl")
    actions = get_POMDP_action(table, belief_array, token_batch, step_batch,
                               closeness_thr=0.4)
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
import pickle
import sys
import time
import multiprocessing
from multiprocessing import Pool, cpu_count
from pathlib import Path
from scipy.stats import gaussian_kde
from typing import Optional

# Julia cannot initialise in a forked process (SIGABRT) when Python is
# statically linked (compiled_modules=False mode).  'spawn' starts each
# worker as a fresh interpreter so Julia initialises cleanly.
# On systems where Python is dynamically linked this is a no-op in practice.
if multiprocessing.get_start_method(allow_none=True) != "spawn":
    multiprocessing.set_start_method("spawn", force=True)
from typing import Any, Dict, List, Tuple


import numpy as np

# ---------------------------------------------------------------------------
# Default constants (all overridable via CLI)
# ---------------------------------------------------------------------------
_DEFAULTS = dict(
    p_slm=0.866,
    p_llm=0.9704,
    avg_token_count=60,
    max_steps=30,
    sarsop_timeout=20.0,
    token_counts="5,10,20,30,40,50,75,100,125,150,175,200,250,300,350,400,450,500",
    belief_step=0.025,
    closeness_thr=0.5,
    cost_per_token=0.25,
    task_reward=1e2,
    # predictor file params (used to construct paths under PREDICTOR_DIR)
    terminal_benchmark="math",
    terminal_split="train",
    obs_benchmark="math-train",
    obs_bin_size=0.05,
)

TERMINAL_STATE = (3, 0, -1)

PREDICTOR_DIR = Path(__file__).resolve().parent / "pomdp_data"

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# These module-level values are set once by the main process and inherited
# by forked workers (copy-on-write).  _worker_init / _solve_single read them.
# ---------------------------------------------------------------------------
_cfg: Dict[str, Any] = {}

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

def _build_belief_grid(
    step: float,
    closeness_thr: float,
) -> List[Tuple[float, float, float]]:
    """Enumerate (p0, p1, p2) on the simplex satisfying p2 > max(p) - closeness_thr."""
    grid: List[Tuple[float, float, float]] = []
    vals = np.round(np.arange(0.0, 1.0 + step, step), 6)
    for p0 in vals:
        for p1 in vals:
            p2 = round(1.0 - p0 - p1, 6)
            if p2 < -1e-9 or p2 > 1.0 + 1e-9:
                continue
            p2 = round(round(p2 / step) * step, 6)
            if abs(p0 + p1 + p2 - 1.0) > 1e-6 or p2 < -1e-9:
                continue
            if p2 > max(p0, p1, p2) - closeness_thr:
                grid.append((round(p0, 6), round(p1, 6), round(p2, 6)))
    return grid


# ---------------------------------------------------------------------------
# Julia bootstrap (only in worker processes that need it)
# ---------------------------------------------------------------------------
_julia_ready = False


def _init_julia() -> None:
    """Lazily initialise PyJulia and import POMDP packages."""
    global _julia_ready
    if _julia_ready:
        return

    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)

    # Each worker is already its own process; Julia only needs 1 thread.
    # Without this, JULIA_NUM_THREADS=auto would try to use all 240 cores
    # per worker, exhausting the system thread limit across 128 workers.
    os.environ["JULIA_NUM_THREADS"] = "1"   # force 1 thread per worker; never let auto pick 240
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.pop("LD_PRELOAD", None)     # tcmalloc (set in /etc/environment on GCP VMs) conflicts with Julia's allocator → SIGABRT

    import subprocess, random
    from julia.api import Julia
    sysimage = os.environ.get("JULIA_SYSIMAGE", "")
    if sysimage and os.path.isfile(sysimage):
        # compiled_modules=False still needed: conda Python is statically
        # linked to libpython, which breaks module compilation even with a
        # sysimage.  The sysimage provides precompiled packages; the flag
        # prevents Julia from trying to compile anything else.
        kw: Dict[str, Any] = {"sysimage": sysimage, "compiled_modules": False}
    else:
        kw: Dict[str, Any] = {"compiled_modules": False}

    # Retry with backoff — concurrent Julia startups on NFS can SIGABRT
    max_retries = 5
    for attempt in range(max_retries):
        try:
            Julia(**kw)
            break
        except subprocess.CalledProcessError:
            if attempt == max_retries - 1:
                raise
            wait = (2 ** attempt) + random.uniform(0, 2)
            logger.warning(
                "Julia init failed (attempt %d/%d), retrying in %.1fs …",
                attempt + 1, max_retries, wait,
            )
            time.sleep(wait)

    global Main, QuickPOMDP, jl_action, jl_support, jl_solve, jl_pdf
    global SARSOPSolver, SparseCat, Deterministic

    from julia import Main as _Main
    from julia.NativeSARSOP import SARSOPSolver as _SARSOPSolver
    from julia.POMDPs import (
        action as _action,
        pdf as _pdf,
        solve as _solve,
        support as _support,
    )
    from julia.POMDPTools import (
        Deterministic as _Deterministic,
        SparseCat as _SparseCat,
    )
    from quickpomdps import QuickPOMDP as _QuickPOMDP

    Main = _Main
    QuickPOMDP = _QuickPOMDP
    jl_action = _action
    jl_support = _support
    jl_pdf = _pdf
    SARSOPSolver = _SARSOPSolver
    SparseCat = _SparseCat
    Deterministic = _Deterministic

    # Patch: NativeSARSOP returns Vector{Any} alphas for some POMDPs,
    # but AlphaVectorPolicy requires Vector{Vector{Float64}}.
    # Cannot fix via sysimage because conda Python is statically linked,
    # making sysimage + PyCall incompatible. This JIT wrapper is necessary.
    _Main.eval("""
    using NativeSARSOP, POMDPs, POMDPTools
    function _typed_solve(solver, pomdp)
        tree = NativeSARSOP.SARSOPTree(solver, pomdp)
        t0 = time()
        iter = 0
        while iter <= solver.max_steps && time()-t0 < solver.max_time &&
              NativeSARSOP.root_diff(tree) > solver.precision
            NativeSARSOP.sample!(solver, tree)
            NativeSARSOP.backup!(tree)
            NativeSARSOP.prune!(solver, tree)
            iter += 1
        end
        if isempty(tree.Γ)
            ns = length(POMDPTools.ordered_states(pomdp))
            alphas = [zeros(Float64, ns)]
            amap = [first(POMDPs.actions(pomdp))]
        else
            alphas = Vector{Vector{Float64}}(getproperty.(tree.Γ, :alpha))
            amap = POMDPTools.ordered_actions(pomdp)[getproperty.(tree.Γ, :action)]
        end
        return POMDPTools.AlphaVectorPolicy(pomdp, alphas, amap)
    end
    """)
    jl_solve = _Main._typed_solve

    _julia_ready = True


# ---------------------------------------------------------------------------
# Predictor loading (per-worker, cached)
# ---------------------------------------------------------------------------
_predictors: Dict[str, Any] = {}


def _load_predictors() -> None:
    if _predictors:
        return
    tb = _cfg.get("terminal_benchmark", _DEFAULTS["terminal_benchmark"])
    ts = _cfg.get("terminal_split",     _DEFAULTS["terminal_split"])
    ob = _cfg.get("obs_benchmark",      _DEFAULTS["obs_benchmark"])
    bs = _cfg.get("obs_bin_size",       _DEFAULTS["obs_bin_size"])
    for name, fname in [
        ("terminal", f"{tb}_{ts}_terminal_predictor_slm.pkl"),
        ("obs_model", f"{ob}_reflected_kde_obs_model.pkl"),
        ("obs_dist",  f"{ob}_obs_distributions_bin_size_{bs}.pkl"),
    ]:
        path = PREDICTOR_DIR / fname
        with open(path, "rb") as f:
            _predictors[name] = pickle.load(f)
        logger.info("Loaded predictor '%s' from %s", name, path)


# ---------------------------------------------------------------------------
# Solve one (token_count, step_no, p0, p1, p2) task
# ---------------------------------------------------------------------------

def _solve_single(args: Tuple) -> Tuple[Tuple, int | None]:
    """Solve one POMDP initialised at (p0, p1, p2) and return the optimal action.

    Returns ``((token_val, step_no, p0, p1, p2), action)`` on success,
    or ``((token_val, step_no, p0, p1, p2), None)`` on failure.
    """
    token_val, step_no, p0, p1, p2, cost_per_token = args
    try:
        return _solve_single_inner(args)
    except Exception as exc:
        logger.warning(
            "Worker failed on (%s, %s, %.3f, %.3f, %.3f): %s",
            token_val, step_no, p0, p1, p2, exc,
        )
        return (token_val, step_no, p0, p1, p2), None


def _solve_single_inner(args: Tuple) -> Tuple[Tuple, int]:
    token_val, step_no, p0, p1, p2, cost_per_token = args

    _init_julia()
    _load_predictors()

    P_SLM = _cfg["p_slm"]
    P_LLM = _cfg["p_llm"]
    AVG_TOKEN_COUNT = _cfg["avg_token_count"]
    MAX_STEPS = _cfg["max_steps"]
    SARSOP_TIMEOUT = _cfg["sarsop_timeout"]
    TASK_REWARD = _cfg["task_reward"]
    TOKENS_LIST = [AVG_TOKEN_COUNT]

    determine_terminal = _predictors["terminal"]
    obs_distributions = _predictors["obs_dist"]

    input_belief = SparseCat(
        [(c, token_val, step_no) for c in [0, 1, 2]],
        [p0, p1, p2],
    )

    # --- POMDP functions -------------------------------------------------------
    def transition(s, a):
        correctness_state, num_tokens, sn = s
        if correctness_state == 3 or sn >= MAX_STEPS:
            return Deterministic(TERMINAL_STATE)
        p_terminal = determine_terminal.predict(sn + 1)
        next_step = sn + 1

        if a == 0:
            if correctness_state == 0:
                next_c = SparseCat([0, 2], [P_SLM, 1 - P_SLM])
            else:
                next_c = Deterministic(1)
        else:
            if correctness_state == 0:
                next_c = SparseCat([0, 1, 2], [
                    P_SLM * P_LLM, 1 - P_LLM, P_LLM * (1 - P_SLM)])
            elif correctness_state == 1:
                next_c = Deterministic(1)
            else:
                next_c = SparseCat([0, 1, 2], [
                    P_SLM * P_LLM, 1 - P_LLM, P_LLM * (1 - P_SLM)])

        states, prob_list = [], []
        for nc in list(jl_support(next_c)):
            p = jl_pdf(next_c, nc)
            states.append((nc, AVG_TOKEN_COUNT, 31)) # Note 30th step will never exist since p_terminal = determine_terminal.predict(30) = 1

            prob_list.append(p * p_terminal)
            states.append((nc, AVG_TOKEN_COUNT, next_step))
            prob_list.append(p * (1 - p_terminal))
        return SparseCat(states, prob_list)

    def observation(a, sp):
        if sp == TERMINAL_STATE:
            return Deterministic((0.0, 0.0, 0, -1))
        correctness_state, num_tokens, sn = sp
        if correctness_state in obs_distributions:
            obs_list, prob_list = obs_distributions[correctness_state]
            observations = [(o1, o2, num_tokens, sn) for o1, o2 in obs_list]
            return SparseCat(observations, prob_list)

    def reward(s, a):
        if s == TERMINAL_STATE:
            return 0.0
        correctness_state, num_tokens, sn = s
        r = 0.0
        if sn == 31:
            if correctness_state == 0:
                r = TASK_REWARD if a == 0 else P_LLM * TASK_REWARD
            elif correctness_state == 2 and a == 1:
                r = P_LLM * TASK_REWARD
        if a == 1:
            r -= cost_per_token * num_tokens
        return r

    # --- Observation space -----------------------------------------------------
    obs_pairs_set = set()
    for obs_pair_list, _ in obs_distributions.values():
        obs_pairs_set.update(obs_pair_list)
    obs_pairs = list(obs_pairs_set)

    observations_space = [
        (float(o1), float(o2), int(tb), int(st))
        for (o1, o2) in obs_pairs
        for tb in TOKENS_LIST
        for st in range(step_no, 32)
    ]
    if token_val != AVG_TOKEN_COUNT:
        observations_space.extend(
            [(o1, o2, token_val, step_no) for (o1, o2) in obs_pairs])
    observations_space.append((0.0, 0.0, 0, -1))

    all_states = [
        (c, n, st)
        for c in [0, 1, 2]
        for n in TOKENS_LIST
        for st in range(step_no, 32)
    ]
    if token_val != AVG_TOKEN_COUNT:
        all_states.extend([(c, token_val, step_no) for c in [0, 1, 2]])

    pomdp = QuickPOMDP(
        states=[TERMINAL_STATE] + all_states,
        actions=[0, 1],
        discount=1 - 1e-6, # Slightly less than 1 for convergence
        isterminal=lambda s: s == TERMINAL_STATE,
        transition=transition,
        observation=observation,
        observations=observations_space,
        reward=reward,
        initialstate=input_belief,
    )

    solver = SARSOPSolver(verbose=False, max_time=SARSOP_TIMEOUT)
    policy = jl_solve(solver, pomdp)
    act = int(jl_action(policy, input_belief))

    return (token_val, step_no, p0, p1, p2), act


# ---------------------------------------------------------------------------
# Worker initialiser
# ---------------------------------------------------------------------------

_init_sem: multiprocessing.Semaphore | None = None


def _worker_init(cfg: Dict[str, Any], sem: multiprocessing.Semaphore) -> None:
    """Pre-warm Julia and predictors in each pool worker."""
    global _cfg, _init_sem
    _cfg = cfg   # spawn workers start with empty _cfg; parent passes its copy
    _init_sem = sem
    # Limit concurrent Julia startups to avoid SIGABRT from NFS contention.
    sem.acquire()
    try:
        _init_julia()
    finally:
        sem.release()
    _load_predictors()


# ---------------------------------------------------------------------------
# Precomputation driver
# ---------------------------------------------------------------------------

def _kill_pool(pool) -> None:
    """SIGKILL all pool workers and force-exit.

    Julia intercepts SIGTERM at the C level, so pool.terminate() hangs
    forever.  SIGKILL the workers directly and os._exit() to skip
    Julia's atexit handlers.
    """
    import signal as _sig
    for w in pool._pool:
        try:
            os.kill(w.pid, _sig.SIGKILL)
        except (ProcessLookupError, OSError):
            pass
    os._exit(0)


def _save_checkpoint(
    table: Dict, output_path: Path, closeness_thr: float, belief_grid: list
) -> None:
    """Atomically write a checkpoint file next to the final output."""
    ckpt_path = output_path.with_suffix(".ckpt")
    tmp = ckpt_path.with_suffix(".ckpt.tmp")
    with open(tmp, "wb") as f:
        pickle.dump(
            {
                "config": _cfg,
                "closeness_thr": closeness_thr,
                "belief_grid": belief_grid,
                "table": table,
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    os.replace(tmp, ckpt_path)  # atomic on POSIX; works across NFS mounts unlike Path.rename()


def _load_checkpoint(
    output_path: Path,
) -> Dict[Tuple[int, int, float, float, float], int]:
    """Load a checkpoint if one exists and its config matches _cfg."""
    ckpt_path = output_path.with_suffix(".ckpt")
    if not ckpt_path.exists():
        return {}
    try:
        with open(ckpt_path, "rb") as f:
            data = pickle.load(f)
        saved_cfg = data.get("config", {})
        # Only resume if key parameters match
        for key in ("p_slm", "p_llm", "avg_token_count", "max_steps",
                     "sarsop_timeout", "task_reward", "token_counts",
                     "belief_step", "cost_per_token"):
            if saved_cfg.get(key) != _cfg.get(key):
                logger.warning(
                    "Checkpoint config mismatch on '%s' — starting fresh.", key
                )
                return {}
        table = data.get("table", {})
        logger.info(
            "Resumed from checkpoint %s with %d entries.", ckpt_path, len(table)
        )
        return table
    except Exception as exc:
        logger.warning("Corrupt checkpoint %s (%s) — starting fresh.", ckpt_path, exc)
        return {}


def precompute_action_table(
    cost_per_token: float = 8e-4,
    closeness_thr: float = 0.4,
    workers: int | None = None,
    chunksize: int = 1,
    output_path: str | Path | None = None,
    checkpoint_every: int = 5000,
) -> Dict[Tuple[int, int, float, float, float], int]:
    """Solve all POMDP instances in parallel and persist the lookup table.

    Each ``(token_count, step_no, p0, p1, p2)`` combination that satisfies
    the closeness constraint gets its own SARSOP solve with the belief
    ``[p0, p1, p2]`` as ``initialstate``.

    Returns
    -------
    table : dict
        ``{(token_count, step_no, p0, p1, p2): action}``
    """
    if workers is None:
        workers = min(cpu_count(), 127)

    token_counts = _cfg["token_counts"]
    max_steps = _cfg["max_steps"]
    belief_step = _cfg["belief_step"]
    step_range = list(range(0, max_steps))

    belief_grid = _build_belief_grid(step=belief_step, closeness_thr=closeness_thr)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Resume from checkpoint if available ---
    table = _load_checkpoint(output_path)
    done_keys = set(table.keys())

    all_tasks = [
        (tc, sn, p0, p1, p2, cost_per_token)
        for (tc, sn), (p0, p1, p2) in itertools.product(
            itertools.product(token_counts, step_range), belief_grid
        )
    ]
    total_all = len(all_tasks)

    # Filter out already-completed tasks
    if done_keys:
        tasks = [t for t in all_tasks if (t[0], t[1], t[2], t[3], t[4]) not in done_keys]
    else:
        tasks = all_tasks

    remaining = len(tasks)
    logger.info(
        "Precomputing %d POMDP instances (%d token_counts × %d steps "
        "× %d beliefs with closeness_thr=%.3f), using %d workers, "
        "chunksize=%d.  %d already done, %d remaining.",
        total_all, len(token_counts), len(step_range), len(belief_grid),
        closeness_thr, workers, chunksize, total_all - remaining, remaining,
    )

    if remaining == 0:
        logger.info("All tasks already completed — skipping to save.")
    else:
        t0 = time.perf_counter()
        completed_this_run = 0
        last_ckpt = 0

        pool_cfg = {**_cfg, "_num_workers": workers}
        # Allow at most 8 concurrent Julia inits to avoid NFS SIGABRT
        init_sem = multiprocessing.Semaphore(8)
        pool = Pool(processes=workers, initializer=_worker_init,
                     initargs=(pool_cfg, init_sem))
        failed_tasks: List[Tuple] = []
        # Stall detection: if no results arrive for STALL_TIMEOUT seconds
        # AFTER the first result, save checkpoint and exit instead of hanging.
        # We don't start the clock until the first result arrives — before that,
        # workers may still be initializing Julia (can take several minutes).
        STALL_TIMEOUT = 600  # 10 minutes
        try:
            it = pool.imap_unordered(_solve_single, tasks, chunksize=chunksize)
            last_result_time: float | None = None  # None = no result yet (init phase)
            while True:
                try:
                    key, act = it.next(timeout=60)
                except StopIteration:
                    break
                except multiprocessing.TimeoutError:
                    if last_result_time is None:
                        # Still in init phase — workers haven't returned anything yet.
                        alive = sum(1 for w in pool._pool if w.is_alive())
                        logger.debug(
                            "Workers still initializing (%d/%d alive) …",
                            alive, workers,
                        )
                        continue
                    stall_secs = time.monotonic() - last_result_time
                    alive = sum(1 for w in pool._pool if w.is_alive())
                    if stall_secs > STALL_TIMEOUT:
                        logger.error(
                            "Pool stalled — %d/%d workers alive, no results "
                            "for %.0fs.  Saving checkpoint and exiting.",
                            alive, workers, stall_secs,
                        )
                        _save_checkpoint(table, output_path, closeness_thr, belief_grid)
                        logger.info(
                            "Checkpoint saved with %d entries.  Re-run to resume.",
                            len(table),
                        )
                        _kill_pool(pool)
                    logger.debug(
                        "Waiting for results (%.0fs, %d/%d workers alive) …",
                        stall_secs, alive, workers,
                    )
                    continue

                last_result_time = time.monotonic()  # update on every result (init + stall)
                completed_this_run += 1
                if act is None:
                    failed_tasks.append(key)
                else:
                    table[key] = act

                if completed_this_run % 500 == 0 or completed_this_run == remaining:
                    elapsed = time.perf_counter() - t0
                    rate = completed_this_run / elapsed
                    eta = (remaining - completed_this_run) / rate if rate > 0 else float("inf")
                    logger.info(
                        "[%d/%d] %.1f tasks/s  ETA %.0fs  table size %d"
                        + (f"  ({len(failed_tasks)} failed)" if failed_tasks else ""),
                        completed_this_run, remaining, rate, eta, len(table),
                    )

                # Periodic checkpoint
                if checkpoint_every > 0 and completed_this_run - last_ckpt >= checkpoint_every:
                    try:
                        _save_checkpoint(table, output_path, closeness_thr, belief_grid)
                        last_ckpt = completed_this_run
                        logger.info("Checkpoint saved (%d entries).", len(table))
                    except Exception as ckpt_exc:
                        logger.warning("Checkpoint save failed (will retry next interval): %s", ckpt_exc)

        except KeyboardInterrupt:
            logger.warning("KeyboardInterrupt — saving checkpoint …")
            _save_checkpoint(table, output_path, closeness_thr, belief_grid)
            logger.info("Checkpoint saved with %d entries.  Re-run to resume.", len(table))
            _kill_pool(pool)
        except Exception as exc:
            # MaybeEncodingError = worker died mid-task.
            # Save progress and exit so user can re-run to resume.
            logger.warning(
                "Pool error (%s: %s) — saving checkpoint …",
                type(exc).__name__, exc,
            )
            _save_checkpoint(table, output_path, closeness_thr, belief_grid)
            logger.info(
                "Checkpoint saved with %d entries (%d failed).  Re-run to resume.",
                len(table), len(failed_tasks),
            )
            _kill_pool(pool)

        if failed_tasks:
            logger.warning(
                "%d tasks failed and will be retried on next run.", len(failed_tasks)
            )

        elapsed = time.perf_counter() - t0
        logger.info("Done in %.1fs — table has %d entries.", elapsed, len(table))

    # --- Save final output ---
    with open(output_path, "wb") as f:
        pickle.dump(
            {
                "config": _cfg,
                "closeness_thr": closeness_thr,
                "belief_grid": belief_grid,
                "table": table,
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    logger.info("Saved to %s", output_path)

    # Clean up checkpoint now that final output is written
    ckpt_path = output_path.with_suffix(".ckpt")
    if ckpt_path.exists():
        ckpt_path.unlink()
        logger.info("Removed checkpoint %s", ckpt_path)

    if remaining > 0:
        _kill_pool(pool)


# ---------------------------------------------------------------------------
# Runtime: load table + vectorised action query
# ---------------------------------------------------------------------------

def load_action_table(path: str | Path) -> Dict:
    """Load a precomputed action table from disk."""
    with open(path, "rb") as f:
        return pickle.load(f)


def _snap_belief(p: np.ndarray, step: float) -> np.ndarray:
    """Snap a belief vector to the nearest simplex grid point."""
    q = np.round(np.round(p / step) * step, 6)
    q = np.clip(q, 0.0, 1.0)
    q[..., -1] = np.round(1.0 - q[..., 0] - q[..., 1], 6)
    q = np.clip(q, 0.0, 1.0)
    return q


def get_POMDP_action(
    table_data: Dict,
    input_belief: np.ndarray,
    token_batch: np.ndarray,
    step_batch: np.ndarray,
    closeness_thr: float = 0.4,
) -> np.ndarray:
    """Vectorised action lookup from the precomputed table.

    Parameters
    ----------
    table_data : dict
        Output of :func:`load_action_table`.
    input_belief : ndarray, shape (B, 3)
        Belief distributions ``[p0, p1, p2]`` per sample.
    token_batch : array-like, shape (B,)
        Token counts per sample (will be clipped to [5, 500]).
    step_batch : array-like, shape (B,)
        Current step index per sample.
    closeness_thr : float
        Threshold on ``max(belief) - belief[2]``; samples where
        ``p2 <= max(p0, p1, p2) - closeness_thr`` are assigned
        action 0 without a table lookup.

    Returns
    -------
    actions : ndarray of int, shape (B,)
    """
    table = table_data["table"]
    cfg = table_data["config"]
    token_counts = cfg["token_counts"]
    max_steps = cfg["max_steps"]
    belief_step = cfg["belief_step"]

    belief = np.asarray(input_belief, dtype=np.float64)
    tokens = np.clip(np.asarray(token_batch, dtype=int),
                     min(token_counts), max(token_counts))
    steps = np.asarray(step_batch, dtype=int)

    max_belief = belief.max(axis=1)
    belief_state_2 = belief[:, -1]

    # p2 > max(p0,p1,p2) - closeness_thr  ⟺  max - p2 < closeness_thr
    mask = (max_belief - belief_state_2) < closeness_thr

    snapped = _snap_belief(belief, step=belief_step)
    actions = np.zeros(len(belief), dtype=int)

    tc_arr = np.array(token_counts)
    for i in np.where(mask)[0]:
        sn = int(steps[i])
        if sn >= max_steps:
            continue
        tc = int(tc_arr[np.argmin(np.abs(tc_arr - tokens[i]))])
        p0, p1, p2 = float(snapped[i, 0]), float(snapped[i, 1]), float(snapped[i, 2])
        key = (tc, sn, p0, p1, p2)
        actions[i] = table.get(key, 0)

    return actions


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_int_list(s: str) -> List[int]:
    """Parse a comma-separated string of ints."""
    return sorted(int(x.strip()) for x in s.split(","))


def main() -> None:
    global _cfg

    p = argparse.ArgumentParser(
        description="Precompute POMDP action lookup table.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- parallelism ---
    p.add_argument("--workers", type=int, default=None,
                   help="Parallel workers (default: min(nproc, 128)).")
    p.add_argument("--chunksize", type=int, default=1,
                   help="Tasks dispatched per worker at a time. Since each "
                        "SARSOP solve takes seconds, chunksize=1 gives the "
                        "best load balancing.")

    # --- POMDP parameters ---
    p.add_argument("--p-slm", type=float, default=_DEFAULTS["p_slm"],
                   help="Correctness probability of the SLM.")
    p.add_argument("--p-llm", type=float, default=_DEFAULTS["p_llm"],
                   help="Correctness probability of the LLM.")
    p.add_argument("--avg-token-count", type=int,
                   default=_DEFAULTS["avg_token_count"],
                   help="Average token count per reasoning step.")
    p.add_argument("--max-steps", type=int, default=_DEFAULTS["max_steps"],
                   help="Maximum number of reasoning steps.")
    p.add_argument("--sarsop-timeout", type=float,
                   default=_DEFAULTS["sarsop_timeout"],
                   help="SARSOP solver timeout per instance (seconds).")
    p.add_argument("--task-reward", type=float,
                   default=_DEFAULTS["task_reward"],
                   help="Reward for a correct final answer.")
    p.add_argument("--token-counts", type=str,
                   default=_DEFAULTS["token_counts"],
                   help="Comma-separated list of token count bins.")
    p.add_argument("--belief-step", type=float,
                   default=_DEFAULTS["belief_step"],
                   help="Discretisation step for belief simplex.")
    p.add_argument("--closeness-thr", type=float,
                   default=_DEFAULTS["closeness_thr"],
                   help="Belief closeness threshold. Only beliefs with "
                        "p2 > max(p0,p1,p2) - thr are precomputed.")
    p.add_argument("--cost-per-token", type=float,
                   default=_DEFAULTS["cost_per_token"],
                   help="Cost per token in the reward function.")

    # --- predictor paths ---
    p.add_argument("--terminal-benchmark", type=str,
                   default=_DEFAULTS["terminal_benchmark"],
                   help="Benchmark used when saving terminal predictors (e.g. 'math').")
    p.add_argument("--terminal-split", type=str,
                   default=_DEFAULTS["terminal_split"],
                   help="Split used when saving terminal predictors (e.g. 'train').")
    p.add_argument("--obs-benchmark", type=str,
                   default=_DEFAULTS["obs_benchmark"],
                   help="Benchmark used for the KDE observation model (e.g. 'omnimath').")
    p.add_argument("--obs-bin-size", type=float,
                   default=_DEFAULTS["obs_bin_size"],
                   help="Bin size used when saving obs distributions.")

    # --- output / checkpointing ---
    p.add_argument("--output", type=str, default=None,
                   help="Output path for the pickle file. "
                        "Default: pomdp_data/pomdp_action_table_cost{ratio}_cthr{thr}.pkl "
                        "where ratio = round(cost_per_token / task_reward, 6).")
    p.add_argument("--checkpoint-every", type=int, default=5000,
                   help="Save a checkpoint every N completed tasks.  "
                        "Set to 0 to disable.  On re-run, resumes "
                        "automatically from the latest checkpoint.")

    args = p.parse_args()

    if args.output is None:
        cost_ratio = round(args.cost_per_token / args.task_reward, 6)
        args.output = str(
            PREDICTOR_DIR
            / f"pomdp_action_table_cost{cost_ratio}_cthr{args.closeness_thr}.pkl"
        )

    log_fmt = "%(asctime)s %(levelname)s %(message)s"
    log_datefmt = "%H:%M:%S"
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    run_ts = time.strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"pomdp_policy_{run_ts}.log"

    logging.basicConfig(
        level=logging.INFO,
        format=log_fmt,
        datefmt=log_datefmt,
        handlers=[
            logging.StreamHandler(),                          # stderr (visible in terminal)
            logging.FileHandler(log_file, encoding="utf-8"), # file
        ],
    )
    logger.info("Logging to %s", log_file)

    # Populate module-level config (inherited by forked workers)
    _cfg.update(
        p_slm=args.p_slm,
        p_llm=args.p_llm,
        avg_token_count=args.avg_token_count,
        max_steps=args.max_steps,
        sarsop_timeout=args.sarsop_timeout,
        task_reward=args.task_reward,
        token_counts=_parse_int_list(args.token_counts),
        belief_step=args.belief_step,
        closeness_thr=args.closeness_thr,
        cost_per_token=args.cost_per_token,
        terminal_benchmark=args.terminal_benchmark,
        terminal_split=args.terminal_split,
        obs_benchmark=args.obs_benchmark,
        obs_bin_size=args.obs_bin_size,
    )

    logger.info("Config: %s", json.dumps(_cfg, indent=2))

    precompute_action_table(
        cost_per_token=args.cost_per_token,
        closeness_thr=args.closeness_thr,
        workers=args.workers,
        chunksize=args.chunksize,
        output_path=args.output,
        checkpoint_every=args.checkpoint_every,
    )


if __name__ == "__main__":
    main()
