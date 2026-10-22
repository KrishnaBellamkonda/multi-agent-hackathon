"""Scientific comparison on real GeneDisco multi-target data.

Six setups (all from MULTI_TARGET_SETUPS):
  - t_cell        : IFNγ + IL-2 (correlated, same cell type, K=2)
  - all_immune    : IFNγ + IL-2 + NK (same lab, different phenotypes, K=3)
  - cross_biology : Tau + SARS-CoV-2 (unrelated biology, K=2)
  - neuro_immune  : IFNγ + Tau (immune vs neuro, K=2)
  - virus_immune  : IFNγ + SARS-CoV-2 (immune vs virology, K=2)
  - diverse       : IFNγ + IL-2 + Tau + SARS-CoV-2 (max diversity, K=4)

Four methods compared:
  - random          : uniform random baseline (lower bound)
  - parego          : ParEGO — random Chebyshev scalarization + single GP (from paper)
  - singleagent     : one LLM agent, all targets visible, direct select_genes() [ablation]
  - multiagent      : K specialist LLMs + text-based moderator (our main method)

GP methods (random, parego) use PCA(20) to reduce 808-dim features.
LLM methods (singleagent, multiagent) use BiologyEnv(use_gp=False) with no surrogate.

Usage:
    cd /Users/ar36/Desktop/mas-hackathon

    # Quick smoke test (1 seed, 3 cycles, t_cell only, all 4 methods):
    python -m ai.experiments.compare_real --quick

    # GP baselines only (no API key needed):
    python -m ai.experiments.compare_real --no-agent --setup t_cell --seeds 42

    # Full comparison, all 6 setups, 2 seeds:
    python -m ai.experiments.compare_real

    # Single setup:
    python -m ai.experiments.compare_real --setup cross_biology

    # Subset of setups:
    python -m ai.experiments.compare_real --setup t_cell cross_biology neuro_immune
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_ED_ROOT = os.path.join(_REPO_ROOT, "ExperimentalDesign")
if _ED_ROOT not in sys.path:
    sys.path.insert(0, _ED_ROOT)

# Load .env
_env_path = os.path.join(_REPO_ROOT, ".env")
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _, _v = _line.partition("=")
                os.environ.setdefault(_k.strip(), _v.strip())

import numpy as np

from maehd.datasets.genedisco import load_multitarget, MULTI_TARGET_SETUPS
from ExperimentalDesign.reproducibility.shared.multitarget_experiment import (
    MultiTargetExperimentConfig,
    run_multitarget_experiment,
)
from ExperimentalDesign.reproducibility.shared.parego_experiment import (
    ParegoExperimentConfig,
    run_parego_experiment,
)
from acua.envs.biology.env import BiologyEnv, BiologyEnvConfig
from acua.agents.biology_orchestrator import BiologyOrchestratorAgent
from acua.agents.biology_singleagent import BiologySingleAgent
from acua.agents.biology_specialist import SpecialistConfig


# ── Experiment defaults ───────────────────────────────────────────────────────

SETUPS: dict[str, dict] = {
    "t_cell": {
        "screen_names": MULTI_TARGET_SETUPS["t_cell"],
        "description": "T cells — correlated targets (IFNγ + IL-2, Schmidt 2021)",
    },
    "all_immune": {
        "screen_names": MULTI_TARGET_SETUPS["all_immune"],
        "description": "All immune — 3 targets (IFNγ + IL-2 + NK, Schmidt/Zhuang 2021)",
    },
    "cross_biology": {
        "screen_names": MULTI_TARGET_SETUPS["cross_biology"],
        "description": "Cross-biology — unrelated targets (Tau + SARS-CoV-2)",
    },
    "neuro_immune": {
        "screen_names": MULTI_TARGET_SETUPS["neuro_immune"],
        "description": "Neuro-immune — different biology (IFNγ + Tau)",
    },
    "virus_immune": {
        "screen_names": MULTI_TARGET_SETUPS["virus_immune"],
        "description": "Virus-immune — immune vs virology (IFNγ + SARS-CoV-2)",
    },
    "diverse": {
        "screen_names": MULTI_TARGET_SETUPS["diverse"],
        "description": "Diverse — 4 targets across immune + neuro + virology",
    },
}

SEEDS = [42, 43]
N_CYCLES = 10
BATCH_SIZE = 20
N_INITIAL = 20
TOPK_PERCENT = 0.05
PCA_DIMS = 20
NOMINATIONS_PER_SPECIALIST = BATCH_SIZE  # each specialist nominates a full batch-worth; moderator trims
POOL_SAMPLE_SIZE = 200
MODEL = "gpt-4o-mini"          # cheap model for fast iteration


# ── Unified result container ──────────────────────────────────────────────────

@dataclass
class RunResult:
    method: str
    setup: str
    seed: int
    screen_names: list[str]
    recall_curves: list[list[float]] = field(default_factory=list)   # [K][cycle]
    final_per_target_recall: list[float] = field(default_factory=list)
    final_mean_recall: float = 0.0
    n_hits_per_target: list[int] = field(default_factory=list)        # total hits in screen
    elapsed_sec: float = 0.0
    success: bool = True
    error_message: str = ""

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "setup": self.setup,
            "seed": self.seed,
            "screen_names": self.screen_names,
            "recall_curves": self.recall_curves,
            "final_per_target_recall": self.final_per_target_recall,
            "final_mean_recall": round(self.final_mean_recall, 4),
            "n_hits_per_target": self.n_hits_per_target,
            "elapsed_sec": round(self.elapsed_sec, 2),
            "success": self.success,
            "error_message": self.error_message,
        }


# ── GP method runners ─────────────────────────────────────────────────────────

def run_random(
    setup_name: str, screen_names: list[str],
    X: np.ndarray, Y: np.ndarray, meta: dict, seed: int,
    n_cycles: int,
) -> RunResult:
    """Random selection baseline (no model)."""
    from sklearn.decomposition import PCA

    t0 = time.perf_counter()
    K = Y.shape[1]

    X_pca = PCA(n_components=min(PCA_DIMS, X.shape[1]), random_state=seed).fit_transform(X).astype(np.float32)

    cfg = MultiTargetExperimentConfig(
        screen_names=screen_names,
        method_name="random",
        batch_size=BATCH_SIZE,
        threshold_pct=TOPK_PERCENT,
        seed=seed,
        n_cycles=n_cycles,
        n_initial=N_INITIAL,
    )
    result = run_multitarget_experiment(cfg, X=X_pca, Y=Y, thresholds=meta["thresholds"],
                                        gene_names=meta["gene_names"], verbose=True)
    elapsed = time.perf_counter() - t0

    recall_curves: list[list[float]] = [[] for _ in range(K)]
    for c in result.cycles:
        for k in range(K):
            recall_curves[k].append(float(c.per_target_recall[k]))
    final = result.final_per_target_recall.tolist() if result.success else [0.0] * K

    return RunResult(
        method="random", setup=setup_name, seed=seed, screen_names=screen_names,
        recall_curves=recall_curves, final_per_target_recall=final,
        final_mean_recall=float(np.mean(final)),
        n_hits_per_target=list(meta["n_hits_per_target"]),
        elapsed_sec=elapsed,
        success=result.success, error_message=result.error_message or "",
    )


def run_parego(
    setup_name: str, screen_names: list[str],
    X: np.ndarray, Y: np.ndarray, meta: dict, seed: int,
    n_cycles: int,
) -> RunResult:
    """ParEGO: random Chebyshev scalarization + single GP."""
    from sklearn.decomposition import PCA

    t0 = time.perf_counter()
    K = Y.shape[1]

    X_pca = PCA(n_components=min(PCA_DIMS, X.shape[1]), random_state=seed).fit_transform(X).astype(np.float32)

    cfg = ParegoExperimentConfig(
        screen_names=screen_names,
        acq_function="probability_hit",
        batch_size=BATCH_SIZE,
        threshold_pct=TOPK_PERCENT,
        seed=seed,
        n_cycles=n_cycles,
        n_initial=N_INITIAL,
    )
    result = run_parego_experiment(cfg, X=X_pca, Y=Y, thresholds=meta["thresholds"],
                                   gene_names=meta["gene_names"], verbose=True)
    elapsed = time.perf_counter() - t0

    recall_curves: list[list[float]] = [[] for _ in range(K)]
    for c in result.cycles:
        for k in range(K):
            recall_curves[k].append(float(c.per_target_recall[k]))
    final = result.final_per_target_recall.tolist() if result.success else [0.0] * K

    return RunResult(
        method="parego", setup=setup_name, seed=seed, screen_names=screen_names,
        recall_curves=recall_curves, final_per_target_recall=final,
        final_mean_recall=float(np.mean(final)),
        n_hits_per_target=list(meta["n_hits_per_target"]),
        elapsed_sec=elapsed,
        success=result.success, error_message=result.error_message or "",
    )


# ── LLM method runners ────────────────────────────────────────────────────────

def _make_env(
    X: np.ndarray, Y: np.ndarray, meta: dict,
    screen_names: list[str], seed: int, n_cycles: int,
) -> BiologyEnv:
    cfg = BiologyEnvConfig(
        use_gp=False,
        batch_size=BATCH_SIZE,
        n_cycles=n_cycles,
        n_initial_random=N_INITIAL,
        pool_sample_size=POOL_SAMPLE_SIZE,
        seed=seed,
    )
    return BiologyEnv(
        X=X, Y=Y,
        thresholds=meta["thresholds"],
        gene_names=meta["gene_names"],
        target_names=screen_names,
        config=cfg,
    )


def _run_llm_episode(
    method_name: str,
    setup_name: str,
    screen_names: list[str],
    env: BiologyEnv,
    agent,
    n_cycles: int,
) -> RunResult:
    """Generic episode loop for any LLM agent (single or multi)."""
    K = len(screen_names)
    recall_curves: list[list[float]] = [[] for _ in range(K)]

    try:
        obs = env.reset()
        done = False
        while not done:
            exp = obs.experiment
            cycle = (exp.cycle + 1) if exp else "?"
            print(f"  [cycle {cycle}/{n_cycles}]", end=" ", flush=True)

            action = agent.get_python_action(obs)
            print(f"→ {action[:60]}{'...' if len(action) > 60 else ''}")

            obs, reward, done, info = env.step(action)

            if obs.target_metrics:
                tm = obs.target_metrics
                n_obs = obs.experiment.n_observed if obs.experiment else 1
                for k in range(K):
                    recall_curves[k].append(float(tm.per_target_recall[k]))
                mean_r = sum(tm.per_target_recall) / K
                # Show hits/tested (empirical hit rate) alongside screen recall
                hits_str = "  ".join(
                    f"{n}={h}/{n_obs}({r:.1%})"
                    for n, h, r in zip(tm.target_names, tm.per_target_hits, tm.per_target_recall)
                )
                n_batch_hits = info.get("per_target_new_hits", [])
                batch_str = f"  batch={sum(n_batch_hits)}/{env._batch_size}" if n_batch_hits else ""
                print(f"    hits/tested: {hits_str}  mean_recall={mean_r:.3f}{batch_str}")

            if info.get("parse_error"):
                print(f"    [env error] {info['parse_error']}")

        env.close()

        if obs.target_metrics:
            final = list(obs.target_metrics.per_target_recall)
        elif recall_curves and recall_curves[0]:
            final = [curve[-1] for curve in recall_curves]
        else:
            final = [0.0] * K

        n_hits = [len(env._true_hits[k]) for k in range(K)]
        return RunResult(
            method=method_name, setup=setup_name, seed=env._seed,
            screen_names=screen_names, recall_curves=recall_curves,
            final_per_target_recall=final, final_mean_recall=float(np.mean(final)),
            n_hits_per_target=n_hits, success=True,
        )

    except Exception as exc:
        env.close()
        print(f"  [ERROR] {exc}")
        return RunResult(
            method=method_name, setup=setup_name, seed=env._seed,
            screen_names=screen_names,
            recall_curves=[[] for _ in range(K)],
            final_per_target_recall=[0.0] * K, final_mean_recall=0.0,
            success=False, error_message=str(exc),
        )


def run_singleagent(
    setup_name: str, screen_names: list[str],
    X: np.ndarray, Y: np.ndarray, meta: dict,
    seed: int, n_cycles: int, api_key: str,
) -> RunResult:
    """Ablation: single LLM agent, all targets visible, direct gene selection."""
    from acua.providers.openai_provider import OpenAIProvider

    t0 = time.perf_counter()
    env = _make_env(X, Y, meta, screen_names, seed, n_cycles)
    provider = OpenAIProvider(model_name=MODEL, api_key=api_key, max_tokens=512)
    agent = BiologySingleAgent(
        target_names=screen_names,
        provider=provider,
        batch_size=BATCH_SIZE,
        keep_last_n_obs=5,
    )
    result = _run_llm_episode("singleagent", setup_name, screen_names, env, agent, n_cycles)
    result.elapsed_sec = time.perf_counter() - t0
    return result


def run_multiagent(
    setup_name: str, screen_names: list[str],
    X: np.ndarray, Y: np.ndarray, meta: dict,
    seed: int, n_cycles: int, api_key: str,
) -> RunResult:
    """Multi-agent: K specialist LLMs + moderator (main method)."""
    from acua.providers.openai_provider import OpenAIProvider

    t0 = time.perf_counter()
    env = _make_env(X, Y, meta, screen_names, seed, n_cycles)
    provider = OpenAIProvider(model_name=MODEL, api_key=api_key, max_tokens=512)
    specialist_configs = [
        SpecialistConfig(name, provider, nominations=NOMINATIONS_PER_SPECIALIST)
        for name in screen_names
    ]
    orchestrator = BiologyOrchestratorAgent(
        moderator_provider=provider,
        specialist_configs=specialist_configs,
        batch_size=BATCH_SIZE,
    )

    # Wrap orchestrator in the standard episode loop
    K = len(screen_names)
    recall_curves: list[list[float]] = [[] for _ in range(K)]
    t0_inner = time.perf_counter()

    try:
        obs = env.reset()
        done = False
        while not done:
            exp = obs.experiment
            cycle = (exp.cycle + 1) if exp else "?"
            print(f"  [cycle {cycle}/{n_cycles}] specialists nominating...")

            action = orchestrator.get_python_action(obs)

            for tgt, nom in orchestrator.last_nominations.items():
                genes_str = ", ".join(nom["genes"][:4])
                print(f"    [{tgt}] → {genes_str}")
            print(f"    [moderator] → {action[:70]}{'...' if len(action) > 70 else ''}")

            obs, reward, done, info = env.step(action)

            if obs.target_metrics:
                tm = obs.target_metrics
                n_obs = obs.experiment.n_observed if obs.experiment else 1
                for k in range(K):
                    recall_curves[k].append(float(tm.per_target_recall[k]))
                mean_r = sum(tm.per_target_recall) / K
                hits_str = "  ".join(
                    f"{n}={h}/{n_obs}({r:.1%})"
                    for n, h, r in zip(tm.target_names, tm.per_target_hits, tm.per_target_recall)
                )
                n_batch_hits = info.get("per_target_new_hits", [])
                batch_str = f"  batch={sum(n_batch_hits)}/{env._batch_size}" if n_batch_hits else ""
                print(f"    hits/tested: {hits_str}  mean_recall={mean_r:.3f}{batch_str}")

            if info.get("parse_error"):
                print(f"    [env error] {info['parse_error']}")

        env.close()
        if obs.target_metrics:
            final = list(obs.target_metrics.per_target_recall)
        elif recall_curves and recall_curves[0]:
            final = [curve[-1] for curve in recall_curves]
        else:
            final = [0.0] * K

        n_hits = [len(env._true_hits[k]) for k in range(K)]
        return RunResult(
            method="multiagent", setup=setup_name, seed=seed,
            screen_names=screen_names, recall_curves=recall_curves,
            final_per_target_recall=final, final_mean_recall=float(np.mean(final)),
            n_hits_per_target=n_hits, elapsed_sec=time.perf_counter() - t0, success=True,
        )

    except Exception as exc:
        env.close()
        print(f"  [ERROR] {exc}")
        return RunResult(
            method="multiagent", setup=setup_name, seed=seed,
            screen_names=screen_names,
            recall_curves=[[] for _ in range(K)],
            final_per_target_recall=[0.0] * K, final_mean_recall=0.0,
            n_hits_per_target=list(meta["n_hits_per_target"]),
            elapsed_sec=time.perf_counter() - t0, success=False, error_message=str(exc),
        )


# ── Reporting ─────────────────────────────────────────────────────────────────

def _mean_curve(curves: list[list[float]]) -> list[float]:
    """Average list of curves (different lengths OK)."""
    if not curves:
        return []
    min_len = min(len(c) for c in curves if c)
    if min_len == 0:
        return []
    return [float(np.mean([c[i] for c in curves if len(c) > i])) for i in range(min_len)]


def print_curve_comparison(all_results: list[RunResult], setup_name: str) -> None:
    from collections import defaultdict

    by_method: dict[str, list[RunResult]] = defaultdict(list)
    for r in all_results:
        if r.setup == setup_name:
            by_method[r.method].append(r)

    if not by_method:
        return

    methods = [m for m in ["random", "parego", "singleagent", "multiagent"] if m in by_method]
    print(f"\n  Recall curve (mean across targets & seeds) — {setup_name}")
    print(f"  {'Cycle':<8}", end="")
    for m in methods:
        print(f"  {m:<15}", end="")
    print()
    print(f"  {'-'*7}", end="")
    for _ in methods:
        print(f"  {'-'*15}", end="")
    print()

    method_curves: dict[str, list[float]] = {}
    for m in methods:
        seed_mean_curves = []
        for r in by_method[m]:
            if not r.success or not r.recall_curves or not r.recall_curves[0]:
                continue
            K = len(r.recall_curves)
            n_cyc = min(len(c) for c in r.recall_curves if c)
            per_cycle = [float(np.mean([r.recall_curves[k][i] for k in range(K)])) for i in range(n_cyc)]
            seed_mean_curves.append(per_cycle)
        method_curves[m] = _mean_curve(seed_mean_curves)

    max_cycles = max((len(c) for c in method_curves.values()), default=0)
    for i in range(max_cycles):
        print(f"  {i+1:<8}", end="")
        for m in methods:
            val = method_curves[m][i] if i < len(method_curves[m]) else float("nan")
            print(f"  {val:<15.3f}", end="")
        print()


def print_summary_table(all_results: list[RunResult], setups: list[str]) -> None:
    """Print summary table.

    Per-target columns show: recall (hits_found / total_hits) and absolute hit count.
    Total genes queried = N_INITIAL + n_cycles * BATCH_SIZE.
    """
    from collections import defaultdict

    n_queried = N_INITIAL + N_CYCLES * BATCH_SIZE
    grouped: dict[tuple, list[RunResult]] = defaultdict(list)
    for r in all_results:
        grouped[(r.setup, r.method)].append(r)

    print("\n" + "=" * 100)
    print(f"FINAL RESULTS  (mean ± std across seeds | {n_queried} genes queried per run)")
    print(f"Per-target: recall = hits_found / total_screen_hits  |  shown as recall (hits_found/total)")
    print("=" * 100)

    for setup_name in setups:
        # Print setup header with screen names and total hits
        first_run = next(
            (r for r in all_results if r.setup == setup_name and r.success), None
        )
        if first_run and first_run.n_hits_per_target:
            totals = "  ".join(
                f"{n}: {h} hits"
                for n, h in zip(first_run.screen_names, first_run.n_hits_per_target)
            )
            print(f"\n  [{setup_name}]  {totals}")
        else:
            print(f"\n  [{setup_name}]")

        print(f"  {'Method':<14} {'Mean recall':>12}  Per-target recall (hits found / total)")
        print(f"  {'-'*13} {'-'*12}  {'-'*55}")

        for method in ["random", "parego", "singleagent", "multiagent"]:
            key = (setup_name, method)
            if key not in grouped:
                continue
            runs = grouped[key]
            success_runs = [r for r in runs if r.success]
            if not success_runs:
                print(f"  {method:<14} {'FAILED':>12}")
                continue

            avg = float(np.mean([r.final_mean_recall for r in success_runs]))
            std = float(np.std([r.final_mean_recall for r in success_runs])) if len(success_runs) > 1 else 0.0
            K = len(success_runs[0].screen_names)

            per_target_parts = []
            for k in range(K):
                recalls = [r.final_per_target_recall[k] for r in success_runs if len(r.final_per_target_recall) > k]
                n_hits_list = [r.n_hits_per_target[k] for r in success_runs if r.n_hits_per_target and len(r.n_hits_per_target) > k]
                if recalls and n_hits_list:
                    r_mean = float(np.mean(recalls))
                    total_h = int(np.mean(n_hits_list))
                    found = int(round(r_mean * total_h))
                    per_target_parts.append(f"{r_mean:.3f} ({found}/{total_h})")
                elif recalls:
                    per_target_parts.append(f"{float(np.mean(recalls)):.3f}")
                else:
                    per_target_parts.append("n/a")

            pt_str = "  ".join(per_target_parts)
            timing = np.mean([r.elapsed_sec for r in success_runs])
            n_ok = len(success_runs)
            n_total = len(runs)
            seed_note = f" [{n_ok}/{n_total} seeds]" if n_ok < n_total else ""
            print(
                f"  {method:<14} {avg:>6.3f} ±{std:<5.3f}  {pt_str}  ({timing:.0f}s{seed_note})"
            )

        print()

    print("=" * 100)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare random / ParEGO / SingleAgent / MultiAgent on real GeneDisco data."
    )
    parser.add_argument("--quick", action="store_true",
                        help="Smoke test: 1 seed, 3 cycles, t_cell only, all methods")
    parser.add_argument("--no-agent", action="store_true",
                        help="Skip LLM methods (GP baselines only, no API key needed)")
    parser.add_argument(
        "--setup", nargs="+",
        choices=list(SETUPS.keys()) + ["all"],
        default=["all"],
        help="One or more setup names, or 'all' (default: all)",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--cycles", type=int, default=N_CYCLES)
    parser.add_argument("--methods", nargs="+",
                        choices=["random", "parego", "singleagent", "multiagent"],
                        default=["random", "parego", "singleagent", "multiagent"],
                        help="Which methods to run (default: all)")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    # --quick overrides other settings
    if args.quick:
        args.seeds = [42]
        args.cycles = 3
        args.setup = ["t_cell"]
        print("Quick mode: 1 seed, 3 cycles, t_cell only")

    api_key: str | None = None
    llm_methods = {"singleagent", "multiagent"}
    running_llm = not args.no_agent and any(m in llm_methods for m in args.methods)
    if running_llm:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("ERROR: OPENAI_API_KEY not set. Use --no-agent to skip LLM methods.")
            sys.exit(1)

    # Resolve setup list: "all" expands to every key in SETUPS
    raw_setups = args.setup if isinstance(args.setup, list) else [args.setup]
    if "all" in raw_setups:
        setups_to_run = list(SETUPS.keys())
    else:
        setups_to_run = [s for s in raw_setups if s in SETUPS]
    methods = args.methods if not args.no_agent else [m for m in args.methods if m not in llm_methods]
    seeds = args.seeds
    n_cycles = args.cycles
    output_path = args.output or os.path.join(os.path.dirname(__file__), "compare_real_results.json")

    print(f"Setups : {setups_to_run}")
    print(f"Methods: {methods}")
    print(f"Seeds  : {seeds}  |  Cycles: {n_cycles}  |  Batch: {BATCH_SIZE}  |  Initial: {N_INITIAL}")
    if running_llm:
        print(f"Model  : {MODEL}  |  Pool sample: {POOL_SAMPLE_SIZE}  |  Nominations/specialist: {NOMINATIONS_PER_SPECIALIST}")
    print()

    all_results: list[RunResult] = []

    for setup_name in setups_to_run:
        info = SETUPS[setup_name]
        screen_names: list[str] = info["screen_names"]

        print(f"\n{'='*72}")
        print(f"SETUP: {setup_name}  —  {info['description']}")
        print(f"Targets: {screen_names}")
        print(f"{'='*72}")

        print("Loading data...", end=" ", flush=True)
        X, Y, meta = load_multitarget(setup=setup_name, feature_set="achilles", topk_percent=TOPK_PERCENT)
        print(f"{meta['n_genes']:,} genes × {meta['d_features']} features × {meta['K']} targets")
        for k, sname in enumerate(meta["screen_names"]):
            print(f"  {sname}: {meta['n_hits_per_target'][k]} hits ({TOPK_PERCENT*100:.0f}% threshold)")

        for seed in seeds:
            if "random" in methods:
                label = f"random / {setup_name} / seed={seed}"
                print(f"\n--- {label} ---")
                result = run_random(setup_name, screen_names, X, Y, meta, seed, n_cycles)
                all_results.append(result)
                status = f"mean={result.final_mean_recall:.3f}" if result.success else f"FAILED: {result.error_message[:60]}"
                print(f"  Done {result.elapsed_sec:.1f}s | {status}")

            if "parego" in methods:
                label = f"parego / {setup_name} / seed={seed}"
                print(f"\n--- {label} ---")
                result = run_parego(setup_name, screen_names, X, Y, meta, seed, n_cycles)
                all_results.append(result)
                status = f"mean={result.final_mean_recall:.3f}" if result.success else f"FAILED: {result.error_message[:60]}"
                print(f"  Done {result.elapsed_sec:.1f}s | {status}")

            if "singleagent" in methods and not args.no_agent:
                label = f"singleagent / {setup_name} / seed={seed}"
                print(f"\n--- {label} ---")
                result = run_singleagent(setup_name, screen_names, X, Y, meta, seed, n_cycles, api_key)
                all_results.append(result)
                status = f"mean={result.final_mean_recall:.3f}" if result.success else f"FAILED: {result.error_message[:60]}"
                print(f"  Done {result.elapsed_sec:.1f}s | {status}")

            if "multiagent" in methods and not args.no_agent:
                label = f"multiagent / {setup_name} / seed={seed}"
                print(f"\n--- {label} ---")
                result = run_multiagent(setup_name, screen_names, X, Y, meta, seed, n_cycles, api_key)
                all_results.append(result)
                status = f"mean={result.final_mean_recall:.3f}" if result.success else f"FAILED: {result.error_message[:60]}"
                print(f"  Done {result.elapsed_sec:.1f}s | {status}")

    # ── Reporting ─────────────────────────────────────────────────────
    for setup_name in setups_to_run:
        print_curve_comparison(all_results, setup_name)

    print_summary_table(all_results, setups_to_run)

    # ── Save ──────────────────────────────────────────────────────────
    output = {
        "config": {
            "setups": setups_to_run,
            "methods": methods,
            "seeds": seeds,
            "n_cycles": n_cycles,
            "batch_size": BATCH_SIZE,
            "n_initial": N_INITIAL,
            "topk_percent": TOPK_PERCENT,
            "pca_dims": PCA_DIMS,
            "model": MODEL if running_llm else None,
        },
        "results": [r.to_dict() for r in all_results],
    }
    with open(output_path, "w") as fh:
        json.dump(output, fh, indent=2)
    print(f"\nResults saved → {output_path}")


if __name__ == "__main__":
    main()
