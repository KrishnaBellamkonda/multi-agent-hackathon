"""Toy-dataset comparison: naive GP baseline vs multi-agent orchestrator.

Two hard synthetic multi-target tasks:

  Task 1 — pathway_multitarget
      n=500 genes, K=2 targets, 8 pathways, d=6 features.
      Targets share only ~25% of pathways → nearly independent hit locations
      (Jaccard ≈ 0.07).  The landscape is multimodal (8 Gaussian pathway bumps).

  Task 2 — correlated_targets
      n=500 genes, K=3 targets, d=6 features, overlap=0.05.
      Three mostly-independent targets.  Budget allocation is the key challenge.

Experiment parameters (same for all methods):
  - topk_percent = 0.05  → ~24 hits per target (top 5%)
  - n_initial    = 20    → random seed before cycle 1
  - batch_size   = 10    → 2% of pool per cycle (deliberately small)
  - n_cycles     = 15    → 170 total queries = 34% of pool
  - seeds        = [42, 43, 44]

Methods compared:
  random          — uniform random (lower bound)
  probability_hit — naive GP baseline: K independent GPyTorch GPs, equal sub-batch
  multiagent      — BiologyOrchestratorAgent: K specialist LLMs + moderator, no GP

Usage:
    cd /Users/ar36/Desktop/mas-hackathon
    # GP baselines only (no API key needed):
    python -m ai.experiments.compare_toy --no-agent

    # Full comparison including multi-agent:
    python -m ai.experiments.compare_toy
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Load .env if present
_env_path = os.path.join(_REPO_ROOT, ".env")
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _, _v = _line.partition("=")
                os.environ.setdefault(_k.strip(), _v.strip())

import numpy as np

from maehd.datasets.synthetic import (
    correlated_targets,
    pathway_multitarget,
    compute_thresholds,
    compute_hit_overlap,
)
from ExperimentalDesign.reproducibility.shared.multitarget_experiment import (
    MultiTargetExperimentConfig,
    run_multitarget_experiment,
)

# ── Experiment parameters ─────────────────────────────────────────────────────

TOPK_PERCENT   = 0.05   # top-5% are "hits"
N_INITIAL      = 20     # random genes observed before cycle 1
BATCH_SIZE     = 10     # genes queried per cycle (2% of n=500)
N_CYCLES       = 15     # acquisition cycles; total = n_initial + n_cycles * batch = 170
SEEDS          = [42, 43, 44]
GP_ITERATIONS  = 100
GP_LR          = 0.1
MODEL          = "gpt-4o-mini"   # cheap dev model for multi-agent

# Hard dataset configurations
DATASETS = {
    "pathway_multitarget": {
        "description": "Multimodal pathway landscape, K=2 near-independent targets",
        "fn": pathway_multitarget,
        "kwargs": dict(
            n_genes=500, n_pathways=8, K=2, d_gene=6,
            pathway_overlap=0.25, noise=0.08,
        ),
        "K": 2,
        "target_names": ["target_A", "target_B"],
    },
    "correlated_targets": {
        "description": "K=3 mostly-independent targets, budget allocation challenge",
        "fn": correlated_targets,
        "kwargs": dict(n=500, d=6, K=3, overlap=0.05, noise=0.05),
        "K": 3,
        "target_names": ["target_A", "target_B", "target_C"],
    },
}


# ── Dataset loader ────────────────────────────────────────────────────────────


def load_dataset(name: str, seed: int) -> tuple:
    """Return (X, Y, thresholds, gene_names, target_names) for one seed."""
    cfg = DATASETS[name]
    X, Y, meta = cfg["fn"](**cfg["kwargs"], seed=seed)
    thresholds = compute_thresholds(Y, topk_percent=TOPK_PERCENT)
    gene_names = [f"GENE_{i:04d}" for i in range(len(X))]
    return X, Y, thresholds, gene_names, cfg["target_names"]


def print_dataset_summary(name: str) -> None:
    """Print hit counts, overlap and difficulty metrics for the dataset."""
    cfg = DATASETS[name]
    X, Y, meta = cfg["fn"](**cfg["kwargs"], seed=42)
    thresholds = compute_thresholds(Y, topk_percent=TOPK_PERCENT)
    info = compute_hit_overlap(Y, thresholds)
    n = len(X)
    total_q = N_INITIAL + N_CYCLES * BATCH_SIZE

    print(f"\n  {name}  [{cfg['description']}]")
    print(f"    n={n}, d={X.shape[1]}, K={Y.shape[1]}, topk={TOPK_PERCENT:.0%}")
    print(f"    Hits per target: {info['per_target_hits']}")
    print(f"    Any-target hits: {info['any_target_hits']}  "
          f"All-target hits: {info['all_target_hits']}")
    if Y.shape[1] == 2:
        print(f"    Jaccard (target overlap): {info['jaccard'][0,1]:.3f}")
    print(f"    Total queries: {total_q} / {n} = {total_q/n:.1%}  "
          f"(random recall ≈ {total_q/n:.1%})")


# ── Naive GP baseline runner ──────────────────────────────────────────────────


def run_naive(
    dataset_name: str,
    method: str,
    seed: int,
    verbose: bool = False,
) -> dict:
    """Run ExperimentalDesign naive baseline (independent GPyTorch GPs).

    Returns dict with keys:
      per_cycle_recall: list of (K,) arrays, one per cycle
      final_mean_recall: float
      method, seed, dataset
    """
    X, Y, thresholds, gene_names, target_names = load_dataset(dataset_name, seed)
    K = Y.shape[1]

    config = MultiTargetExperimentConfig(
        screen_names=target_names,
        feature_set="achilles",   # irrelevant when X is supplied directly
        method_name=method,
        batch_size=BATCH_SIZE,
        threshold_pct=TOPK_PERCENT,
        seed=seed,
        n_cycles=N_CYCLES,
        n_initial=N_INITIAL,
        gp_iterations=GP_ITERATIONS,
        gp_lr=GP_LR,
    )

    result = run_multitarget_experiment(
        config=config,
        X=X,
        Y=Y,
        thresholds=thresholds,
        gene_names=gene_names,
        verbose=verbose,
    )

    if not result.success:
        print(f"    WARNING: naive run failed: {result.error_message}")
        return {
            "method": method, "seed": seed, "dataset": dataset_name,
            "per_cycle_mean_recall": [0.0] * N_CYCLES,
            "per_cycle_per_target": [[0.0] * K for _ in range(N_CYCLES)],
            "final_mean_recall": 0.0,
            "success": False,
        }

    per_cycle_mean = [c.mean_recall for c in result.cycles]
    per_cycle_per_target = [c.per_target_recall.tolist() for c in result.cycles]

    return {
        "method": method,
        "seed": seed,
        "dataset": dataset_name,
        "per_cycle_mean_recall": per_cycle_mean,
        "per_cycle_per_target": per_cycle_per_target,
        "final_mean_recall": result.final_mean_recall,
        "success": True,
    }


# ── Multi-agent runner ────────────────────────────────────────────────────────


def run_multiagent(
    dataset_name: str,
    seed: int,
    api_key: str,
    verbose: bool = False,
) -> dict:
    """Run BiologyOrchestratorAgent on the dataset.

    Returns dict with same structure as run_naive().
    """
    from acua.envs.biology.env import BiologyEnv, BiologyEnvConfig
    from acua.agents.biology_orchestrator import BiologyOrchestratorAgent
    from acua.agents.biology_specialist import SpecialistConfig
    from acua.providers.openai_provider import OpenAIProvider
    import maehd.metrics as maehd_metrics

    X, Y, thresholds, gene_names, target_names = load_dataset(dataset_name, seed)
    K = Y.shape[1]

    # Env in multi-agent mode (no GP surrogate)
    cfg = BiologyEnvConfig(
        use_gp=False,
        batch_size=BATCH_SIZE,
        n_cycles=N_CYCLES,
        n_initial_random=N_INITIAL,
        pool_sample_size=60,   # genes shown per specialist per cycle
        seed=seed,
    )
    env = BiologyEnv(
        X=X, Y=Y, thresholds=thresholds,
        gene_names=gene_names, target_names=target_names,
        config=cfg,
    )

    # Shared provider — all specialists and moderator use the same model
    # to keep API costs low during benchmarking.
    provider = OpenAIProvider(
        model_name=MODEL,
        api_key=api_key,
        max_tokens=600,
    )

    specialist_configs = [
        SpecialistConfig(
            target_name=target_names[k],
            provider=provider,
            # Each specialist nominates batch_size // K + 1 genes so there are
            # enough nominations to fill the batch after deduplication.
            nominations=max(3, BATCH_SIZE // K + 2),
        )
        for k in range(K)
    ]

    orchestrator = BiologyOrchestratorAgent(
        moderator_provider=provider,
        specialist_configs=specialist_configs,
        batch_size=BATCH_SIZE,
    )

    obs = env.reset()
    per_cycle_mean: list[float] = []
    per_cycle_per_target: list[list[float]] = []
    done = False

    while not done:
        action = orchestrator.get_python_action(obs)
        obs, reward, done, info = env.step(action)

        if obs.target_metrics:
            tm = obs.target_metrics
            per_cycle_mean.append(sum(tm.per_target_recall) / len(tm.per_target_recall))
            per_cycle_per_target.append(list(tm.per_target_recall))

        if verbose and obs.target_metrics:
            tm = obs.target_metrics
            recalls = "  ".join(
                f"{n}={r:.3f}" for n, r in zip(tm.target_names, tm.per_target_recall)
            )
            print(f"      cycle {obs.experiment.cycle}: {recalls}")

        if info.get("parse_error") and verbose:
            print(f"      [env error] {info['parse_error']}")

    final_mean = per_cycle_mean[-1] if per_cycle_mean else 0.0
    env.close()

    return {
        "method": "multiagent",
        "seed": seed,
        "dataset": dataset_name,
        "per_cycle_mean_recall": per_cycle_mean,
        "per_cycle_per_target": per_cycle_per_target,
        "final_mean_recall": final_mean,
        "success": True,
    }


# ── Aggregation and reporting ─────────────────────────────────────────────────


def aggregate(results: list[dict]) -> dict:
    """Mean ± std of final_mean_recall across seeds."""
    finals = [r["final_mean_recall"] for r in results if r.get("success")]
    if not finals:
        return {"mean": 0.0, "std": 0.0, "n": 0}
    return {
        "mean": float(np.mean(finals)),
        "std": float(np.std(finals)),
        "n": len(finals),
    }


def mean_curve(results: list[dict]) -> list[float]:
    """Per-cycle mean recall averaged across seeds."""
    curves = [r["per_cycle_mean_recall"] for r in results if r.get("success")]
    if not curves:
        return []
    min_len = min(len(c) for c in curves)
    arr = np.array([c[:min_len] for c in curves])
    return arr.mean(axis=0).tolist()


def print_results_table(all_results: dict) -> None:
    """Print a compact comparison table."""
    print("\n" + "=" * 70)
    print("RESULTS  (mean ± std final recall across seeds)")
    print("=" * 70)
    print(f"{'Dataset':<30}  {'Method':<20}  {'Recall':<12}  {'n seeds'}")
    print("-" * 70)

    for dataset_name, method_results in all_results.items():
        for method, results in sorted(method_results.items()):
            agg = aggregate(results)
            mark = " *" if method == "multiagent" else ""
            print(
                f"  {dataset_name:<28}  {method:<20}  "
                f"{agg['mean']:.3f} ± {agg['std']:.3f}  "
                f"  n={agg['n']}{mark}"
            )
        print()

    print("* = multi-agent (no GP surrogate, LLM-only)")
    print("=" * 70)


def print_cycle_curves(all_results: dict) -> None:
    """Print per-cycle recall curves side by side."""
    print("\nPer-cycle mean recall (averaged over seeds):")
    for dataset_name, method_results in all_results.items():
        print(f"\n  {dataset_name}:")
        # Header
        methods = sorted(method_results.keys())
        header = f"  {'cycle':>5}" + "".join(f"  {m[:14]:>14}" for m in methods)
        print(header)

        curves = {m: mean_curve(method_results[m]) for m in methods}
        max_len = max((len(c) for c in curves.values()), default=0)

        for i in range(max_len):
            row = f"  {i+1:>5}"
            for m in methods:
                c = curves[m]
                val = c[i] if i < len(c) else float("nan")
                row += f"  {val:>14.3f}"
            print(row)


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare naive GP vs multi-agent on toy datasets.")
    parser.add_argument("--no-agent", action="store_true",
                        help="Skip multi-agent runs (no OpenAI API key required).")
    parser.add_argument("--datasets", nargs="+",
                        choices=list(DATASETS.keys()),
                        default=list(DATASETS.keys()),
                        help="Datasets to run (default: all).")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS,
                        help="Random seeds.")
    parser.add_argument("--output", default="results/compare_toy.json",
                        help="Path to save JSON results.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # Suppress GPyTorch noise
    warnings.filterwarnings("ignore", category=UserWarning)

    api_key = os.environ.get("OPENAI_API_KEY")
    run_agent = not args.no_agent
    if run_agent and not api_key:
        print("WARNING: OPENAI_API_KEY not set — skipping multi-agent runs.")
        run_agent = False

    print("=" * 70)
    print("HARD TOY DATASET COMPARISON")
    print(f"  Methods : random, probability_hit"
          + (", multiagent" if run_agent else " (multiagent SKIPPED — no API key)"))
    print(f"  Seeds   : {args.seeds}")
    print(f"  Budget  : initial={N_INITIAL}, batch={BATCH_SIZE}, cycles={N_CYCLES}")
    print(f"  Hit rate: top {TOPK_PERCENT:.0%}")
    print("=" * 70)

    for name in args.datasets:
        print_dataset_summary(name)

    all_results: dict[str, dict[str, list]] = {}
    methods_to_run = ["random", "probability_hit"]
    if run_agent:
        methods_to_run.append("multiagent")

    for dataset_name in args.datasets:
        all_results[dataset_name] = {m: [] for m in methods_to_run}
        print(f"\n{'─'*70}")
        print(f"Dataset: {dataset_name}")
        print(f"{'─'*70}")

        for method in methods_to_run:
            print(f"\n  Method: {method}")
            for seed in args.seeds:
                t0 = time.time()
                print(f"    seed={seed} ... ", end="", flush=True)

                try:
                    if method == "multiagent":
                        result = run_multiagent(
                            dataset_name, seed, api_key,
                            verbose=args.verbose,
                        )
                    else:
                        result = run_naive(
                            dataset_name, method, seed,
                            verbose=args.verbose,
                        )
                except Exception as exc:
                    print(f"FAILED: {exc}")
                    result = {
                        "method": method, "seed": seed, "dataset": dataset_name,
                        "per_cycle_mean_recall": [], "per_cycle_per_target": [],
                        "final_mean_recall": 0.0, "success": False,
                        "error": str(exc),
                    }

                elapsed = time.time() - t0
                status = "✓" if result.get("success") else "✗"
                print(f"{status}  final_recall={result['final_mean_recall']:.3f}  "
                      f"({elapsed:.1f}s)")

                all_results[dataset_name][method].append(result)

    # ── Report ────────────────────────────────────────────────────────────────
    print_results_table(all_results)
    print_cycle_curves(all_results)

    # Save results
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
