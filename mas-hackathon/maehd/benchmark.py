"""
Benchmark runner for multi-target hit discovery.

Usage:
    python -m maehd.benchmark --dataset pathway_multitarget --K 3 --seeds 42 43 44 45 46

Runs multi-agent (shared budget) vs independent baselines across seeds.
Prints summary table and saves results.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from maehd.datasets.synthetic import REGISTRY as DATASET_REGISTRY
from maehd.datasets.synthetic import get_dataset
from maehd.experiment import ExperimentConfig, run_experiment, run_independent_baselines


def run_benchmark(
    dataset_name: str,
    K: int = 3,
    seeds: list[int] = (42, 43, 44, 45, 46),
    n_initial: int = 10,
    batch_size: int = 20,
    n_cycles: int = 15,
    topk_percent: float = 0.10,
    mt_policies: list[str] = ("max_poh", "mean_poh", "joint_poh", "mt_ucb", "mt_thompson"),
    ind_policies: list[str] = ("poh", "ucb", "thompson", "random"),
    gp_lengthscale: float = 0.3,
    gp_amplitude: float = 1.0,
    gp_noise: float = 0.01,
    output_dir: str | None = None,
    verbose: bool = True,
):
    """Run full benchmark: multi-agent policies vs independent baselines."""

    if verbose:
        print(f"{'='*60}")
        print(f"Benchmark: {dataset_name} | K={K} | seeds={list(seeds)}")
        print(f"Budget: {n_cycles} cycles × {batch_size} batch = {n_cycles * batch_size} total")
        print(f"{'='*60}")

    all_results = {}

    for seed in seeds:
        if verbose:
            print(f"\n{'─'*60}")
            print(f"Seed {seed}")
            print(f"{'─'*60}")

        # Generate dataset
        ds_kwargs = {"K": K, "seed": seed}
        if dataset_name == "pathway_multitarget":
            ds_kwargs["n_genes"] = 200
        elif dataset_name in ("correlated_targets", "independent_targets"):
            ds_kwargs["n"] = 200

        X, Y, meta = get_dataset(dataset_name, **ds_kwargs)

        base_config = ExperimentConfig(
            n_initial=n_initial,
            batch_size=batch_size,
            n_cycles=n_cycles,
            topk_percent=topk_percent,
            gp_lengthscale=gp_lengthscale,
            gp_amplitude=gp_amplitude,
            gp_noise=gp_noise,
            seed=seed,
            optimize_gp=True,
            use_feature_agent=False,
            use_bandit=False,
        )

        seed_results = {}

        # Multi-agent policies (shared budget)
        for mt_pol in mt_policies:
            if verbose:
                print(f"\n--- Multi-agent: {mt_pol} ---")
            cfg = ExperimentConfig(
                n_initial=n_initial,
                batch_size=batch_size,
                n_cycles=n_cycles,
                topk_percent=topk_percent,
                gp_lengthscale=gp_lengthscale,
                gp_amplitude=gp_amplitude,
                gp_noise=gp_noise,
                policy_name=mt_pol,
                seed=seed,
                optimize_gp=True,
                use_feature_agent=True,
                use_bandit=False,
                use_supervised_proj=True,
                use_diversity=True,
            )
            result = run_experiment(X, Y, cfg, verbose=verbose)
            seed_results[f"mt_{mt_pol}"] = result

        # Multi-agent with adaptive policy (bandit-based)
        if verbose:
            print(f"\n--- Multi-agent: adaptive ---")
        cfg_adaptive = ExperimentConfig(
            n_initial=n_initial,
            batch_size=batch_size,
            n_cycles=n_cycles,
            topk_percent=topk_percent,
            gp_lengthscale=gp_lengthscale,
            gp_amplitude=gp_amplitude,
            gp_noise=gp_noise,
            policy_name=None,  # let bandit decide
            seed=seed,
            optimize_gp=True,
            use_feature_agent=True,
            use_bandit=True,
            use_supervised_proj=True,
            use_diversity=True,
        )
        result_adaptive = run_experiment(X, Y, cfg_adaptive, verbose=verbose)
        seed_results["mt_adaptive"] = result_adaptive

        # Independent baselines (split budget)
        ind_results = run_independent_baselines(
            X, Y, base_config, policies=ind_policies, verbose=verbose,
        )
        for pol_name, res in ind_results.items():
            seed_results[f"ind_{pol_name}"] = res

        all_results[seed] = seed_results

    # Summary table
    if verbose:
        _print_summary(all_results, seeds, K)

    # Save results
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        _save_results(all_results, seeds, K, output_dir, dataset_name)

    return all_results


def _print_summary(all_results, seeds, K):
    """Print summary table of results."""
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    # Collect all method names
    methods = sorted(all_results[seeds[0]].keys())

    # Header
    header = f"{'Method':<25}"
    for k in range(K):
        header += f" {'Recall_'+str(k):>10}"
    header += f" {'Any':>10} {'Queries':>8}"
    print(header)
    print("-" * len(header))

    for method in methods:
        recalls = []
        any_recalls = []
        queries = []
        for seed in seeds:
            r = all_results[seed][method]
            recalls.append(r.final_recall)
            obs = np.where(r.total_hits > -np.inf)[0]  # always true
            # Compute any-target recall from hits
            any_r = 0.0
            for k in range(K):
                any_r = max(any_r, r.final_recall[k])
            any_recalls.append(np.mean(r.final_recall))
            queries.append(r.total_queries)

        mean_recall = np.mean(recalls, axis=0)
        std_recall = np.std(recalls, axis=0)
        mean_any = np.mean(any_recalls)
        mean_q = np.mean(queries)

        row = f"{method:<25}"
        for k in range(K):
            row += f" {mean_recall[k]:.3f}±{std_recall[k]:.3f}"
        row += f" {mean_any:>10.3f} {mean_q:>8.0f}"
        print(row)


def _save_results(all_results, seeds, K, output_dir, dataset_name):
    """Save results to JSON."""
    save_data = {}
    for seed in seeds:
        seed_data = {}
        for method, result in all_results[seed].items():
            seed_data[method] = {
                "final_recall": result.final_recall.tolist(),
                "total_hits": result.total_hits.tolist(),
                "total_queries": result.total_queries,
            }
        save_data[str(seed)] = seed_data

    path = os.path.join(output_dir, f"benchmark_{dataset_name}.json")
    with open(path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {path}")


def main():
    parser = argparse.ArgumentParser(description="Multi-target hit discovery benchmark")
    parser.add_argument("--dataset", default="pathway_multitarget",
                        choices=list(DATASET_REGISTRY.keys()))
    parser.add_argument("--K", type=int, default=3)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--n-initial", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--n-cycles", type=int, default=15)
    parser.add_argument("--topk-percent", type=float, default=0.10)
    parser.add_argument("--gp-lengthscale", type=float, default=0.3)
    parser.add_argument("--gp-amplitude", type=float, default=1.0)
    parser.add_argument("--gp-noise", type=float, default=0.01)
    parser.add_argument("--output-dir", default="results/benchmark_maehd")
    args = parser.parse_args()

    run_benchmark(
        dataset_name=args.dataset,
        K=args.K,
        seeds=args.seeds,
        n_initial=args.n_initial,
        batch_size=args.batch_size,
        n_cycles=args.n_cycles,
        topk_percent=args.topk_percent,
        gp_lengthscale=args.gp_lengthscale,
        gp_amplitude=args.gp_amplitude,
        gp_noise=args.gp_noise,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
