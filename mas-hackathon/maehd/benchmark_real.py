"""
Benchmark on real GeneDisco data.

Three-way comparison:
1. CLASSIC multi-target: shared budget, multi-output GP, fixed policy, no agents
2. LLM AGENT multi-target: same GP + an LLM agent deciding policy/weights/batch each cycle
3. INDEPENDENT baselines: budget split per target, single-output GP

Usage:
    python -m maehd.benchmark_real --setup t_cell --seeds 42 43 44 45 46

    # With LLM agent (requires Ollama running):
    python -m maehd.benchmark_real --setup t_cell --seeds 42 43 44 45 46 --with-agent
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

from maehd.datasets.genedisco import (
    MULTI_TARGET_SETUPS,
    load_multitarget,
    load_multitarget_fusion,
)
from maehd.datasets.synthetic import compute_thresholds
from maehd.experiment import (
    ExperimentConfig,
    run_experiment,
    run_experiment_hybrid_policy,
    run_experiment_llm_policy,
    run_experiment_with_agent,
    run_independent_baselines,
)


def run_real_benchmark(
    setup: str = "t_cell",
    feature_set: str = "achilles",
    fusion: bool = False,
    seeds: list[int] = (42, 43, 44, 45, 46),
    n_initial: int = 20,
    batch_size: int = 30,
    n_cycles: int = 15,
    topk_percent: float = 0.10,
    pca_dim: int = 20,
    gp_lengthscale: float = 1.0,
    gp_amplitude: float = 1.0,
    gp_noise: float = 0.05,
    output_dir: str = "results/benchmark_real",
    with_agent: bool = False,
    no_batch_scoring: bool = False,
    no_gene_embeddings: bool = False,
    ollama_url: str = "http://127.0.0.1:11434/v1",
    ollama_model: str = "llama3.1:8b",
    verbose: bool = True,
):
    """Run benchmark on real GeneDisco data.

    Three-way comparison:
    1. CLASSIC: fixed multi-target policies with no agent intelligence
    2. LLM AGENT: an LLM agent reasons each cycle about policy choice
    3. INDEPENDENT: single-target baselines (budget split per target)
    """

    # Load data
    if fusion:
        print(f"Loading GeneDisco data: setup={setup}, FUSION features (achilles+string+ccle)")
        X_raw, Y, meta = load_multitarget_fusion(setup, topk_percent=topk_percent)
        print(f"  Feature dims: {meta.get('dim_breakdown', {})}")
    else:
        print(f"Loading GeneDisco data: setup={setup}, features={feature_set}")
        X_raw, Y, meta = load_multitarget(setup, feature_set, topk_percent=topk_percent)
    n, K = Y.shape
    print(f"  n_genes={n}, d_raw={X_raw.shape[1]}, K={K}")
    print(f"  screens: {meta['screen_names']}")
    print(f"  hits per target: {meta['n_hits_per_target']}")

    # PCA + normalize
    print(f"  PCA: {X_raw.shape[1]} → {pca_dim} dims")
    pca = PCA(n_components=pca_dim, random_state=42)
    X_pca = pca.fit_transform(X_raw)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X_pca).astype(np.float32)
    print(f"  Variance explained: {pca.explained_variance_ratio_.sum():.3f}")

    # Set up LLM agents if requested
    agent_client = None
    biology_agent = None
    X_enriched = None  # features augmented with LLM biology priors
    if with_agent:
        from maehd.agents.llm_client import OllamaClient
        from maehd.agents.llm_agent import ExperimentAgent
        from maehd.agents.biology_agent import BiologyKnowledgeAgent
        agent_client = OllamaClient(
            base_url=ollama_url, model=ollama_model,
        )
        if agent_client.is_available():
            print(f"  LLM agent: connected to {ollama_model} at {ollama_url}")

            # ── Biology Knowledge Agent: extract prior features ──
            biology_agent = BiologyKnowledgeAgent(client=agent_client)
            print(f"\n  Querying LLM for biology knowledge...")
            do_batch = not no_batch_scoring
            if do_batch:
                print(f"  Dense batch scoring enabled — will score all {n} genes per target")
            X_prior = biology_agent.create_prior_features(
                gene_names=meta["gene_names"],
                screen_names=meta["screen_names"],
                do_batch_scoring=do_batch,
                verbose=True,
            )
            n_prior_genes = int((X_prior.sum(axis=1) > 0).sum())
            print(f"  Biology priors: {X_prior.shape[1]} features, "
                  f"{n_prior_genes}/{n} genes annotated")

            # Feature recommendation
            feat_rec = biology_agent.recommend_features(meta["screen_names"])
            print(f"  Feature recommendation: {feat_rec.get('recommended_feature', 'N/A')}")
            print(f"    Reasoning: {feat_rec.get('reasoning', 'N/A')[:120]}...")

            # ── Gene Functional Embeddings: 8-axis scoring ──
            X_func = None
            if not no_gene_embeddings:
                from maehd.agents.gene_embeddings import compute_llm_gene_embeddings
                print(f"\n  Computing LLM gene functional embeddings (8 axes)...")
                X_func = compute_llm_gene_embeddings(
                    gene_names=meta["gene_names"],
                    client=agent_client,
                    verbose=True,
                )
                print(f"  Functional embeddings: {X_func.shape}")

            # Enrich features: PCA + biology priors + (optional) functional embeddings
            parts = [X, X_prior]
            desc = [f"{X.shape[1]} PCA", f"{X_prior.shape[1]} biology"]
            if X_func is not None:
                parts.append(X_func)
                desc.append(f"{X_func.shape[1]} functional")
            X_enriched = np.hstack(parts)
            print(f"  Enriched features: {' + '.join(desc)} = {X_enriched.shape[1]} total")
        else:
            print(f"  WARNING: Ollama not reachable at {ollama_url} — agent will use fallback")

    all_results = {}

    for seed in seeds:
        print(f"\n{'─'*60}")
        print(f"Seed {seed}")
        print(f"{'─'*60}")

        seed_results = {}

        # ── 1. CLASSIC multi-target: NO agents, NO fancy features ──
        # Same GP, same budget, same scoring — but pure math, no LLM.
        for mt_pol in ["max_poh", "mean_poh", "coverage_poh"]:
            label = f"classic_{mt_pol}"
            if verbose:
                print(f"\n--- {label} ---")
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
                use_feature_agent=False,
                use_bandit=False,
                use_correlated_gp=False,
                use_supervised_proj=False,
                use_diversity=False,
            )
            result = run_experiment(X, Y, cfg, verbose=verbose)
            seed_results[label] = result

        # ── 1b. BIOLOGY FEATURES + fixed policy (isolates feature contribution) ──
        if with_agent and X_enriched is not None:
            for mt_pol in ["max_poh", "mean_poh"]:
                label = f"bio_{mt_pol}"
                if verbose:
                    print(f"\n--- {label} ---")
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
                    use_feature_agent=False,
                    use_bandit=False,
                    use_correlated_gp=False,
                    use_supervised_proj=False,
                    use_diversity=False,
                )
                result = run_experiment(X_enriched, Y, cfg, verbose=verbose)
                seed_results[label] = result

        # ── 2. LLM AGENT: the real agentic framework ──
        if with_agent:
            label = "llm_agent"
            if verbose:
                print(f"\n--- {label} ---")

            # Build biology context for the per-cycle agent
            bio_ctx = ""
            if biology_agent is not None:
                bio_ctx = biology_agent.get_phenotype_context(meta["screen_names"])

            agent = ExperimentAgent(
                client=agent_client,
                screen_names=meta.get("screen_names"),
                biology_context=bio_ctx,
            )
            cfg = ExperimentConfig(
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
                use_correlated_gp=False,
                use_supervised_proj=False,
                use_diversity=False,
            )
            # Use enriched features (PCA + biology priors) if available
            X_agent = X_enriched if X_enriched is not None else X
            result = run_experiment_with_agent(X_agent, Y, cfg, agent, verbose=verbose)
            seed_results[label] = result

            # Log agent decisions
            if verbose and agent.history:
                print(f"  Agent decisions:")
                for d in agent.history:
                    llm_tag = "LLM" if d.used_llm else "fallback"
                    print(f"    {d.policy} ({llm_tag}): {d.reasoning[:80]}...")

        # ── 2b. LLM-AS-POLICY: the LLM directly selects genes ──
        if with_agent:
            from maehd.agents.llm_policy import LLMPolicyAgent

            label = "llm_policy"
            if verbose:
                print(f"\n--- {label} ---")

            # Build biology context for the LLM policy agent
            bio_ctx_policy = ""
            if biology_agent is not None:
                bio_ctx_policy = biology_agent.get_phenotype_context(meta["screen_names"])

            llm_pol_agent = LLMPolicyAgent(
                client=agent_client,
                biology_context=bio_ctx_policy,
                screen_names=meta.get("screen_names"),
                n_candidates=100,
            )
            cfg = ExperimentConfig(
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
                use_correlated_gp=False,
                use_supervised_proj=False,
                use_diversity=False,
            )
            result = run_experiment_llm_policy(
                X, Y, meta["gene_names"], cfg, llm_pol_agent, verbose=verbose,
            )
            seed_results[label] = result

            if verbose and llm_pol_agent.history:
                n_llm = sum(1 for h in llm_pol_agent.history if h.get("used_llm"))
                print(f"  LLM policy: {n_llm}/{len(llm_pol_agent.history)} cycles used LLM")

        # ── 2c. HYBRID LLM: GP exploit + LLM explore (bounded risk) ──
        if with_agent:
            from maehd.agents.llm_policy_hybrid import HybridLLMPolicy

            label = "hybrid_llm"
            if verbose:
                print(f"\n--- {label} ---")

            bio_ctx_hybrid = ""
            if biology_agent is not None:
                bio_ctx_hybrid = biology_agent.get_phenotype_context(meta["screen_names"])

            hybrid_agent = HybridLLMPolicy(
                client=agent_client,
                biology_context=bio_ctx_hybrid,
                screen_names=meta.get("screen_names"),
                exploit_frac=0.7,
                tiebreaker_pool_size=60,
            )
            cfg = ExperimentConfig(
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
                use_correlated_gp=False,
                use_supervised_proj=False,
                use_diversity=False,
            )
            result = run_experiment_hybrid_policy(
                X, Y, meta["gene_names"], cfg, hybrid_agent, verbose=verbose,
            )
            seed_results[label] = result

            if verbose and hybrid_agent.history:
                n_llm = sum(1 for h in hybrid_agent.history if h.get("used_llm"))
                print(f"  Hybrid: {n_llm}/{len(hybrid_agent.history)} LLM, "
                      f"{len(hybrid_agent._hit_genes)} hits memorized")

        # ── 3. INDEPENDENT baselines ──
        base_cfg = ExperimentConfig(
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
            use_diversity=False,
        )
        ind_results = run_independent_baselines(
            X, Y, base_cfg,
            policies=["poh", "ucb", "thompson", "random"],
            verbose=verbose,
        )
        for pol_name, res in ind_results.items():
            seed_results[f"ind_{pol_name}"] = res

        all_results[seed] = seed_results

    # Summary
    _print_summary(all_results, seeds, K, meta["screen_names"])

    # Save
    os.makedirs(output_dir, exist_ok=True)
    _save_results(all_results, seeds, output_dir, setup, meta["feature_set"])

    return all_results


def _print_summary(all_results, seeds, K, screen_names):
    print(f"\n{'='*80}")
    print("SUMMARY — Real Data")
    print(f"{'='*80}")

    methods = sorted(all_results[seeds[0]].keys())

    header = f"{'Method':<22}"
    for sn in screen_names:
        short = sn.split("_")[-1][:8]
        header += f" {short:>12}"
    header += f" {'MeanRecall':>12} {'#Hits':>8} {'HitRatio':>10} {'Queries':>8}"
    print(header)
    print("-" * len(header))

    for method in methods:
        recalls = np.array([all_results[s][method].final_recall for s in seeds])
        hits = np.array([all_results[s][method].total_hits.sum() for s in seeds])
        queries = np.array([all_results[s][method].total_queries for s in seeds])
        mean_r = recalls.mean(axis=0)
        std_r = recalls.std(axis=0)
        mean_all = mean_r.mean()
        mean_hits = hits.mean()
        mean_q = queries.mean()
        hit_ratio = mean_hits / max(mean_q, 1)

        row = f"{method:<22}"
        for k in range(K):
            row += f" {mean_r[k]:.3f}±{std_r[k]:.3f}"
        row += f" {mean_all:>12.3f} {mean_hits:>8.0f} {hit_ratio:>10.3f} {mean_q:>8.0f}"
        print(row)


def _save_results(all_results, seeds, output_dir, setup, feature_set):
    save_data = {}
    for seed in seeds:
        seed_data = {}
        for method, result in all_results[seed].items():
            total_hits_sum = int(result.total_hits.sum())
            hit_ratio = total_hits_sum / max(result.total_queries, 1)
            seed_data[method] = {
                "final_recall": result.final_recall.tolist(),
                "total_hits": result.total_hits.tolist(),
                "total_hits_sum": total_hits_sum,
                "hit_ratio": round(hit_ratio, 4),
                "total_queries": result.total_queries,
            }
        save_data[str(seed)] = seed_data

    path = os.path.join(output_dir, f"benchmark_{setup}_{feature_set}.json")
    with open(path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--setup", default="t_cell",
                        choices=list(MULTI_TARGET_SETUPS.keys()))
    parser.add_argument("--feature-set", default="achilles")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--n-initial", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=30)
    parser.add_argument("--n-cycles", type=int, default=15)
    parser.add_argument("--pca-dim", type=int, default=20)
    parser.add_argument("--gp-lengthscale", type=float, default=1.0)
    parser.add_argument("--gp-noise", type=float, default=0.05)
    parser.add_argument("--output-dir", default="results/benchmark_real")
    parser.add_argument("--with-agent", action="store_true",
                        help="Run LLM agent (requires Ollama running)")
    parser.add_argument("--fusion", action="store_true",
                        help="Use fused features (achilles+string+ccle)")
    parser.add_argument("--no-batch-scoring", action="store_true",
                        help="Skip dense batch scoring (faster, sparse bio only)")
    parser.add_argument("--no-gene-embeddings", action="store_true",
                        help="Skip LLM gene functional embeddings (8 axes)")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434/v1")
    parser.add_argument("--ollama-model", default="llama3.1:8b")
    args = parser.parse_args()

    run_real_benchmark(
        setup=args.setup,
        feature_set=args.feature_set,
        fusion=args.fusion,
        seeds=args.seeds,
        n_initial=args.n_initial,
        batch_size=args.batch_size,
        n_cycles=args.n_cycles,
        pca_dim=args.pca_dim,
        gp_lengthscale=args.gp_lengthscale,
        gp_noise=args.gp_noise,
        output_dir=args.output_dir,
        with_agent=args.with_agent,
        no_batch_scoring=args.no_batch_scoring,
        no_gene_embeddings=args.no_gene_embeddings,
        ollama_url=args.ollama_url,
        ollama_model=args.ollama_model,
    )


if __name__ == "__main__":
    main()
