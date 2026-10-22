"""
Main experiment loop for multi-target hit discovery.

Runs the four-agent pipeline over multiple cycles:
    for each cycle:
        0. (optional) Feature Agent → learn feature weights from observed data
        1. Fit multi-output GP on (possibly re-weighted) features
        2. Run agent pipeline → select batch of genes
        3. Query those genes (reveal true responses for ALL targets)
        4. Update observed set, update bandit with hit-rate reward
        5. Record metrics

Supports both toy (synthetic) and real (GeneDisco) datasets.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from maehd.agents.pipeline import (
    CycleState,
    PolicyBandit,
    apply_feature_weights,
    correlation_agent,
    feature_agent,
    run_agent_pipeline,
    supervised_projection,
)
from maehd.datasets.synthetic import compute_thresholds
from maehd.metrics import (
    all_target_recall,
    any_target_recall,
    budget_efficiency,
    hit_count,
    per_target_recall,
)
from maehd.models.gp import CorrelatedMultiOutputGP, MultiOutputGP


@dataclass
class ExperimentConfig:
    """Configuration for a multi-target experiment."""

    n_initial: int = 10          # initial random observations
    batch_size: int = 20         # genes per cycle
    n_cycles: int = 15           # number of active learning cycles
    topk_percent: float = 0.10   # top 10% = hits
    gp_lengthscale: float = 0.3
    gp_amplitude: float = 1.0
    gp_noise: float = 0.01
    policy_name: str | None = None  # None = adaptive (let agents decide)
    seed: int = 42
    optimize_gp: bool = True     # optimize GP hyperparams via marginal likelihood
    use_feature_agent: bool = True  # use target-aware feature re-weighting
    use_bandit: bool = True      # use Thompson Sampling bandit for policy selection
    use_correlated_gp: bool = False  # use ICM multi-output GP (cross-target transfer)
    use_supervised_proj: bool = True  # supervised dimensionality reduction after warm-up
    use_diversity: bool = True    # diverse batch selection


@dataclass
class CycleLog:
    """Metrics recorded at each cycle."""

    cycle: int
    n_observed: int
    per_target_recall: np.ndarray   # (K,)
    any_recall: float
    all_recall: float
    hits_found: np.ndarray          # (K,)
    efficiency: float
    policy_used: str
    weights: np.ndarray             # (K,)


@dataclass
class ExperimentResult:
    """Full result of an experiment run."""

    config: ExperimentConfig
    cycle_logs: list[CycleLog] = field(default_factory=list)
    total_hits: np.ndarray = field(default_factory=lambda: np.array([]))
    final_recall: np.ndarray = field(default_factory=lambda: np.array([]))
    total_queries: int = 0


def run_experiment(
    X: np.ndarray,
    Y: np.ndarray,
    config: ExperimentConfig,
    verbose: bool = True,
) -> ExperimentResult:
    """Run a single multi-target active learning experiment.

    Parameters
    ----------
    X : (n, d) gene features
    Y : (n, K) multi-target responses
    config : experiment configuration

    Returns
    -------
    ExperimentResult with per-cycle metrics
    """
    rng = np.random.default_rng(config.seed)
    n, K = Y.shape

    # Compute hit thresholds
    thresholds = compute_thresholds(Y, config.topk_percent)
    true_hits = [np.where(Y[:, k] > thresholds[k])[0] for k in range(K)]

    if verbose:
        print(f"Dataset: n={n}, d={X.shape[1]}, K={K}")
        for k in range(K):
            print(f"  Target {k}: {len(true_hits[k])} hits (τ={thresholds[k]:.3f})")

    # Initial random observations
    all_indices = np.arange(n)
    initial = rng.choice(all_indices, size=min(config.n_initial, n), replace=False)
    obs_mask = np.zeros(n, dtype=bool)
    obs_mask[initial] = True

    total_budget = config.n_cycles * config.batch_size
    budget_remaining = total_budget

    # Build GP — correlated or independent
    if config.use_correlated_gp:
        gp = CorrelatedMultiOutputGP(
            K=K,
            lengthscale=config.gp_lengthscale,
            amplitude=config.gp_amplitude,
            noise=config.gp_noise,
        )
    else:
        gp = MultiOutputGP(
            K=K,
            lengthscale=config.gp_lengthscale,
            amplitude=config.gp_amplitude,
            noise=config.gp_noise,
        )

    # Initialize bandit if adaptive + bandit enabled
    bandit = None
    if config.use_bandit and config.policy_name is None:
        bandit = PolicyBandit()

    result = ExperimentResult(config=config)
    feat_weights = None  # will be learned online
    current_recall = np.zeros(K)  # track per-target recall for coverage_poh

    for cycle in range(config.n_cycles):
        obs_idx = np.where(obs_mask)[0]
        pool_idx = np.where(~obs_mask)[0]

        if len(pool_idx) == 0:
            if verbose:
                print(f"  Cycle {cycle}: pool exhausted")
            break

        # Feature processing pipeline:
        # 1. Try supervised projection (after enough observations)
        # 2. Fall back to feature agent weights
        # 3. Fall back to raw features
        X_fit = X  # default: raw features

        if config.use_supervised_proj and len(obs_idx) >= 50:
            X_proj = supervised_projection(
                X[obs_idx], Y[obs_idx], X,
                n_components=min(10, X.shape[1]),
                min_obs=50,
            )
            if X_proj is not None:
                X_fit = X_proj
        elif config.use_feature_agent and len(obs_idx) >= 15:
            feat_weights = feature_agent(X[obs_idx], Y[obs_idx])
            X_fit = apply_feature_weights(X, feat_weights)

        # Fit GP on processed features
        gp.fit(X_fit[obs_idx], Y[obs_idx], optimize=config.optimize_gp)

        # Build cycle state — use processed features for GP prediction
        state = CycleState(
            X_pool=X_fit[pool_idx],
            X_obs=X_fit[obs_idx],
            Y_obs=Y[obs_idx],
            pool_indices=pool_idx,
            obs_indices=obs_idx,
            thresholds=thresholds,
            cycle=cycle,
            budget_remaining=budget_remaining,
            gp=gp,
            recall_so_far=current_recall.copy(),
        )

        # Count hits before this cycle (for bandit reward)
        hits_before = hit_count(obs_idx, true_hits).sum() if bandit else 0

        # Run agent pipeline
        decision = run_agent_pipeline(
            state,
            batch_size=config.batch_size,
            policy_name=config.policy_name,
            rng=rng,
            bandit=bandit,
            use_feature_agent=False,  # already applied above
            recall_so_far=current_recall,
            use_diversity=config.use_diversity,
        )

        # Query selected genes
        selected_global = pool_idx[decision.query_indices]
        obs_mask[selected_global] = True
        budget_remaining -= len(selected_global)

        # Update bandit with hit-rate reward
        if bandit is not None:
            all_observed_now = np.where(obs_mask)[0]
            hits_after = hit_count(all_observed_now, true_hits).sum()
            new_hits = hits_after - hits_before
            reward = new_hits / max(len(selected_global), 1)
            bandit.update(decision.policy_name, reward)

        # Update current recall for coverage-aware policies
        all_observed = np.where(obs_mask)[0]
        current_recall = per_target_recall(all_observed, true_hits)

        # Record metrics
        cycle_log = CycleLog(
            cycle=cycle,
            n_observed=len(all_observed),
            per_target_recall=current_recall.copy(),
            any_recall=any_target_recall(all_observed, true_hits),
            all_recall=all_target_recall(all_observed, true_hits),
            hits_found=hit_count(all_observed, true_hits),
            efficiency=budget_efficiency(all_observed, true_hits),
            policy_used=decision.policy_name,
            weights=decision.weights.copy(),
        )
        result.cycle_logs.append(cycle_log)

        if verbose:
            recall_str = ", ".join(
                f"{r:.3f}" for r in cycle_log.per_target_recall
            )
            print(
                f"  Cycle {cycle}: obs={cycle_log.n_observed}, "
                f"recall=[{recall_str}], "
                f"any={cycle_log.any_recall:.3f}, "
                f"policy={cycle_log.policy_used}"
            )

    # Final summary
    final_observed = np.where(obs_mask)[0]
    result.total_hits = hit_count(final_observed, true_hits)
    result.final_recall = per_target_recall(final_observed, true_hits)
    result.total_queries = int(obs_mask.sum()) - config.n_initial

    if verbose:
        print(f"\nFinal: {result.total_queries} queries")
        for k in range(K):
            print(
                f"  Target {k}: {result.total_hits[k]}/{len(true_hits[k])} hits "
                f"(recall {result.final_recall[k]:.3f})"
            )
        print(f"  Any-target recall: {any_target_recall(final_observed, true_hits):.3f}")

    return result


# --------------------------------------------------------------------------- #
#                     Independent baselines                                    #
# --------------------------------------------------------------------------- #


def run_independent_baselines(
    X: np.ndarray,
    Y: np.ndarray,
    config: ExperimentConfig,
    policies: list[str] = ("poh", "ucb", "thompson", "random"),
    verbose: bool = True,
) -> dict[str, ExperimentResult]:
    """Run single-target baselines: each target gets its own budget.

    This is the "no resource sharing" baseline. Total budget is split
    evenly across K targets. Each target runs independently with a
    single-output GP and single-target acquisition.

    Returns dict mapping policy_name → ExperimentResult.
    """
    from maehd.models.gp import SingleOutputGP
    from maehd.policies.acquisition import (
        expected_improvement_tau,
        probability_of_hit,
        thompson_sample,
        ucb,
    )

    rng = np.random.default_rng(config.seed)
    n, K = Y.shape
    thresholds = compute_thresholds(Y, config.topk_percent)
    true_hits = [np.where(Y[:, k] > thresholds[k])[0] for k in range(K)]

    # Split budget across targets
    budget_per_target = (config.n_cycles * config.batch_size) // K
    batch_per_target = max(1, config.batch_size // K)
    n_cycles_ind = budget_per_target // batch_per_target

    results = {}

    for policy_str in policies:
        if verbose:
            print(f"\n=== Independent baseline: {policy_str} ===")

        total_observed = set()

        # Per-target independent runs
        for k in range(K):
            gp_k = SingleOutputGP(
                config.gp_lengthscale, config.gp_amplitude, config.gp_noise,
            )

            # Initial random obs
            init_k = rng.choice(n, size=min(config.n_initial, n), replace=False)
            obs_mask_k = np.zeros(n, dtype=bool)
            obs_mask_k[init_k] = True
            total_observed.update(init_k.tolist())

            for cyc in range(n_cycles_ind):
                obs_idx = np.where(obs_mask_k)[0]
                pool_idx = np.where(~obs_mask_k)[0]
                if len(pool_idx) == 0:
                    break

                gp_k.fit(X[obs_idx], Y[obs_idx, k], optimize=config.optimize_gp)
                mu, sigma = gp_k.predict(X[pool_idx])

                if policy_str == "poh":
                    scores = probability_of_hit(mu, sigma, thresholds[k])
                elif policy_str == "ucb":
                    scores = ucb(mu, sigma)
                elif policy_str == "thompson":
                    scores = thompson_sample(mu, sigma, rng)
                elif policy_str == "random":
                    scores = rng.random(len(pool_idx))
                elif policy_str == "ei_tau":
                    scores = expected_improvement_tau(mu, sigma, thresholds[k])
                else:
                    raise ValueError(f"Unknown policy: {policy_str}")

                selected = np.argsort(scores)[::-1][:batch_per_target]
                selected_global = pool_idx[selected]
                obs_mask_k[selected_global] = True
                total_observed.update(selected_global.tolist())

        # Evaluate
        all_obs = np.array(sorted(total_observed))
        result = ExperimentResult(config=config)
        result.total_hits = hit_count(all_obs, true_hits)
        result.final_recall = per_target_recall(all_obs, true_hits)
        result.total_queries = len(all_obs)

        if verbose:
            for k_target in range(K):
                print(
                    f"  Target {k_target}: {result.total_hits[k_target]}/{len(true_hits[k_target])} "
                    f"(recall {result.final_recall[k_target]:.3f})"
                )
            print(f"  Any-target recall: {any_target_recall(all_obs, true_hits):.3f}")

        results[policy_str] = result

    return results


# --------------------------------------------------------------------------- #
#             LLM Agent-driven experiment loop                                 #
# --------------------------------------------------------------------------- #


def run_experiment_with_agent(
    X: np.ndarray,
    Y: np.ndarray,
    config: ExperimentConfig,
    agent: "ExperimentAgent",
    verbose: bool = True,
) -> ExperimentResult:
    """Run experiment with an LLM agent making per-cycle decisions.

    The agent observes the full experiment state each cycle and decides:
    - Which acquisition policy to use
    - How to weight the K targets
    - The batch size for this cycle

    This is the proper agentic framework — the LLM reasons about
    the experiment and adapts strategy based on observed outcomes.

    Parameters
    ----------
    X : (n, d) gene features
    Y : (n, K) multi-target responses
    config : experiment configuration (policy_name is ignored — agent decides)
    agent : ExperimentAgent instance (with or without LLM client)
    verbose : print per-cycle progress

    Returns
    -------
    ExperimentResult with per-cycle metrics
    """
    from maehd.agents.llm_agent import ExperimentAgent  # type check
    from maehd.policies import acquisition as acq

    rng = np.random.default_rng(config.seed)
    n, K = Y.shape

    # Compute hit thresholds
    thresholds = compute_thresholds(Y, config.topk_percent)
    true_hits = [np.where(Y[:, k] > thresholds[k])[0] for k in range(K)]

    if verbose:
        print(f"Dataset: n={n}, d={X.shape[1]}, K={K}")
        for k in range(K):
            print(f"  Target {k}: {len(true_hits[k])} hits (τ={thresholds[k]:.3f})")

    # Initial random observations
    initial = rng.choice(n, size=min(config.n_initial, n), replace=False)
    obs_mask = np.zeros(n, dtype=bool)
    obs_mask[initial] = True

    total_budget = config.n_cycles * config.batch_size
    budget_remaining = total_budget

    # Build GP (always independent — simpler for agent loop)
    gp = MultiOutputGP(
        K=K,
        lengthscale=config.gp_lengthscale,
        amplitude=config.gp_amplitude,
        noise=config.gp_noise,
    )

    result = ExperimentResult(config=config)
    current_recall = np.zeros(K)
    last_hit_rate = 0.0

    for cycle in range(config.n_cycles):
        obs_idx = np.where(obs_mask)[0]
        pool_idx = np.where(~obs_mask)[0]

        if len(pool_idx) == 0:
            if verbose:
                print(f"  Cycle {cycle}: pool exhausted")
            break

        # Fit GP
        gp.fit(X[obs_idx], Y[obs_idx], optimize=config.optimize_gp)

        # Compute state info for the agent
        mu_pool, sigma_pool = gp.predict(X[pool_idx])
        mean_sigma_per_target = sigma_pool.mean(axis=0)  # (K,)

        # Target correlation
        if len(obs_idx) >= 3:
            corr = np.corrcoef(Y[obs_idx].T)
            corr = np.nan_to_num(corr, nan=0.0)
            np.fill_diagonal(corr, 1.0)
        else:
            corr = np.eye(K)

        hits_found = hit_count(obs_idx, true_hits).astype(float)
        total_hits_per_target = np.array([len(th) for th in true_hits], dtype=float)

        # ── Ask the LLM agent for a decision ──
        decision = agent.decide(
            cycle=cycle,
            n_cycles=config.n_cycles,
            n_observed=len(obs_idx),
            n_total=n,
            K=K,
            per_target_recall=current_recall,
            per_target_hits=hits_found,
            per_target_total_hits=total_hits_per_target,
            thresholds=thresholds,
            gp_mean_sigma=mean_sigma_per_target,
            target_correlation=corr,
            last_cycle_hit_rate=last_hit_rate,
            default_batch_size=config.batch_size,
            budget_remaining=budget_remaining,
        )

        # ── Execute the agent's decision ──
        actual_batch = min(decision.batch_size, budget_remaining, len(pool_idx))
        policy_name = decision.policy
        weights = decision.target_weights

        # Score candidates using the chosen policy
        if policy_name == "max_poh":
            scores = acq.max_poh(mu_pool, sigma_pool, thresholds)
        elif policy_name == "mean_poh":
            scores = acq.mean_poh(mu_pool, sigma_pool, thresholds, weights)
        elif policy_name == "coverage_poh":
            scores = acq.coverage_poh(
                mu_pool, sigma_pool, thresholds, current_recall,
            )
        elif policy_name == "mt_thompson":
            scores = acq.multi_target_thompson(
                mu_pool, sigma_pool, thresholds, rng,
            )
        else:
            scores = acq.max_poh(mu_pool, sigma_pool, thresholds)

        # Select top batch
        selected = np.argsort(scores)[::-1][:actual_batch]

        # Query selected genes
        selected_global = pool_idx[selected]
        hits_before = hit_count(obs_idx, true_hits).sum()
        obs_mask[selected_global] = True
        budget_remaining -= len(selected_global)

        # Compute hit rate for this cycle (for next cycle's agent input)
        all_observed = np.where(obs_mask)[0]
        hits_after = hit_count(all_observed, true_hits).sum()
        new_hits = hits_after - hits_before
        last_hit_rate = new_hits / max(len(selected_global), 1)

        # Update recall
        current_recall = per_target_recall(all_observed, true_hits)

        # Log
        cycle_log = CycleLog(
            cycle=cycle,
            n_observed=len(all_observed),
            per_target_recall=current_recall.copy(),
            any_recall=any_target_recall(all_observed, true_hits),
            all_recall=all_target_recall(all_observed, true_hits),
            hits_found=hit_count(all_observed, true_hits),
            efficiency=budget_efficiency(all_observed, true_hits),
            policy_used=policy_name,
            weights=weights.copy(),
        )
        result.cycle_logs.append(cycle_log)

        if verbose:
            recall_str = ", ".join(f"{r:.3f}" for r in cycle_log.per_target_recall)
            llm_tag = "LLM" if decision.used_llm else "fallback"
            print(
                f"  Cycle {cycle}: obs={cycle_log.n_observed}, "
                f"recall=[{recall_str}], "
                f"any={cycle_log.any_recall:.3f}, "
                f"policy={policy_name} ({llm_tag})"
            )

    # Final summary
    final_observed = np.where(obs_mask)[0]
    result.total_hits = hit_count(final_observed, true_hits)
    result.final_recall = per_target_recall(final_observed, true_hits)
    result.total_queries = int(obs_mask.sum()) - config.n_initial

    if verbose:
        print(f"\nFinal: {result.total_queries} queries")
        for k in range(K):
            print(
                f"  Target {k}: {result.total_hits[k]}/{len(true_hits[k])} hits "
                f"(recall {result.final_recall[k]:.3f})"
            )
        print(f"  Any-target recall: {any_target_recall(final_observed, true_hits):.3f}")

    return result


# --------------------------------------------------------------------------- #
#          LLM-as-Policy experiment loop (agent IS the acquisition fn)         #
# --------------------------------------------------------------------------- #


def run_experiment_llm_policy(
    X: np.ndarray,
    Y: np.ndarray,
    gene_names: list[str],
    config: ExperimentConfig,
    llm_agent: "LLMPolicyAgent",
    verbose: bool = True,
) -> ExperimentResult:
    """Run experiment where the LLM directly selects genes (agent IS the policy).

    Instead of the LLM choosing which math formula to run, here the LLM
    sees the top GP candidates *with gene names* and picks which to query,
    using biology knowledge to break ties among statistically similar genes.

    Parameters
    ----------
    X : (n, d) gene features
    Y : (n, K) multi-target responses
    gene_names : list of n gene symbols (HUGO names)
    config : experiment configuration
    llm_agent : LLMPolicyAgent instance
    verbose : print per-cycle progress

    Returns
    -------
    ExperimentResult with per-cycle metrics
    """
    rng = np.random.default_rng(config.seed)
    n, K = Y.shape

    thresholds = compute_thresholds(Y, config.topk_percent)
    true_hits = [np.where(Y[:, k] > thresholds[k])[0] for k in range(K)]
    gene_names_arr = np.array(gene_names)

    if verbose:
        print(f"Dataset: n={n}, d={X.shape[1]}, K={K}")
        for k in range(K):
            print(f"  Target {k}: {len(true_hits[k])} hits (τ={thresholds[k]:.3f})")

    # Initial random observations
    initial = rng.choice(n, size=min(config.n_initial, n), replace=False)
    obs_mask = np.zeros(n, dtype=bool)
    obs_mask[initial] = True

    total_budget = config.n_cycles * config.batch_size
    budget_remaining = total_budget

    gp = MultiOutputGP(
        K=K,
        lengthscale=config.gp_lengthscale,
        amplitude=config.gp_amplitude,
        noise=config.gp_noise,
    )

    result = ExperimentResult(config=config)
    current_recall = np.zeros(K)
    last_hit_rate = 0.0

    for cycle in range(config.n_cycles):
        obs_idx = np.where(obs_mask)[0]
        pool_idx = np.where(~obs_mask)[0]

        if len(pool_idx) == 0:
            if verbose:
                print(f"  Cycle {cycle}: pool exhausted")
            break

        # Fit GP
        gp.fit(X[obs_idx], Y[obs_idx], optimize=config.optimize_gp)

        # GP posteriors for pool
        mu_pool, sigma_pool = gp.predict(X[pool_idx])
        actual_batch = min(config.batch_size, budget_remaining, len(pool_idx))

        # ── LLM selects genes directly ──
        selected_in_pool = llm_agent.select_batch(
            gene_names=gene_names_arr[pool_idx],
            mu_pool=mu_pool,
            sigma_pool=sigma_pool,
            thresholds=thresholds,
            pool_indices=pool_idx,
            batch_size=actual_batch,
            cycle=cycle,
            n_cycles=config.n_cycles,
            n_observed=len(obs_idx),
            n_total=n,
            budget_remaining=budget_remaining,
            per_target_recall=current_recall,
            last_hit_rate=last_hit_rate,
            rng=rng,
        )

        # Query selected genes
        selected_global = pool_idx[selected_in_pool]
        obs_mask[selected_global] = True
        budget_remaining -= len(selected_global)

        # Compute hit rate for next cycle
        all_observed = np.where(obs_mask)[0]
        hits_now = hit_count(all_observed, true_hits).sum()
        hits_before = hit_count(obs_idx, true_hits).sum()
        new_hits = hits_now - hits_before
        last_hit_rate = new_hits / max(len(selected_global), 1)

        current_recall = per_target_recall(all_observed, true_hits)

        # Determine if LLM was used this cycle
        used_llm = (
            len(llm_agent.history) > 0
            and llm_agent.history[-1].get("cycle") == cycle
            and llm_agent.history[-1].get("used_llm", False)
        )

        cycle_log = CycleLog(
            cycle=cycle,
            n_observed=len(all_observed),
            per_target_recall=current_recall.copy(),
            any_recall=any_target_recall(all_observed, true_hits),
            all_recall=all_target_recall(all_observed, true_hits),
            hits_found=hit_count(all_observed, true_hits),
            efficiency=budget_efficiency(all_observed, true_hits),
            policy_used="llm_policy" if used_llm else "llm_policy_fallback",
            weights=np.ones(K) / K,
        )
        result.cycle_logs.append(cycle_log)

        if verbose:
            recall_str = ", ".join(f"{r:.3f}" for r in cycle_log.per_target_recall)
            tag = "LLM" if used_llm else "fallback"
            print(
                f"  Cycle {cycle}: obs={cycle_log.n_observed}, "
                f"recall=[{recall_str}], "
                f"any={cycle_log.any_recall:.3f}, "
                f"hit_rate={last_hit_rate:.3f} ({tag})"
            )

    # Final summary
    final_observed = np.where(obs_mask)[0]
    result.total_hits = hit_count(final_observed, true_hits)
    result.final_recall = per_target_recall(final_observed, true_hits)
    result.total_queries = int(obs_mask.sum()) - config.n_initial

    if verbose:
        print(f"\nFinal: {result.total_queries} queries")
        for k in range(K):
            print(
                f"  Target {k}: {result.total_hits[k]}/{len(true_hits[k])} hits "
                f"(recall {result.final_recall[k]:.3f})"
            )
        print(f"  Any-target recall: {any_target_recall(final_observed, true_hits):.3f}")

        # LLM usage stats
        if llm_agent.history:
            n_llm = sum(1 for h in llm_agent.history if h.get("used_llm"))
            n_fb = len(llm_agent.history) - n_llm
            print(f"  LLM usage: {n_llm} LLM / {n_fb} fallback cycles")

    return result


# --------------------------------------------------------------------------- #
#       Hybrid LLM Policy: GP exploitation + LLM exploration                   #
# --------------------------------------------------------------------------- #


def run_experiment_hybrid_policy(
    X: np.ndarray,
    Y: np.ndarray,
    gene_names: list[str],
    config: ExperimentConfig,
    hybrid_agent: "HybridLLMPolicy",
    verbose: bool = True,
) -> ExperimentResult:
    """Run experiment with hybrid policy: GP picks exploit slots, LLM picks explore.

    The GP selects ~70% of each batch (pure max_poh — never worse than classic).
    The LLM fills the remaining ~30% from a tiebreaker pool of borderline
    candidates where biology reasoning can break ties.

    After each cycle, the LLM gets feedback on which picks were hits,
    building an in-context memory of what gene families work.
    """
    rng = np.random.default_rng(config.seed)
    n, K = Y.shape

    thresholds = compute_thresholds(Y, config.topk_percent)
    true_hits = [np.where(Y[:, k] > thresholds[k])[0] for k in range(K)]
    gene_names_arr = np.array(gene_names)

    if verbose:
        print(f"Dataset: n={n}, d={X.shape[1]}, K={K}")
        for k in range(K):
            print(f"  Target {k}: {len(true_hits[k])} hits (τ={thresholds[k]:.3f})")

    initial = rng.choice(n, size=min(config.n_initial, n), replace=False)
    obs_mask = np.zeros(n, dtype=bool)
    obs_mask[initial] = True

    total_budget = config.n_cycles * config.batch_size
    budget_remaining = total_budget

    gp = MultiOutputGP(
        K=K,
        lengthscale=config.gp_lengthscale,
        amplitude=config.gp_amplitude,
        noise=config.gp_noise,
    )

    result = ExperimentResult(config=config)
    current_recall = np.zeros(K)
    last_hit_rate = 0.0

    for cycle in range(config.n_cycles):
        obs_idx = np.where(obs_mask)[0]
        pool_idx = np.where(~obs_mask)[0]

        if len(pool_idx) == 0:
            if verbose:
                print(f"  Cycle {cycle}: pool exhausted")
            break

        gp.fit(X[obs_idx], Y[obs_idx], optimize=config.optimize_gp)
        mu_pool, sigma_pool = gp.predict(X[pool_idx])
        actual_batch = min(config.batch_size, budget_remaining, len(pool_idx))

        # Hybrid selection: GP exploit + LLM explore
        selected_in_pool = hybrid_agent.select_batch(
            gene_names=gene_names_arr[pool_idx],
            mu_pool=mu_pool,
            sigma_pool=sigma_pool,
            thresholds=thresholds,
            pool_indices=pool_idx,
            batch_size=actual_batch,
            cycle=cycle,
            n_cycles=config.n_cycles,
            n_observed=len(obs_idx),
            n_total=n,
            budget_remaining=budget_remaining,
            per_target_recall=current_recall,
            last_hit_rate=last_hit_rate,
            rng=rng,
        )

        # Query selected genes
        selected_global = pool_idx[selected_in_pool]
        obs_mask[selected_global] = True
        budget_remaining -= len(selected_global)

        # Feed back outcomes to the hybrid agent (cycle memory)
        was_hit = np.zeros(len(selected_global), dtype=bool)
        for k in range(K):
            was_hit |= Y[selected_global, k] > thresholds[k]
        hybrid_agent.update_cycle_outcomes(
            gene_names=gene_names_arr[selected_global].tolist(),
            was_hit=was_hit,
        )

        # Stats
        all_observed = np.where(obs_mask)[0]
        hits_now = hit_count(all_observed, true_hits).sum()
        hits_before = hit_count(obs_idx, true_hits).sum()
        new_hits = hits_now - hits_before
        last_hit_rate = new_hits / max(len(selected_global), 1)

        current_recall = per_target_recall(all_observed, true_hits)

        used_llm = (
            len(hybrid_agent.history) > 0
            and hybrid_agent.history[-1].get("cycle") == cycle
            and hybrid_agent.history[-1].get("used_llm", False)
        )

        cycle_log = CycleLog(
            cycle=cycle,
            n_observed=len(all_observed),
            per_target_recall=current_recall.copy(),
            any_recall=any_target_recall(all_observed, true_hits),
            all_recall=all_target_recall(all_observed, true_hits),
            hits_found=hit_count(all_observed, true_hits),
            efficiency=budget_efficiency(all_observed, true_hits),
            policy_used="hybrid_llm" if used_llm else "hybrid_fallback",
            weights=np.ones(K) / K,
        )
        result.cycle_logs.append(cycle_log)

        if verbose:
            recall_str = ", ".join(f"{r:.3f}" for r in cycle_log.per_target_recall)
            tag = "LLM" if used_llm else "fallback"
            h = hybrid_agent.history[-1] if hybrid_agent.history else {}
            print(
                f"  Cycle {cycle}: obs={cycle_log.n_observed}, "
                f"recall=[{recall_str}], "
                f"any={cycle_log.any_recall:.3f}, "
                f"hit_rate={last_hit_rate:.3f} "
                f"({tag}, exploit={h.get('n_exploit', '?')}/explore={h.get('n_explore', '?')})"
            )

    final_observed = np.where(obs_mask)[0]
    result.total_hits = hit_count(final_observed, true_hits)
    result.final_recall = per_target_recall(final_observed, true_hits)
    result.total_queries = int(obs_mask.sum()) - config.n_initial

    if verbose:
        print(f"\nFinal: {result.total_queries} queries")
        for k in range(K):
            print(
                f"  Target {k}: {result.total_hits[k]}/{len(true_hits[k])} hits "
                f"(recall {result.final_recall[k]:.3f})"
            )
        print(f"  Any-target recall: {any_target_recall(final_observed, true_hits):.3f}")
        if hybrid_agent.history:
            n_llm = sum(1 for h in hybrid_agent.history if h.get("used_llm"))
            n_fb = len(hybrid_agent.history) - n_llm
            print(f"  Hybrid: {n_llm} LLM / {n_fb} fallback cycles")
            print(f"  Hit memory: {len(hybrid_agent._hit_genes)} hits, "
                  f"{len(hybrid_agent._miss_genes)} misses accumulated")

    return result
