"""ParEGO multi-objective Bayesian optimization for gene hit discovery.

Knowles (2006) ParEGO adapted for batch multi-target CRISPR screens.

Algorithm per acquisition cycle
---------------------------------
1. Sample a random weight vector λ uniformly from the K-simplex.
2. Compute the augmented Chebyshev scalarization for every OBSERVED gene:

     z_i = max_k( λ_k · y_{i,k} ) + ρ · Σ_k λ_k · y_{i,k}

   where ρ = 0.05 mixes the Chebyshev term with a weighted-sum penalty to
   remove dominated solutions on the boundary of the simplex.

3. Fit a SINGLE Gaussian Process on (X_obs, z_obs).

4. Apply an acquisition function (probability_hit by default) on z to select
   the next batch of genes.

How this differs from the naive independent-GP baseline
--------------------------------------------------------
- Uses ONE GP instead of K independent GPs.
- Cross-target information is implicitly captured by the scalarization.
- Budget is *dynamically* allocated: a cycle with high λ_k for target k
  de-prioritises other targets, creating natural diversity across an episode.
- Follows the Pareto-front literature (ParEGO) while staying practical for
  the discrete hit-discovery setting.

Reference
---------
Knowles, J. (2006). ParEGO: A Hybrid Algorithm with On-Line Landscape
Approximation for Expensive Multiobjective Optimization Problems.
IEEE Transactions on Evolutionary Computation.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .acquisition import get_acquisition_function
from .experiment import predict_gp, train_gp
from .metrics import compute_hit_recovery_ratio


# ── Scalarization ─────────────────────────────────────────────────────────────

_RHO = 0.05  # augmented Chebyshev mixing coefficient (from original ParEGO paper)


def _sample_weight_vector(K: int, rng: np.random.RandomState) -> np.ndarray:
    """Sample λ uniformly from the K-simplex (Dirichlet(1,...,1))."""
    x = rng.exponential(1.0, size=K)
    return x / x.sum()


def _augmented_chebyshev(Y: np.ndarray, lam: np.ndarray) -> np.ndarray:
    """Compute augmented Chebyshev scalarization.

    Parameters
    ----------
    Y   : (n, K) observed responses (normalized to [0,1])
    lam : (K,) weight vector, Σlam=1, lam_k > 0

    Returns
    -------
    z : (n,) scalarized values
    """
    # Chebyshev term: max over targets of (λ_k * y_k)
    cheb = np.max(lam[None, :] * Y, axis=1)          # (n,)
    # Weighted-sum penalty term
    wsum = np.dot(Y, lam)                              # (n,)
    return cheb + _RHO * wsum


# ── Data containers ───────────────────────────────────────────────────────────


@dataclass
class ParegoExperimentConfig:
    """Configuration for a ParEGO multi-objective BO experiment.

    Parameters
    ----------
    screen_names  : K screen names (e.g. ["schmidt_2021_ifng", "schmidt_2021_il2"])
    feature_set   : which feature matrix to use ("achilles", "string", "ccle")
    acq_function  : acquisition function applied to the scalarized GP
    batch_size    : total genes queried per cycle
    threshold_pct : fraction of genes counted as hits (top-k percent)
    seed          : random seed
    n_cycles      : number of acquisition cycles
    n_initial     : random genes observed before cycle 1
    gp_iterations : Adam steps for GP marginal likelihood training
    gp_lr         : learning rate for GP training
    rho           : augmented Chebyshev mixing coefficient
    data_dir      : path to ExperimentalDesign/data/ (auto-resolved if None)
    """

    screen_names: List[str]
    feature_set: str = "achilles"
    acq_function: str = "probability_hit"
    batch_size: int = 20
    threshold_pct: float = 0.10
    seed: int = 42
    n_cycles: int = 15
    n_initial: int = 10
    gp_iterations: int = 100
    gp_lr: float = 0.1
    rho: float = _RHO
    data_dir: Optional[str] = None


@dataclass
class ParegoCycleResults:
    """Results from one ParEGO acquisition cycle."""

    cycle: int
    weight_vector: np.ndarray                    # (K,) sampled λ
    acquired_indices: np.ndarray                 # newly queried indices this cycle
    cumulative_acquired: np.ndarray              # all queried so far
    per_target_recall: np.ndarray                # (K,) hit recall per target
    mean_recall: float
    scalarized_threshold: float                  # z threshold used in acquisition


@dataclass
class ParegoExperimentResults:
    """Complete results from a ParEGO experiment."""

    config: ParegoExperimentConfig
    screen_names: List[str]
    cycles: List[ParegoCycleResults]
    final_per_target_recall: np.ndarray
    final_mean_recall: float
    success: bool
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": {**self.config.__dict__},
            "screen_names": self.screen_names,
            "cycles": [
                {
                    "cycle": c.cycle,
                    "weight_vector": c.weight_vector.tolist(),
                    "acquired_indices": c.acquired_indices.tolist(),
                    "cumulative_acquired": c.cumulative_acquired.tolist(),
                    "per_target_recall": c.per_target_recall.tolist(),
                    "mean_recall": c.mean_recall,
                    "scalarized_threshold": c.scalarized_threshold,
                }
                for c in self.cycles
            ],
            "final_per_target_recall": self.final_per_target_recall.tolist(),
            "final_mean_recall": self.final_mean_recall,
            "success": self.success,
            "error_message": self.error_message,
        }


# ── Core runner ───────────────────────────────────────────────────────────────


def run_parego_experiment(
    config: ParegoExperimentConfig,
    X: Optional[np.ndarray] = None,
    Y: Optional[np.ndarray] = None,
    thresholds: Optional[np.ndarray] = None,
    gene_names: Optional[List[str]] = None,
    output_dir: Optional[Path] = None,
    verbose: bool = False,
) -> ParegoExperimentResults:
    """Run ParEGO (random scalarization + single GP) on a multi-target dataset.

    Data can be supplied directly as numpy arrays or loaded from GeneDisco
    .h5 files via config.screen_names and config.feature_set.

    Parameters
    ----------
    config    : experiment configuration
    X         : (n, d) gene feature matrix [optional — loaded if None]
    Y         : (n, K) multi-target responses [optional]
    thresholds: (K,) per-target hit thresholds [optional]
    gene_names: list of n gene name strings [optional]
    output_dir: if given, saves results.pickle here
    verbose   : print cycle-by-cycle progress
    """
    import pickle
    import torch

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    rng = np.random.RandomState(config.seed)

    resolved_screen_names = list(config.screen_names)

    try:
        # ── Data loading ──────────────────────────────────────────────
        if X is None or Y is None:
            from maehd.datasets.genedisco import load_multitarget, MULTI_TARGET_SETUPS

            screen_arg: Any = config.screen_names
            if (
                isinstance(screen_arg, list)
                and len(screen_arg) == 1
                and screen_arg[0] in MULTI_TARGET_SETUPS
            ):
                screen_arg = screen_arg[0]

            X, Y, meta = load_multitarget(
                setup=screen_arg,
                feature_set=config.feature_set,
                cache_directory=config.data_dir,
                topk_percent=config.threshold_pct,
            )
            thresholds = meta["thresholds"]
            gene_names = meta["gene_names"]
            resolved_screen_names = meta["screen_names"]

            if verbose:
                print(
                    f"  Loaded {X.shape[0]} genes × {X.shape[1]} features, "
                    f"{Y.shape[1]} targets"
                )

        n, d = X.shape
        K = Y.shape[1]

        if thresholds is None:
            thresholds = np.array(
                [
                    np.percentile(Y[:, k], 100.0 * (1.0 - config.threshold_pct))
                    for k in range(K)
                ]
            )

        target_names = (
            resolved_screen_names
            if len(resolved_screen_names) == K
            else [f"target_{k}" for k in range(K)]
        )

        true_hits = [np.where(Y[:, k] > thresholds[k])[0] for k in range(K)]

        if verbose:
            for k, name in enumerate(target_names):
                print(f"  {name}: {len(true_hits[k])} hits (threshold={thresholds[k]:.4f})")

        acq_func = get_acquisition_function(config.acq_function)

        # ── Initial random seed ───────────────────────────────────────
        all_indices = np.arange(n)
        acquired = rng.choice(n, size=min(config.n_initial, n), replace=False).copy()

        cycles: List[ParegoCycleResults] = []

        # ── Acquisition loop ──────────────────────────────────────────
        for cycle in range(config.n_cycles):
            pool = np.setdiff1d(all_indices, acquired)
            if len(pool) == 0:
                if verbose:
                    print("  Pool exhausted — stopping early.")
                break

            # 1. Sample random weight vector for this cycle
            lam = _sample_weight_vector(K, rng)

            if verbose:
                lam_str = " ".join(f"{v:.2f}" for v in lam)
                print(
                    f"  Cycle {cycle + 1}/{config.n_cycles} | "
                    f"observed={len(acquired)} | pool={len(pool)} | "
                    f"λ=[{lam_str}]"
                )

            # 2. Compute scalarized z for observed genes
            Y_obs = Y[acquired]                          # (n_obs, K)
            z_obs = _augmented_chebyshev(Y_obs, lam)     # (n_obs,)

            # Scalarized threshold: what z value would correspond to a "hit"
            # We use the weighted combination of per-target thresholds.
            z_threshold = float(
                np.max(lam * thresholds) + config.rho * float(np.dot(lam, thresholds))
            )

            # 3. Fit a single GP on scalarized objective
            try:
                model, lik, norm = train_gp(
                    X[acquired],
                    z_obs,
                    n_iterations=config.gp_iterations,
                    lr=config.gp_lr,
                )
                gp_mean, gp_std = predict_gp(model, lik, X[pool], norm)
            except Exception as exc:
                warnings.warn(
                    f"GP training failed at cycle {cycle + 1}: {exc}. "
                    "Falling back to random selection."
                )
                gp_mean = np.zeros(len(pool))
                gp_std = np.ones(len(pool))

            # 4. Acquire top batch_size genes
            n_select = min(config.batch_size, len(pool))
            local_idx = acq_func.acquire(
                X[pool],
                gp_mean,
                gp_std,
                batch_size=n_select,
                threshold=z_threshold,
            )
            local_idx = np.asarray(local_idx)[:n_select]
            new_acquired = pool[local_idx]
            acquired = np.concatenate([acquired, new_acquired])

            # ── Per-target recall ─────────────────────────────────────
            per_target_recall = np.array(
                [compute_hit_recovery_ratio(acquired, true_hits[k]) for k in range(K)]
            )
            mean_recall = float(per_target_recall.mean())

            cycles.append(
                ParegoCycleResults(
                    cycle=cycle + 1,
                    weight_vector=lam,
                    acquired_indices=new_acquired,
                    cumulative_acquired=acquired.copy(),
                    per_target_recall=per_target_recall,
                    mean_recall=mean_recall,
                    scalarized_threshold=z_threshold,
                )
            )

            if verbose:
                n_obs = len(acquired)
                hits_acquired = [int(np.isin(acquired, true_hits[k]).sum()) for k in range(K)]
                recalls = "  ".join(
                    f"{name}={h}/{n_obs}({r:.1%})"
                    for name, h, r in zip(target_names, hits_acquired, per_target_recall)
                )
                print(f"    hits/tested: {recalls}  mean_recall={mean_recall:.3f}")

        # ── Collect final results ─────────────────────────────────────
        if cycles:
            final_recall = cycles[-1].per_target_recall
        else:
            final_recall = np.zeros(K)

        result = ParegoExperimentResults(
            config=config,
            screen_names=target_names,
            cycles=cycles,
            final_per_target_recall=final_recall,
            final_mean_recall=float(final_recall.mean()),
            success=True,
        )

    except Exception as exc:
        warnings.warn(f"ParEGO experiment failed: {exc}")
        result = ParegoExperimentResults(
            config=config,
            screen_names=resolved_screen_names,
            cycles=[],
            final_per_target_recall=np.zeros(len(config.screen_names)),
            final_mean_recall=0.0,
            success=False,
            error_message=str(exc),
        )

    if output_dir is not None:
        import pickle
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "parego_results.pickle", "wb") as fh:
            pickle.dump(result.to_dict(), fh)

    return result
