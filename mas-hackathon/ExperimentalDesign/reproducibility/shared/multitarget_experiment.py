"""
Multi-target experiment runner -- naive independent-GP baseline.

Strategy:
    - K independent GPs, one per target (same GPyTorch ExactGP used elsewhere)
    - Each target is allocated an equal sub-batch: sub_batch = batch_size // K
    - The batch for a cycle = UNION of per-target selections
      (deduplicated; padded with random genes if union < batch_size)
    - Querying a gene reveals ALL K target values simultaneously (shared oracle)
    - After every batch, every GP is updated with the full observed set

This is the "fixed sub-batch" baseline against which the maehd/ai agent
framework (with joint multi-target GP and LLM-driven acquisition) is compared.
"""

from __future__ import annotations

import pickle
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .acquisition import get_acquisition_function
from .experiment import predict_gp, train_gp
from .metrics import compute_hit_recovery_ratio


# ── Data containers ───────────────────────────────────────────────────


@dataclass
class MultiTargetExperimentConfig:
    """Configuration for a multi-target independent-GP experiment.

    Parameters
    ----------
    screen_names : K screen names (e.g. ["schmidt_2021_ifng", "schmidt_2021_il2"])
                   OR a MULTI_TARGET_SETUPS key (e.g. "t_cell")
    feature_set  : which feature matrix to use ("achilles", "string", "ccle")
    method_name  : acquisition function applied independently per target
    batch_size   : total genes queried per cycle across all K targets
    threshold_pct: fraction of genes counted as hits (top-k percent)
    seed         : random seed
    n_cycles     : number of acquisition cycles
    n_initial    : random genes observed before cycle 1
    gp_iterations: Adam steps for GPyTorch marginal likelihood training
    gp_lr        : learning rate for GP training
    data_dir     : path to ExperimentalDesign/data/ (auto-resolved if None)
    """

    screen_names: List[str]
    feature_set: str = "achilles"
    method_name: str = "probability_hit"
    batch_size: int = 20
    threshold_pct: float = 0.10
    seed: int = 42
    n_cycles: int = 15
    n_initial: int = 10
    gp_iterations: int = 100
    gp_lr: float = 0.1
    data_dir: Optional[str] = None


@dataclass
class MultiTargetCycleResults:
    """Results from one acquisition cycle."""

    cycle: int
    sub_batch_size: int                        # genes allocated per target
    acquired_indices: np.ndarray               # newly queried indices this cycle
    cumulative_acquired: np.ndarray            # all queried so far
    per_target_recall: np.ndarray              # (K,) RecoveredHitsRatio per target
    mean_recall: float
    # local (within-pool) indices each target's GP selected
    per_target_local_selections: Dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class MultiTargetExperimentResults:
    """Complete results from a multi-target experiment."""

    config: MultiTargetExperimentConfig
    screen_names: List[str]
    cycles: List[MultiTargetCycleResults]
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
                    "sub_batch_size": c.sub_batch_size,
                    "acquired_indices": c.acquired_indices.tolist(),
                    "cumulative_acquired": c.cumulative_acquired.tolist(),
                    "per_target_recall": c.per_target_recall.tolist(),
                    "mean_recall": c.mean_recall,
                }
                for c in self.cycles
            ],
            "final_per_target_recall": self.final_per_target_recall.tolist(),
            "final_mean_recall": self.final_mean_recall,
            "success": self.success,
            "error_message": self.error_message,
        }


# ── Core runner ───────────────────────────────────────────────────────


def _load_data(
    config: MultiTargetExperimentConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
    """Load (X, Y, thresholds, gene_names, resolved_screen_names) from disk."""
    from maehd.datasets.genedisco import load_multitarget, MULTI_TARGET_SETUPS

    screen_arg: Any = config.screen_names
    # Accept a single string that is a MULTI_TARGET_SETUPS key
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
    return X, Y, meta["thresholds"], meta["gene_names"], meta["screen_names"]


def run_multitarget_experiment(
    config: MultiTargetExperimentConfig,
    X: Optional[np.ndarray] = None,
    Y: Optional[np.ndarray] = None,
    thresholds: Optional[np.ndarray] = None,
    gene_names: Optional[List[str]] = None,
    output_dir: Optional[Path] = None,
    verbose: bool = False,
) -> MultiTargetExperimentResults:
    """Run the naive multi-target baseline (K independent GPs, fixed sub-batches).

    Data can be supplied directly as numpy arrays (X, Y, thresholds) or
    loaded automatically from the GeneDisco .h5 files via config.screen_names
    and config.feature_set.

    Parameters
    ----------
    config       : experiment configuration
    X            : (n, d) gene feature matrix  [optional — loaded if None]
    Y            : (n, K) multi-target response matrix  [optional]
    thresholds   : (K,) per-target hit thresholds  [optional]
    gene_names   : list of n gene name strings  [optional]
    output_dir   : if given, saves results.pickle here
    verbose      : print cycle-by-cycle progress

    Returns
    -------
    MultiTargetExperimentResults
    """
    import torch

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    resolved_screen_names = list(config.screen_names)

    try:
        # ── Data loading ──────────────────────────────────────────────
        if X is None or Y is None:
            X, Y, thresholds, gene_names, resolved_screen_names = _load_data(config)
            if verbose:
                print(
                    f"  Loaded {X.shape[0]} genes x {X.shape[1]} features, "
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

        # True hit indices per target
        true_hits = [np.where(Y[:, k] > thresholds[k])[0] for k in range(K)]

        if verbose:
            for k, name in enumerate(target_names):
                print(f"  {name}: {len(true_hits[k])} hits (threshold={thresholds[k]:.4f})")

        acq_func = get_acquisition_function(config.method_name)

        # sub_batch = floor(batch_size / K), remainder handled by padding
        sub_batch = max(1, config.batch_size // K)

        # ── Initial random seed ───────────────────────────────────────
        all_indices = np.arange(n)
        acquired = np.random.choice(
            n, size=min(config.n_initial, n), replace=False
        ).copy()

        cycles: List[MultiTargetCycleResults] = []

        # ── Acquisition loop ──────────────────────────────────────────
        for cycle in range(config.n_cycles):
            pool = np.setdiff1d(all_indices, acquired)
            if len(pool) == 0:
                if verbose:
                    print("  Pool exhausted — stopping early.")
                break

            if verbose:
                print(
                    f"  Cycle {cycle + 1}/{config.n_cycles} | "
                    f"observed={len(acquired)} | pool={len(pool)}"
                )

            union_global: set = set()
            per_target_local: Dict[str, np.ndarray] = {}

            for k in range(K):
                name_k = target_names[k]

                # Train single-output GP for target k on all observed data
                try:
                    model_k, lik_k, norm_k = train_gp(
                        X[acquired],
                        Y[acquired, k],
                        n_iterations=config.gp_iterations,
                        lr=config.gp_lr,
                    )
                    gp_mean, gp_std = predict_gp(model_k, lik_k, X[pool], norm_k)
                except Exception as exc:
                    warnings.warn(
                        f"GP training failed for {name_k} (cycle {cycle + 1}): {exc}. "
                        "Falling back to random selection."
                    )
                    gp_mean = np.zeros(len(pool))
                    gp_std = np.ones(len(pool))

                # Acquire top sub_batch local indices from this target's GP
                n_select = min(sub_batch, len(pool))
                local_idx = acq_func.acquire(
                    X[pool],
                    gp_mean,
                    gp_std,
                    batch_size=n_select,
                    threshold=float(thresholds[k]),
                )
                local_idx = np.asarray(local_idx)[:n_select]

                per_target_local[name_k] = local_idx
                union_global.update(pool[local_idx].tolist())

            # ── Build final batch from union, trimmed / padded to batch_size ──
            union_arr = np.array(sorted(union_global), dtype=int)

            if len(union_arr) > config.batch_size:
                # More candidates than budget: sub-sample uniformly
                union_arr = np.random.choice(union_arr, config.batch_size, replace=False)

            elif len(union_arr) < config.batch_size:
                # Fewer candidates (due to deduplication or small pool): pad randomly
                leftover = np.setdiff1d(pool, union_arr)
                n_pad = min(config.batch_size - len(union_arr), len(leftover))
                if n_pad > 0:
                    pad = np.random.choice(leftover, n_pad, replace=False)
                    union_arr = np.concatenate([union_arr, pad])

            new_acquired = union_arr
            acquired = np.concatenate([acquired, new_acquired])

            # ── Per-target recall ─────────────────────────────────────
            per_target_recall = np.array(
                [compute_hit_recovery_ratio(acquired, true_hits[k]) for k in range(K)]
            )
            mean_recall = float(per_target_recall.mean())

            cycles.append(
                MultiTargetCycleResults(
                    cycle=cycle + 1,
                    sub_batch_size=sub_batch,
                    acquired_indices=new_acquired,
                    cumulative_acquired=acquired.copy(),
                    per_target_recall=per_target_recall,
                    mean_recall=mean_recall,
                    per_target_local_selections=per_target_local,
                )
            )

            if verbose:
                n_obs = len(acquired)
                hits_acquired = [int(np.isin(acquired, true_hits[k]).sum()) for k in range(K)]
                recalls = "  ".join(
                    f"{name_k}={h}/{n_obs}({r:.1%})"
                    for name_k, h, r in zip(target_names, hits_acquired, per_target_recall)
                )
                print(f"    hits/tested: {recalls}  mean_recall={mean_recall:.3f}")

        # ── Collect results ───────────────────────────────────────────
        if cycles:
            final_recall = cycles[-1].per_target_recall
        else:
            final_recall = np.zeros(K)

        result = MultiTargetExperimentResults(
            config=config,
            screen_names=target_names,
            cycles=cycles,
            final_per_target_recall=final_recall,
            final_mean_recall=float(final_recall.mean()),
            success=True,
        )

    except Exception as exc:
        warnings.warn(f"Multi-target experiment failed: {exc}")
        result = MultiTargetExperimentResults(
            config=config,
            screen_names=resolved_screen_names,
            cycles=[],
            final_per_target_recall=np.zeros(len(config.screen_names)),
            final_mean_recall=0.0,
            success=False,
            error_message=str(exc),
        )

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "multitarget_results.pickle", "wb") as fh:
            pickle.dump(result.to_dict(), fh)

    return result
