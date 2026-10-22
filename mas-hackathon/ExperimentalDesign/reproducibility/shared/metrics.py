"""
Performance metrics for hit discovery evaluation.

Single-target metrics (RecoveredHitsRatio, SMAPE, SpearmanRho, ClusterRecovery)
and multi-target extensions (per-target recall, any/all-target recall).
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
from scipy.stats import spearmanr
from sklearn.cluster import DBSCAN


@dataclass
class MetricResults:
    """Container for metric results with aggregation support."""
    RecoveredHitsRatio_Cumulative: float
    HitClusters_RecoveryRatio: float
    SMAPE_Unseen: float
    SpearmanRho: float
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'RecoveredHitsRatio_Cumulative': self.RecoveredHitsRatio_Cumulative,
            'HitClusters_RecoveryRatio': self.HitClusters_RecoveryRatio,
            'SMAPE_Unseen': self.SMAPE_Unseen,
            'SpearmanRho': self.SpearmanRho,
        }


def compute_hit_recovery_ratio(acquired_indices: np.ndarray, 
                                true_hit_indices: np.ndarray) -> float:
    """Compute the fraction of true hits that were acquired."""
    if len(true_hit_indices) == 0:
        return 0.0
    
    acquired_set = set(acquired_indices)
    hits_recovered = sum(1 for idx in true_hit_indices if idx in acquired_set)
    return hits_recovered / len(true_hit_indices)


def compute_smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Symmetric Mean Absolute Percentage Error.
    SMAPE = 100 * mean(|y_true - y_pred| / (|y_true| + |y_pred| + eps))
    """
    if len(y_true) == 0:
        return 0.0
    
    eps = 1e-8
    denominator = np.abs(y_true) + np.abs(y_pred) + eps
    smape = 100 * np.mean(np.abs(y_true - y_pred) / denominator)
    return smape


def compute_spearman_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Spearman rank correlation."""
    if len(y_true) < 2:
        return 0.0
    
    corr, _ = spearmanr(y_true, y_pred)
    return 0.0 if np.isnan(corr) else corr


def compute_hit_cluster_recovery(X: np.ndarray, 
                                  acquired_indices: np.ndarray,
                                  true_hit_indices: np.ndarray,
                                  eps: float = 0.5,
                                  min_samples: int = 2) -> float:
    """
    Compute fraction of hit clusters that were discovered.
    Uses DBSCAN to cluster true hits, then checks coverage.
    """
    if len(true_hit_indices) == 0:
        return 0.0
    
    X_hits = X[true_hit_indices]
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(X_hits)
    labels = clustering.labels_
    
    unique_clusters = set(labels) - {-1}
    
    if len(unique_clusters) == 0:
        unique_clusters = set(range(len(true_hit_indices)))
        labels = np.arange(len(true_hit_indices))
    
    acquired_set = set(acquired_indices)
    clusters_discovered = 0
    
    for cluster_id in unique_clusters:
        cluster_indices = true_hit_indices[labels == cluster_id]
        if any(idx in acquired_set for idx in cluster_indices):
            clusters_discovered += 1
    
    return clusters_discovered / len(unique_clusters)


def compute_metrics(X: np.ndarray, y: np.ndarray,
                    acquired_indices: np.ndarray,
                    true_hit_indices: np.ndarray,
                    y_pred: Optional[np.ndarray] = None,
                    unseen_indices: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Compute all evaluation metrics.
    
    Args:
        X: Feature matrix
        y: True target values
        acquired_indices: Indices of acquired points
        true_hit_indices: Indices of true hits
        y_pred: GP predictions (optional)
        unseen_indices: Indices of unseen points for SMAPE (optional)
    
    Returns:
        Dictionary of metric names to values
    """
    metrics = {}
    
    metrics['RecoveredHitsRatio_Cumulative'] = compute_hit_recovery_ratio(
        acquired_indices, true_hit_indices
    )
    
    metrics['HitClusters_RecoveryRatio'] = compute_hit_cluster_recovery(
        X, acquired_indices, true_hit_indices
    )
    
    if y_pred is not None and unseen_indices is not None and len(unseen_indices) > 0:
        y_unseen = y[unseen_indices]
        y_pred_unseen = y_pred[unseen_indices]
        
        metrics['SMAPE_Unseen'] = compute_smape(y_unseen, y_pred_unseen)
        metrics['SpearmanRho'] = compute_spearman_correlation(y_unseen, y_pred_unseen)
    else:
        metrics['SMAPE_Unseen'] = np.nan
        metrics['SpearmanRho'] = np.nan
    
    return metrics


def aggregate_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """Aggregate metrics across multiple random seeds."""
    if not metrics_list:
        return {}
    
    aggregated = {}
    metric_names = metrics_list[0].keys()
    
    for name in metric_names:
        values = [m[name] for m in metrics_list if not np.isnan(m.get(name, np.nan))]
        if values:
            aggregated[name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'n': len(values)
            }
    
    return aggregated


# ── Multi-target metrics ──────────────────────────────────────────────


@dataclass
class MultiTargetMetricResults:
    """Per-target and aggregate recall metrics for multi-target experiments."""

    per_target_recall: np.ndarray        # (K,) RecoveredHitsRatio per target
    mean_recall: float                   # mean across targets
    any_target_recall: float             # fraction of union-of-hits recovered
    all_target_recall: float             # fraction of intersection-of-hits recovered
    target_names: List[str]

    def to_dict(self) -> Dict[str, float]:
        result: Dict[str, float] = {
            "mean_recall": self.mean_recall,
            "any_target_recall": self.any_target_recall,
            "all_target_recall": self.all_target_recall,
        }
        for name, r in zip(self.target_names, self.per_target_recall):
            result[f"recall_{name}"] = float(r)
        return result


def compute_multitarget_metrics(
    acquired_indices: np.ndarray,
    true_hits_per_target: List[np.ndarray],
    target_names: Optional[List[str]] = None,
) -> MultiTargetMetricResults:
    """Compute per-target and aggregate recall for a multi-target experiment.

    Parameters
    ----------
    acquired_indices     : indices of all genes queried so far
    true_hits_per_target : list of K arrays, each containing true hit indices
    target_names         : optional list of K target name strings

    Returns
    -------
    MultiTargetMetricResults
    """
    K = len(true_hits_per_target)
    if target_names is None:
        target_names = [f"target_{k}" for k in range(K)]

    acquired_set = set(acquired_indices.tolist())

    per_target = np.array(
        [compute_hit_recovery_ratio(acquired_indices, hits) for hits in true_hits_per_target]
    )

    # Any-target: gene is a "hit" if it's a hit for at least one target
    union_hits: set = set()
    for hits in true_hits_per_target:
        union_hits.update(hits.tolist())
    any_recall = (
        len(acquired_set & union_hits) / len(union_hits) if union_hits else 1.0
    )

    # All-target: gene must be a hit for every target
    if true_hits_per_target:
        intersection_hits = set(true_hits_per_target[0].tolist())
        for hits in true_hits_per_target[1:]:
            intersection_hits &= set(hits.tolist())
    else:
        intersection_hits = set()
    all_recall = (
        len(acquired_set & intersection_hits) / len(intersection_hits)
        if intersection_hits
        else 1.0
    )

    return MultiTargetMetricResults(
        per_target_recall=per_target,
        mean_recall=float(per_target.mean()),
        any_target_recall=float(any_recall),
        all_target_recall=float(all_recall),
        target_names=list(target_names),
    )


def aggregate_multitarget_metrics(
    results_list: List[MultiTargetMetricResults],
) -> Dict[str, Dict[str, float]]:
    """Aggregate MultiTargetMetricResults across multiple seeds."""
    if not results_list:
        return {}

    all_dicts = [r.to_dict() for r in results_list]
    keys = list(all_dicts[0].keys())
    aggregated: Dict[str, Dict[str, float]] = {}
    for key in keys:
        vals = [d[key] for d in all_dicts if not np.isnan(d.get(key, np.nan))]
        if vals:
            aggregated[key] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "n": len(vals),
            }
    return aggregated


if __name__ == "__main__":
    print("Testing metrics...")
    np.random.seed(42)
    n = 100
    X = np.random.randn(n, 2)
    y = np.random.randn(n)
    
    true_hits = np.where(y > np.percentile(y, 90))[0]
    acquired = np.random.choice(n, size=20, replace=False)
    y_pred = y + 0.1 * np.random.randn(n)
    unseen = np.setdiff1d(np.arange(n), acquired)
    
    metrics = compute_metrics(X, y, acquired, true_hits, y_pred, unseen)
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
