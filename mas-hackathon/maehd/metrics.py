"""
Metrics for multi-target hit discovery.

All metrics work on K targets simultaneously.
"""
from __future__ import annotations

import numpy as np


def per_target_recall(
    discovered: np.ndarray, true_hits: list[np.ndarray],
) -> np.ndarray:
    """Fraction of true hits discovered, per target.

    Parameters
    ----------
    discovered : set of queried gene indices
    true_hits : list of K arrays, each containing indices of true hits

    Returns
    -------
    recall : (K,) recall per target
    """
    discovered_set = set(discovered)
    K = len(true_hits)
    recall = np.zeros(K)
    for k in range(K):
        hits_k = set(true_hits[k])
        if len(hits_k) == 0:
            recall[k] = 1.0  # vacuously true
        else:
            recall[k] = len(discovered_set & hits_k) / len(hits_k)
    return recall


def any_target_recall(
    discovered: np.ndarray, true_hits: list[np.ndarray],
) -> float:
    """Fraction of union-of-hits discovered.

    A gene is a "union hit" if it's a hit for ANY target.
    """
    discovered_set = set(discovered)
    union_hits = set()
    for hits_k in true_hits:
        union_hits.update(hits_k)
    if len(union_hits) == 0:
        return 1.0
    return len(discovered_set & union_hits) / len(union_hits)


def all_target_recall(
    discovered: np.ndarray, true_hits: list[np.ndarray],
) -> float:
    """Fraction of intersection-of-hits discovered.

    A gene is an "intersection hit" if it's a hit for ALL targets.
    """
    discovered_set = set(discovered)
    if len(true_hits) == 0:
        return 1.0
    intersection_hits = set(true_hits[0])
    for hits_k in true_hits[1:]:
        intersection_hits &= set(hits_k)
    if len(intersection_hits) == 0:
        return 1.0
    return len(discovered_set & intersection_hits) / len(intersection_hits)


def hit_count(
    discovered: np.ndarray, true_hits: list[np.ndarray],
) -> np.ndarray:
    """Number of true hits discovered, per target.

    Returns
    -------
    counts : (K,) integer counts
    """
    discovered_set = set(discovered)
    K = len(true_hits)
    counts = np.zeros(K, dtype=int)
    for k in range(K):
        counts[k] = len(discovered_set & set(true_hits[k]))
    return counts


def budget_efficiency(
    discovered: np.ndarray, true_hits: list[np.ndarray],
) -> float:
    """Ratio of unique hits found to total queries made.

    Counts a gene once even if it's a hit for multiple targets.
    Higher = more efficient use of query budget.
    """
    if len(discovered) == 0:
        return 0.0
    union_hits = set()
    for hits_k in true_hits:
        union_hits.update(hits_k)
    discovered_set = set(discovered)
    return len(discovered_set & union_hits) / len(discovered)
