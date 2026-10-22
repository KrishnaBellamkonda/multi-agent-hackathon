"""
Multi-target synthetic datasets for benchmarking.

Each dataset returns (X, Y, meta) where:
  X: (n, d) feature matrix
  Y: (n, K) multi-target response (K targets)
  meta: dict with dataset info, ground-truth structure, etc.

Datasets are designed so that:
  1. Targets share SOME but not all hit genes (controllable overlap)
  2. A GP with RBF kernel can learn each target well
  3. Structure is fully interpretable (known pathways, known overlap)
  4. Difficulty is tunable via noise, dimensionality, overlap fraction

Three main scenarios:
  - correlated_targets: smooth functions with controlled correlation
  - pathway_multitarget: pathway-structured with target-specific activation
  - independent_targets: baseline with zero cross-target structure
"""
from __future__ import annotations

import numpy as np
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════════
#  1.  CORRELATED TARGETS  (simplest multi-target test)
# ═══════════════════════════════════════════════════════════════════════════

def correlated_targets(
    n: int = 500,
    d: int = 4,
    K: int = 2,
    overlap: float = 0.3,
    noise: float = 0.05,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, dict]:
    r"""Multi-target dataset with controlled hit-region overlap.

    Each target k has a "center of effect" c_k in R^d and a response:

        f_k(x) = shared(x) + target_k(x)

    where shared(x) is common across targets (controls overlap) and
    target_k(x) is target-specific.

    Specifically:
        shared(x)   = A * exp(-||x - c_shared||^2 / (2 * sigma_s^2))
        target_k(x) = (1-A) * exp(-||x - c_k||^2 / (2 * sigma_k^2))
                     + gene_effect_k(x)

    where A = overlap controls the fraction of variation that is shared.

    Parameters
    ----------
    n : int
        Number of genes (data points).
    d : int
        Feature dimensionality.
    K : int
        Number of targets (phenotypes).
    overlap : float in [0, 1]
        Fraction of response variance that is shared across targets.
        0 = fully independent, 1 = identical targets.
    noise : float
        Observation noise σ.
    seed : int
        Random seed.

    Returns
    -------
    X : (n, d) feature matrix
    Y : (n, K) multi-target responses
    meta : dict with ground-truth info
    """
    rng = np.random.default_rng(seed)

    # Gene features: uniform in [0,1]^d
    X = rng.uniform(0, 1, size=(n, d))

    # Shared center (one for all targets)
    c_shared = rng.uniform(0.3, 0.7, size=d)
    sigma_s = 0.4

    # Per-target centers (spread around the unit hypercube)
    target_centers = []
    for k in range(K):
        angle = 2 * np.pi * k / K
        offset = 0.25 * np.array([np.cos(angle), np.sin(angle)] + [0.0] * (d - 2))
        c_k = np.clip(0.5 + offset, 0.1, 0.9)
        target_centers.append(c_k)
    target_centers = np.array(target_centers)  # (K, d)
    sigma_k = 0.3

    # Random per-target frequency vectors for gene-specific modulation
    freq_vectors = rng.uniform(1.0, 4.0, size=(K, d))

    Y = np.zeros((n, K))
    for k in range(K):
        # Shared component
        dist_shared = np.sum((X - c_shared) ** 2, axis=1)
        shared = overlap * np.exp(-dist_shared / (2 * sigma_s ** 2))

        # Target-specific component
        dist_k = np.sum((X - target_centers[k]) ** 2, axis=1)
        target_specific = (1 - overlap) * np.exp(-dist_k / (2 * sigma_k ** 2))

        # Gene-specific modulation (smooth, learnable by GP)
        gene_mod = 0.2 * np.sin(np.pi * X @ freq_vectors[k])

        Y[:, k] = shared + target_specific + gene_mod

    # Add noise
    Y += noise * rng.standard_normal(Y.shape)

    # Normalize each target to [0, 1]
    for k in range(K):
        yk = Y[:, k]
        Y[:, k] = (yk - yk.min()) / (yk.max() - yk.min() + 1e-8)

    meta = {
        "name": "correlated_targets",
        "n": n,
        "d": d,
        "K": K,
        "overlap": overlap,
        "noise": noise,
        "seed": seed,
        "c_shared": c_shared,
        "target_centers": target_centers,
        "freq_vectors": freq_vectors,
    }
    return X, Y, meta


# ═══════════════════════════════════════════════════════════════════════════
#  2.  PATHWAY MULTI-TARGET  (biologically structured)
# ═══════════════════════════════════════════════════════════════════════════

def pathway_multitarget(
    n_genes: int = 200,
    n_pathways: int = 6,
    K: int = 3,
    d_gene: int = 4,
    pathway_overlap: float = 0.3,
    noise: float = 0.05,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, dict]:
    r"""Pathway-structured multi-target dataset.

    Genes belong to pathways. Each target k activates a SUBSET of
    pathways. Shared pathways create cross-target hit overlap.

    Structure:
        - n_pathways pathways, each a cluster in R^d_gene
        - Each target activates ceil(pathway_overlap * n_pathways) shared
          pathways + some target-specific ones
        - Gene response: f_k(gene) = Σ_p  w_{k,p} · pathway_effect_p(gene)
        - Pathway effect is a Gaussian bump around pathway center

    This mimics real biology: immune targets share TCR signaling
    pathway but diverge in downstream effectors (STAT vs NFAT).

    Parameters
    ----------
    n_genes : int
        Number of genes (pool size).
    n_pathways : int
        Number of biological pathways.
    K : int
        Number of targets (phenotypes).
    d_gene : int
        Gene embedding dimensionality.
    pathway_overlap : float in [0, 1]
        Fraction of pathways shared by all targets.
    noise : float
        Observation noise.
    seed : int
        Random seed.

    Returns
    -------
    X : (n_genes, d_gene) gene feature matrix
    Y : (n_genes, K) multi-target responses
    meta : dict with pathway assignments, activation matrix, etc.
    """
    rng = np.random.default_rng(seed)

    # ── Pathway centers in gene-embedding space ──
    # Use well-separated centers via Sobol-like placement
    pathway_centers = np.zeros((n_pathways, d_gene))
    for p in range(n_pathways):
        for dim in range(d_gene):
            pathway_centers[p, dim] = (
                (p * 0.618033988749895 + dim * 0.3247) % 1.0  # golden ratio spacing
            )
    # Scale to [0.1, 0.9]
    pathway_centers = 0.1 + 0.8 * pathway_centers

    # ── Assign genes to pathways ──
    genes_per_pathway = n_genes // n_pathways
    gene_pathway = np.repeat(np.arange(n_pathways), genes_per_pathway)
    remaining = n_genes - len(gene_pathway)
    if remaining > 0:
        gene_pathway = np.concatenate([gene_pathway, rng.integers(0, n_pathways, remaining)])
    rng.shuffle(gene_pathway)

    # ── Gene embeddings: clustered around pathway centers ──
    spread = 0.08
    X = np.zeros((n_genes, d_gene))
    for i in range(n_genes):
        X[i] = pathway_centers[gene_pathway[i]] + rng.normal(0, spread, d_gene)
    X = np.clip(X, 0, 1)

    # ── Pathway activation matrix: which targets activate which pathways ──
    # Shape: (K, n_pathways), binary
    n_shared = max(1, int(pathway_overlap * n_pathways))
    n_specific = max(1, (n_pathways - n_shared) // K)

    # Shared pathways (first n_shared)
    activation = np.zeros((K, n_pathways), dtype=float)
    activation[:, :n_shared] = 1.0

    # Target-specific pathways
    specific_start = n_shared
    for k in range(K):
        start = specific_start + k * n_specific
        end = min(start + n_specific, n_pathways)
        activation[k, start:end] = 1.0

    # Random modulation of pathway strengths per target
    pathway_strengths = rng.uniform(0.5, 1.5, size=(K, n_pathways))
    activation = activation * pathway_strengths

    # ── Pathway effect widths ──
    pathway_widths = rng.uniform(0.15, 0.30, size=n_pathways)

    # ── Generate responses ──
    Y = np.zeros((n_genes, K))
    for k in range(K):
        for i in range(n_genes):
            total = 0.0
            for p in range(n_pathways):
                if activation[k, p] < 1e-6:
                    continue
                dist_sq = np.sum((X[i] - pathway_centers[p]) ** 2)
                bump = np.exp(-dist_sq / (2 * pathway_widths[p] ** 2))
                total += activation[k, p] * bump
            Y[i, k] = total

    # Add noise
    Y += noise * rng.standard_normal(Y.shape)

    # Normalize each target to [0, 1]
    for k in range(K):
        yk = Y[:, k]
        Y[:, k] = (yk - yk.min()) / (yk.max() - yk.min() + 1e-8)

    meta = {
        "name": "pathway_multitarget",
        "n_genes": n_genes,
        "n_pathways": n_pathways,
        "K": K,
        "d_gene": d_gene,
        "pathway_overlap": pathway_overlap,
        "noise": noise,
        "seed": seed,
        "pathway_centers": pathway_centers,
        "gene_pathway": gene_pathway,
        "activation": activation,
        "pathway_widths": pathway_widths,
    }
    return X, Y, meta


# ═══════════════════════════════════════════════════════════════════════════
#  3.  INDEPENDENT TARGETS  (negative control)
# ═══════════════════════════════════════════════════════════════════════════

def independent_targets(
    n: int = 500,
    d: int = 4,
    K: int = 2,
    noise: float = 0.05,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Multi-target dataset with NO cross-target structure.

    Each target is generated from a different random function with
    no shared components. This is the negative control: multi-target
    resource sharing should provide NO benefit here (but also no harm).
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, size=(n, d))

    Y = np.zeros((n, K))
    for k in range(K):
        # Random center and frequency per target
        c_k = rng.uniform(0.2, 0.8, size=d)
        freq_k = rng.uniform(2.0, 6.0, size=d)
        dist_k = np.sum((X - c_k) ** 2, axis=1)
        Y[:, k] = np.exp(-dist_k / 0.3) + 0.3 * np.sin(X @ freq_k)

    Y += noise * rng.standard_normal(Y.shape)

    for k in range(K):
        yk = Y[:, k]
        Y[:, k] = (yk - yk.min()) / (yk.max() - yk.min() + 1e-8)

    meta = {
        "name": "independent_targets",
        "n": n, "d": d, "K": K, "noise": noise, "seed": seed,
    }
    return X, Y, meta


# ═══════════════════════════════════════════════════════════════════════════
#  4.  REGISTRY
# ═══════════════════════════════════════════════════════════════════════════

REGISTRY = {
    "correlated_targets": correlated_targets,
    "pathway_multitarget": pathway_multitarget,
    "independent_targets": independent_targets,
}


def get_dataset(name: str, **kwargs) -> tuple[np.ndarray, np.ndarray, dict]:
    """Get a multi-target dataset by name."""
    if name not in REGISTRY:
        raise ValueError(f"Unknown dataset '{name}'. Available: {list(REGISTRY.keys())}")
    return REGISTRY[name](**kwargs)


def compute_thresholds(
    Y: np.ndarray, topk_percent: float = 0.1
) -> np.ndarray:
    """Compute per-target hit thresholds (top topk_percent).

    Returns
    -------
    thresholds : (K,) array where threshold[k] is the value above which
                 a gene is a "hit" for target k.
    """
    K = Y.shape[1]
    thresholds = np.zeros(K)
    for k in range(K):
        sorted_vals = np.sort(Y[:, k])
        idx = int(len(sorted_vals) * (1 - topk_percent))
        thresholds[k] = sorted_vals[idx]
    return thresholds


def compute_hit_overlap(Y: np.ndarray, thresholds: np.ndarray) -> dict:
    """Compute pairwise hit overlap (Jaccard) between targets.

    Useful for understanding cross-target structure.
    """
    K = Y.shape[1]
    hits = [set(np.where(Y[:, k] > thresholds[k])[0]) for k in range(K)]

    jaccard = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            if i == j:
                jaccard[i, j] = 1.0
            else:
                union = len(hits[i] | hits[j])
                jaccard[i, j] = len(hits[i] & hits[j]) / union if union > 0 else 0.0

    # Any-target hits (hit in at least one target)
    any_hits = set()
    for h in hits:
        any_hits |= h

    # All-target hits (hit in every target)
    all_hits = hits[0].copy()
    for h in hits[1:]:
        all_hits &= h

    return {
        "jaccard": jaccard,
        "per_target_hits": [len(h) for h in hits],
        "any_target_hits": len(any_hits),
        "all_target_hits": len(all_hits),
    }
