"""
GeneDisco data loader for multi-target experiments.

Loads real experimental screening data from GeneDisco as multi-target
datasets: multiple phenotypic screens share the same set of gene
perturbations and gene-level features.

Available multi-target setups:
    - "t_cell": IFNγ + IL-2 from Schmidt 2021 (T cells)
    - "all_immune": IFNγ + IL-2 + NK killing (immune cells)
    - "diverse": IFNγ + IL-2 + Tau + SARS-CoV-2 (all cell types)

Each gene perturbation simultaneously reveals responses for ALL targets.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
import h5py

_ED_ROOT = Path(__file__).resolve().parent.parent.parent / "ExperimentalDesign"

_SCREEN_FILES = {
    "schmidt_2021_ifng": "schmidt_2021_ifng.h5",
    "schmidt_2021_il2": "schmidt_2021_il2.h5",
    "zhuang_2019_nk": "zhuang_2019.h5",
    "sanchez_2021_tau": "sanchez_2021_neurons_tau.h5",
    "zhu_2021_sarscov2": "zhu_2021_sarscov2_host_factors.h5",
}

_FEATURE_FILES = {
    "achilles": "achilles.h5",
    "string": "string_embedding.h5",
    "ccle": "ccle_protein_quantification.h5",
}

# Pre-defined multi-target setups
MULTI_TARGET_SETUPS = {
    # Same lab, same cell type — highly correlated targets
    "t_cell": ["schmidt_2021_ifng", "schmidt_2021_il2"],
    # Same lab + different immune cells
    "all_immune": ["schmidt_2021_ifng", "schmidt_2021_il2", "zhuang_2019_nk"],
    # Cross-biology: completely unrelated phenotypes/cell types
    "cross_biology": ["sanchez_2021_tau", "zhu_2021_sarscov2"],
    # Immune meets neuro — different biology, different cells
    "neuro_immune": ["schmidt_2021_ifng", "sanchez_2021_tau"],
    # Immune meets virology
    "virus_immune": ["schmidt_2021_ifng", "zhu_2021_sarscov2"],
    # Everything — maximum diversity
    "full_diverse": [
        "schmidt_2021_ifng", "schmidt_2021_il2",
        "zhuang_2019_nk", "sanchez_2021_tau", "zhu_2021_sarscov2",
    ],
    # Legacy
    "diverse": [
        "schmidt_2021_ifng", "schmidt_2021_il2",
        "sanchez_2021_tau", "zhu_2021_sarscov2",
    ],
}


def _load_h5(path: str | Path) -> tuple[np.ndarray, list[str]]:
    """Load covariates and gene names from an H5 file."""
    with h5py.File(str(path), "r") as f:
        data = f["covariates"][:]
        rownames = f["rownames"].asstr()[:]
    return data, list(rownames)


def load_multitarget(
    setup: str | list[str] = "t_cell",
    feature_set: str = "achilles",
    cache_directory: str | Path | None = None,
    topk_percent: float = 0.10,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load a multi-target GeneDisco dataset.

    Parameters
    ----------
    setup : name of predefined setup or list of screen names
    feature_set : "achilles", "string", or "ccle"
    cache_directory : path to GeneDisco data files
    topk_percent : fraction of genes considered hits per target

    Returns
    -------
    X : (n_genes, d) feature matrix for common genes
    Y : (n_genes, K) response matrix (one column per target)
    meta : dict with gene names, screen names, thresholds, etc.
    """
    if cache_directory is None:
        cache_directory = _ED_ROOT / "data"
    cache_directory = Path(cache_directory)

    # Resolve setup
    if isinstance(setup, str):
        if setup not in MULTI_TARGET_SETUPS:
            raise ValueError(
                f"Unknown setup: {setup}. "
                f"Available: {list(MULTI_TARGET_SETUPS.keys())}"
            )
        screen_names = MULTI_TARGET_SETUPS[setup]
    else:
        screen_names = list(setup)

    K = len(screen_names)

    # Load features
    feat_path = cache_directory / _FEATURE_FILES[feature_set]
    feat_data, feat_genes = _load_h5(feat_path)
    feat_idx = {g: i for i, g in enumerate(feat_genes)}

    # Load each screen and find common genes
    screen_data_list = []
    screen_gene_sets = [set(feat_genes)]

    for sname in screen_names:
        if sname not in _SCREEN_FILES:
            raise ValueError(f"Unknown screen: {sname}")
        spath = cache_directory / _SCREEN_FILES[sname]
        sdata, sgenes = _load_h5(spath)
        sdx = {g: i for i, g in enumerate(sgenes)}
        screen_data_list.append((sdata, sdx))
        screen_gene_sets.append(set(sgenes))

    # Common genes across features AND all screens
    common_genes = sorted(set.intersection(*screen_gene_sets))
    n_genes = len(common_genes)

    # Build X (features)
    fi = [feat_idx[g] for g in common_genes]
    X = feat_data[fi].astype(np.float32)

    # Handle NaN features
    nan_mask = np.isnan(X)
    if nan_mask.any():
        col_means = np.nanmean(X, axis=0)
        inds = np.where(nan_mask)
        X[inds] = np.take(col_means, inds[1])

    # Build Y (multi-target responses)
    Y = np.zeros((n_genes, K), dtype=np.float32)
    for k, (sdata, sdx) in enumerate(screen_data_list):
        si = [sdx[g] for g in common_genes]
        Y[:, k] = sdata[si].flatten()

    # Normalize each target to [0, 1]
    for k in range(K):
        yk = Y[:, k]
        lo, hi = yk.min(), yk.max()
        if hi > lo:
            Y[:, k] = (yk - lo) / (hi - lo)

    # Compute thresholds
    thresholds = np.zeros(K)
    n_hits_per_target = []
    for k in range(K):
        n_topk = max(1, int(n_genes * topk_percent))
        sorted_yk = np.sort(Y[:, k])[::-1]
        thresholds[k] = sorted_yk[n_topk - 1]
        n_hits_per_target.append(int((Y[:, k] > thresholds[k]).sum()))

    meta = {
        "gene_names": common_genes,
        "screen_names": screen_names,
        "feature_set": feature_set,
        "n_genes": n_genes,
        "K": K,
        "thresholds": thresholds,
        "n_hits_per_target": n_hits_per_target,
        "d_features": X.shape[1],
    }

    return X, Y, meta


def load_multitarget_fusion(
    setup: str | list[str] = "t_cell",
    feature_sets: list[str] | None = None,
    cache_directory: str | Path | None = None,
    topk_percent: float = 0.10,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load multi-target data with FUSED features from multiple sources.

    Loads each feature set independently, finds common genes, concatenates
    feature matrices, then returns the fused feature matrix.

    Parameters
    ----------
    setup : predefined setup name or list of screen names
    feature_sets : list of feature sources to combine (default: all three)
    cache_directory : path to GeneDisco data files
    topk_percent : fraction of genes considered hits per target

    Returns
    -------
    X : (n_genes, sum_of_dims) concatenated feature matrix
    Y : (n_genes, K) response matrix
    meta : dict with additional info including per-source dimensions
    """
    if feature_sets is None:
        feature_sets = ["achilles", "string"]

    if cache_directory is None:
        cache_directory = _ED_ROOT / "data"
    cache_directory = Path(cache_directory)

    # Resolve setup
    if isinstance(setup, str):
        if setup not in MULTI_TARGET_SETUPS:
            raise ValueError(f"Unknown setup: {setup}")
        screen_names = MULTI_TARGET_SETUPS[setup]
    else:
        screen_names = list(setup)

    K = len(screen_names)

    # Load all feature sets and find common genes
    feat_matrices = {}
    feat_gene_sets = []
    for fs in feature_sets:
        fpath = cache_directory / _FEATURE_FILES[fs]
        fdata, fgenes = _load_h5(fpath)
        feat_matrices[fs] = (fdata, {g: i for i, g in enumerate(fgenes)})
        feat_gene_sets.append(set(fgenes))

    # Load screens
    screen_data_list = []
    screen_gene_sets = list(feat_gene_sets)
    for sname in screen_names:
        spath = cache_directory / _SCREEN_FILES[sname]
        sdata, sgenes = _load_h5(spath)
        screen_data_list.append((sdata, {g: i for i, g in enumerate(sgenes)}))
        screen_gene_sets.append(set(sgenes))

    common_genes = sorted(set.intersection(*screen_gene_sets))
    n_genes = len(common_genes)

    # Build concatenated X
    x_parts = []
    dim_breakdown = {}
    for fs in feature_sets:
        fdata, fidx = feat_matrices[fs]
        fi = [fidx[g] for g in common_genes]
        xfs = fdata[fi].astype(np.float32)
        # Handle NaN
        nan_mask = np.isnan(xfs)
        if nan_mask.any():
            col_means = np.nanmean(xfs, axis=0)
            inds = np.where(nan_mask)
            xfs[inds] = np.take(col_means, inds[1])
        x_parts.append(xfs)
        dim_breakdown[fs] = xfs.shape[1]

    X = np.hstack(x_parts)

    # Build Y
    Y = np.zeros((n_genes, K), dtype=np.float32)
    for k, (sdata, sdx) in enumerate(screen_data_list):
        si = [sdx[g] for g in common_genes]
        Y[:, k] = sdata[si].flatten()

    for k in range(K):
        yk = Y[:, k]
        lo, hi = yk.min(), yk.max()
        if hi > lo:
            Y[:, k] = (yk - lo) / (hi - lo)

    thresholds = np.zeros(K)
    n_hits_per_target = []
    for k in range(K):
        n_topk = max(1, int(n_genes * topk_percent))
        sorted_yk = np.sort(Y[:, k])[::-1]
        thresholds[k] = sorted_yk[n_topk - 1]
        n_hits_per_target.append(int((Y[:, k] > thresholds[k]).sum()))

    meta = {
        "gene_names": common_genes,
        "screen_names": screen_names,
        "feature_set": "+".join(feature_sets),
        "n_genes": n_genes,
        "K": K,
        "thresholds": thresholds,
        "n_hits_per_target": n_hits_per_target,
        "d_features": X.shape[1],
        "dim_breakdown": dim_breakdown,
    }

    return X, Y, meta
