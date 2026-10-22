#!/usr/bin/env python3
"""
=============================================================================
GENEDISCO DATASET COMPLEXITY ANALYSIS
=============================================================================

Analyze complexity metrics for GeneDisco (real biological) datasets to
understand dataset difficulty and its relationship to method performance.

Complexity metrics analyzed:
- Dataset dimensionality and effective dimensionality
- Hit distribution and clustering patterns
- Local smoothness and function complexity
- Feature correlations and redundancy
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any

from scipy import stats
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# =============================================================================
# DATACLASSES
# =============================================================================

@dataclass
class GeneDiscoDatasetAnalysis:
    """Analysis results for a GeneDisco dataset."""
    dataset_name: str
    feature_set: str
    n_genes: int
    n_features: int
    threshold_pct: float
    n_hits: int
    hit_fraction: float
    
    # Hit distribution metrics
    hit_avg_distance: float
    hit_std_distance: float
    hit_spread: float
    n_hit_clusters: int
    cluster_ratio: float
    hit_centroid_distance: float
    
    # Smoothness metrics
    local_smoothness: float
    distance_y_correlation: float
    
    # Dimensionality metrics
    effective_dim_50: int
    effective_dim_95: int
    first_pc_variance: float
    
    # Score statistics
    score_mean: float
    score_std: float
    score_range: float
    
    def to_dict(self) -> dict:
        return asdict(self)


# =============================================================================
# COMPLEXITY METRICS
# =============================================================================

def compute_hit_clusters(
    X_scaled: np.ndarray,
    hit_indices: np.ndarray,
    method: str = 'dbscan'
) -> Tuple[int, float]:
    """
    Compute the number of hit clusters.
    
    Args:
        X_scaled: Scaled feature matrix
        hit_indices: Indices of hit samples
        method: Clustering method ('dbscan' or 'hierarchical')
    
    Returns:
        Tuple of (n_clusters, cluster_ratio)
    """
    n_hits = len(hit_indices)
    if n_hits < 2:
        return (1 if n_hits > 0 else 0), 1.0
    
    X_hits = X_scaled[hit_indices]
    
    if method == 'dbscan':
        # Adaptive eps based on k-NN distances
        nn = NearestNeighbors(n_neighbors=min(5, n_hits))
        nn.fit(X_hits)
        distances, _ = nn.kneighbors(X_hits)
        eps = np.percentile(distances[:, -1], 90)
        
        dbscan = DBSCAN(eps=max(eps, 0.1), min_samples=2)
        labels = dbscan.fit_predict(X_hits)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_clusters = max(1, n_clusters)  # At least 1 cluster
        
    else:  # hierarchical
        if n_hits > 1000:
            # Sample for large datasets
            sample_idx = np.random.choice(n_hits, 1000, replace=False)
            X_hits = X_hits[sample_idx]
        
        Z = linkage(X_hits, method='ward')
        labels = fcluster(Z, 5, criterion='maxclust')
        n_clusters = len(set(labels))
    
    cluster_ratio = n_clusters / n_hits
    return n_clusters, cluster_ratio


def compute_local_smoothness(
    X_scaled: np.ndarray,
    y: np.ndarray,
    n_neighbors: int = 10
) -> float:
    """
    Compute local smoothness as average local y-variation.
    
    Lower values indicate smoother functions.
    """
    n_samples = len(y)
    k = min(n_neighbors, n_samples - 1)
    
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(X_scaled)
    _, indices = nn.kneighbors(X_scaled)
    
    local_variations = []
    for i in range(n_samples):
        neighbor_y = y[indices[i]]
        local_variations.append(np.std(neighbor_y))
    
    return float(np.mean(local_variations))


def compute_distance_y_correlation(
    X_scaled: np.ndarray,
    y: np.ndarray,
    max_samples: int = 500
) -> float:
    """
    Compute correlation between pairwise distances and y-differences.
    
    Higher values indicate that nearby points have similar y values (smoother).
    """
    n_samples = len(y)
    
    if n_samples > max_samples:
        sample_idx = np.random.choice(n_samples, max_samples, replace=False)
        X_sample = X_scaled[sample_idx]
        y_sample = y[sample_idx]
    else:
        X_sample = X_scaled
        y_sample = y
    
    n = len(y_sample)
    dist_matrix = squareform(pdist(X_sample))
    y_diff_matrix = np.abs(y_sample[:, None] - y_sample[None, :])
    
    # Upper triangle only (excluding diagonal)
    upper_tri = np.triu_indices(n, k=1)
    dist_flat = dist_matrix[upper_tri]
    y_diff_flat = y_diff_matrix[upper_tri]
    
    corr, _ = stats.spearmanr(dist_flat, y_diff_flat)
    return float(corr) if not np.isnan(corr) else 0.0


def compute_effective_dimensionality(X_scaled: np.ndarray) -> Tuple[int, int, float]:
    """
    Compute effective dimensionality using PCA.
    
    Returns:
        Tuple of (dim_50%, dim_95%, first_pc_variance)
    """
    n_features = X_scaled.shape[1]
    
    if n_features < 2:
        return 1, 1, 1.0
    
    pca = PCA()
    pca.fit(X_scaled)
    
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    
    dim_50 = np.searchsorted(cumvar, 0.50) + 1
    dim_95 = np.searchsorted(cumvar, 0.95) + 1
    first_pc = float(pca.explained_variance_ratio_[0])
    
    return int(dim_50), int(dim_95), first_pc


def analyze_genedisco_dataset(
    X: np.ndarray,
    y: np.ndarray,
    dataset_name: str,
    feature_set: str,
    threshold_pct: float = 0.10
) -> GeneDiscoDatasetAnalysis:
    """
    Analyze complexity metrics for a GeneDisco dataset.
    
    Args:
        X: Feature matrix (n_genes, n_features)
        y: Target values (n_genes,)
        dataset_name: Name of the dataset
        feature_set: Feature set name
        threshold_pct: Top percentage for hits
    
    Returns:
        GeneDiscoDatasetAnalysis with all metrics
    """
    n_genes, n_features = X.shape
    y = y.flatten()
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Hit identification
    k = max(1, int(threshold_pct * n_genes))
    threshold = np.partition(y, -k)[-k]
    hit_indices = np.where(y >= threshold)[0]
    n_hits = len(hit_indices)
    
    # Hit distribution metrics
    if n_hits > 1:
        X_hits = X_scaled[hit_indices]
        
        # Pairwise distances
        if n_hits <= 1000:
            hit_distances = pdist(X_hits, metric='euclidean')
        else:
            sample_idx = np.random.choice(n_hits, 1000, replace=False)
            hit_distances = pdist(X_hits[sample_idx], metric='euclidean')
        
        hit_avg_distance = float(np.mean(hit_distances))
        hit_std_distance = float(np.std(hit_distances))
        
        # Hit spread (covariance trace)
        hit_cov = np.cov(X_hits.T)
        if n_features == 1:
            hit_spread = float(hit_cov)
        else:
            hit_spread = float(np.trace(hit_cov))
        
        # Hit centroid distance from overall centroid
        overall_centroid = np.mean(X_scaled, axis=0)
        hit_centroid = np.mean(X_hits, axis=0)
        hit_centroid_distance = float(np.linalg.norm(hit_centroid - overall_centroid))
    else:
        hit_avg_distance = 0.0
        hit_std_distance = 0.0
        hit_spread = 0.0
        hit_centroid_distance = 0.0
    
    # Clustering
    n_clusters, cluster_ratio = compute_hit_clusters(X_scaled, hit_indices)
    
    # Smoothness
    local_smoothness = compute_local_smoothness(X_scaled, y)
    distance_y_corr = compute_distance_y_correlation(X_scaled, y)
    
    # Dimensionality
    dim_50, dim_95, first_pc = compute_effective_dimensionality(X_scaled)
    
    return GeneDiscoDatasetAnalysis(
        dataset_name=dataset_name,
        feature_set=feature_set,
        n_genes=n_genes,
        n_features=n_features,
        threshold_pct=threshold_pct,
        n_hits=n_hits,
        hit_fraction=n_hits / n_genes,
        
        hit_avg_distance=hit_avg_distance,
        hit_std_distance=hit_std_distance,
        hit_spread=hit_spread,
        n_hit_clusters=n_clusters,
        cluster_ratio=cluster_ratio,
        hit_centroid_distance=hit_centroid_distance,
        
        local_smoothness=local_smoothness,
        distance_y_correlation=distance_y_corr,
        
        effective_dim_50=dim_50,
        effective_dim_95=dim_95,
        first_pc_variance=first_pc,
        
        score_mean=float(np.mean(y)),
        score_std=float(np.std(y)),
        score_range=float(np.max(y) - np.min(y)),
    )


# =============================================================================
# MAIN ANALYSIS FUNCTIONS
# =============================================================================

def analyze_genedisco_complexity(
    cache_dir: Path,
    output_dir: Path,
    datasets: List[str] = None,
    feature_sets: List[str] = None,
    threshold_pcts: List[float] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Analyze complexity of GeneDisco datasets.
    
    Args:
        cache_dir: Directory with GeneDisco data
        output_dir: Output directory for results
        datasets: List of dataset names (default: all)
        feature_sets: List of feature sets (default: ['achilles', 'string'])
        threshold_pcts: List of threshold percentages (default: [0.05, 0.10, 0.20])
        verbose: Print progress
    
    Returns:
        DataFrame with complexity metrics for all datasets
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Defaults
    if datasets is None:
        datasets = [
            'schmidt_2021_ifng',
            'schmidt_2021_il2',
            'zhuang_2019_nk',
            'sanchez_2021_tau',
            'zhu_2021_sarscov2',
        ]
    
    if feature_sets is None:
        feature_sets = ['achilles', 'string']
    
    if threshold_pcts is None:
        threshold_pcts = [0.05, 0.10, 0.20]
    
    results = []
    
    for dataset_name in datasets:
        for feature_set in feature_sets:
            if verbose:
                print(f"Analyzing {dataset_name} / {feature_set}...")
            
            try:
                # Load data
                X, y = load_genedisco_data(cache_dir, dataset_name, feature_set)
                
                for threshold in threshold_pcts:
                    analysis = analyze_genedisco_dataset(
                        X, y, dataset_name, feature_set, threshold
                    )
                    results.append(analysis.to_dict())
                    
            except Exception as e:
                if verbose:
                    print(f"  Error: {e}")
                continue
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Save results
    df.to_csv(output_dir / 'genedisco_complexity.csv', index=False)
    
    # Generate summary report
    if len(df) > 0:
        generate_complexity_report(df, output_dir)
    
    if verbose:
        print(f"\nResults saved to {output_dir}")
    
    return df


def load_genedisco_data(
    cache_dir: Path,
    dataset_name: str,
    feature_set: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load GeneDisco dataset from cache.
    
    This attempts to load the data using the activehit.data module.
    """
    try:
        from activehit.data.genedisco import load_dataset
        return load_dataset(cache_dir, dataset_name, feature_set)
    except ImportError:
        # Fallback: try loading from pickle cache
        cache_file = Path(cache_dir) / f"{dataset_name}_{feature_set}.pkl"
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            return data['X'], data['y']
        else:
            raise FileNotFoundError(
                f"Could not load {dataset_name}/{feature_set}. "
                f"Install activehit package or ensure data is cached."
            )


def generate_complexity_report(df: pd.DataFrame, output_dir: Path):
    """Generate a text summary report of complexity analysis."""
    
    report_lines = [
        "=" * 80,
        "GENEDISCO DATASET COMPLEXITY ANALYSIS REPORT",
        "=" * 80,
        "",
        f"Analyzed {df['dataset_name'].nunique()} datasets",
        f"Feature sets: {df['feature_set'].unique().tolist()}",
        f"Thresholds: {df['threshold_pct'].unique().tolist()}",
        "",
        "-" * 80,
        "DATASET SUMMARIES",
        "-" * 80,
    ]
    
    # Summary by dataset
    for dataset in df['dataset_name'].unique():
        subset = df[df['dataset_name'] == dataset]
        
        report_lines.append(f"\n{dataset}:")
        report_lines.append(f"  Genes: {subset['n_genes'].iloc[0]}")
        report_lines.append(f"  Features: {subset['n_features'].iloc[0]}")
        report_lines.append(f"  Effective dim (95%): {subset['effective_dim_95'].iloc[0]}")
        report_lines.append(f"  Hit clusters (avg): {subset['n_hit_clusters'].mean():.1f}")
        report_lines.append(f"  Local smoothness: {subset['local_smoothness'].mean():.4f}")
    
    report_lines.extend([
        "",
        "-" * 80,
        "COMPLEXITY RANKINGS (at 10% threshold)",
        "-" * 80,
    ])
    
    # Rank by different metrics
    df_10 = df[df['threshold_pct'] == 0.10]
    if len(df_10) > 0:
        # Most clustered (harder)
        report_lines.append("\nMost clustered hits (hardest):")
        sorted_df = df_10.sort_values('n_hit_clusters', ascending=False)
        for _, row in sorted_df.head(3).iterrows():
            report_lines.append(f"  {row['dataset_name']}/{row['feature_set']}: "
                              f"{row['n_hit_clusters']} clusters")
        
        # Least smooth (harder)
        report_lines.append("\nLeast smooth functions (hardest):")
        sorted_df = df_10.sort_values('local_smoothness', ascending=False)
        for _, row in sorted_df.head(3).iterrows():
            report_lines.append(f"  {row['dataset_name']}/{row['feature_set']}: "
                              f"{row['local_smoothness']:.4f}")
    
    report_lines.append("\n" + "=" * 80)
    
    # Write report
    with open(output_dir / 'complexity_report.txt', 'w') as f:
        f.write("\n".join(report_lines))


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Analyze GeneDisco dataset complexity'
    )
    parser.add_argument('--cache-dir', type=str, required=True,
                       help='GeneDisco data cache directory')
    parser.add_argument('--output-dir', type=str, default='output/complexity',
                       help='Output directory')
    parser.add_argument('--datasets', nargs='+', default=None,
                       help='Dataset names to analyze')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress output')
    
    args = parser.parse_args()
    
    analyze_genedisco_complexity(
        cache_dir=Path(args.cache_dir),
        output_dir=Path(args.output_dir),
        datasets=args.datasets,
        verbose=not args.quiet
    )


if __name__ == '__main__':
    main()
