#!/usr/bin/env python3
"""
================================================================================
Dataset Complexity Analysis for Real GeneDisco Datasets
================================================================================

This script analyzes the complexity of real GeneDisco datasets using metrics
similar to the toy dataset complexity analysis. It examines:
- Feature dimensionality
- Hit distribution and clustering
- Local smoothness of the target function
- Distance-target correlations

Usage:
    python analyze_complexity.py

Author: Complexity Analysis
Date: 2026-01-24
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ML imports
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr, pearsonr
from scipy.spatial.distance import cdist

import sys
SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR.parent.parent))
sys.path.insert(0, str(SCRIPT_DIR.parent.parent.parent / "DiscoBAX"))

# GeneDisco imports
from genedisco.datasets.features.achilles import Achilles
from genedisco.datasets.features.string_embedding import STRINGEmbedding  
from genedisco.datasets.features.ccle_protein_quantification import CCLEProteinQuantification
from genedisco.datasets.screens.schmidt_2021_t_cells_ifng import Schmidt2021TCellsIFNg
from genedisco.datasets.screens.schmidt_2021_t_cells_il2 import Schmidt2021TCellsIL2
from genedisco.datasets.screens.zhuang_2019_nk_cancer import Zhuang2019NKCancer
from genedisco.datasets.screens.sanchez_2021_neurons_tau import Sanchez2021NeuronsTau
from genedisco.datasets.screens.zhu_2021_sarscov2_host_factors import Zhu2021SARSCoV2HostFactors


# =============================================================================
# Configuration
# =============================================================================

# Datasets
SCREENS = {
    'schmidt_2021_ifng': Schmidt2021TCellsIFNg,
    'schmidt_2021_il2': Schmidt2021TCellsIL2,
    'zhuang_2019_nk': Zhuang2019NKCancer,
    'sanchez_2021_tau': Sanchez2021NeuronsTau,
    'zhu_2021_sarscov2': Zhu2021SARSCoV2HostFactors,
}

FEATURES = {
    'achilles': Achilles,
    'string': STRINGEmbedding,
    'ccle': CCLEProteinQuantification,
}

# Threshold for hits (top 10%)
THRESHOLD_PCT = 10.0

# Data cache directory
CACHE_DIR = "/Users/ar36/Desktop/Experimental Design/DiscoBAX/data"


# =============================================================================
# Complexity Metrics
# =============================================================================

def load_dataset(screen_name: str, feature_name: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load a dataset and return X, y, and gene names."""
    
    feature_class = FEATURES[feature_name]
    screen_class = SCREENS[screen_name]
    
    X_data = feature_class.load_data(CACHE_DIR)
    Y_data = screen_class.load_data(CACHE_DIR)
    
    # Get common genes
    common_genes = sorted(list(
        set(X_data.get_row_names()).intersection(set(Y_data.get_row_names()))
    ))
    
    # Subset to common genes
    X_data = X_data.subset(common_genes)
    Y_data = Y_data.subset(common_genes)
    
    # Extract arrays
    X = np.array(X_data.get_data()[0]).squeeze()
    y = np.array(Y_data.get_data()[0]).flatten()
    
    return X, y, common_genes


def compute_local_smoothness(X: np.ndarray, y: np.ndarray, k: int = 10) -> Tuple[float, float]:
    """
    Compute local smoothness: how much y varies among k-nearest neighbors.
    Lower values = smoother function (easier to learn).
    """
    n_samples = len(X)
    k = min(k + 1, n_samples)
    
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(X)
    _, indices = nn.kneighbors(X)
    
    smoothness_scores = []
    for i in range(n_samples):
        neighbor_idx = indices[i, 1:]  # Exclude self
        y_diffs = np.abs(y[neighbor_idx] - y[i])
        smoothness_scores.append(np.mean(y_diffs))
    
    return np.mean(smoothness_scores), np.std(smoothness_scores)


def compute_hit_clustering(X_hits: np.ndarray, eps_percentile: float = 50) -> Dict[str, Any]:
    """
    Analyze clustering of hit points in feature space.
    More clusters = harder to find all hits.
    """
    if len(X_hits) < 2:
        return {
            'n_clusters_dbscan': 1,
            'n_noise_hits': 0,
            'n_clusters_hier_2': 1,
            'n_clusters_hier_3': 1,
            'n_clusters_hier_5': 1,
        }
    
    # Normalize hits
    scaler = StandardScaler()
    X_hits_norm = scaler.fit_transform(X_hits)
    
    # Compute pairwise distances for eps estimation
    if len(X_hits_norm) > 100:
        sample_idx = np.random.choice(len(X_hits_norm), 100, replace=False)
        sample = X_hits_norm[sample_idx]
    else:
        sample = X_hits_norm
    
    dists = cdist(sample, sample)
    np.fill_diagonal(dists, np.inf)
    min_dists = np.min(dists, axis=1)
    eps = np.percentile(min_dists, eps_percentile)
    eps = max(eps, 0.5)
    
    # DBSCAN clustering
    dbscan = DBSCAN(eps=eps, min_samples=2)
    dbscan_labels = dbscan.fit_predict(X_hits_norm)
    n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    n_clusters_dbscan = max(1, n_clusters_dbscan)
    n_noise = sum(dbscan_labels == -1)
    
    # Hierarchical clustering at different levels
    hier_results = {}
    for n_clusters in [2, 3, 5]:
        if len(X_hits_norm) >= n_clusters:
            hier = AgglomerativeClustering(n_clusters=n_clusters)
            hier_labels = hier.fit_predict(X_hits_norm)
            hier_results[f'n_clusters_hier_{n_clusters}'] = len(set(hier_labels))
        else:
            hier_results[f'n_clusters_hier_{n_clusters}'] = min(n_clusters, len(X_hits_norm))
    
    return {
        'n_clusters_dbscan': n_clusters_dbscan,
        'n_noise_hits': n_noise,
        **hier_results,
    }


def compute_hit_spread(X: np.ndarray, X_hits: np.ndarray) -> Dict[str, float]:
    """
    Compute how spread out hits are in feature space.
    """
    if len(X_hits) < 2:
        return {
            'hit_avg_distance': 0.0,
            'hit_std_distance': 0.0,
            'hit_max_distance': 0.0,
            'hit_spread': 0.0,
            'hit_centroid_distance': 0.0,
        }
    
    # Normalize all data
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X)
    X_hits_norm = scaler.transform(X_hits)
    
    # Pairwise distances among hits
    hit_dists = cdist(X_hits_norm, X_hits_norm)
    upper_tri = hit_dists[np.triu_indices_from(hit_dists, k=1)]
    
    # Distance from hits to overall centroid
    centroid = np.mean(X_norm, axis=0)
    centroid_dists = np.linalg.norm(X_hits_norm - centroid, axis=1)
    
    return {
        'hit_avg_distance': np.mean(upper_tri),
        'hit_std_distance': np.std(upper_tri),
        'hit_max_distance': np.max(upper_tri) if len(upper_tri) > 0 else 0.0,
        'hit_spread': np.var(upper_tri) if len(upper_tri) > 0 else 0.0,
        'hit_centroid_distance': np.mean(centroid_dists),
    }


def compute_dimensionality(X: np.ndarray) -> Dict[str, float]:
    """
    Compute effective dimensionality metrics.
    """
    n_samples, n_features = X.shape
    n_components = min(50, n_samples, n_features)
    
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X)
    
    pca = PCA(n_components=n_components)
    pca.fit(X_norm)
    
    # Cumulative variance
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    
    # Effective dimensionality at 95% variance
    effective_dim_95 = np.searchsorted(cumvar, 0.95) + 1
    effective_dim_90 = np.searchsorted(cumvar, 0.90) + 1
    effective_dim_80 = np.searchsorted(cumvar, 0.80) + 1
    
    return {
        'effective_dim_95': effective_dim_95,
        'effective_dim_90': effective_dim_90,
        'effective_dim_80': effective_dim_80,
        'first_pc_variance': pca.explained_variance_ratio_[0],
        'first_5pc_variance': sum(pca.explained_variance_ratio_[:5]),
        'first_10pc_variance': sum(pca.explained_variance_ratio_[:10]),
    }


def compute_correlation_metrics(X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """
    Compute correlations between features/distances and target.
    """
    n_samples, n_features = X.shape
    
    # Distance from centroid vs y
    centroid = np.mean(X, axis=0)
    dists = np.linalg.norm(X - centroid, axis=1)
    dist_y_spearman, _ = spearmanr(dists, y)
    dist_y_pearson, _ = pearsonr(dists, y)
    
    # Feature-y correlations
    feature_corrs = []
    for j in range(min(n_features, 100)):  # Sample features if too many
        corr, _ = spearmanr(X[:, j], y)
        if not np.isnan(corr):
            feature_corrs.append(abs(corr))
    
    return {
        'distance_y_spearman': abs(dist_y_spearman) if not np.isnan(dist_y_spearman) else 0.0,
        'distance_y_pearson': abs(dist_y_pearson) if not np.isnan(dist_y_pearson) else 0.0,
        'max_feature_y_corr': max(feature_corrs) if feature_corrs else 0.0,
        'mean_feature_y_corr': np.mean(feature_corrs) if feature_corrs else 0.0,
        'median_feature_y_corr': np.median(feature_corrs) if feature_corrs else 0.0,
    }


def analyze_dataset(screen_name: str, feature_name: str) -> Dict[str, Any]:
    """
    Compute all complexity metrics for a dataset.
    """
    print(f"Analyzing {screen_name} + {feature_name}...", flush=True)
    
    try:
        X, y, gene_names = load_dataset(screen_name, feature_name)
    except Exception as e:
        print(f"  Error loading: {e}")
        return None
    
    n_samples, n_features = X.shape
    
    # Basic statistics
    threshold = np.percentile(y, 100 - THRESHOLD_PCT)
    hits_mask = y >= threshold
    n_hits = hits_mask.sum()
    X_hits = X[hits_mask]
    y_hits = y[hits_mask]
    
    # Compute metrics
    local_smooth, local_smooth_std = compute_local_smoothness(X, y, k=10)
    hit_clustering = compute_hit_clustering(X_hits)
    hit_spread = compute_hit_spread(X, X_hits)
    dimensionality = compute_dimensionality(X)
    correlations = compute_correlation_metrics(X, y)
    
    result = {
        'dataset': screen_name,
        'feature_set': feature_name,
        'n_samples': n_samples,
        'n_features': n_features,
        'threshold_pct': THRESHOLD_PCT,
        'threshold_value': threshold,
        'n_hits': n_hits,
        'hit_fraction': n_hits / n_samples,
        
        # Target statistics
        'y_mean': np.mean(y),
        'y_std': np.std(y),
        'y_min': np.min(y),
        'y_max': np.max(y),
        'y_range': np.max(y) - np.min(y),
        'y_hits_mean': np.mean(y_hits),
        'y_hits_std': np.std(y_hits),
        
        # Local smoothness
        'local_smoothness': local_smooth,
        'local_smoothness_std': local_smooth_std,
        
        # Hit clustering
        **hit_clustering,
        
        # Hit spread
        **hit_spread,
        
        # Dimensionality
        **dimensionality,
        
        # Correlations
        **correlations,
    }
    
    return result


# =============================================================================
# Visualization
# =============================================================================

def generate_plots(df: pd.DataFrame, output_dir: Path) -> None:
    """Generate complexity analysis plots."""
    
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("Warning: matplotlib not available, skipping plots")
        return
    
    # 1. Complexity heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Select key metrics for heatmap
    metrics = ['local_smoothness', 'n_clusters_dbscan', 'hit_spread', 
               'effective_dim_95', 'distance_y_spearman', 'max_feature_y_corr']
    
    pivot_data = df.pivot_table(
        index='dataset',
        columns='feature_set',
        values=metrics,
        aggfunc='mean'
    )
    
    # Normalize each metric
    pivot_norm = pivot_data.copy()
    for metric in metrics:
        if metric in pivot_norm.columns.get_level_values(0):
            for fs in pivot_norm[metric].columns:
                col = pivot_norm[(metric, fs)]
                pivot_norm[(metric, fs)] = (col - col.min()) / (col.max() - col.min() + 1e-10)
    
    sns.heatmap(pivot_norm, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax)
    ax.set_title('Dataset Complexity Metrics (Normalized)')
    plt.tight_layout()
    plt.savefig(output_dir / 'complexity_heatmap.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'complexity_heatmap.pdf', bbox_inches='tight')
    plt.close()
    print("Saved: complexity_heatmap.png/pdf")
    
    # 2. Hit clusters by dataset
    fig, ax = plt.subplots(figsize=(10, 6))
    
    df_plot = df.copy()
    df_plot['dataset_feature'] = df_plot['dataset'] + '\n(' + df_plot['feature_set'] + ')'
    
    x = np.arange(len(df_plot))
    bars = ax.bar(x, df_plot['n_clusters_dbscan'], color='steelblue', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(df_plot['dataset_feature'], rotation=45, ha='right')
    ax.set_ylabel('Number of Hit Clusters (DBSCAN)')
    ax.set_title('Hit Clustering by Dataset-Feature Combination')
    
    for bar, n_hits in zip(bars, df_plot['n_hits']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
               f'n={n_hits}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'hit_clusters_by_dataset.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: hit_clusters_by_dataset.png")
    
    # 3. Local smoothness comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(df_plot))
    width = 0.35
    
    ax.bar(x, df_plot['local_smoothness'], color='coral', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(df_plot['dataset_feature'], rotation=45, ha='right')
    ax.set_ylabel('Local Smoothness (lower = easier)')
    ax.set_title('Target Function Local Smoothness')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'local_smoothness_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: local_smoothness_comparison.png")
    
    # 4. Feature dimensionality
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Raw dimensions
    ax = axes[0]
    x = np.arange(len(df_plot))
    ax.bar(x, df_plot['n_features'], color='teal', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(df_plot['dataset_feature'], rotation=45, ha='right')
    ax.set_ylabel('Number of Features')
    ax.set_title('Raw Feature Dimensionality')
    
    # Effective dimensions
    ax = axes[1]
    ax.bar(x, df_plot['effective_dim_95'], color='purple', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(df_plot['dataset_feature'], rotation=45, ha='right')
    ax.set_ylabel('Effective Dimensions (95% variance)')
    ax.set_title('Effective Dimensionality')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'dimensionality_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: dimensionality_comparison.png")
    
    # 5. Correlation analysis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(df_plot))
    width = 0.25
    
    ax.bar(x - width, df_plot['distance_y_spearman'], width, label='Distance-Y Corr', color='blue', alpha=0.7)
    ax.bar(x, df_plot['max_feature_y_corr'], width, label='Max Feature-Y Corr', color='green', alpha=0.7)
    ax.bar(x + width, df_plot['mean_feature_y_corr'], width, label='Mean Feature-Y Corr', color='orange', alpha=0.7)
    
    ax.set_xticks(x)
    ax.set_xticklabels(df_plot['dataset_feature'], rotation=45, ha='right')
    ax.set_ylabel('Correlation (absolute)')
    ax.set_title('Feature-Target Correlations')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'correlation_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: correlation_analysis.png")
    
    # 6. Correlation matrix of complexity metrics
    fig, ax = plt.subplots(figsize=(12, 10))
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
                cmap='RdBu_r', center=0, ax=ax, annot_kws={'size': 7})
    ax.set_title('Correlation Between Complexity Metrics')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_correlation_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: metrics_correlation_matrix.png")


def generate_report(df: pd.DataFrame, output_dir: Path) -> None:
    """Generate a text report of the complexity analysis."""
    
    report_lines = [
        "=" * 80,
        "DATASET COMPLEXITY ANALYSIS - REAL GENEDISCO DATASETS",
        "=" * 80,
        "",
        f"Analysis Date: 2026-01-24",
        f"Threshold: Top {THRESHOLD_PCT}% as hits",
        f"Number of datasets analyzed: {len(df)}",
        "",
        "-" * 80,
        "SUMMARY BY DATASET",
        "-" * 80,
    ]
    
    for dataset in df['dataset'].unique():
        report_lines.append(f"\n### {dataset}")
        df_ds = df[df['dataset'] == dataset]
        
        for _, row in df_ds.iterrows():
            report_lines.append(f"\n  Feature Set: {row['feature_set']}")
            report_lines.append(f"    Samples: {row['n_samples']}, Features: {row['n_features']}")
            report_lines.append(f"    Hits: {row['n_hits']} ({100*row['hit_fraction']:.1f}%)")
            report_lines.append(f"    Hit Clusters (DBSCAN): {row['n_clusters_dbscan']}")
            report_lines.append(f"    Local Smoothness: {row['local_smoothness']:.4f}")
            report_lines.append(f"    Effective Dim (95%): {row['effective_dim_95']}")
            report_lines.append(f"    Distance-Y Corr: {row['distance_y_spearman']:.4f}")
            report_lines.append(f"    Max Feature-Y Corr: {row['max_feature_y_corr']:.4f}")
    
    # Compare feature sets
    report_lines.extend([
        "",
        "-" * 80,
        "FEATURE SET COMPARISON",
        "-" * 80,
    ])
    
    for fs in df['feature_set'].unique():
        df_fs = df[df['feature_set'] == fs]
        report_lines.append(f"\n### {fs}")
        report_lines.append(f"  Avg Features: {df_fs['n_features'].mean():.0f}")
        report_lines.append(f"  Avg Effective Dim: {df_fs['effective_dim_95'].mean():.1f}")
        report_lines.append(f"  Avg Local Smoothness: {df_fs['local_smoothness'].mean():.4f}")
        report_lines.append(f"  Avg Hit Clusters: {df_fs['n_clusters_dbscan'].mean():.1f}")
        report_lines.append(f"  Avg Distance-Y Corr: {df_fs['distance_y_spearman'].mean():.4f}")
    
    # Key findings
    report_lines.extend([
        "",
        "-" * 80,
        "KEY FINDINGS",
        "-" * 80,
        "",
    ])
    
    # Easiest dataset (lowest local smoothness)
    easiest = df.loc[df['local_smoothness'].idxmin()]
    report_lines.append(f"1. Easiest to learn (lowest local smoothness):")
    report_lines.append(f"   {easiest['dataset']} + {easiest['feature_set']} (smoothness={easiest['local_smoothness']:.4f})")
    
    # Most clustered hits
    most_clustered = df.loc[df['n_clusters_dbscan'].idxmax()]
    report_lines.append(f"\n2. Most clustered hits:")
    report_lines.append(f"   {most_clustered['dataset']} + {most_clustered['feature_set']} (clusters={most_clustered['n_clusters_dbscan']})")
    
    # Highest dimensionality
    highest_dim = df.loc[df['effective_dim_95'].idxmax()]
    report_lines.append(f"\n3. Highest effective dimensionality:")
    report_lines.append(f"   {highest_dim['dataset']} + {highest_dim['feature_set']} (dim={highest_dim['effective_dim_95']})")
    
    # Best feature-target correlation
    best_corr = df.loc[df['max_feature_y_corr'].idxmax()]
    report_lines.append(f"\n4. Best feature-target correlation:")
    report_lines.append(f"   {best_corr['dataset']} + {best_corr['feature_set']} (corr={best_corr['max_feature_y_corr']:.4f})")
    
    report_lines.append("")
    report_lines.append("=" * 80)
    
    report = "\n".join(report_lines)
    
    with open(output_dir / "complexity_analysis_report.txt", 'w') as f:
        f.write(report)
    
    print("Saved: complexity_analysis_report.txt")
    print("\n" + report)


# =============================================================================
# Main
# =============================================================================

def main():
    output_dir = SCRIPT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("REAL DATASET COMPLEXITY ANALYSIS")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    print(f"Threshold: Top {THRESHOLD_PCT}%")
    print(f"Datasets: {len(SCREENS)}")
    print(f"Feature sets: {len(FEATURES)}")
    print()
    
    # Analyze all combinations
    results = []
    
    for screen_name in SCREENS:
        for feature_name in FEATURES:
            result = analyze_dataset(screen_name, feature_name)
            if result is not None:
                results.append(result)
    
    if not results:
        print("Error: No datasets analyzed successfully")
        return
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Save CSV
    df.to_csv(output_dir / "dataset_complexity.csv", index=False)
    print(f"\nSaved: dataset_complexity.csv ({len(df)} rows)")
    
    # Generate plots
    print("\nGenerating plots...")
    generate_plots(df, output_dir)
    
    # Generate report
    print("\nGenerating report...")
    generate_report(df, output_dir)
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
