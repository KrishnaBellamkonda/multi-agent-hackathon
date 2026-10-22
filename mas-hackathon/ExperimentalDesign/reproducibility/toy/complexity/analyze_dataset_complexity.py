#!/usr/bin/env python3
"""
=============================================================================
DATASET COMPLEXITY AND PERFORMANCE CORRELATION ANALYSIS
=============================================================================

This script analyzes:
1. Dataset complexity metrics (dimensionality, smoothness, hit distribution)
2. Number of hit clusters at different thresholds
3. Correlation between complexity and method performance
4. Effects of batch size and threshold on performance

Author: Andrea Rubbi, University of Cambridge
"""

import os
import sys
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import DBSCAN, KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Add repo root to path
sys.path.insert(0, str(Path(os.path.dirname(os.path.abspath(__file__))).parent.parent.parent))

from reproducibility.shared.datasets import generate_dataset, list_datasets


# =============================================================================
# DATASET COMPLEXITY METRICS
# =============================================================================

def compute_dataset_complexity(X: np.ndarray, y: np.ndarray, threshold_pct: float = 0.1) -> dict:
    """
    Compute various complexity metrics for a dataset.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        y: Target values (n_samples,)
        threshold_pct: Top percentage to consider as hits (0.1 = top 10%)
    
    Returns:
        Dictionary of complexity metrics
    """
    n_samples, n_features = X.shape
    y = y.flatten()
    
    # Normalize X for distance calculations
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 1. Basic statistics
    metrics = {
        'n_samples': n_samples,
        'n_features': n_features,
        'y_mean': float(np.mean(y)),
        'y_std': float(np.std(y)),
        'y_min': float(np.min(y)),
        'y_max': float(np.max(y)),
        'y_range': float(np.max(y) - np.min(y)),
    }
    
    # 2. Hit threshold and hit indices
    k = max(1, int(threshold_pct * n_samples))
    hit_threshold = np.partition(y, -k)[-k]
    hit_indices = np.where(y >= hit_threshold)[0]
    n_hits = len(hit_indices)
    
    metrics['threshold_value'] = float(hit_threshold)
    metrics['n_hits'] = n_hits
    metrics['hit_fraction'] = n_hits / n_samples
    
    # 3. Hit distribution in feature space
    if n_hits > 1:
        X_hits = X_scaled[hit_indices]
        
        # Average pairwise distance between hits
        if n_hits <= 1000:  # Avoid memory issues
            hit_distances = pdist(X_hits, metric='euclidean')
            metrics['hit_avg_distance'] = float(np.mean(hit_distances))
            metrics['hit_std_distance'] = float(np.std(hit_distances))
            metrics['hit_max_distance'] = float(np.max(hit_distances))
        else:
            # Sample for large datasets
            sample_idx = np.random.choice(n_hits, 1000, replace=False)
            hit_distances = pdist(X_hits[sample_idx], metric='euclidean')
            metrics['hit_avg_distance'] = float(np.mean(hit_distances))
            metrics['hit_std_distance'] = float(np.std(hit_distances))
            metrics['hit_max_distance'] = float(np.max(hit_distances))
        
        # Hit spread (trace of covariance)
        hit_cov = np.cov(X_hits.T)
        if n_features == 1:
            metrics['hit_spread'] = float(hit_cov)
        else:
            metrics['hit_spread'] = float(np.trace(hit_cov))
    else:
        metrics['hit_avg_distance'] = 0.0
        metrics['hit_std_distance'] = 0.0
        metrics['hit_max_distance'] = 0.0
        metrics['hit_spread'] = 0.0
    
    # 4. Clustering of hits (DBSCAN)
    if n_hits > 2:
        # Use DBSCAN with eps based on typical distances
        nn = NearestNeighbors(n_neighbors=min(5, n_hits))
        nn.fit(X_scaled[hit_indices])
        distances, _ = nn.kneighbors(X_scaled[hit_indices])
        eps = np.percentile(distances[:, -1], 90)  # 90th percentile of k-NN distances
        
        dbscan = DBSCAN(eps=max(eps, 0.1), min_samples=2)
        hit_labels = dbscan.fit_predict(X_scaled[hit_indices])
        
        n_clusters = len(set(hit_labels)) - (1 if -1 in hit_labels else 0)
        n_noise = np.sum(hit_labels == -1)
        
        metrics['n_hit_clusters_dbscan'] = n_clusters
        metrics['n_noise_hits'] = n_noise
        metrics['cluster_ratio'] = n_clusters / n_hits if n_hits > 0 else 0
        
        # Also try hierarchical clustering with different thresholds
        if n_hits <= 1000:
            Z = linkage(X_scaled[hit_indices], method='ward')
            for n_clust in [2, 3, 5, 10]:
                if n_hits >= n_clust:
                    labels = fcluster(Z, n_clust, criterion='maxclust')
                    metrics[f'n_hit_clusters_hier_{n_clust}'] = len(set(labels))
    else:
        metrics['n_hit_clusters_dbscan'] = 1 if n_hits > 0 else 0
        metrics['n_noise_hits'] = 0
        metrics['cluster_ratio'] = 1.0 if n_hits > 0 else 0
    
    # 5. Local smoothness (how much y varies with neighbors)
    nn = NearestNeighbors(n_neighbors=min(10, n_samples - 1))
    nn.fit(X_scaled)
    distances, indices = nn.kneighbors(X_scaled)
    
    # Local variation: std of y among neighbors
    local_variations = []
    for i in range(n_samples):
        neighbor_y = y[indices[i]]
        local_variations.append(np.std(neighbor_y))
    
    metrics['local_smoothness'] = float(np.mean(local_variations))
    metrics['local_smoothness_std'] = float(np.std(local_variations))
    
    # 6. Global smoothness (Spearman correlation between distance and y difference)
    if n_samples <= 500:
        dist_matrix = squareform(pdist(X_scaled))
        y_diff_matrix = np.abs(y[:, None] - y[None, :])
        
        # Get upper triangle (excluding diagonal)
        upper_tri = np.triu_indices(n_samples, k=1)
        dist_flat = dist_matrix[upper_tri]
        y_diff_flat = y_diff_matrix[upper_tri]
        
        # Correlation between distance and y difference
        corr, _ = stats.spearmanr(dist_flat, y_diff_flat)
        metrics['distance_y_correlation'] = float(corr) if not np.isnan(corr) else 0.0
    else:
        # Sample for large datasets
        sample_idx = np.random.choice(n_samples, 500, replace=False)
        X_sample = X_scaled[sample_idx]
        y_sample = y[sample_idx]
        
        dist_matrix = squareform(pdist(X_sample))
        y_diff_matrix = np.abs(y_sample[:, None] - y_sample[None, :])
        
        upper_tri = np.triu_indices(500, k=1)
        dist_flat = dist_matrix[upper_tri]
        y_diff_flat = y_diff_matrix[upper_tri]
        
        corr, _ = stats.spearmanr(dist_flat, y_diff_flat)
        metrics['distance_y_correlation'] = float(corr) if not np.isnan(corr) else 0.0
    
    # 7. Hit concentration (how concentrated are hits in feature space)
    # Ratio of hit centroid distance to overall centroid
    overall_centroid = np.mean(X_scaled, axis=0)
    hit_centroid = np.mean(X_scaled[hit_indices], axis=0)
    centroid_distance = np.linalg.norm(hit_centroid - overall_centroid)
    metrics['hit_centroid_distance'] = float(centroid_distance)
    
    # 8. Effective dimensionality (via PCA explained variance)
    if n_features > 1:
        from sklearn.decomposition import PCA
        pca = PCA()
        pca.fit(X_scaled)
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        # Dimensions needed for 95% variance
        eff_dim_95 = np.searchsorted(cumvar, 0.95) + 1
        metrics['effective_dim_95'] = int(eff_dim_95)
        metrics['first_pc_variance'] = float(pca.explained_variance_ratio_[0])
    else:
        metrics['effective_dim_95'] = 1
        metrics['first_pc_variance'] = 1.0
    
    return metrics


def analyze_all_datasets(seeds: list = [0, 1, 2, 3, 4], n_samples: int = 300) -> pd.DataFrame:
    """Analyze complexity for all datasets across seeds and thresholds."""
    
    datasets = list_datasets()
    thresholds = [0.05, 0.10, 0.20]  # 5%, 10%, 20%
    
    all_results = []
    
    for dataset_name in datasets:
        print(f"\nAnalyzing {dataset_name}...")
        
        for seed in seeds:
            ds = generate_dataset(dataset_name, n_samples=n_samples, seed=seed)
            X = np.asarray(ds.X)
            y = np.asarray(ds.y).flatten()
            
            for threshold in thresholds:
                metrics = compute_dataset_complexity(X, y, threshold_pct=threshold)
                metrics['dataset'] = dataset_name
                metrics['seed'] = seed
                metrics['threshold_pct'] = threshold * 100
                all_results.append(metrics)
    
    return pd.DataFrame(all_results)


# =============================================================================
# PERFORMANCE CORRELATION ANALYSIS
# =============================================================================

def load_performance_data(ablation_dir: str) -> pd.DataFrame:
    """Load performance data from ablation results."""
    csv_path = os.path.join(ablation_dir, 'all_cycles_metrics.csv')
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    
    # Otherwise load from pickle files
    raise FileNotFoundError(
        f"No all_cycles_metrics.csv found in {ablation_dir}. "
        "Re-run the ablation study to generate outputs."
    )


def merge_complexity_performance(complexity_df: pd.DataFrame, performance_df: pd.DataFrame) -> pd.DataFrame:
    """Merge complexity metrics with performance data."""
    
    # Get final cycle performance
    max_cycle = performance_df['cycle'].max()
    perf_final = performance_df[performance_df['cycle'] == max_cycle].copy()
    
    # Rename threshold column for merging
    if 'threshold' in perf_final.columns:
        perf_final['threshold_pct'] = perf_final['threshold']
    
    # Aggregate performance by dataset, threshold, seed
    perf_agg = perf_final.groupby(['dataset', 'threshold_pct', 'seed', 'method', 'batch_size']).agg({
        'RecoveredHitsRatio_Cumulative': 'mean',
        'SMAPE_Unseen': 'mean',
        'SpearmanRho': 'mean',
        'HitClusters_RecoveryRatio': 'mean',
    }).reset_index()
    
    # Merge with complexity
    merged = perf_agg.merge(
        complexity_df,
        on=['dataset', 'threshold_pct', 'seed'],
        how='left'
    )
    
    return merged


def compute_correlations(merged_df: pd.DataFrame) -> pd.DataFrame:
    """Compute correlations between complexity and performance."""
    
    complexity_cols = [
        'n_features', 'y_std', 'y_range',
        'hit_avg_distance', 'hit_spread', 'n_hit_clusters_dbscan',
        'cluster_ratio', 'local_smoothness', 'distance_y_correlation',
        'hit_centroid_distance', 'effective_dim_95'
    ]
    
    performance_cols = [
        'RecoveredHitsRatio_Cumulative', 'SMAPE_Unseen', 
        'SpearmanRho', 'HitClusters_RecoveryRatio'
    ]
    
    # Filter to valid columns
    complexity_cols = [c for c in complexity_cols if c in merged_df.columns]
    performance_cols = [c for c in performance_cols if c in merged_df.columns]
    
    correlations = []
    
    for comp_col in complexity_cols:
        for perf_col in performance_cols:
            valid = merged_df[[comp_col, perf_col]].dropna()
            if len(valid) > 10:
                corr, pval = stats.spearmanr(valid[comp_col], valid[perf_col])
                correlations.append({
                    'complexity_metric': comp_col,
                    'performance_metric': perf_col,
                    'spearman_corr': corr,
                    'p_value': pval,
                    'n_samples': len(valid),
                })
    
    return pd.DataFrame(correlations)


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_dataset_complexity_summary(complexity_df: pd.DataFrame, output_dir: str):
    """Plot summary of dataset complexity metrics."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Aggregate across seeds
    agg = complexity_df.groupby(['dataset', 'threshold_pct']).agg({
        'n_features': 'first',
        'n_hits': 'mean',
        'n_hit_clusters_dbscan': 'mean',
        'hit_avg_distance': 'mean',
        'hit_spread': 'mean',
        'local_smoothness': 'mean',
        'distance_y_correlation': 'mean',
        'cluster_ratio': 'mean',
    }).reset_index()
    
    # 1. Number of clusters by dataset and threshold
    fig, ax = plt.subplots(figsize=(10, 6))
    pivot = agg.pivot(index='dataset', columns='threshold_pct', values='n_hit_clusters_dbscan')
    pivot.plot(kind='bar', ax=ax, width=0.8)
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Number of Hit Clusters (DBSCAN)')
    ax.set_title('Hit Clusters by Dataset and Threshold')
    ax.legend(title='Threshold (%)', labels=['5%', '10%', '20%'])
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'hit_clusters_by_dataset.png'), dpi=150)
    plt.close()
    
    # 2. Complexity heatmap
    metrics_for_heatmap = ['n_features', 'n_hit_clusters_dbscan', 'hit_avg_distance', 
                          'local_smoothness', 'distance_y_correlation']
    metrics_for_heatmap = [m for m in metrics_for_heatmap if m in agg.columns]
    
    # Normalize for comparison
    agg_10 = agg[agg['threshold_pct'] == 10].set_index('dataset')[metrics_for_heatmap]
    agg_norm = (agg_10 - agg_10.min()) / (agg_10.max() - agg_10.min() + 1e-10)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(agg_norm.T, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax)
    ax.set_title('Dataset Complexity Metrics (Normalized, 10% threshold)')
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Complexity Metric')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'complexity_heatmap.png'), dpi=150)
    plt.close()
    
    # 3. Hit spread vs clusters
    fig, ax = plt.subplots(figsize=(8, 6))
    for thresh in agg['threshold_pct'].unique():
        subset = agg[agg['threshold_pct'] == thresh]
        ax.scatter(subset['hit_spread'], subset['n_hit_clusters_dbscan'], 
                  label=f'{int(thresh)}%', s=100, alpha=0.7)
        for _, row in subset.iterrows():
            ax.annotate(row['dataset'], (row['hit_spread'], row['n_hit_clusters_dbscan']),
                       fontsize=8, alpha=0.7)
    ax.set_xlabel('Hit Spread (Covariance Trace)')
    ax.set_ylabel('Number of Hit Clusters')
    ax.set_title('Hit Distribution: Spread vs Clusters')
    ax.legend(title='Threshold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'hit_spread_vs_clusters.png'), dpi=150)
    plt.close()
    
    print(f"Saved complexity plots to {output_dir}")


def plot_performance_correlations(merged_df: pd.DataFrame, correlations_df: pd.DataFrame, output_dir: str):
    """Plot performance correlations with complexity."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Correlation heatmap
    pivot = correlations_df.pivot(
        index='complexity_metric', 
        columns='performance_metric', 
        values='spearman_corr'
    )
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdBu_r', center=0, 
                vmin=-1, vmax=1, ax=ax)
    ax.set_title('Correlation: Complexity vs Performance')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'), dpi=150)
    plt.close()
    
    # 2. Performance by number of clusters
    if 'n_hit_clusters_dbscan' in merged_df.columns:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        metrics = ['RecoveredHitsRatio_Cumulative', 'SMAPE_Unseen', 
                   'SpearmanRho', 'HitClusters_RecoveryRatio']
        titles = ['Hit Recovery', 'SMAPE (Unseen)', 'Spearman ρ', 'Cluster Recovery']
        
        for ax, metric, title in zip(axes.flatten(), metrics, titles):
            if metric not in merged_df.columns:
                continue
            
            # Group by cluster count
            merged_df['cluster_bin'] = pd.cut(merged_df['n_hit_clusters_dbscan'], 
                                              bins=[0, 2, 5, 10, 50], 
                                              labels=['1-2', '3-5', '6-10', '11+'])
            
            for method in merged_df['method'].unique():
                subset = merged_df[merged_df['method'] == method]
                agg = subset.groupby('cluster_bin')[metric].agg(['mean', 'std']).reset_index()
                ax.errorbar(range(len(agg)), agg['mean'], yerr=agg['std'], 
                           label=method, marker='o', capsize=3)
            
            ax.set_xlabel('Number of Hit Clusters')
            ax.set_ylabel(title)
            ax.set_title(f'{title} vs Cluster Count')
            ax.set_xticks(range(4))
            ax.set_xticklabels(['1-2', '3-5', '6-10', '11+'])
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'performance_by_clusters.png'), dpi=150)
        plt.close()
    
    # 3. Performance by batch size and threshold
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    metrics = ['RecoveredHitsRatio_Cumulative', 'HitClusters_RecoveryRatio']
    
    for idx, metric in enumerate(metrics):
        if metric not in merged_df.columns:
            continue
        
        # By batch size
        ax = axes[idx, 0]
        for method in merged_df['method'].unique():
            subset = merged_df[merged_df['method'] == method]
            agg = subset.groupby('batch_size')[metric].agg(['mean', 'std']).reset_index()
            ax.errorbar(agg['batch_size'], agg['mean'], yerr=agg['std'],
                       label=method, marker='o', capsize=3)
        ax.set_xlabel('Batch Size')
        ax.set_ylabel(metric.replace('_', ' '))
        ax.set_title(f'{metric.replace("_", " ")} by Batch Size')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # By threshold
        ax = axes[idx, 1]
        for method in merged_df['method'].unique():
            subset = merged_df[merged_df['method'] == method]
            agg = subset.groupby('threshold_pct')[metric].agg(['mean', 'std']).reset_index()
            ax.errorbar(agg['threshold_pct'], agg['mean'], yerr=agg['std'],
                       label=method, marker='o', capsize=3)
        ax.set_xlabel('Threshold (%)')
        ax.set_ylabel(metric.replace('_', ' '))
        ax.set_title(f'{metric.replace("_", " ")} by Threshold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_by_batch_threshold.png'), dpi=150)
    plt.close()
    
    # 4. Method ranking by dataset complexity
    if 'local_smoothness' in merged_df.columns:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Rank datasets by complexity (local smoothness)
        dataset_complexity = merged_df.groupby('dataset')['local_smoothness'].mean().sort_values()
        
        for method in merged_df['method'].unique():
            perf_by_dataset = []
            for dataset in dataset_complexity.index:
                subset = merged_df[(merged_df['method'] == method) & (merged_df['dataset'] == dataset)]
                if len(subset) > 0:
                    perf_by_dataset.append(subset['RecoveredHitsRatio_Cumulative'].mean())
                else:
                    perf_by_dataset.append(np.nan)
            ax.plot(range(len(dataset_complexity)), perf_by_dataset, 
                   marker='o', label=method, linewidth=2)
        
        ax.set_xticks(range(len(dataset_complexity)))
        ax.set_xticklabels([f"{d}\n(smooth={dataset_complexity[d]:.2f})" 
                          for d in dataset_complexity.index], rotation=45, ha='right')
        ax.set_xlabel('Dataset (ordered by complexity)')
        ax.set_ylabel('Hit Recovery Ratio')
        ax.set_title('Method Performance vs Dataset Complexity')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'method_ranking_by_complexity.png'), dpi=150)
        plt.close()
    
    print(f"Saved performance correlation plots to {output_dir}")


def generate_summary_report(complexity_df: pd.DataFrame, correlations_df: pd.DataFrame, 
                           merged_df: pd.DataFrame, output_dir: str):
    """Generate a text summary report."""
    
    report_path = os.path.join(output_dir, 'complexity_analysis_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("DATASET COMPLEXITY AND PERFORMANCE CORRELATION ANALYSIS\n")
        f.write("=" * 80 + "\n\n")
        
        # Dataset complexity summary
        f.write("1. DATASET COMPLEXITY SUMMARY\n")
        f.write("-" * 40 + "\n\n")
        
        agg = complexity_df.groupby('dataset').agg({
            'n_features': 'first',
            'n_hits': 'mean',
            'n_hit_clusters_dbscan': 'mean',
            'hit_avg_distance': 'mean',
            'local_smoothness': 'mean',
            'distance_y_correlation': 'mean',
        }).round(3)
        
        f.write(agg.to_string())
        f.write("\n\n")
        
        # Complexity ranking
        f.write("Dataset Complexity Ranking (by local smoothness, higher = harder):\n")
        ranking = agg['local_smoothness'].sort_values(ascending=False)
        for i, (dataset, val) in enumerate(ranking.items(), 1):
            f.write(f"  {i}. {dataset}: {val:.4f}\n")
        f.write("\n")
        
        # Cluster count summary
        f.write("2. HIT CLUSTER COUNT BY THRESHOLD\n")
        f.write("-" * 40 + "\n\n")
        
        cluster_pivot = complexity_df.groupby(['dataset', 'threshold_pct'])['n_hit_clusters_dbscan'].mean().unstack()
        f.write(cluster_pivot.round(1).to_string())
        f.write("\n\n")
        
        # Top correlations
        f.write("3. TOP CORRELATIONS (Complexity vs Performance)\n")
        f.write("-" * 40 + "\n\n")
        
        top_corr = correlations_df.sort_values('spearman_corr', key=abs, ascending=False).head(15)
        for _, row in top_corr.iterrows():
            sig = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else ""
            f.write(f"  {row['complexity_metric']:25s} vs {row['performance_metric']:35s}: "
                   f"r={row['spearman_corr']:+.3f} {sig}\n")
        f.write("\n  (* p<0.05, ** p<0.01, *** p<0.001)\n\n")
        
        # Method performance by complexity
        f.write("4. METHOD PERFORMANCE SUMMARY\n")
        f.write("-" * 40 + "\n\n")
        
        if merged_df is not None and len(merged_df) > 0:
            method_perf = merged_df.groupby('method').agg({
                'RecoveredHitsRatio_Cumulative': ['mean', 'std'],
                'HitClusters_RecoveryRatio': ['mean', 'std'],
            }).round(3)
            f.write(method_perf.to_string())
            f.write("\n\n")
            
            # Best method by dataset
            f.write("Best Method by Dataset (Hit Recovery):\n")
            for dataset in merged_df['dataset'].unique():
                subset = merged_df[merged_df['dataset'] == dataset]
                best = subset.groupby('method')['RecoveredHitsRatio_Cumulative'].mean().idxmax()
                best_val = subset.groupby('method')['RecoveredHitsRatio_Cumulative'].mean().max()
                f.write(f"  {dataset}: {best} ({best_val:.3f})\n")
            f.write("\n")
        
        # Batch size and threshold effects
        f.write("5. BATCH SIZE AND THRESHOLD EFFECTS\n")
        f.write("-" * 40 + "\n\n")
        
        if merged_df is not None and len(merged_df) > 0:
            f.write("Hit Recovery by Batch Size:\n")
            batch_perf = merged_df.groupby('batch_size')['RecoveredHitsRatio_Cumulative'].mean()
            for batch, perf in batch_perf.items():
                f.write(f"  Batch {batch}: {perf:.3f}\n")
            f.write("\n")
            
            f.write("Hit Recovery by Threshold:\n")
            thresh_perf = merged_df.groupby('threshold_pct')['RecoveredHitsRatio_Cumulative'].mean()
            for thresh, perf in thresh_perf.items():
                f.write(f"  {int(thresh)}%: {perf:.3f}\n")
            f.write("\n")
        
        f.write("=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")
    
    print(f"Saved summary report to {report_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze dataset complexity and performance correlations')
    parser.add_argument('--ablation_dir', type=str, default='./ablation_toy_20260121_171434',
                       help='Path to ablation results directory')
    parser.add_argument('--output_dir', type=str, default='./complexity_analysis',
                       help='Output directory for analysis results')
    parser.add_argument('--seeds', type=str, default='0,1,2,3,4',
                       help='Comma-separated seeds to analyze')
    parser.add_argument('--n_samples', type=int, default=300,
                       help='Number of samples per dataset')
    args = parser.parse_args()
    
    seeds = [int(s) for s in args.seeds.split(',')]
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 70)
    print("DATASET COMPLEXITY ANALYSIS")
    print("=" * 70)
    
    # 1. Analyze dataset complexity
    print("\n1. Analyzing dataset complexity...")
    complexity_df = analyze_all_datasets(seeds=seeds, n_samples=args.n_samples)
    complexity_df.to_csv(os.path.join(args.output_dir, 'dataset_complexity.csv'), index=False)
    print(f"   Saved complexity metrics to {args.output_dir}/dataset_complexity.csv")
    
    # 2. Load performance data
    print("\n2. Loading performance data...")
    if os.path.exists(args.ablation_dir):
        performance_df = load_performance_data(args.ablation_dir)
        print(f"   Loaded {len(performance_df)} performance records")
    else:
        print(f"   Warning: Ablation directory not found: {args.ablation_dir}")
        performance_df = None
    
    # 3. Merge and analyze correlations
    merged_df = None
    correlations_df = None
    
    if performance_df is not None and len(performance_df) > 0:
        print("\n3. Computing performance correlations...")
        merged_df = merge_complexity_performance(complexity_df, performance_df)
        merged_df.to_csv(os.path.join(args.output_dir, 'merged_complexity_performance.csv'), index=False)
        
        correlations_df = compute_correlations(merged_df)
        correlations_df.to_csv(os.path.join(args.output_dir, 'correlations.csv'), index=False)
        print(f"   Computed {len(correlations_df)} correlations")
    
    # 4. Generate plots
    print("\n4. Generating plots...")
    plot_dataset_complexity_summary(complexity_df, args.output_dir)
    
    if merged_df is not None and correlations_df is not None:
        plot_performance_correlations(merged_df, correlations_df, args.output_dir)
    
    # 5. Generate summary report
    print("\n5. Generating summary report...")
    generate_summary_report(complexity_df, correlations_df, merged_df, args.output_dir)
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"Results saved to: {args.output_dir}")
    print("  - dataset_complexity.csv")
    print("  - correlations.csv")
    print("  - merged_complexity_performance.csv")
    print("  - complexity_analysis_report.txt")
    print("  - *.png (plots)")
    print("=" * 70)


if __name__ == "__main__":
    main()
