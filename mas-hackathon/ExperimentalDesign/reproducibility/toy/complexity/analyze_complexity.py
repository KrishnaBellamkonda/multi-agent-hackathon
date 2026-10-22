"""
Dataset Complexity Analysis for Toy Datasets.

Analyzes complexity metrics such as dimensionality, smoothness, 
hit distribution, and clustering structure.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from reproducibility.shared.datasets import generate_dataset, list_datasets


def compute_dataset_complexity(X: np.ndarray, y: np.ndarray, 
                                threshold_pct: float = 0.1) -> Dict:
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
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Basic statistics
    metrics = {
        'n_samples': n_samples,
        'n_features': n_features,
        'y_mean': float(np.mean(y)),
        'y_std': float(np.std(y)),
        'y_range': float(np.max(y) - np.min(y)),
    }
    
    # Hit threshold and indices
    k = max(1, int(threshold_pct * n_samples))
    hit_threshold = np.partition(y, -k)[-k]
    hit_indices = np.where(y >= hit_threshold)[0]
    n_hits = len(hit_indices)
    
    metrics['threshold_value'] = float(hit_threshold)
    metrics['n_hits'] = n_hits
    metrics['hit_fraction'] = n_hits / n_samples
    
    # Hit distribution in feature space
    if n_hits > 1:
        X_hits = X_scaled[hit_indices]
        hit_distances = pdist(X_hits, metric='euclidean')
        metrics['hit_avg_distance'] = float(np.mean(hit_distances))
        metrics['hit_std_distance'] = float(np.std(hit_distances))
        
        hit_cov = np.cov(X_hits.T)
        metrics['hit_spread'] = float(np.trace(hit_cov) if n_features > 1 else hit_cov)
    else:
        metrics['hit_avg_distance'] = 0.0
        metrics['hit_std_distance'] = 0.0
        metrics['hit_spread'] = 0.0
    
    # Clustering of hits
    if n_hits > 2:
        nn = NearestNeighbors(n_neighbors=min(5, n_hits))
        nn.fit(X_scaled[hit_indices])
        distances, _ = nn.kneighbors(X_scaled[hit_indices])
        eps = np.percentile(distances[:, -1], 90)
        
        dbscan = DBSCAN(eps=max(eps, 0.1), min_samples=2)
        hit_labels = dbscan.fit_predict(X_scaled[hit_indices])
        
        n_clusters = len(set(hit_labels)) - (1 if -1 in hit_labels else 0)
        metrics['n_hit_clusters'] = max(1, n_clusters)
        metrics['cluster_ratio'] = n_clusters / n_hits if n_hits > 0 else 0
    else:
        metrics['n_hit_clusters'] = 1 if n_hits > 0 else 0
        metrics['cluster_ratio'] = 1.0 if n_hits > 0 else 0
    
    # Local smoothness
    nn = NearestNeighbors(n_neighbors=min(10, n_samples - 1))
    nn.fit(X_scaled)
    distances, indices = nn.kneighbors(X_scaled)
    
    local_variations = [np.std(y[indices[i]]) for i in range(n_samples)]
    metrics['local_smoothness'] = float(np.mean(local_variations))
    
    # Global smoothness
    if n_samples <= 500:
        dist_matrix = squareform(pdist(X_scaled))
        y_diff_matrix = np.abs(y[:, None] - y[None, :])
        upper_tri = np.triu_indices(n_samples, k=1)
        corr, _ = stats.spearmanr(dist_matrix[upper_tri], y_diff_matrix[upper_tri])
        metrics['distance_y_correlation'] = float(corr) if not np.isnan(corr) else 0.0
    else:
        sample_idx = np.random.choice(n_samples, 500, replace=False)
        dist_matrix = squareform(pdist(X_scaled[sample_idx]))
        y_sample = y[sample_idx]
        y_diff_matrix = np.abs(y_sample[:, None] - y_sample[None, :])
        upper_tri = np.triu_indices(500, k=1)
        corr, _ = stats.spearmanr(dist_matrix[upper_tri], y_diff_matrix[upper_tri])
        metrics['distance_y_correlation'] = float(corr) if not np.isnan(corr) else 0.0
    
    # Effective dimensionality
    if n_features > 1:
        pca = PCA()
        pca.fit(X_scaled)
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        metrics['effective_dim_95'] = int(np.searchsorted(cumvar, 0.95) + 1)
    else:
        metrics['effective_dim_95'] = 1
    
    return metrics


def analyze_all_datasets(seeds: List[int] = [0, 1, 2, 3, 4], 
                          n_samples: int = 300,
                          thresholds: List[float] = [0.05, 0.10, 0.20]) -> pd.DataFrame:
    """Analyze complexity for all datasets across seeds and thresholds."""
    
    datasets = list_datasets()
    all_results = []
    
    for dataset_name in datasets:
        print(f"Analyzing {dataset_name}...")
        
        for seed in seeds:
            ds = generate_dataset(dataset_name, n_samples=n_samples, seed=seed)
            X, y = ds.X, ds.y
            
            for threshold in thresholds:
                metrics = compute_dataset_complexity(X, y, threshold_pct=threshold)
                metrics['dataset'] = dataset_name
                metrics['seed'] = seed
                metrics['threshold_pct'] = threshold * 100
                all_results.append(metrics)
    
    return pd.DataFrame(all_results)


def plot_complexity_heatmap(df: pd.DataFrame, output_path: Optional[Path] = None):
    """Plot complexity metrics as heatmap."""
    
    metrics_cols = ['n_features', 'y_std', 'hit_spread', 'n_hit_clusters', 
                    'local_smoothness', 'distance_y_correlation', 'effective_dim_95']
    
    # Filter to available columns
    metrics_cols = [c for c in metrics_cols if c in df.columns]
    
    # Aggregate by dataset
    agg = df.groupby('dataset')[metrics_cols].mean()
    
    # Normalize for visualization
    agg_norm = (agg - agg.min()) / (agg.max() - agg.min() + 1e-8)
    
    plt.figure(figsize=(12, 6))
    sns.heatmap(agg_norm, annot=agg.round(2), fmt='', cmap='YlOrRd', 
                linewidths=0.5, cbar_kws={'label': 'Normalized Value'})
    plt.title('Dataset Complexity Metrics')
    plt.ylabel('Dataset')
    plt.xlabel('Metric')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_hit_clusters(df: pd.DataFrame, output_path: Optional[Path] = None):
    """Plot number of hit clusters by dataset and threshold."""
    
    plt.figure(figsize=(10, 6))
    
    for dataset in df['dataset'].unique():
        subset = df[df['dataset'] == dataset]
        agg = subset.groupby('threshold_pct')['n_hit_clusters'].mean()
        plt.plot(agg.index, agg.values, 'o-', label=dataset, markersize=8)
    
    plt.xlabel('Threshold (%)')
    plt.ylabel('Number of Hit Clusters')
    plt.title('Hit Clusters by Dataset and Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze dataset complexity')
    parser.add_argument('--output-dir', type=str, default='output/complexity',
                       help='Output directory')
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2, 3, 4],
                       help='Random seeds')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Analyzing dataset complexity...")
    df = analyze_all_datasets(seeds=args.seeds)
    
    df.to_csv(output_dir / 'dataset_complexity.csv', index=False)
    print(f"Saved to {output_dir / 'dataset_complexity.csv'}")
    
    plot_complexity_heatmap(df, output_dir / 'complexity_heatmap.png')
    plot_hit_clusters(df, output_dir / 'hit_clusters_by_dataset.png')
    
    print("Done!")
