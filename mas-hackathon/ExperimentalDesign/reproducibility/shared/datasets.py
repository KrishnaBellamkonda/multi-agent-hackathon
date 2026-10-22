"""
Dataset generation for ActiveHit reproducibility studies.

This module provides functions to generate synthetic datasets for benchmarking
acquisition methods in Bayesian optimization.

IMPORTANT: These dataset generators match the implementations in
activehit/data/toy_datasets.py to ensure reproducibility.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any, List
from dataclasses import dataclass


@dataclass
class Dataset:
    """Container for dataset with features and targets."""
    X: np.ndarray
    y: np.ndarray
    name: str
    n_samples: int
    n_features: int
    
    def get_hits(self, threshold_pct: float) -> np.ndarray:
        """Get indices of points above the threshold percentile."""
        threshold = np.percentile(self.y, 100 - threshold_pct)
        return np.where(self.y >= threshold)[0]
    
    def get_threshold_value(self, threshold_pct: float) -> float:
        """Get the threshold value for a given percentile."""
        return np.percentile(self.y, 100 - threshold_pct)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def generate_sine(n_samples: int = 300, noise_std: float = 0.05, 
                  seed: Optional[int] = None) -> Dataset:
    """
    Generate 1D sinusoidal dataset.
    y = sin(2π * x) + noise
    """
    rng = np.random.default_rng(seed)
    X = np.linspace(0, 1, n_samples).reshape(-1, 1)
    y = np.sin(2 * np.pi * X[:, 0]) + noise_std * rng.standard_normal(n_samples)
    return Dataset(X=X, y=y, name="sine", n_samples=n_samples, n_features=1)


def generate_sine2d(n_samples: int = 300, noise_std: float = 0.03,
                    seed: Optional[int] = None) -> Dataset:
    """
    Generate 2D sinusoidal dataset.
    y = sin(2π * x1) * sin(2π * x2) + noise, normalized to [0, 1]
    """
    rng = np.random.default_rng(seed)
    side = int(np.ceil(np.sqrt(n_samples)))
    x1 = np.linspace(0, 1, side)
    x2 = np.linspace(0, 1, side)
    X1, X2 = np.meshgrid(x1, x2)
    X = np.stack([X1.ravel(), X2.ravel()], axis=1)[:n_samples]
    
    y = np.sin(2 * np.pi * X[:, 0]) * np.sin(2 * np.pi * X[:, 1])
    y = (y - y.min()) / (y.max() - y.min() + 1e-8)
    y = y + noise_std * rng.standard_normal(len(y))
    return Dataset(X=X, y=y, name="sine2d", n_samples=n_samples, n_features=2)


def generate_branin(n_samples: int = 300, noise_std: float = 0.02,
                    seed: Optional[int] = None) -> Dataset:
    """Generate Branin-Hoo test function dataset."""
    rng = np.random.default_rng(seed)
    side = int(np.ceil(np.sqrt(n_samples)))
    x1 = np.linspace(0, 1, side)
    x2 = np.linspace(0, 1, side)
    X1, X2 = np.meshgrid(x1, x2)
    X = np.stack([X1.ravel(), X2.ravel()], axis=1)[:n_samples]
    
    x1_scaled = X[:, 0] * 15 - 5
    x2_scaled = X[:, 1] * 15
    
    a, b, c = 1.0, 5.1 / (4 * np.pi**2), 5 / np.pi
    r, s, t = 6.0, 10.0, 1 / (8 * np.pi)
    
    term1 = a * (x2_scaled - b * x1_scaled**2 + c * x1_scaled - r)**2
    term2 = s * (1 - t) * np.cos(x1_scaled)
    
    y_raw = -(term1 + term2 + s)
    y = (y_raw - y_raw.min()) / (y_raw.max() - y_raw.min() + 1e-8)
    y = y + noise_std * rng.standard_normal(len(y))
    return Dataset(X=X, y=y, name="branin", n_samples=n_samples, n_features=2)


def generate_perturbation_pathways(n_samples: int = 300, noise_std: float = 0.08,
                                   seed: Optional[int] = None) -> Dataset:
    """Generate pathway-based perturbation dataset (4D)."""
    rng = np.random.default_rng(seed)
    n = n_samples
    n_genes = 40
    n_pathways = 4
    d_gene = 2
    d_context = 2
    pathway_effect_scale = 2.5
    gene_effect_scale = 0.4
    
    pathway_centers = np.array([
        [0.2, 0.2], [0.8, 0.2], [0.2, 0.8], [0.8, 0.8],
    ])[:n_pathways]
    
    genes_per_pathway = n_genes // n_pathways
    gene_pathway_ids = np.repeat(np.arange(n_pathways), genes_per_pathway)
    if len(gene_pathway_ids) < n_genes:
        gene_pathway_ids = np.concatenate([
            gene_pathway_ids,
            rng.integers(0, n_pathways, size=n_genes - len(gene_pathway_ids))
        ])
    
    gene_spread = 0.12
    gene_embeddings = np.zeros((n_genes, d_gene))
    for i in range(n_genes):
        pathway_id = gene_pathway_ids[i]
        center = pathway_centers[pathway_id]
        gene_embeddings[i] = center + rng.normal(0, gene_spread, size=d_gene)
    gene_embeddings = np.clip(gene_embeddings, 0.0, 1.0)
    
    pathway_amplitudes = rng.uniform(0.8, 1.5, size=n_pathways) * pathway_effect_scale
    pathway_widths = rng.uniform(0.15, 0.25, size=n_pathways)
    
    side = int(np.ceil(np.sqrt(n)))
    u1 = np.linspace(0, 1, side)
    u2 = np.linspace(0, 1, side)
    U1, U2 = np.meshgrid(u1, u2)
    context_all = np.stack([U1.ravel(), U2.ravel()], axis=1)[:n]
    
    gene_ids = rng.integers(0, n_genes, size=n)
    
    X = np.zeros((n, d_gene + d_context))
    X[:, :d_gene] = gene_embeddings[gene_ids]
    X[:, d_gene:] = context_all
    
    y = np.zeros(n)
    for i in range(n):
        gene_id = gene_ids[i]
        gene_pos = gene_embeddings[gene_id]
        context = context_all[i]
        pathway_id = gene_pathway_ids[gene_id]
        
        center = pathway_centers[pathway_id]
        dist_sq = np.sum((context - center) ** 2)
        pathway_effect = pathway_amplitudes[pathway_id] * np.exp(
            -dist_sq / (2 * pathway_widths[pathway_id] ** 2)
        )
        gene_effect = gene_effect_scale * np.sin(4 * np.pi * gene_pos[0]) * np.cos(4 * np.pi * gene_pos[1])
        interaction = 0.3 * np.sin(2 * np.pi * (gene_pos[0] + context[0])) * \
                           np.cos(2 * np.pi * (gene_pos[1] + context[1]))
        
        y[i] = pathway_effect + gene_effect + interaction
    
    y = y + noise_std * rng.standard_normal(n)
    y = y - y.min() + 0.1
    
    order = np.lexsort((context_all[:, 1], context_all[:, 0]))
    X = X[order]
    y = y[order]
    
    return Dataset(X=X, y=y, name="perturbation_pathways", n_samples=n_samples, n_features=4)


def generate_perturbation_sem(n_samples: int = 300, noise_std: float = 0.08,
                              seed: Optional[int] = None) -> Dataset:
    """Generate structural equation model (SEM) perturbation dataset (6D)."""
    rng = np.random.default_rng(seed)
    n = n_samples
    n_genes = 50
    d_gene = 2
    d_cell = 2
    d_env = 2
    
    gene_embeddings = rng.uniform(0, 1, size=(n_genes, d_gene))
    gene_ids = rng.integers(0, n_genes, size=n)
    cell_states = rng.uniform(0, 1, size=(n, d_cell))
    environments = rng.uniform(0, 1, size=(n, d_env))
    
    X = np.zeros((n, d_gene + d_cell + d_env))
    X[:, :d_gene] = gene_embeddings[gene_ids]
    X[:, d_gene:d_gene+d_cell] = cell_states
    X[:, d_gene+d_cell:] = environments
    
    y = np.zeros(n)
    for i in range(n):
        g = gene_embeddings[gene_ids[i]]
        c = cell_states[i]
        e = environments[i]
        
        gene_effect = 1.5 * np.sin(2 * np.pi * g[0]) * np.cos(2 * np.pi * g[1])
        cell_effect = 0.8 * np.sin(np.pi * c[0]) + 0.4 * np.cos(np.pi * c[1])
        env_effect = 0.6 * np.sin(np.pi * e[0]) * np.cos(np.pi * e[1])
        interaction = 0.5 * np.sin(np.pi * (g[0] + c[0])) * np.cos(np.pi * (g[1] + c[1]))
        
        y[i] = gene_effect + cell_effect + env_effect + interaction
    
    y = y + noise_std * rng.standard_normal(n)
    y = y - y.min() + 0.1
    
    return Dataset(X=X, y=y, name="perturbation_sem", n_samples=n_samples, n_features=6)


# Dataset registry
DATASET_GENERATORS = {
    "sine": generate_sine,
    "sine2d": generate_sine2d,
    "branin": generate_branin,
    "perturbation_pathways": generate_perturbation_pathways,
    "perturbation_sem": generate_perturbation_sem,
}


def generate_dataset(name: str, n_samples: int = 300, noise_std: float = None,
                     seed: Optional[int] = None) -> Dataset:
    """Generate a dataset by name."""
    if name not in DATASET_GENERATORS:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASET_GENERATORS.keys())}")
    
    if noise_std is not None:
        return DATASET_GENERATORS[name](n_samples=n_samples, noise_std=noise_std, seed=seed)
    return DATASET_GENERATORS[name](n_samples=n_samples, seed=seed)


def list_datasets() -> List[str]:
    """Get list of all available dataset names."""
    return list(DATASET_GENERATORS.keys())


if __name__ == "__main__":
    print("Testing dataset generation...")
    for name in list_datasets():
        ds = generate_dataset(name, seed=42)
        print(f"  {name}: X={ds.X.shape}, y=[{ds.y.min():.2f}, {ds.y.max():.2f}]")
