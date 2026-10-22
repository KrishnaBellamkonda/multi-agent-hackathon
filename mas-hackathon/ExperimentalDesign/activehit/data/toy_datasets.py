"""Unified toy dataset generators for DiscoBAX experiments.

This module provides all synthetic datasets for benchmarking acquisition functions.
These are the datasets described in the manuscript.

DATASET CATEGORIES:
==================

1. SIMPLE (baseline/debugging):
   - sine: 1D sine wave - simplest test case
   - branin: 2D Branin-Hoo function - standard optimization benchmark
   
2. PERTURBATION (pseudo-biological, learnable):
   - perturbation_sem: SEM-based perturbation with gene embeddings (learnable)
   - perturbation_pathways: Pathway-structured perturbations (learnable)

Each perturbation dataset models:
   - Gene perturbations with continuous embeddings
   - Cell state (intrinsic) factors  
   - Environmental (extrinsic) factors
   - Gene × context interactions
   - Phenotype readout

Key design principles:
   - Fully continuous features (no one-hot encoding)
   - GP-learnable with RBF kernel
   - Biologically interpretable structure
   - Smooth functions for reliable optimization

Usage:
    from activehit.data.toy_datasets import get_dataset, list_datasets
    
    # Get specific dataset
    X, y, meta = get_dataset('perturbation_pathways', n=500, seed=42)
    
    # List all available datasets
    print(list_datasets())
"""
from __future__ import annotations

import numpy as np
from typing import Tuple, Optional, Dict, Any, List


def _rng(seed: Optional[int]):
    """Create numpy random generator."""
    return np.random.default_rng(seed)


# =============================================================================
# SIMPLE DATASETS (baseline benchmarks)
# =============================================================================

def sine(n: int = 500, seed: int = 0) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """1D sine wave - simplest baseline.
    
    y = sin(2π * x) + noise
    
    Features: [x] (1D)
    """
    rng = _rng(seed)
    X = np.linspace(0, 1, n).reshape(-1, 1)
    y = np.sin(2 * np.pi * X[:, 0]) + 0.05 * rng.standard_normal(n)
    
    meta = {'dim': 1, 'name': 'sine', 'category': 'simple'}
    return X, y, meta


def branin(n: int = 500, seed: int = 0) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """2D Branin-Hoo function - standard optimization benchmark.
    
    Standard test function with 3 global minima.
    Rescaled to unit square [0,1]² and inverted for maximization.
    
    Features: [x1, x2] (2D)
    """
    rng = _rng(seed)
    
    # Generate grid points
    side = int(np.ceil(np.sqrt(n)))
    x1 = np.linspace(0, 1, side)
    x2 = np.linspace(0, 1, side)
    X1, X2 = np.meshgrid(x1, x2)
    X = np.stack([X1.ravel(), X2.ravel()], axis=1)[:n]
    
    # Branin function (rescaled to [0,1]²)
    # Original domain: x1 ∈ [-5, 10], x2 ∈ [0, 15]
    x1_scaled = X[:, 0] * 15 - 5
    x2_scaled = X[:, 1] * 15
    
    a, b, c = 1.0, 5.1 / (4 * np.pi**2), 5 / np.pi
    r, s, t = 6.0, 10.0, 1 / (8 * np.pi)
    
    term1 = a * (x2_scaled - b * x1_scaled**2 + c * x1_scaled - r)**2
    term2 = s * (1 - t) * np.cos(x1_scaled)
    
    # Invert and normalize for maximization
    y_raw = -(term1 + term2 + s)  # Negate for maximization
    y = (y_raw - y_raw.min()) / (y_raw.max() - y_raw.min() + 1e-8)
    y = y + 0.02 * rng.standard_normal(len(y))
    
    meta = {'dim': 2, 'name': 'branin', 'category': 'simple'}
    return X, y, meta


def sine2d(n: int = 500, seed: int = 0) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """2D sinusoidal surface - smooth multi-modal test function.
    
    y = sin(2π * x1) * sin(2π * x2) + noise
    
    Features: [x1, x2] (2D)
    
    This creates a smooth egg-carton pattern with multiple peaks/troughs.
    Good for testing acquisition functions on 2D landscapes.
    """
    rng = _rng(seed)
    
    # Generate grid points
    side = int(np.ceil(np.sqrt(n)))
    x1 = np.linspace(0, 1, side)
    x2 = np.linspace(0, 1, side)
    X1, X2 = np.meshgrid(x1, x2)
    X = np.stack([X1.ravel(), X2.ravel()], axis=1)[:n]
    
    # 2D sine product
    y = np.sin(2 * np.pi * X[:, 0]) * np.sin(2 * np.pi * X[:, 1])
    # Normalize to [0, 1] and add noise
    y = (y - y.min()) / (y.max() - y.min() + 1e-8)
    y = y + 0.03 * rng.standard_normal(len(y))
    
    meta = {'dim': 2, 'name': 'sine2d', 'category': 'simple'}
    return X, y, meta


# =============================================================================
# PERTURBATION DATASETS (pseudo-biological, learnable)
# =============================================================================

def perturbation_sem(
    n: int = 600,
    seed: int = 0,
    n_genes: int = 50,
    d_gene: int = 2,      # Gene embedding dimension
    d_cell: int = 2,      # Cell state dimension
    d_env: int = 2,       # Environment dimension
    noise_sigma: float = 0.08,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """SEM-based perturbation dataset with gene embeddings.
    
    Models single-cell perturbation experiments where:
    - Each gene has a learned 2D embedding
    - Cell states vary continuously (e.g., cell cycle, differentiation)
    - Environment varies (e.g., culture conditions)
    - Phenotype = f(gene, cell_state, environment)
    
    Features: [gene_emb (2D), cell_state (2D), environment (2D)] = 6D total
    
    This is LEARNABLE because:
    - All features are continuous
    - Similar genes (nearby in embedding space) have similar effects
    - Smooth function over the feature space
    """
    rng = _rng(seed)
    
    # Gene embeddings (clustered by functional similarity)
    gene_embeddings = rng.uniform(0, 1, size=(n_genes, d_gene))
    
    # Sample genes uniformly
    gene_ids = rng.integers(0, n_genes, size=n)
    
    # Cell state: continuous latent (models cell cycle, differentiation, etc.)
    cell_states = rng.uniform(0, 1, size=(n, d_cell))
    
    # Environment: continuous latent (models culture conditions)
    environments = rng.uniform(0, 1, size=(n, d_env))
    
    # Build feature matrix
    X = np.zeros((n, d_gene + d_cell + d_env))
    X[:, :d_gene] = gene_embeddings[gene_ids]
    X[:, d_gene:d_gene+d_cell] = cell_states
    X[:, d_gene+d_cell:] = environments
    
    # Generate phenotype with interpretable structure
    y = np.zeros(n)
    for i in range(n):
        g = gene_embeddings[gene_ids[i]]
        c = cell_states[i]
        e = environments[i]
        
        # Main effects (additive)
        gene_effect = 1.5 * np.sin(2 * np.pi * g[0]) * np.cos(2 * np.pi * g[1])
        cell_effect = 0.8 * np.sin(np.pi * c[0]) + 0.4 * np.cos(np.pi * c[1])
        env_effect = 0.6 * np.sin(np.pi * e[0]) * np.cos(np.pi * e[1])
        
        # Gene × cell interaction
        interaction = 0.5 * np.sin(np.pi * (g[0] + c[0])) * np.cos(np.pi * (g[1] + c[1]))
        
        y[i] = gene_effect + cell_effect + env_effect + interaction
    
    y = y + noise_sigma * rng.standard_normal(n)
    y = y - y.min() + 0.1  # Make positive
    
    meta = {
        'dim': d_gene + d_cell + d_env,
        'name': 'perturbation_sem',
        'category': 'perturbation',
        'n_genes': n_genes,
        'gene_embeddings': gene_embeddings,
        'gene_ids': gene_ids,
        'feature_slices': {
            'gene': slice(0, d_gene),
            'cell': slice(d_gene, d_gene + d_cell),
            'env': slice(d_gene + d_cell, d_gene + d_cell + d_env),
        }
    }
    return X, y, meta


# Alias for backwards compatibility
perturbation_simple = perturbation_sem


def perturbation_pathways(
    n: int = 600,
    seed: int = 0,
    n_genes: int = 40,
    n_pathways: int = 4,
    d_gene: int = 2,
    d_context: int = 2,
    pathway_effect_scale: float = 2.5,
    gene_effect_scale: float = 0.4,
    noise_sigma: float = 0.08,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Pathway-structured perturbation dataset.
    
    Models perturbation experiments with pathway structure:
    - Genes belong to biological pathways (clusters in embedding space)
    - Pathway membership determines baseline effect
    - Gene-specific effects modulate pathway effect
    - Context (cell state + environment) modulates phenotype
    
    Features: [gene_emb (2D), context (2D)] = 4D total
    
    Key biological analogy:
    - Genes in same pathway have similar 2D positions
    - Perturbing any gene in an active pathway has large effect
    - Context determines which pathways are active
    
    This is LEARNABLE because:
    - Pathway structure creates smooth regions
    - Similar genes → similar effects
    - GP with RBF kernel captures spatial structure
    """
    rng = _rng(seed)
    
    # Pathway centers (well-separated in 2D embedding space)
    pathway_centers = np.array([
        [0.2, 0.2], [0.8, 0.2],
        [0.2, 0.8], [0.8, 0.8],
    ])[:n_pathways]
    
    # Assign genes to pathways
    genes_per_pathway = n_genes // n_pathways
    gene_pathway_ids = np.repeat(np.arange(n_pathways), genes_per_pathway)
    if len(gene_pathway_ids) < n_genes:
        gene_pathway_ids = np.concatenate([
            gene_pathway_ids,
            rng.integers(0, n_pathways, size=n_genes - len(gene_pathway_ids))
        ])
    
    # Place genes around pathway centers (genes in same pathway cluster together)
    gene_spread = 0.12
    gene_embeddings = np.zeros((n_genes, d_gene))
    for i in range(n_genes):
        pathway_id = gene_pathway_ids[i]
        center = pathway_centers[pathway_id]
        gene_embeddings[i] = center + rng.normal(0, gene_spread, size=d_gene)
    gene_embeddings = np.clip(gene_embeddings, 0.0, 1.0)
    
    # Pathway effect parameters
    pathway_amplitudes = rng.uniform(0.8, 1.5, size=n_pathways) * pathway_effect_scale
    pathway_widths = rng.uniform(0.15, 0.25, size=n_pathways)
    
    # Generate context positions (grid for visualization)
    side = int(np.ceil(np.sqrt(n)))
    u1 = np.linspace(0, 1, side)
    u2 = np.linspace(0, 1, side)
    U1, U2 = np.meshgrid(u1, u2)
    context_all = np.stack([U1.ravel(), U2.ravel()], axis=1)[:n]
    
    # Sample genes
    gene_ids = rng.integers(0, n_genes, size=n)
    
    # Build feature matrix
    X = np.zeros((n, d_gene + d_context))
    X[:, :d_gene] = gene_embeddings[gene_ids]
    X[:, d_gene:] = context_all
    
    # Generate phenotype
    y = np.zeros(n)
    for i in range(n):
        gene_id = gene_ids[i]
        gene_pos = gene_embeddings[gene_id]
        context = context_all[i]
        pathway_id = gene_pathway_ids[gene_id]
        
        # Pathway effect: Gaussian bump centered at pathway center
        # When context is near pathway center → strong effect
        center = pathway_centers[pathway_id]
        dist_sq = np.sum((context - center) ** 2)
        pathway_effect = pathway_amplitudes[pathway_id] * np.exp(
            -dist_sq / (2 * pathway_widths[pathway_id] ** 2)
        )
        
        # Gene-specific modulation
        gene_effect = gene_effect_scale * np.sin(4 * np.pi * gene_pos[0]) * np.cos(4 * np.pi * gene_pos[1])
        
        # Context-gene interaction
        interaction = 0.3 * np.sin(2 * np.pi * (gene_pos[0] + context[0])) * \
                           np.cos(2 * np.pi * (gene_pos[1] + context[1]))
        
        y[i] = pathway_effect + gene_effect + interaction
    
    y = y + noise_sigma * rng.standard_normal(n)
    y = y - y.min() + 0.1
    
    # Sort by context for visualization
    order = np.lexsort((context_all[:, 1], context_all[:, 0]))
    X = X[order]
    y = y[order]
    gene_ids = gene_ids[order]
    context_all = context_all[order]
    
    meta = {
        'dim': d_gene + d_context,
        'name': 'perturbation_pathways',
        'category': 'perturbation',
        'n_genes': n_genes,
        'n_pathways': n_pathways,
        'gene_embeddings': gene_embeddings,
        'gene_pathway_ids': gene_pathway_ids,
        'pathway_centers': pathway_centers,
        'gene_ids': gene_ids,
        'context_positions': context_all,
        'feature_slices': {
            'gene': slice(0, d_gene),
            'context': slice(d_gene, d_gene + d_context),
        }
    }
    return X, y, meta


# =============================================================================
# DATASET REGISTRY
# =============================================================================

DATASET_REGISTRY = {
    # Simple (baseline only)
    'sine': sine,
    'sine2d': sine2d,
    'branin': branin,
    # Perturbation (pseudo-biological) - RECOMMENDED for method evaluation
    'perturbation_sem': perturbation_sem,
    'perturbation_simple': perturbation_simple,  # Alias for backwards compatibility
    'perturbation_pathways': perturbation_pathways,
}

DATASET_CATEGORIES = {
    'simple': ['sine', 'sine2d', 'branin'],
    'perturbation': ['perturbation_sem', 'perturbation_pathways'],
}

# Default datasets for ablation (only ones where methods differentiate well)
DEFAULT_ABLATION_DATASETS = ['perturbation_sem', 'perturbation_pathways']


def list_datasets() -> List[str]:
    """Return list of all available dataset names."""
    return list(DATASET_REGISTRY.keys())


def list_datasets_by_category() -> Dict[str, List[str]]:
    """Return datasets organized by category."""
    return DATASET_CATEGORIES.copy()


def get_dataset(
    name: str,
    n: int = 500,
    seed: int = 0,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Get a dataset by name.
    
    Args:
        name: Dataset name (see list_datasets())
        n: Number of samples
        seed: Random seed
        **kwargs: Additional dataset-specific parameters
    
    Returns:
        X: Feature matrix (n, d)
        y: Target values (n,)
        meta: Metadata dict with 'name', 'category', 'dim', etc.
    """
    if name not in DATASET_REGISTRY:
        available = ', '.join(list_datasets())
        raise ValueError(f"Unknown dataset '{name}'. Available: {available}")
    
    return DATASET_REGISTRY[name](n=n, seed=seed, **kwargs)


def get_dataset_info(name: str) -> Dict:
    """Get information about a dataset without generating it."""
    info = {
        'sine': {
            'dim': 1, 'category': 'simple',
            'description': '1D sine wave - simplest baseline'
        },
        'sine2d': {
            'dim': 2, 'category': 'simple',
            'description': '2D sinusoidal surface - smooth multi-modal'
        },
        'branin': {
            'dim': 2, 'category': 'simple', 
            'description': '2D Branin-Hoo function - optimization benchmark'
        },
        'perturbation_sem': {
            'dim': 6, 'category': 'perturbation',
            'description': 'SEM-based perturbation with embeddings (6D: gene+cell+env)'
        },
        'perturbation_simple': {
            'dim': 6, 'category': 'perturbation',
            'description': 'Alias for perturbation_sem'
        },
        'perturbation_pathways': {
            'dim': 4, 'category': 'perturbation',
            'description': 'Pathway-structured perturbations (4D: gene+context)'
        },
    }
    if name not in info:
        raise ValueError(f"Unknown dataset '{name}'")
    return info[name]


# For backwards compatibility
__all__ = [
    'sine', 'sine2d', 'branin',
    'perturbation_sem', 'perturbation_simple', 'perturbation_pathways',
    'get_dataset', 'list_datasets', 'list_datasets_by_category',
    'DATASET_REGISTRY', 'DATASET_CATEGORIES',
]
