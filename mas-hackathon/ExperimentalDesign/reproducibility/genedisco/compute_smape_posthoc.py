#!/usr/bin/env python3
"""
Compute SMAPE post-hoc for GeneDisco experiments.

This script loads saved models and computes prediction error (SMAPE) on 
unseen data points for each cycle.
"""

import os
import sys
import json
import pickle
import tarfile
import tempfile
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric Mean Absolute Percentage Error."""
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    # Avoid division by zero
    mask = denominator > 1e-10
    if not np.any(mask):
        return 0.0
    
    errors = np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]
    return float(np.mean(errors) * 100)  # Percentage


def load_genedisco_dataset(cache_dir: str, dataset_name: str, feature_set: str):
    """Load GeneDisco dataset."""
    try:
        import h5py
        
        # Map dataset names to file names
        dataset_file_map = {
            'sanchez_2021_tau': 'sanchez_2021_neurons_tau',
            'schmidt_2021_ifng': 'schmidt_2021_ifng',
            'schmidt_2021_il2': 'schmidt_2021_il2',
            'zhu_2021_sarscov2': 'zhu_2021_sarscov2_host_factors',
            'zhuang_2019_nk': 'zhuang_2019',
        }
        dataset_file_name = dataset_file_map.get(dataset_name, dataset_name)
        
        # Load features
        feature_file = os.path.join(cache_dir, f'{feature_set}.h5')
        if not os.path.exists(feature_file):
            feature_file = os.path.join(cache_dir, f'{feature_set}_embedding.h5')
        
        with h5py.File(feature_file, 'r') as f:
            # Check for 'rownames' or 'row_names'
            row_key = 'rownames' if 'rownames' in f.keys() else 'row_names'
            data_key = 'covariates' if 'covariates' in f.keys() else 'data'
            feature_genes = [g.decode() if isinstance(g, bytes) else g for g in f[row_key][:]]
            features = f[data_key][:]
        
        # Load targets
        target_file = os.path.join(cache_dir, f'{dataset_file_name}.h5')
        with h5py.File(target_file, 'r') as f:
            row_key = 'rownames' if 'rownames' in f.keys() else 'row_names'
            data_key = 'covariates' if 'covariates' in f.keys() else 'data'
            target_genes = [g.decode() if isinstance(g, bytes) else g for g in f[row_key][:]]
            targets = f[data_key][:].flatten()
        
        # Find common genes
        feature_gene_set = set(feature_genes)
        common_genes = [g for g in target_genes if g in feature_gene_set]
        
        # Build aligned arrays
        feature_idx = {g: i for i, g in enumerate(feature_genes)}
        target_idx = {g: i for i, g in enumerate(target_genes)}
        
        X = np.array([features[feature_idx[g]] for g in common_genes])
        y = np.array([targets[target_idx[g]] for g in common_genes])
        
        return X, y, common_genes
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise


def load_model_and_predict(model_path: str, X: np.ndarray) -> np.ndarray:
    """Load saved model and make predictions."""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Extract outer tarball
        with tarfile.open(model_path, 'r:gz') as tar:
            tar.extractall(tmpdir)
        
        # Extract submodel
        submodel_path = os.path.join(tmpdir, 'model', 'submodel.tar.gz')
        with tarfile.open(submodel_path, 'r:gz') as tar:
            tar.extractall(os.path.join(tmpdir, 'submodel'))
        
        # Load PyTorch model
        model_pt = os.path.join(tmpdir, 'submodel', 'model', 'model.pt')
        
        # Try to load and run the model (skip pickle files that reference discobax module)
        try:
            # Load state dict
            state_dict = torch.load(model_pt, map_location='cpu', weights_only=False)
            
            # Infer model architecture from state dict
            input_dim = X.shape[1]
            
            # Build a simple MLP matching the state dict
            layers = []
            layer_sizes = []
            
            for key in sorted(state_dict.keys()):
                if 'weight' in key and 'batch' not in key.lower():
                    layer_sizes.append(state_dict[key].shape)
            
            if layer_sizes:
                # Reconstruct MLP
                model = build_mlp_from_state_dict(state_dict, input_dim)
                model.load_state_dict(state_dict, strict=False)
                model.eval()
                
                X_tensor = torch.tensor(X, dtype=torch.float32)
                with torch.no_grad():
                    pred = model(X_tensor)
                    if isinstance(pred, tuple):
                        pred = pred[0]
                    return pred.numpy().flatten()
            
        except Exception as e:
            print(f"  Warning: Could not load model directly: {e}")
    
    # Fallback: return None to indicate prediction failed
    return None


def build_mlp_from_state_dict(state_dict: dict, input_dim: int) -> torch.nn.Module:
    """Build MLP architecture from state dict keys."""
    import torch.nn as nn
    
    # Parse layer dimensions from state dict
    layer_dims = []
    for key in sorted(state_dict.keys()):
        if 'weight' in key and 'bn' not in key.lower() and 'batch' not in key.lower():
            out_dim, in_dim = state_dict[key].shape
            if not layer_dims:
                layer_dims.append(in_dim)
            layer_dims.append(out_dim)
    
    if not layer_dims:
        # Fallback simple model
        return nn.Sequential(nn.Linear(input_dim, 1))
    
    # Build sequential model
    layers = []
    for i in range(len(layer_dims) - 1):
        layers.append(nn.Linear(layer_dims[i], layer_dims[i+1]))
        if i < len(layer_dims) - 2:  # Not last layer
            layers.append(nn.ReLU())
    
    return nn.Sequential(*layers)


def compute_smape_for_experiment(exp_dir: str, cache_dir: str) -> Dict[str, Any]:
    """Compute SMAPE for all cycles of an experiment."""
    
    exp_path = Path(exp_dir)
    results_json = exp_path / 'results.json'
    
    if not results_json.exists():
        return None
    
    with open(results_json) as f:
        config = json.load(f)['config']
    
    dataset_name = config['dataset_name']
    feature_set = config['feature_set_name']
    
    # Find the nested experiment directory
    nested_dirs = list(exp_path.glob('*_bayesian_mlp_*'))
    if not nested_dirs:
        return None
    
    base_dir = nested_dirs[0]
    method_dirs = list(base_dir.glob('*_bayesian_mlp_*'))
    if not method_dirs:
        return None
    
    cycle_base = method_dirs[0]
    
    # Load dataset
    try:
        X, y, genes = load_genedisco_dataset(cache_dir, dataset_name, feature_set)
        gene_to_idx = {g: i for i, g in enumerate(genes)}
    except Exception as e:
        print(f"  Could not load dataset: {e}")
        return None
    
    # Process each cycle
    cycle_results = []
    acquired_indices = set()
    
    for cycle in range(config['n_cycles']):
        cycle_dir = cycle_base / f'cycle_{cycle}'
        
        if not cycle_dir.exists():
            continue
        
        # Load selected indices for this cycle
        selected_path = cycle_dir / 'selected_indices.pkl'
        if selected_path.exists():
            with open(selected_path, 'rb') as f:
                selected_genes = pickle.load(f)
            
            # Convert gene names to indices
            new_indices = set()
            for g in selected_genes:
                g_str = str(g)
                if g_str in gene_to_idx:
                    new_indices.add(gene_to_idx[g_str])
            
            acquired_indices.update(new_indices)
        
        # Compute SMAPE on unseen data
        all_indices = set(range(len(y)))
        unseen_indices = list(all_indices - acquired_indices)
        
        if len(unseen_indices) == 0:
            continue
        
        # Try to load model and predict
        model_path = cycle_dir / 'model.tar.gz'
        if model_path.exists():
            try:
                X_unseen = X[unseen_indices]
                y_unseen = y[unseen_indices]
                
                predictions = load_model_and_predict(str(model_path), X_unseen)
                
                if predictions is not None:
                    smape_value = smape(y_unseen, predictions)
                    
                    # Also compute Spearman correlation
                    from scipy.stats import spearmanr
                    spearman_rho, _ = spearmanr(y_unseen, predictions)
                    
                    cycle_results.append({
                        'cycle': cycle,
                        'n_acquired': len(acquired_indices),
                        'n_unseen': len(unseen_indices),
                        'SMAPE_Unseen': smape_value,
                        'SpearmanRho': spearman_rho,
                    })
                else:
                    cycle_results.append({
                        'cycle': cycle,
                        'n_acquired': len(acquired_indices),
                        'n_unseen': len(unseen_indices),
                        'SMAPE_Unseen': None,
                        'SpearmanRho': None,
                    })
                    
            except Exception as e:
                print(f"  Cycle {cycle} error: {e}")
                continue
    
    return {
        'config': config,
        'cycles': cycle_results,
    }


def process_all_experiments(experiments_dir: str, cache_dir: str, output_file: str):
    """Process all experiments and save results."""
    
    exp_path = Path(experiments_dir)
    all_results = []
    
    # Get all experiment directories
    exp_dirs = sorted([d for d in exp_path.iterdir() if d.is_dir()])
    
    print(f"Processing {len(exp_dirs)} experiments...")
    
    for i, exp_dir in enumerate(exp_dirs):
        exp_name = exp_dir.name
        print(f"[{i+1}/{len(exp_dirs)}] {exp_name}")
        
        result = compute_smape_for_experiment(str(exp_dir), cache_dir)
        
        if result and result['cycles']:
            config = result['config']
            for cycle_data in result['cycles']:
                row = {
                    'dataset': config['dataset_name'],
                    'feature_set': config['feature_set_name'],
                    'method': config['method_name'],
                    'seed': config['seed'],
                    **cycle_data
                }
                all_results.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    # Save
    df.to_csv(output_file, index=False)
    print(f"\nSaved {len(df)} rows to {output_file}")
    
    return df


def create_summary_tables(df: pd.DataFrame, output_dir: str):
    """Create summary tables with mean ± std."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Filter to valid SMAPE values
    df_valid = df[df['SMAPE_Unseen'].notna()].copy()
    
    if len(df_valid) == 0:
        print("No valid SMAPE values found")
        return
    
    # Final cycle performance
    max_cycle = df_valid.groupby(['dataset', 'feature_set', 'method', 'seed'])['cycle'].max()
    df_final = df_valid.merge(
        max_cycle.reset_index().rename(columns={'cycle': 'max_cycle'}),
        on=['dataset', 'feature_set', 'method', 'seed']
    )
    df_final = df_final[df_final['cycle'] == df_final['max_cycle']]
    
    # Aggregate by dataset, feature_set, method
    summary = df_final.groupby(['dataset', 'feature_set', 'method']).agg({
        'SMAPE_Unseen': ['mean', 'std'],
        'SpearmanRho': ['mean', 'std'],
    }).round(3)
    
    # Flatten columns
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    summary = summary.reset_index()
    
    # Format as mean ± std
    summary['SMAPE'] = summary.apply(
        lambda r: f"{r['SMAPE_Unseen_mean']:.2f} ± {r['SMAPE_Unseen_std']:.2f}", axis=1
    )
    summary['Spearman'] = summary.apply(
        lambda r: f"{r['SpearmanRho_mean']:.3f} ± {r['SpearmanRho_std']:.3f}", axis=1
    )
    
    # Save
    summary.to_csv(output_path / 'smape_summary.csv', index=False)
    
    # Pivot table by method
    pivot = summary.pivot_table(
        index=['dataset', 'feature_set'],
        columns='method',
        values='SMAPE',
        aggfunc='first'
    )
    pivot.to_csv(output_path / 'smape_by_method.csv')
    
    print(f"\nSummary tables saved to {output_path}")
    print("\nSMAPE by method (final cycle):")
    print(pivot.to_string())


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Compute SMAPE post-hoc')
    parser.add_argument('--experiments-dir', type=str, 
                       default='output/genedisco_ablation/results/experiments',
                       help='Directory with experiment results')
    parser.add_argument('--cache-dir', type=str, required=True,
                       help='GeneDisco data cache directory (directory with .h5 files)')
    parser.add_argument('--output', type=str,
                       default='output/genedisco_ablation/smape_results.csv',
                       help='Output CSV file')
    
    args = parser.parse_args()
    
    df = process_all_experiments(args.experiments_dir, args.cache_dir, args.output)
    
    if len(df) > 0:
        create_summary_tables(df, os.path.dirname(args.output))
