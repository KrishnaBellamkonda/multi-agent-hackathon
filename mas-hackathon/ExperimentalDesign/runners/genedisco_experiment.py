#!/usr/bin/env python3
"""
Runner for GeneDisco (real biological screen) experiments.

Runs one or more acquisition methods on real CRISPR screen datasets
using an MC Dropout neural network surrogate.
Default values target a single representative run; pass additional flags to sweep.

Usage:
    python runners/genedisco_experiment.py                              # default single run
    python runners/genedisco_experiment.py --quick                      # 1 seed, 3 cycles
    python runners/genedisco_experiment.py --datasets schmidt_2021_ifng
    python runners/genedisco_experiment.py --methods probability_hit,random
    python runners/genedisco_experiment.py --batch_sizes 100,200 --seeds 0,1,2  # sweep
    python runners/genedisco_experiment.py --export_only --ablation_dir ./output/run_XYZ

Datasets available: schmidt_2021_ifng, schmidt_2021_il2, zhuang_2019_nk,
                    sanchez_2021_tau, zhu_2021_sarscov2
"""

import os
import sys
import json
import pickle
import argparse
import subprocess
import itertools
from datetime import datetime
from pathlib import Path
from collections import defaultdict, Counter
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


# =============================================================================
# DEFAULT CONFIGURATION  (single representative run — override via CLI)
# =============================================================================

# Core methods used in the paper
METHODS = [
    "probability_hit",
    "thompson_hit",
    "thompson_sampling",
    "top_k",
    "random",
]

# Datasets available: schmidt_2021_ifng, schmidt_2021_il2, zhuang_2019_nk,
#                     sanchez_2021_tau, zhu_2021_sarscov2
DATASETS = ["schmidt_2021_ifng"]

# Single batch size for a default run; pass "--batch_sizes 100,200,500" to sweep
BATCH_SIZES = [200]

# Hit threshold: top X% of screen are hits
HIT_THRESHOLDS = [10]

# Fixed parameters
SEEDS = [0]  # Add more seeds (e.g. 0,1,2,3,4) for statistical robustness
NUM_CYCLES = 10
TOTAL_POINTS = 300
MODEL_NAME = "bayesian_mlp"

# Plotting configuration
METHOD_NAMES = {
    'thompson_hit': 'Thompson Hit',
    'thompson_sampling': 'Thompson Sampling',
    'top_k': 'Top-K',
    'probability_hit': 'Probability Hit',
    'random': 'Random',
}

METHOD_COLORS = {
    'thompson_hit': '#E63946',
    'thompson_sampling': '#457B9D',
    'top_k': '#2A9D8F',
    'probability_hit': '#F4A261',
    'random': '#6C757D',
}

THRESHOLD_COLORS = {5: '#2A9D8F', 10: '#E9C46A', 20: '#E76F51'}

KEY_METRICS = [
    'RecoveredHitsRatio_Cumulative',
    'SMAPE_Unseen',
    'SpearmanRho',
    'HitClusters_RecoveryRatio',
]


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

def run_single_experiment(
    dataset: str,
    method: str,
    batch_size: int,
    threshold: int,
    seed: int,
    num_cycles: int,
    total_points: int,
    output_dir: str,
    dry_run: bool = False,
) -> dict:
    """Run a single experiment configuration."""
    
    exp_output_dir = os.path.join(
        output_dir, dataset, f"batch{batch_size}", f"thresh{threshold}", method, f"seed{seed}"
    )
    fig_dir = os.path.join(exp_output_dir, "figures")
    
    os.makedirs(exp_output_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    
    # Build command
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cmd = [
        sys.executable,
        os.path.join(repo_root, "activehit/apps/genedisco_experiment.py"),
        f"--dataset_name={dataset}",
        f"--acquisition_function_name={method}",
        f"--acquisition_batch_size={batch_size}",
        f"--num_active_learning_cycles={num_cycles}",
        f"--seed={seed}",
        f"--output_directory={exp_output_dir}",
        f"--model_name={MODEL_NAME}",
        f"--cache_directory=./data",
        f"--feature_set_name=achilles",
        f"--hit_threshold={threshold}",
    ]
    
    result = {
        "dataset": dataset,
        "method": method,
        "batch_size": batch_size,
        "threshold": threshold,
        "seed": seed,
        "num_cycles": num_cycles,
        "output_dir": exp_output_dir,
        "command": " ".join(cmd),
        "status": "pending",
    }
    
    if dry_run:
        print(f"[DRY-RUN] {' '.join(cmd)}")
        result["status"] = "dry-run"
        return result
    
    try:
        proc = subprocess.run(
            cmd,
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=3600,
        )
        
        with open(os.path.join(exp_output_dir, "stdout.log"), "w") as f:
            f.write(proc.stdout)
        with open(os.path.join(exp_output_dir, "stderr.log"), "w") as f:
            f.write(proc.stderr)
        
        if proc.returncode == 0:
            result["status"] = "success"
        else:
            result["status"] = "failed"
            result["error"] = proc.stderr[-500:] if proc.stderr else "Unknown error"
                
    except subprocess.TimeoutExpired:
        result["status"] = "timeout"
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
    
    return result


def run_ablation(
    datasets: List[str],
    methods: List[str],
    batch_sizes: List[int],
    thresholds: List[int],
    seeds: List[int],
    num_cycles: int,
    total_points: int,
    output_dir: str,
    dry_run: bool = False,
) -> List[dict]:
    """Run the complete ablation study."""
    
    total_experiments = len(datasets) * len(methods) * len(batch_sizes) * len(thresholds) * len(seeds)
    
    print(f"\n{'='*70}")
    print(f"ABLATION STUDY")
    print(f"{'='*70}")
    print(f"Datasets ({len(datasets)}): {', '.join(datasets)}")
    print(f"Methods ({len(methods)}): {', '.join(methods)}")
    print(f"Batch sizes ({len(batch_sizes)}): {', '.join(map(str, batch_sizes))}")
    print(f"Thresholds ({len(thresholds)}): {', '.join(map(str, thresholds))}%")
    print(f"Seeds ({len(seeds)}): {', '.join(map(str, seeds))}")
    print(f"Cycles: {num_cycles}")
    print(f"Total experiments: {total_experiments}")
    print(f"Output: {output_dir}")
    print(f"{'='*70}\n")
    
    # Save configuration
    config = {
        "datasets": datasets,
        "methods": methods,
        "batch_sizes": batch_sizes,
        "thresholds": thresholds,
        "seeds": seeds,
        "num_cycles": num_cycles,
        "total_points": total_points,
        "model_name": MODEL_NAME,
        "timestamp": datetime.now().isoformat(),
    }
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    # Run experiments
    results = []
    completed = 0
    
    for dataset, batch_size, threshold, method, seed in itertools.product(
        datasets, batch_sizes, thresholds, methods, seeds
    ):
        completed += 1
        status_char = "▶" if not dry_run else "○"
        print(f"[{completed}/{total_experiments}] {status_char} {dataset} | batch={batch_size} | thresh={threshold}% | {method} | seed={seed}", end="")
        
        result = run_single_experiment(
            dataset=dataset,
            method=method,
            batch_size=batch_size,
            threshold=threshold,
            seed=seed,
            num_cycles=num_cycles,
            total_points=total_points,
            output_dir=output_dir,
            dry_run=dry_run,
        )
        results.append(result)
        
        if result["status"] == "success":
            print(" ✓")
        elif result["status"] == "dry-run":
            print(" (dry-run)")
        else:
            print(f" ✗ ({result['status']})")
    
    # Save results
    with open(os.path.join(output_dir, "ablation_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    success = sum(1 for r in results if r["status"] == "success")
    failed = sum(1 for r in results if r["status"] == "failed")
    
    print(f"\n{'='*70}")
    print(f"ABLATION COMPLETE")
    print(f"{'='*70}")
    print(f"Success: {success}/{total_experiments} ({100*success/total_experiments:.1f}%)")
    if failed > 0:
        print(f"Failed: {failed}")
    print(f"{'='*70}\n")
    
    return results


# =============================================================================
# DATA LOADING AND EXPORT
# =============================================================================

def find_results_files(ablation_dir: str) -> List[str]:
    """Find all results.pickle files recursively."""
    results_files = []
    for root, dirs, files in os.walk(ablation_dir):
        if 'results.pickle' in files:
            results_files.append(os.path.join(root, 'results.pickle'))
    return results_files


def parse_experiment_path(results_path: str, ablation_dir: str) -> dict:
    """Parse experiment metadata from file path."""
    rel_path = os.path.relpath(results_path, ablation_dir)
    parts = rel_path.split(os.sep)
    
    meta = {
        'dataset': None,
        'batch_size': None,
        'threshold': None,
        'method': None,
        'seed': None,
    }
    
    for part in parts:
        if part.startswith('batch'):
            try:
                meta['batch_size'] = int(part.replace('batch', ''))
            except:
                pass
        elif part.startswith('thresh'):
            try:
                meta['threshold'] = int(part.replace('thresh', ''))
            except:
                pass
        elif part.startswith('seed'):
            try:
                meta['seed'] = int(part.replace('seed', ''))
            except:
                pass
        elif part in METHOD_NAMES:
            meta['method'] = part
        elif part in DATASETS:
            meta['dataset'] = part
    
    return meta


def load_all_cycle_data(ablation_dir: str) -> pd.DataFrame:
    """Load all experiment results into a DataFrame."""
    results_files = find_results_files(ablation_dir)
    print(f"Found {len(results_files)} results.pickle files")
    
    all_rows = []
    
    for results_path in results_files:
        meta = parse_experiment_path(results_path, ablation_dir)
        
        if None in [meta['dataset'], meta['batch_size'], meta['method'], meta['seed']]:
            continue
        
        try:
            with open(results_path, 'rb') as f:
                cycle_results = pickle.load(f)
        except Exception as e:
            continue
        
        if not isinstance(cycle_results, list):
            continue
        
        for cycle_idx, cycle_data in enumerate(cycle_results):
            if not isinstance(cycle_data, dict):
                continue
            
            row = {
                'dataset': meta['dataset'],
                'batch_size': meta['batch_size'],
                'threshold': meta['threshold'],
                'method': meta['method'],
                'seed': meta['seed'],
                'cycle': cycle_idx + 1,
            }
            
            for key, value in cycle_data.items():
                if isinstance(value, str):
                    continue
                try:
                    row[key] = float(value)
                except:
                    pass
            
            all_rows.append(row)
    
    df = pd.DataFrame(all_rows)
    print(f"Loaded {len(df)} cycle records from {len(results_files)} experiments")
    return df


def export_to_csv(df: pd.DataFrame, output_path: str):
    """Export DataFrame to CSV."""
    sort_cols = ['dataset', 'batch_size', 'threshold', 'method', 'seed', 'cycle']
    sort_cols = [c for c in sort_cols if c in df.columns]
    df = df.sort_values(sort_cols)
    df.to_csv(output_path, index=False)
    print(f"Exported {len(df)} rows to {output_path}")


def generate_summary_stats(df: pd.DataFrame, output_dir: str) -> pd.DataFrame:
    """Generate summary statistics CSV."""
    max_cycle = df['cycle'].max()
    df_final = df[df['cycle'] == max_cycle]
    
    has_threshold = 'threshold' in df_final.columns and df_final['threshold'].notna().any()
    
    agg_cols = [c for c in KEY_METRICS if c in df_final.columns]
    
    summary_rows = []
    
    if has_threshold:
        groupby_cols = ['dataset', 'batch_size', 'threshold', 'method']
    else:
        groupby_cols = ['dataset', 'batch_size', 'method']
    
    for keys, group in df_final.groupby(groupby_cols):
        if has_threshold:
            dataset, batch_size, threshold, method = keys
            row = {
                'dataset': dataset,
                'batch_size': batch_size,
                'threshold': int(threshold) if threshold is not None else None,
                'method': method,
                'n_seeds': len(group),
            }
        else:
            dataset, batch_size, method = keys
            row = {
                'dataset': dataset,
                'batch_size': batch_size,
                'method': method,
                'n_seeds': len(group),
            }
        for col in agg_cols:
            if col in group.columns:
                vals = group[col].dropna()
                if len(vals) > 0:
                    row[f'{col}_mean'] = vals.mean()
                    row[f'{col}_std'] = vals.std()
        summary_rows.append(row)
    
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(output_dir, 'summary_final_cycle.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary stats to {summary_path}")
    
    return summary_df


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_learning_curves_by_batch_threshold(df: pd.DataFrame, output_dir: str):
    """Generate learning curve plots organized by batch size and threshold."""
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    datasets = sorted(df['dataset'].unique())
    batch_sizes = sorted(df['batch_size'].unique())
    methods = sorted(df['method'].unique())
    
    if 'threshold' in df.columns and df['threshold'].notna().any():
        thresholds = sorted(df['threshold'].dropna().unique())
    else:
        thresholds = [None]
    
    metrics_to_plot = [
        ('RecoveredHitsRatio_Cumulative', 'Hit Recovery Ratio', (0, 1.05)),
        ('SMAPE_Unseen', 'SMAPE (Unseen)', None),
        ('SpearmanRho', 'Spearman ρ', (-0.2, 1.05)),
        ('HitClusters_RecoveryRatio', 'Hit Cluster Recovery', (0, 1.05)),
    ]
    
    for batch_size in batch_sizes:
        for threshold in thresholds:
            if threshold is not None:
                subdir = f'batch_{batch_size}_thresh_{int(threshold)}'
            else:
                subdir = f'batch_{batch_size}'
            batch_dir = os.path.join(plots_dir, subdir)
            os.makedirs(batch_dir, exist_ok=True)
            
            df_subset = df[df['batch_size'] == batch_size]
            if threshold is not None:
                df_subset = df_subset[df_subset['threshold'] == threshold]
            
            for dataset in datasets:
                df_ds = df_subset[df_subset['dataset'] == dataset]
                
                for metric, ylabel, ylim in metrics_to_plot:
                    if metric not in df_ds.columns:
                        continue
                    
                    fig, ax = plt.subplots(figsize=(8, 5))
                    
                    for method in methods:
                        df_method = df_ds[df_ds['method'] == method]
                        if len(df_method) == 0:
                            continue
                        
                        agg = df_method.groupby('cycle')[metric].agg(['mean', 'std']).reset_index()
                        
                        color = METHOD_COLORS.get(method, '#333333')
                        label = METHOD_NAMES.get(method, method)
                        
                        ax.plot(agg['cycle'], agg['mean'], color=color, linewidth=2,
                               label=label, marker='o', markersize=4)
                        ax.fill_between(agg['cycle'],
                                       agg['mean'] - agg['std'],
                                       agg['mean'] + agg['std'],
                                       color=color, alpha=0.15)
                    
                    ax.set_xlabel('Cycle', fontsize=11)
                    ax.set_ylabel(ylabel, fontsize=11)
                    title = f'{dataset} - Batch {batch_size}'
                    if threshold is not None:
                        title += f' - Thresh {int(threshold)}%'
                    ax.set_title(title, fontsize=12, fontweight='bold')
                    ax.legend(loc='best', fontsize=9)
                    ax.grid(True, alpha=0.3)
                    
                    if ylim:
                        ax.set_ylim(ylim)
                    
                    plt.tight_layout()
                    filename = f'{dataset}_{metric}.png'
                    fig.savefig(os.path.join(batch_dir, filename), dpi=150, bbox_inches='tight')
                    plt.close(fig)
        
        print(f"  Saved learning curves to: plots/batch_{batch_size}_*/")


def plot_method_comparison_barplots(df: pd.DataFrame, output_dir: str):
    """Generate method comparison barplots by batch size and threshold."""
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    max_cycle = df['cycle'].max()
    df_final = df[df['cycle'] == max_cycle]
    
    datasets = sorted(df_final['dataset'].unique())
    batch_sizes = sorted(df_final['batch_size'].unique())
    methods = sorted(df_final['method'].unique())
    
    if 'threshold' in df_final.columns and df_final['threshold'].notna().any():
        thresholds = sorted(df_final['threshold'].dropna().unique())
    else:
        thresholds = [None]
    
    metrics_to_plot = [
        ('RecoveredHitsRatio_Cumulative', 'Hit Recovery Ratio'),
        ('SMAPE_Unseen', 'SMAPE (Unseen)'),
        ('SpearmanRho', 'Spearman ρ'),
        ('HitClusters_RecoveryRatio', 'Hit Cluster Recovery'),
    ]
    
    for batch_size in batch_sizes:
        for threshold in thresholds:
            df_subset = df_final[df_final['batch_size'] == batch_size]
            if threshold is not None:
                df_subset = df_subset[df_subset['threshold'] == threshold]
            
            if len(df_subset) == 0:
                continue
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            axes = axes.flatten()
            
            for ax, (metric, ylabel) in zip(axes, metrics_to_plot):
                if metric not in df_subset.columns:
                    continue
                
                x_positions = np.arange(len(datasets))
                bar_width = 0.8 / len(methods)
                
                for i, method in enumerate(methods):
                    values = []
                    errors = []
                    for dataset in datasets:
                        df_dm = df_subset[(df_subset['dataset'] == dataset) & (df_subset['method'] == method)]
                        if len(df_dm) > 0 and metric in df_dm.columns:
                            vals = df_dm[metric].dropna()
                            values.append(vals.mean() if len(vals) > 0 else 0)
                            errors.append(vals.std() if len(vals) > 0 else 0)
                        else:
                            values.append(0)
                            errors.append(0)
                    
                    color = METHOD_COLORS.get(method, '#333333')
                    label = METHOD_NAMES.get(method, method)
                    offset = (i - len(methods)/2 + 0.5) * bar_width
                    
                    ax.bar(x_positions + offset, values, bar_width * 0.9,
                          label=label, color=color, alpha=0.85, yerr=errors, capsize=2)
                
                ax.set_xlabel('Dataset', fontsize=10)
                ax.set_ylabel(ylabel, fontsize=10)
                ax.set_xticks(x_positions)
                ax.set_xticklabels(datasets, rotation=45, ha='right', fontsize=9)
                ax.set_title(ylabel, fontsize=11, fontweight='bold')
                ax.grid(True, axis='y', alpha=0.3)
            
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.02),
                      ncol=len(methods), fontsize=10)
            
            title = f'Method Comparison - Batch Size {batch_size}'
            if threshold is not None:
                title += f' - Threshold {int(threshold)}%'
            fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
            plt.tight_layout(rect=[0, 0.06, 1, 0.95])
            
            if threshold is not None:
                filename = f'method_comparison_batch{batch_size}_thresh{int(threshold)}.png'
            else:
                filename = f'method_comparison_batch{batch_size}.png'
            fig.savefig(os.path.join(plots_dir, filename), dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"  Saved: plots/{filename}")


def plot_threshold_comparison(df: pd.DataFrame, output_dir: str):
    """Generate plots comparing performance across thresholds."""
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    if 'threshold' not in df.columns or not df['threshold'].notna().any():
        print("  No threshold data - skipping threshold comparison plots")
        return
    
    max_cycle = df['cycle'].max()
    df_final = df[df['cycle'] == max_cycle]
    
    batch_sizes = sorted(df_final['batch_size'].unique())
    methods = sorted(df_final['method'].unique())
    thresholds = sorted(df_final['threshold'].dropna().unique())
    
    metrics_to_plot = [
        ('RecoveredHitsRatio_Cumulative', 'Hit Recovery Ratio'),
        ('SMAPE_Unseen', 'SMAPE (Unseen)'),
        ('SpearmanRho', 'Spearman ρ'),
        ('HitClusters_RecoveryRatio', 'Hit Cluster Recovery'),
    ]
    
    for metric, ylabel in metrics_to_plot:
        if metric not in df_final.columns:
            continue
        
        n_batches = len(batch_sizes)
        fig, axes = plt.subplots(1, n_batches, figsize=(6*n_batches, 5), sharey=True)
        if n_batches == 1:
            axes = [axes]
        
        for ax, batch_size in zip(axes, batch_sizes):
            df_batch = df_final[df_final['batch_size'] == batch_size]
            
            x_positions = np.arange(len(methods))
            bar_width = 0.8 / len(thresholds)
            
            for i, threshold in enumerate(thresholds):
                df_thresh = df_batch[df_batch['threshold'] == threshold]
                
                values = []
                errors = []
                for method in methods:
                    df_m = df_thresh[df_thresh['method'] == method]
                    if len(df_m) > 0 and metric in df_m.columns:
                        vals = df_m[metric].dropna()
                        values.append(vals.mean() if len(vals) > 0 else 0)
                        errors.append(vals.std() if len(vals) > 0 else 0)
                    else:
                        values.append(0)
                        errors.append(0)
                
                color = THRESHOLD_COLORS.get(int(threshold), '#333333')
                offset = (i - len(thresholds)/2 + 0.5) * bar_width
                
                ax.bar(x_positions + offset, values, bar_width * 0.9,
                      label=f'{int(threshold)}% threshold', color=color, alpha=0.85,
                      yerr=errors, capsize=2)
            
            ax.set_xlabel('Method', fontsize=10)
            ax.set_xticks(x_positions)
            ax.set_xticklabels([METHOD_NAMES.get(m, m) for m in methods], rotation=45, ha='right', fontsize=9)
            ax.set_title(f'Batch Size {batch_size}', fontsize=11, fontweight='bold')
            ax.grid(True, axis='y', alpha=0.3)
        
        axes[0].set_ylabel(ylabel, fontsize=11)
        
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.02),
                  ncol=len(thresholds), fontsize=10)
        
        fig.suptitle(f'{ylabel} by Threshold', fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0.08, 1, 0.95])
        
        filename = f'{metric}_by_threshold.png'
        fig.savefig(os.path.join(plots_dir, filename), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: plots/{filename}")


def generate_all_plots(df: pd.DataFrame, output_dir: str):
    """Generate all plots."""
    print("\nGenerating plots...")
    
    print("\n1. Learning curves by batch size and threshold...")
    plot_learning_curves_by_batch_threshold(df, output_dir)
    
    print("\n2. Method comparison barplots...")
    plot_method_comparison_barplots(df, output_dir)
    
    print("\n3. Threshold comparison plots...")
    plot_threshold_comparison(df, output_dir)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Comprehensive ablation study runner for toy experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full ablation with all defaults
  python runners/genedisco_experiment.py
  
  # Quick test (1 seed, 3 cycles, reduced config)
  python runners/genedisco_experiment.py --quick
  
  # Custom configuration
  python runners/genedisco_experiment.py --datasets sine,branin --batch_sizes 5,10 --thresholds 10,20
  
  # Only export/plot from existing run
  python runners/genedisco_experiment.py --export_only --ablation_dir ./ablation_toy_YYYYMMDD_HHMMSS
  
  # Dry run (show what would be executed)
  python runners/genedisco_experiment.py --dry-run
"""
    )
    
    parser.add_argument("--datasets", type=str, default=",".join(DATASETS),
                       help=f"Comma-separated datasets. Default: {','.join(DATASETS)}")
    parser.add_argument("--methods", type=str, default=",".join(METHODS),
                       help=f"Comma-separated methods. Default: {','.join(METHODS)}")
    parser.add_argument("--batch_sizes", type=str, default=",".join(map(str, BATCH_SIZES)),
                       help=f"Comma-separated batch sizes. Default: {','.join(map(str, BATCH_SIZES))}")
    parser.add_argument("--thresholds", type=str, default=",".join(map(str, HIT_THRESHOLDS)),
                       help=f"Comma-separated hit thresholds (%%). Default: {','.join(map(str, HIT_THRESHOLDS))}")
    parser.add_argument("--seeds", type=str, default=",".join(map(str, SEEDS)),
                       help=f"Comma-separated seeds. Default: {','.join(map(str, SEEDS))}")
    parser.add_argument("--num_cycles", type=int, default=NUM_CYCLES,
                       help=f"Number of cycles. Default: {NUM_CYCLES}")
    parser.add_argument("--total_points", type=int, default=TOTAL_POINTS,
                       help=f"Total pool size. Default: {TOTAL_POINTS}")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory. Default: ./ablation_<timestamp>")
    parser.add_argument("--quick", action="store_true",
                       help="Quick test mode: 1 seed, 3 cycles, reduced config")
    parser.add_argument("--dry-run", action="store_true",
                       help="Print commands without running")
    parser.add_argument("--export_only", action="store_true",
                       help="Only export results and generate plots (skip running experiments)")
    parser.add_argument("--ablation_dir", type=str, default=None,
                       help="Ablation directory for --export_only mode")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Parse configuration
    datasets = [d.strip() for d in args.datasets.split(",")]
    methods = [m.strip() for m in args.methods.split(",")]
    batch_sizes = [int(b.strip()) for b in args.batch_sizes.split(",")]
    thresholds = [int(t.strip()) for t in args.thresholds.split(",")]
    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    
    # Quick mode
    if args.quick:
        seeds = [0]
        args.num_cycles = 3
        datasets = datasets[:2]
        batch_sizes = batch_sizes[:1]
        thresholds = [10]
        print("[QUICK MODE] Reduced configuration for testing\n")
    
    # Determine output directory
    if args.export_only:
        if not args.ablation_dir:
            print("ERROR: --export_only requires --ablation_dir")
            sys.exit(1)
        output_dir = args.ablation_dir
    else:
        if args.output_dir:
            output_dir = args.output_dir
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"./ablation_{timestamp}"
    
    # Run experiments or just export
    if not args.export_only:
        results = run_ablation(
            datasets=datasets,
            methods=methods,
            batch_sizes=batch_sizes,
            thresholds=thresholds,
            seeds=seeds,
            num_cycles=args.num_cycles,
            total_points=args.total_points,
            output_dir=output_dir,
            dry_run=args.dry_run,
        )
        
        if args.dry_run:
            print("\n[DRY-RUN] No experiments were executed.")
            return
    
    # Load and export results
    print("\nLoading and exporting results...")
    df = load_all_cycle_data(output_dir)
    
    if len(df) == 0:
        print("No data found!")
        return
    
    # Export CSV
    csv_path = os.path.join(output_dir, 'all_cycles_metrics.csv')
    export_to_csv(df, csv_path)
    
    # Generate summary
    generate_summary_stats(df, output_dir)
    
    # Generate plots
    generate_all_plots(df, output_dir)
    
    print(f"\n{'='*70}")
    print("ALL DONE!")
    print(f"{'='*70}")
    print(f"Results directory: {output_dir}")
    print(f"  - all_cycles_metrics.csv: Full results ({len(df)} rows)")
    print(f"  - summary_final_cycle.csv: Aggregated final cycle stats")
    print(f"  - plots/: All generated figures")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
