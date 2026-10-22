"""
Toy dataset ablation study runner.

Run complete ablation studies on synthetic datasets to evaluate
acquisition method performance across different conditions.
"""

import os
import sys
import json
import pickle
import argparse
import csv
from pathlib import Path
from datetime import datetime
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Optional
import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from reproducibility.shared.datasets import list_datasets, generate_dataset
from reproducibility.shared.acquisition import list_acquisition_functions
from reproducibility.shared.experiment import run_experiment, ExperimentConfig


# Default configuration
DEFAULT_CONFIG = {
    'datasets': ['sine', 'sine2d', 'branin', 'perturbation_pathways', 'perturbation_sem'],
    'methods': ['probability_hit', 'thompson_sampling', 'thompson_hit', 'top_k', 'random'],
    'batch_sizes': [5, 10],
    'thresholds': [5, 10],
    'seeds': [42, 43, 44],
    'n_cycles': 10,
    'n_samples': 300,
    'n_initial': 22,
}


def run_ablation_study(
    output_dir: Path,
    datasets: List[str] = None,
    methods: List[str] = None,
    batch_sizes: List[int] = None,
    thresholds: List[float] = None,
    seeds: List[int] = None,
    n_cycles: int = 10,
    n_samples: int = 300,
    n_initial: int = 22,
    n_workers: int = 1,
    verbose: bool = True
) -> Dict:
    """
    Run complete ablation study.
    
    Args:
        output_dir: Directory to save results
        datasets: List of dataset names
        methods: List of acquisition method names
        batch_sizes: List of batch sizes
        thresholds: List of threshold percentages
        seeds: List of random seeds
        n_cycles: Number of acquisition cycles
        n_samples: Number of samples per dataset
        n_initial: Initial training samples
        n_workers: Number of parallel workers
        verbose: Whether to print progress
    
    Returns:
        Summary dictionary with results
    """
    # Use defaults if not specified
    datasets = datasets or DEFAULT_CONFIG['datasets']
    methods = methods or DEFAULT_CONFIG['methods']
    batch_sizes = batch_sizes or DEFAULT_CONFIG['batch_sizes']
    thresholds = thresholds or DEFAULT_CONFIG['thresholds']
    seeds = seeds or DEFAULT_CONFIG['seeds']
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate all configurations
    configs = []
    for dataset, method, batch_size, threshold, seed in product(
        datasets, methods, batch_sizes, thresholds, seeds
    ):
        config = ExperimentConfig(
            dataset_name=dataset,
            method_name=method,
            batch_size=batch_size,
            threshold_pct=threshold,
            seed=seed,
            n_cycles=n_cycles,
            n_samples=n_samples,
            n_initial=n_initial
        )
        configs.append(config)
    
    total = len(configs)
    if verbose:
        print(f"Running {total} experiments...")
        print(f"  Datasets: {datasets}")
        print(f"  Methods: {methods}")
        print(f"  Batch sizes: {batch_sizes}")
        print(f"  Thresholds: {thresholds}")
        print(f"  Seeds: {seeds}")
        print(f"  Output: {output_dir}")
    
    # Save study configuration
    study_config = {
        'datasets': datasets,
        'methods': methods,
        'batch_sizes': batch_sizes,
        'thresholds': thresholds,
        'seeds': seeds,
        'n_cycles': n_cycles,
        'n_samples': n_samples,
        'n_initial': n_initial,
        'timestamp': datetime.now().isoformat(),
    }
    with open(output_dir / 'study_config.json', 'w') as f:
        json.dump(study_config, f, indent=2)
    
    # Run experiments
    results = []
    completed = 0
    failed = 0
    
    def run_single(config):
        folder_name = (f"{config.dataset_name}_{config.method_name}_"
                      f"batch{config.batch_size}_thresh{int(config.threshold_pct)}_"
                      f"seed{config.seed}")
        exp_output_dir = output_dir / folder_name
        return run_experiment(config, output_dir=exp_output_dir, verbose=False)
    
    if n_workers > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(run_single, c): c for c in configs}
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                completed += 1
                if not result.success:
                    failed += 1
                if verbose and completed % 10 == 0:
                    print(f"  Progress: {completed}/{total} ({failed} failed)")
    else:
        for i, config in enumerate(configs):
            result = run_single(config)
            results.append(result)
            completed += 1
            if not result.success:
                failed += 1
            if verbose and (i + 1) % 10 == 0:
                print(f"  Progress: {completed}/{total} ({failed} failed)")
    
    # Generate summary
    summary = {
        'total_experiments': total,
        'completed': completed,
        'failed': failed,
        'success_rate': (completed - failed) / total if total > 0 else 0,
        'timestamp': datetime.now().isoformat(),
    }
    
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Write flat CSV: one row per (experiment, cycle)
    rows = []
    for result in results:
        if not result.success:
            continue
        cfg = result.config
        for cycle_result in result.cycles:
            row = {
                'dataset': cfg.dataset_name,
                'method': cfg.method_name,
                'batch_size': cfg.batch_size,
                'threshold_pct': cfg.threshold_pct,
                'seed': cfg.seed,
                'cycle': cycle_result.cycle,
            }
            row.update(cycle_result.metrics)
            rows.append(row)
    if rows:
        fieldnames = list(rows[0].keys())
        with open(output_dir / 'results.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    if verbose:
        print(f"\nCompleted: {completed}/{total} ({failed} failed)")
        print(f"Results saved to: {output_dir}")
    
    return summary


def main():
    parser = argparse.ArgumentParser(description='Run toy dataset ablation study')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: output/ablation_TIMESTAMP)')
    parser.add_argument('--datasets', nargs='+', default=None,
                       help='Dataset names')
    parser.add_argument('--methods', nargs='+', default=None,
                       help='Acquisition method names')
    parser.add_argument('--batch-sizes', nargs='+', type=int, default=None,
                       help='Batch sizes')
    parser.add_argument('--thresholds', nargs='+', type=float, default=None,
                       help='Threshold percentages')
    parser.add_argument('--seeds', nargs='+', type=int, default=None,
                       help='Random seeds')
    parser.add_argument('--n-cycles', type=int, default=10,
                       help='Number of acquisition cycles')
    parser.add_argument('--n-workers', type=int, default=1,
                       help='Number of parallel workers')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress output')
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path('output') / f'ablation_{timestamp}'
    else:
        output_dir = Path(args.output_dir)
    
    run_ablation_study(
        output_dir=output_dir,
        datasets=args.datasets,
        methods=args.methods,
        batch_sizes=args.batch_sizes,
        thresholds=args.thresholds,
        seeds=args.seeds,
        n_cycles=args.n_cycles,
        n_workers=args.n_workers,
        verbose=not args.quiet
    )


if __name__ == '__main__':
    main()
