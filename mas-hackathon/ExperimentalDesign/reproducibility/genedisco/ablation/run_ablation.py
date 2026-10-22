"""
GeneDisco Ablation Study Runner.

Run acquisition method comparisons on real GeneDisco datasets.
"""

import os
import sys
import json
import pickle
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from itertools import product
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

# Study configuration
STUDY_CONFIG = {
    'datasets': [
        'schmidt_2021_ifng',
        'schmidt_2021_il2', 
        'zhuang_2019_nk',
        'sanchez_2021_tau',
        'zhu_2021_sarscov2',
    ],
    'feature_sets': ['achilles', 'string'],
    'methods': ['probability_hit', 'thompson_hit', 'thompson_sampling', 'top_k', 'random'],
    'batch_size': 200,
    'threshold_pct': 10.0,
    'n_cycles': 10,
    'seeds': [0, 1, 2, 3, 4],
}


@dataclass
class ExperimentConfig:
    """Configuration for a single GeneDisco experiment."""
    dataset_name: str
    feature_set_name: str
    method_name: str
    batch_size: int
    threshold_pct: float
    seed: int
    n_cycles: int


def generate_all_configs(config: Dict = None) -> List[ExperimentConfig]:
    """Generate all experiment configurations."""
    config = config or STUDY_CONFIG
    
    configs = []
    for dataset, features, method, seed in product(
        config['datasets'],
        config['feature_sets'],
        config['methods'],
        config['seeds']
    ):
        configs.append(ExperimentConfig(
            dataset_name=dataset,
            feature_set_name=features,
            method_name=method,
            batch_size=config['batch_size'],
            threshold_pct=config['threshold_pct'],
            seed=seed,
            n_cycles=config['n_cycles']
        ))
    return configs


def run_single_experiment(config: ExperimentConfig,
                          output_dir: Path,
                          cache_dir: Path,
                          verbose: bool = False) -> Dict[str, Any]:
    """Run a single GeneDisco experiment."""
    
    exp_name = (f"{config.dataset_name}_{config.feature_set_name}_{config.method_name}_"
                f"seed{config.seed}")
    exp_dir = output_dir / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = exp_dir / "results.pickle"
    if results_file.exists():
        if verbose:
            print(f"  [SKIP] {exp_name}")
        with open(results_file, 'rb') as f:
            return pickle.load(f)
    
    if verbose:
        print(f"  Running: {exp_name}")
    
    # Build command to run GeneDisco experiment
    cmd = [
        sys.executable, "-m", "activehit.apps.genedisco_experiment",
        "--dataset_name", config.dataset_name,
        "--feature_set_name", config.feature_set_name,
        "--acquisition_function", config.method_name,
        "--batch_size", str(config.batch_size),
        "--threshold_pct", str(config.threshold_pct),
        "--num_cycles", str(config.n_cycles),
        "--seed", str(config.seed),
        "--output_directory", str(exp_dir),
        "--cache_directory", str(cache_dir),
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        
        if result.returncode != 0:
            return {
                'success': False,
                'error': result.stderr,
                'config': asdict(config)
            }
        
        if results_file.exists():
            with open(results_file, 'rb') as f:
                return pickle.load(f)
        else:
            return {
                'success': True,
                'config': asdict(config),
                'note': 'Completed but no results file found'
            }
            
    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'error': 'Timeout after 1 hour',
            'config': asdict(config)
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'config': asdict(config)
        }


def run_genedisco_ablation(
    output_dir: Path,
    cache_dir: Path,
    datasets: List[str] = None,
    methods: List[str] = None,
    seeds: List[int] = None,
    n_workers: int = 1,
    verbose: bool = True
) -> Dict:
    """
    Run GeneDisco ablation study.
    
    Args:
        output_dir: Directory to save results
        cache_dir: Directory with GeneDisco data cache
        datasets: List of dataset names (default: all)
        methods: List of method names (default: all)
        seeds: List of random seeds (default: [0,1,2,3,4])
        n_workers: Number of parallel workers
        verbose: Whether to print progress
    
    Returns:
        Summary dictionary
    """
    output_dir = Path(output_dir)
    cache_dir = Path(cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build config
    config = STUDY_CONFIG.copy()
    if datasets:
        config['datasets'] = datasets
    if methods:
        config['methods'] = methods
    if seeds:
        config['seeds'] = seeds
    
    # Generate all configs
    all_configs = generate_all_configs(config)
    total = len(all_configs)
    
    if verbose:
        print(f"Running {total} experiments...")
        print(f"  Datasets: {config['datasets']}")
        print(f"  Methods: {config['methods']}")
        print(f"  Seeds: {config['seeds']}")
    
    # Save study config
    study_meta = {
        'config': config,
        'timestamp': datetime.now().isoformat(),
        'total_experiments': total,
    }
    with open(output_dir / 'study_config.json', 'w') as f:
        json.dump(study_meta, f, indent=2)
    
    # Run experiments
    results = []
    completed = 0
    failed = 0
    
    for exp_config in all_configs:
        result = run_single_experiment(exp_config, output_dir, cache_dir, verbose)
        results.append(result)
        completed += 1
        if not result.get('success', False):
            failed += 1
    
    # Summary
    summary = {
        'total': total,
        'completed': completed,
        'failed': failed,
        'success_rate': (completed - failed) / total if total > 0 else 0,
        'timestamp': datetime.now().isoformat(),
    }
    
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    if verbose:
        print(f"\nCompleted: {completed}/{total} ({failed} failed)")
        print(f"Results saved to: {output_dir}")
    
    return summary


def main():
    parser = argparse.ArgumentParser(description='Run GeneDisco ablation study')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory')
    parser.add_argument('--cache-dir', type=str, required=True,
                       help='GeneDisco data cache directory')
    parser.add_argument('--datasets', nargs='+', default=None,
                       help='Dataset names')
    parser.add_argument('--methods', nargs='+', default=None,
                       help='Method names')
    parser.add_argument('--seeds', nargs='+', type=int, default=None,
                       help='Random seeds')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress output')
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path('output') / f'genedisco_ablation_{timestamp}'
    else:
        output_dir = Path(args.output_dir)
    
    run_genedisco_ablation(
        output_dir=output_dir,
        cache_dir=Path(args.cache_dir),
        datasets=args.datasets,
        methods=args.methods,
        seeds=args.seeds,
        verbose=not args.quiet
    )


if __name__ == '__main__':
    main()
