#!/usr/bin/env python3
"""
Run a single experiment configuration.

Usage:
    python run_single_config.py \
        --dataset schmidt_2021_ifng \
        --feature_set string \
        --method probability_hit \
        --batch_size 100 \
        --seed 42
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def parse_args():
    parser = argparse.ArgumentParser(description="Run single GeneDisco experiment")
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name')
    parser.add_argument('--feature_set', type=str, required=True,
                        help='Feature set (achilles or string)')
    parser.add_argument('--method', type=str, required=True,
                        help='Acquisition method')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Batch size')
    parser.add_argument('--threshold', type=float, default=10.0,
                        help='Threshold percentage')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--n_cycles', type=int, default=10,
                        help='Number of cycles')
    parser.add_argument('--cache_dir', type=str, default='../../DiscoBAX/data',
                        help='Cache directory')
    parser.add_argument('--output_dir', type=str, default='./output/single',
                        help='Output directory')
    return parser.parse_args()


def main():
    args = parse_args()
    
    from reproducibility.genedisco.ablation.run_ablation import ExperimentConfig, run_single_experiment
    
    config = ExperimentConfig(
        dataset_name=args.dataset,
        feature_set_name=args.feature_set,
        method_name=args.method,
        batch_size=args.batch_size,
        threshold_pct=args.threshold,
        seed=args.seed,
        n_cycles=args.n_cycles
    )
    
    print(f"Running single experiment:")
    print(f"  Dataset: {config.dataset_name}")
    print(f"  Feature set: {config.feature_set_name}")
    print(f"  Method: {config.method_name}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Threshold: {config.threshold_pct}%")
    print(f"  Seed: {config.seed}")
    print(f"  Cycles: {config.n_cycles}")
    
    result = run_single_experiment(
        config,
        Path(args.output_dir),
        Path(args.cache_dir),
        verbose=True
    )
    
    if result['success']:
        print("\n✓ Experiment completed successfully!")
        if result.get('metrics'):
            print("Metrics:")
            for k, v in result['metrics'].items():
                print(f"  {k}: {v}")
    else:
        print("\n✗ Experiment failed!")
        if 'error' in result:
            print(f"  Error: {result['error']}")
        if 'stderr' in result:
            print(f"  Stderr: {result['stderr'][:500]}")


if __name__ == "__main__":
    main()
