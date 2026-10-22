#!/usr/bin/env python3
"""Runner for toy (synthetic dataset) experiments.

Runs one or more acquisition methods on synthetic datasets using a GP surrogate.
Default values target a single representative run; pass additional flags to sweep.

Usage:
    python runners/toy_experiment.py                          # default single run
    python runners/toy_experiment.py --quick                  # 1 seed, 3 cycles
    python runners/toy_experiment.py --datasets sine,branin   # specific datasets
    python runners/toy_experiment.py --methods probability_hit,thompson_hit
    python runners/toy_experiment.py --batch_sizes 5,10 --seeds 0,1,2  # sweep
    python runners/toy_experiment.py --dry-run                # print commands only
"""

import os
import sys
import argparse
import subprocess
import itertools
import time
from datetime import datetime
from pathlib import Path
import json

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving
import matplotlib.pyplot as plt
import pandas as pd


# ============================================================================
# DEFAULT CONFIGURATION  (single representative run — override via CLI)
# ============================================================================

# Core methods used in the paper.  Add extras from the list below as needed:
#   "discobax"    DiscoBAX (slow — BAX with subset_select objective)
#   "straddle"    Straddle heuristic (Bryan et al. 2006; Gotovos et al. 2013)
#   "truvar"      TruVaR thresholded bandit (Bogunovic et al. 2016)
#   "levelset_bax" Level-Set BAX
#   "eitau"       Expected Improvement above Threshold
METHODS = [
    "probability_hit",
    "thompson_hit",
    "thompson_sampling",
    "top_k",
    "random",
]

# Datasets available: sine, sine2d, branin, perturbation_sem, perturbation_pathways
DATASETS = ["sine"]

# Single batch size for a default run; pass "--batch_sizes 5,10,15" to sweep
BATCH_SIZES = [10]

# Hit threshold: top X% of pool are hits
HIT_THRESHOLDS = [10]

# Fixed parameters
TOTAL_POINTS = 300  # Pool size
NUM_CYCLES = 10
REPORT_EVERY = 5
SEEDS = [0]  # Add more seeds (e.g. 0,1,2,3,4) for statistical robustness

# Model
MODEL_NAME = "gp"


def parse_args():
    parser = argparse.ArgumentParser(description="Run toy experiment ablation study")
    parser.add_argument("--methods", type=str, default=",".join(METHODS),
                        help=f"Comma-separated methods. Default: {','.join(METHODS)}")
    parser.add_argument("--datasets", type=str, default=",".join(DATASETS),
                        help=f"Comma-separated datasets. Default: {','.join(DATASETS)}")
    parser.add_argument("--batch_sizes", type=str, default=",".join(map(str, BATCH_SIZES)),
                        help=f"Comma-separated batch sizes. Default: {','.join(map(str, BATCH_SIZES))}")
    parser.add_argument("--hit_thresholds", type=str, default=",".join(map(str, HIT_THRESHOLDS)),
                        help=f"Comma-separated hit threshold percentiles (top X%%). Default: {','.join(map(str, HIT_THRESHOLDS))}")
    parser.add_argument("--seeds", type=str, default=",".join(map(str, SEEDS)),
                        help=f"Comma-separated seeds. Default: {','.join(map(str, SEEDS))}")
    parser.add_argument("--num_cycles", type=int, default=NUM_CYCLES,
                        help=f"Number of cycles. Default: {NUM_CYCLES}")
    parser.add_argument("--total_points", type=int, default=TOTAL_POINTS,
                        help=f"Total dataset points. Default: {TOTAL_POINTS}")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory. Default: ./ablation_toy_<timestamp>")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test mode: 1 seed, 3 cycles, first dataset/batch only")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without running")
    parser.add_argument("--parallel", type=int, default=1,
                        help="Number of parallel processes (default: 1 = sequential)")
    return parser.parse_args()


def run_single_experiment(
    dataset: str,
    method: str,
    batch_size: int,
    seed: int,
    num_cycles: int,
    total_points: int,
    output_dir: str,
    hit_threshold_pct: int = 20,
    dry_run: bool = False,
) -> dict:
    """Run a single experiment configuration."""
    
    # Create output subdirectory
    exp_name = f"{dataset}_batch{batch_size}_{method}_seed{seed}_thresh{hit_threshold_pct}"
    exp_output_dir = os.path.join(output_dir, dataset, f"batch{batch_size}", f"thresh{hit_threshold_pct}", method, f"seed{seed}")
    fig_dir = os.path.join(exp_output_dir, "figures")
    
    os.makedirs(exp_output_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    
    # Build command
    cmd = [
        sys.executable,
        "activehit/apps/toy_experiment.py",
        f"--dataset_name={dataset}",
        f"--acquisition_function_name={method}",
        f"--acquisition_batch_size={batch_size}",
        f"--num_active_learning_cycles={num_cycles}",
        f"--num_samples={total_points}",
        f"--seed={seed}",
        f"--output_directory={exp_output_dir}",
        f"--save_fig_dir={fig_dir}",
        f"--model_name={MODEL_NAME}",
        f"--cache_directory=./data",
        f"--hit_threshold_percentile={hit_threshold_pct}",
    ]
    
    result = {
        "dataset": dataset,
        "method": method,
        "batch_size": batch_size,
        "hit_threshold_pct": hit_threshold_pct,
        "seed": seed,
        "num_cycles": num_cycles,
        "output_dir": exp_output_dir,
        "command": " ".join(cmd),
        "status": "pending",
        "runtime_seconds": None,  # Will be filled in after experiment
    }
    
    if dry_run:
        print(f"[DRY-RUN] {' '.join(cmd)}")
        result["status"] = "dry-run"
        return result
    
    print(f"\n{'='*60}")
    print(f"Running: {exp_name}")
    print(f"  Dataset: {dataset}, Method: {method}, Batch: {batch_size}, Seed: {seed}")
    print(f"  Output: {exp_output_dir}")
    print(f"{'='*60}")
    
    # Track runtime for each experiment
    start_time = time.time()
    
    try:
        proc = subprocess.run(
            cmd,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout per experiment
        )
        
        # Record runtime
        end_time = time.time()
        runtime = end_time - start_time
        result["runtime_seconds"] = runtime
        
        # Save logs
        with open(os.path.join(exp_output_dir, "stdout.log"), "w") as f:
            f.write(proc.stdout)
        with open(os.path.join(exp_output_dir, "stderr.log"), "w") as f:
            f.write(proc.stderr)
        
        if proc.returncode == 0:
            result["status"] = "success"
            print(f"  ✓ Completed successfully in {runtime:.1f}s")
        else:
            result["status"] = "failed"
            result["error"] = proc.stderr[-500:] if proc.stderr else "Unknown error"
            print(f"  ✗ Failed (return code {proc.returncode}) after {runtime:.1f}s")
            if proc.stderr:
                print(f"    Error: {proc.stderr[-200:]}")
                
    except subprocess.TimeoutExpired:
        end_time = time.time()
        result["runtime_seconds"] = end_time - start_time
        result["status"] = "timeout"
        print(f"  ✗ Timeout after 1 hour")
    except Exception as e:
        end_time = time.time()
        result["runtime_seconds"] = end_time - start_time
        result["status"] = "error"
        result["error"] = str(e)
        print(f"  ✗ Error: {e}")
    
    return result


def load_metric_from_csv(csv_path: str, metric: str = "RecoveredHitsRatio_Cumulative_mean") -> float:
    """Load a specific metric from a summary CSV file."""
    try:
        import csv
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if metric in row:
                    return float(row[metric])
    except Exception:
        pass
    return float('nan')


def load_toy_experiment_metrics(exp_output_dir: str, method: str, num_cycles: int = 10, report_every: int = 5) -> dict:
    """Load metrics from toy experiment CSV files (per-cycle summaries).
    
    Uses summary_cycle_N_agg.csv files which contain RecoveredHitsRatio_Cumulative,
    the correct "hits recovered" metric that matches the tables.
    
    Returns dict with:
    - hits_ratio_per_cycle: list of RecoveredHitsRatio_Cumulative values per reporting cycle
    - final_hits_ratio: final RecoveredHitsRatio_Cumulative value
    - cycles: list of cycle numbers where metrics were recorded
    """
    import csv
    import glob
    
    hits_ratio = []
    cycles = []
    
    # Load from per-cycle CSVs (e.g., summary_cycle_5_agg.csv, summary_cycle_10_agg.csv)
    for cycle in range(report_every, num_cycles + 1, report_every):
        csv_path = os.path.join(exp_output_dir, f"summary_cycle_{cycle}_agg.csv")
        if os.path.exists(csv_path):
            try:
                with open(csv_path, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if 'RecoveredHitsRatio_Cumulative_mean' in row:
                            val = float(row['RecoveredHitsRatio_Cumulative_mean'])
                            hits_ratio.append(val)
                            cycles.append(cycle)
                            break
            except Exception as e:
                pass
    
    # Also try loading final value from summary_agg.csv
    final_csv = os.path.join(exp_output_dir, "summary_agg.csv")
    final_hits = None
    if os.path.exists(final_csv):
        try:
            with open(final_csv, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if 'RecoveredHitsRatio_Cumulative_mean' in row:
                        final_hits = float(row['RecoveredHitsRatio_Cumulative_mean'])
                        break
        except:
            pass
    
    if not hits_ratio and final_hits is None:
        return None
    
    return {
        'hits_ratio_per_cycle': hits_ratio,
        'cycles': cycles,
        'final_hits_ratio': final_hits if final_hits is not None else (hits_ratio[-1] if hits_ratio else None),
    }


def generate_metric_tables(results: list, output_dir: str):
    """Generate metric comparison tables from experiment results."""
    import csv
    from collections import defaultdict
    
    # Collect metrics from successful experiments
    metrics_data = defaultdict(lambda: defaultdict(list))
    
    for r in results:
        if r["status"] != "success":
            continue
        
        # Find summary_agg.csv
        exp_dir = r.get("output_dir", "")
        csv_path = os.path.join(exp_dir, "summary_agg.csv")
        
        if not os.path.exists(csv_path):
            continue
        
        dataset = r["dataset"]
        method = r["method"]
        batch_size = r["batch_size"]
        
        # Load RecoveredHitsRatio_Cumulative
        hit_ratio = load_metric_from_csv(csv_path, "RecoveredHitsRatio_Cumulative_mean")
        if not np.isnan(hit_ratio):
            metrics_data[(dataset, batch_size)][method].append(hit_ratio)
    
    if not metrics_data:
        print("No metric data found for tables.")
        return
    
    # Generate tables per dataset
    datasets = sorted(set(k[0] for k in metrics_data.keys()))
    batch_sizes = sorted(set(k[1] for k in metrics_data.keys()))
    methods = sorted(set(m for data in metrics_data.values() for m in data.keys()))
    
    table_path = os.path.join(output_dir, "metric_tables.md")
    latex_path = os.path.join(output_dir, "metric_tables.tex")
    
    with open(table_path, 'w') as f:
        f.write("# Metric Comparison Tables\n\n")
        f.write("## RecoveredHitsRatio (Cumulative) - Mean ± Std\n\n")
        
        for dataset in datasets:
            f.write(f"### {dataset}\n\n")
            f.write("| Method | " + " | ".join(f"Batch {b}" for b in batch_sizes) + " |\n")
            f.write("|" + "--------|" * (len(batch_sizes) + 1) + "\n")
            
            for method in methods:
                row = [method]
                for batch_size in batch_sizes:
                    values = metrics_data.get((dataset, batch_size), {}).get(method, [])
                    if values:
                        mean = np.mean(values)
                        std = np.std(values) if len(values) > 1 else 0
                        row.append(f"{mean:.3f} ± {std:.3f}")
                    else:
                        row.append("-")
                f.write("| " + " | ".join(row) + " |\n")
            f.write("\n")
    
    # Generate LaTeX version
    with open(latex_path, 'w') as f:
        f.write("% RecoveredHitsRatio (Cumulative) Tables\n\n")
        for dataset in datasets:
            f.write(f"% Dataset: {dataset}\n")
            f.write("\\begin{table}[h]\n\\centering\n")
            f.write("\\caption{Hit Recovery Ratio on " + dataset + "}\n")
            f.write("\\begin{tabular}{l" + "c" * len(batch_sizes) + "}\n")
            f.write("\\toprule\n")
            f.write("Method & " + " & ".join(f"Batch {b}" for b in batch_sizes) + " \\\\\n")
            f.write("\\midrule\n")
            
            for method in methods:
                row = [method.replace("_", "\\_")]
                for batch_size in batch_sizes:
                    values = metrics_data.get((dataset, batch_size), {}).get(method, [])
                    if values:
                        mean = np.mean(values)
                        std = np.std(values) if len(values) > 1 else 0
                        row.append(f"{mean:.3f} $\\pm$ {std:.3f}")
                    else:
                        row.append("-")
                f.write(" & ".join(row) + " \\\\\n")
            
            f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n\n")
    
    print(f"  Metric tables saved to: {table_path}")
    print(f"  LaTeX tables saved to: {latex_path}")


def generate_learning_curves(results: list, output_dir: str, num_cycles: int = 10):
    """Generate learning curve plots showing hit recovery over cycles."""
    from collections import defaultdict
    
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Method colors and styles - including DiscoBAX
    METHOD_COLORS = {
        'thompson_hit': '#E63946',
        'thompson_sampling': '#457B9D', 
        'top_k': '#2A9D8F',
        'probability_hit': '#F4A261',
        'random': '#6C757D',
        'discobax': '#9B59B6',  # Purple for DiscoBAX
        'straddle': '#1ABC9C',  # Teal for Straddle
        'truvar': '#E74C3C',  # Red-orange for TruVaR
        'levelset_bax': '#3498DB',  # Blue for Level-Set BAX
        'eitau': '#8E44AD',  # Purple for EI-tau
    }
    
    METHOD_NAMES = {
        'thompson_hit': 'Thompson Hit',
        'thompson_sampling': 'Thompson Sampling',
        'top_k': 'Top-K',
        'probability_hit': 'P(Hit)',
        'random': 'Random',
        'discobax': 'DiscoBAX',
        'straddle': 'Straddle',
        'truvar': 'TruVaR',
        'levelset_bax': 'Level-Set BAX',
        'eitau': 'EI-τ',
    }
    
    # Collect per-cycle metrics from CSV files
    cycle_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    final_metrics = defaultdict(lambda: defaultdict(list))  # For bar charts
    runtime_data = defaultdict(list)  # For runtime comparison
    
    # Determine report_every from num_cycles (typically 5 for 10 cycles)
    report_every = 5  # Default, matches toy_experiment.py
    
    for r in results:
        if r["status"] != "success":
            continue
        
        dataset = r["dataset"]
        method = r["method"]
        batch_size = r["batch_size"]
        exp_dir = r.get("output_dir", "")
        runtime = r.get("runtime_seconds", 0)
        
        # Collect runtime
        runtime_data[method].append(runtime)
        
        # Load metrics from CSV files (correct RecoveredHitsRatio metric)
        metrics = load_toy_experiment_metrics(exp_dir, method, num_cycles, report_every)
        if metrics:
            # Store per-cycle data using actual cycle numbers
            for cycle, ratio in zip(metrics.get('cycles', []), metrics.get('hits_ratio_per_cycle', [])):
                cycle_data[(dataset, batch_size)][method][cycle].append(ratio)
            
            # Store final metrics
            if metrics.get('final_hits_ratio') is not None:
                final_metrics[(dataset, batch_size, method)]['hits_ratio'].append(metrics['final_hits_ratio'])
    
    if not cycle_data:
        print("No cycle data found for plots.")
        return
    
    # Generate plots per dataset
    datasets = sorted(set(k[0] for k in cycle_data.keys()))
    batch_sizes = sorted(set(k[1] for k in cycle_data.keys()))
    methods = sorted(set(m for data in cycle_data.values() for m in data.keys()))
    
    # 1. Learning curves plot
    for dataset in datasets:
        fig, axes = plt.subplots(1, len(batch_sizes), figsize=(5*len(batch_sizes), 4), sharey=True)
        if len(batch_sizes) == 1:
            axes = [axes]
        
        for ax_idx, batch_size in enumerate(batch_sizes):
            ax = axes[ax_idx]
            
            for method in methods:
                cycles_available = sorted(cycle_data.get((dataset, batch_size), {}).get(method, {}).keys())
                if not cycles_available:
                    continue
                
                means = []
                stds = []
                for cycle in cycles_available:
                    values = cycle_data[(dataset, batch_size)][method][cycle]
                    means.append(np.mean(values))
                    stds.append(np.std(values) if len(values) > 1 else 0)
                
                color = METHOD_COLORS.get(method, '#333333')
                label = METHOD_NAMES.get(method, method)
                ax.errorbar(cycles_available, means, yerr=stds, 
                           label=label, color=color, marker='o', capsize=3, linewidth=2)
            
            ax.set_xlabel('Cycle')
            ax.set_title(f'Batch Size = {batch_size}')
            if ax_idx == 0:
                ax.set_ylabel('Hits Recovered Ratio')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1.05)
        
        # Add legend to last subplot
        axes[-1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        fig.suptitle(f'Hit Recovery on {dataset}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        plot_path = os.path.join(plots_dir, f"learning_curves_{dataset}.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {plot_path}")
    
    # 2. Generate bar chart comparing methods across all datasets
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(datasets))
    width = 0.12
    
    for i, method in enumerate(methods):
        means = []
        stds = []
        for dataset in datasets:
            # Average across batch sizes for final cycle
            all_values = []
            for batch_size in batch_sizes:
                max_cycle = max(cycle_data.get((dataset, batch_size), {}).get(method, {}).keys(), default=0)
                if max_cycle:
                    values = cycle_data[(dataset, batch_size)][method][max_cycle]
                    all_values.extend(values)
            if all_values:
                means.append(np.mean(all_values))
                stds.append(np.std(all_values) if len(all_values) > 1 else 0)
            else:
                means.append(0)
                stds.append(0)
        
        color = METHOD_COLORS.get(method, '#333333')
        label = METHOD_NAMES.get(method, method)
        ax.bar(x + i*width - width*len(methods)/2, means, width, 
               label=label, color=color, yerr=stds, capsize=2)
    
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Hits Recovered Ratio (Final)')
    ax.set_title('Method Comparison Across Datasets')
    ax.set_xticks(x)
    ax.set_xticklabels([d.replace('_', '\n') for d in datasets])
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')
    
    bar_path = os.path.join(plots_dir, "method_comparison_bar.png")
    plt.savefig(bar_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {bar_path}")
    
    # 3. Generate PERFORMANCE vs RUNTIME scatter plot (key plot for paper)
    if runtime_data:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for method in methods:
            # Get mean performance across all datasets
            all_perf = []
            for dataset in datasets:
                for batch_size in batch_sizes:
                    max_cycle = max(cycle_data.get((dataset, batch_size), {}).get(method, {}).keys(), default=0)
                    if max_cycle:
                        all_perf.extend(cycle_data[(dataset, batch_size)][method][max_cycle])
            
            mean_perf = np.mean(all_perf) if all_perf else 0
            mean_runtime = np.mean(runtime_data[method]) if runtime_data[method] else 0
            
            color = METHOD_COLORS.get(method, '#333333')
            label = METHOD_NAMES.get(method, method)
            
            ax.scatter(mean_runtime, mean_perf, s=200, c=color, label=label, edgecolors='black', linewidths=1.5)
            ax.annotate(label, (mean_runtime, mean_perf), textcoords="offset points", 
                       xytext=(5, 5), fontsize=10)
        
        ax.set_xlabel('Mean Runtime (seconds)', fontsize=12)
        ax.set_ylabel('Hits Recovered Ratio', fontsize=12)
        ax.set_title('Performance vs Runtime: DiscoBAX is Slow Without Performance Gain', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.1)
        
        scatter_path = os.path.join(plots_dir, "performance_vs_runtime.png")
        plt.savefig(scatter_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {scatter_path}")
        
        # 4. Generate efficiency bar chart (performance / runtime)
        fig, ax = plt.subplots(figsize=(10, 6))
        
        efficiencies = []
        for method in methods:
            all_perf = []
            for dataset in datasets:
                for batch_size in batch_sizes:
                    max_cycle = max(cycle_data.get((dataset, batch_size), {}).get(method, {}).keys(), default=0)
                    if max_cycle:
                        all_perf.extend(cycle_data[(dataset, batch_size)][method][max_cycle])
            
            mean_perf = np.mean(all_perf) if all_perf else 0
            mean_runtime = np.mean(runtime_data[method]) if runtime_data[method] else 1
            efficiency = mean_perf / mean_runtime  # Performance per second
            efficiencies.append((method, efficiency, mean_perf, mean_runtime))
        
        # Sort by efficiency
        efficiencies.sort(key=lambda x: -x[1])
        
        methods_sorted = [e[0] for e in efficiencies]
        eff_values = [e[1] for e in efficiencies]
        colors = [METHOD_COLORS.get(m, '#333333') for m in methods_sorted]
        labels = [METHOD_NAMES.get(m, m) for m in methods_sorted]
        
        bars = ax.bar(range(len(methods_sorted)), eff_values, color=colors)
        ax.set_xticks(range(len(methods_sorted)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel('Efficiency (Hits Ratio / Runtime)', fontsize=12)
        ax.set_title('Method Efficiency: Our Methods >> DiscoBAX', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, (method, eff, perf, runtime) in zip(bars, efficiencies):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                   f'{eff:.3f}', ha='center', va='bottom', fontsize=9)
        
        eff_path = os.path.join(plots_dir, "method_efficiency.png")
        plt.savefig(eff_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {eff_path}")


def generate_summary_report(results: list, output_dir: str, report_every: int = 5, num_cycles: int = 10):
    """Generate a summary report of all experiments."""
    
    report_path = os.path.join(output_dir, "ablation_summary.md")
    json_path = os.path.join(output_dir, "ablation_results.json")
    
    # Save raw results as JSON
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    
    # Count statistics
    total = len(results)
    success = sum(1 for r in results if r["status"] == "success")
    failed = sum(1 for r in results if r["status"] == "failed")
    
    # Generate markdown report
    with open(report_path, "w") as f:
        f.write("# Toy Experiment Ablation Study Results\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"## Summary\n\n")
        f.write(f"- Total experiments: {total}\n")
        f.write(f"- Successful: {success}\n")
        f.write(f"- Failed: {failed}\n")
        f.write(f"- Report every {report_every} cycles\n\n")
        
        f.write("## Configuration\n\n")
        datasets_used = sorted(set(r["dataset"] for r in results))
        methods_used = sorted(set(r["method"] for r in results))
        batches_used = sorted(set(r["batch_size"] for r in results))
        seeds_used = sorted(set(r["seed"] for r in results))
        
        f.write(f"- **Datasets**: {', '.join(datasets_used)}\n")
        f.write(f"- **Methods**: {', '.join(methods_used)}\n")
        f.write(f"- **Batch sizes**: {', '.join(map(str, batches_used))}\n")
        f.write(f"- **Seeds**: {', '.join(map(str, seeds_used))}\n\n")
        
        f.write("## Results by Dataset\n\n")
        for dataset in datasets_used:
            f.write(f"### {dataset}\n\n")
            ds_results = [r for r in results if r["dataset"] == dataset]
            
            # Create a table
            f.write("| Method | Batch Size | Seed | Status |\n")
            f.write("|--------|------------|------|--------|\n")
            for r in sorted(ds_results, key=lambda x: (x["method"], x["batch_size"], x["seed"])):
                status_icon = "✓" if r["status"] == "success" else "✗"
                f.write(f"| {r['method']} | {r['batch_size']} | {r['seed']} | {status_icon} {r['status']} |\n")
            f.write("\n")
        
        if failed > 0:
            f.write("## Failed Experiments\n\n")
            for r in results:
                if r["status"] != "success":
                    f.write(f"- **{r['dataset']}** / {r['method']} / batch={r['batch_size']} / seed={r['seed']}\n")
                    if "error" in r:
                        f.write(f"  - Error: {r['error'][:200]}\n")
            f.write("\n")
        
        # Add runtime summary section
        f.write("## Runtime Analysis\n\n")
        f.write("Average runtime per method (seconds):\n\n")
        f.write("| Method | Mean Runtime (s) | Std (s) | Min (s) | Max (s) | N |\n")
        f.write("|--------|------------------|---------|---------|---------|---|\n")
        
        from collections import defaultdict
        runtime_by_method = defaultdict(list)
        for r in results:
            if r["status"] == "success" and r.get("runtime_seconds") is not None:
                runtime_by_method[r["method"]].append(r["runtime_seconds"])
        
        # Sort by mean runtime (slowest first) to highlight discobax
        method_stats = []
        for method, runtimes in runtime_by_method.items():
            if runtimes:
                method_stats.append((method, np.mean(runtimes), np.std(runtimes), min(runtimes), max(runtimes), len(runtimes)))
        
        for method, mean_rt, std_rt, min_rt, max_rt, n in sorted(method_stats, key=lambda x: -x[1]):
            f.write(f"| {method} | {mean_rt:.1f} | {std_rt:.1f} | {min_rt:.1f} | {max_rt:.1f} | {n} |\n")
        f.write("\n")
        
        # Add speedup comparison vs discobax if present
        if "discobax" in runtime_by_method and runtime_by_method["discobax"]:
            discobax_mean = np.mean(runtime_by_method["discobax"])
            f.write("### Speedup vs DiscoBAX\n\n")
            f.write("| Method | Speedup Factor |\n")
            f.write("|--------|----------------|\n")
            for method, mean_rt, _, _, _, _ in sorted(method_stats, key=lambda x: x[1]):
                if method != "discobax" and mean_rt > 0:
                    speedup = discobax_mean / mean_rt
                    f.write(f"| {method} | {speedup:.1f}x faster |\n")
            f.write("\n")
    
    print(f"\n{'='*60}")
    print(f"Ablation study complete!")
    print(f"  Summary report: {report_path}")
    print(f"  JSON results: {json_path}")
    print(f"  Success rate: {success}/{total} ({100*success/total:.1f}%)")
    print(f"{'='*60}")
    
    # Generate tables and plots
    if success > 0:
        try:
            generate_metric_tables(results, output_dir)
            generate_learning_curves(results, output_dir, num_cycles)
            print(f"  Generated tables and plots in: {output_dir}")
        except Exception as e:
            print(f"Warning: Could not generate tables/plots: {e}")
            import traceback
            traceback.print_exc()


def main():
    args = parse_args()
    
    # Parse arguments
    methods = [m.strip() for m in args.methods.split(",")]
    datasets = [d.strip() for d in args.datasets.split(",")]
    batch_sizes = [int(b.strip()) for b in args.batch_sizes.split(",")]
    hit_thresholds = [int(t.strip()) for t in args.hit_thresholds.split(",")]
    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    
    # Quick mode overrides
    if args.quick:
        seeds = [0]
        args.num_cycles = 3
        datasets = datasets[:1]
        batch_sizes = batch_sizes[:1]
        hit_thresholds = [20]  # Only use default threshold
        print("[QUICK MODE] Running with reduced configuration")
    
    # Output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"./ablation_toy_{timestamp}"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate total experiments
    total_experiments = len(datasets) * len(methods) * len(batch_sizes) * len(hit_thresholds) * len(seeds)
    
    print(f"\n{'='*60}")
    print(f"TOY EXPERIMENT ABLATION STUDY")
    print(f"{'='*60}")
    print(f"Datasets ({len(datasets)}): {', '.join(datasets)}")
    print(f"Methods ({len(methods)}): {', '.join(methods)}")
    print(f"Batch sizes ({len(batch_sizes)}): {', '.join(map(str, batch_sizes))}")
    print(f"Hit thresholds ({len(hit_thresholds)}): {', '.join(map(str, hit_thresholds))}%")
    print(f"Seeds ({len(seeds)}): {', '.join(map(str, seeds))}")
    print(f"Cycles: {args.num_cycles}, Report every: {REPORT_EVERY}")
    print(f"Total points per run: {args.total_points}")
    print(f"Total experiments: {total_experiments}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}")
    
    if args.dry_run:
        print("\n[DRY-RUN MODE] Commands will be printed but not executed.\n")
    
    # Save configuration
    config = {
        "datasets": datasets,
        "methods": methods,
        "batch_sizes": batch_sizes,
        "hit_thresholds": hit_thresholds,
        "seeds": seeds,
        "num_cycles": args.num_cycles,
        "total_points": args.total_points,
        "report_every": REPORT_EVERY,
        "model_name": MODEL_NAME,
        "timestamp": datetime.now().isoformat(),
    }
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    # Run experiments
    results = []
    completed = 0
    
    # Iterate over all combinations (including threshold)
    for dataset, batch_size, hit_thresh, method, seed in itertools.product(
        datasets, batch_sizes, hit_thresholds, methods, seeds
    ):
        completed += 1
        print(f"\n[{completed}/{total_experiments}] ", end="")
        
        result = run_single_experiment(
            dataset=dataset,
            method=method,
            batch_size=batch_size,
            seed=seed,
            num_cycles=args.num_cycles,
            total_points=args.total_points,
            output_dir=output_dir,
            hit_threshold_pct=hit_thresh,
            dry_run=args.dry_run,
        )
        results.append(result)
    
    # Generate summary report
    if not args.dry_run:
        generate_summary_report(results, output_dir, REPORT_EVERY)
    else:
        print(f"\n[DRY-RUN] Would run {total_experiments} experiments")


if __name__ == "__main__":
    main()
