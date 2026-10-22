"""
ActiveHit Reproducibility Studies
==================================

This package contains all code for reproducing the experiments in the ActiveHit paper.

Structure:
---------
- shared/     : Common utilities (datasets, metrics, acquisition functions)
- toy/        : Toy dataset experiments
  - ablation/ : Ablation studies on synthetic data
  - complexity/: Complexity analysis of synthetic data
- genedisco/  : GeneDisco (real biological) experiments
  - ablation/ : Ablation studies on GeneDisco datasets
  - complexity/: Complexity analysis of GeneDisco datasets

Quick Start:
-----------
# Run toy ablation study
from reproducibility.toy.ablation import run_ablation_study
run_ablation_study(output_dir='output/toy_ablation')

# Run GeneDisco ablation study
from reproducibility.genedisco.ablation import run_genedisco_ablation
run_genedisco_ablation(
    output_dir='output/genedisco_ablation',
    cache_dir='path/to/genedisco/data'
)

# Analyze dataset complexity
from reproducibility.toy.complexity import analyze_toy_complexity
from reproducibility.genedisco.complexity import analyze_genedisco_complexity
"""

__version__ = '1.0.0'
__author__ = 'Arash Mehrjou'

# Import subpackages for easy access
from . import shared
from . import toy
from . import genedisco
