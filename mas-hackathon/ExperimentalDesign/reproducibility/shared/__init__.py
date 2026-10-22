"""
Shared utilities for ActiveHit reproducibility studies.

This module provides common functionality for both toy and GeneDisco experiments:
- Dataset loading and generation
- Metric computation (single-target and multi-target)
- Acquisition functions
- Experiment orchestration (single-target and multi-target)
"""

from .datasets import Dataset, generate_dataset, list_datasets
from .metrics import (
    compute_metrics,
    compute_multitarget_metrics,
    aggregate_multitarget_metrics,
    MetricResults,
    MultiTargetMetricResults,
)
from .acquisition import (
    get_acquisition_function,
    list_acquisition_functions,
    ACQUISITION_FUNCTIONS,
    AcquisitionFunction,
    RandomAcquisition,
    ThompsonSamplingAcquisition,
    ProbabilityHitAcquisition,
    ThompsonHitAcquisition,
    TopKAcquisition,
)
from .experiment import run_experiment, ExperimentConfig
from .multitarget_experiment import (
    run_multitarget_experiment,
    MultiTargetExperimentConfig,
    MultiTargetExperimentResults,
    MultiTargetCycleResults,
)

__all__ = [
    # Datasets
    "Dataset",
    "generate_dataset",
    "list_datasets",

    # Metrics — single target
    "compute_metrics",
    "MetricResults",

    # Metrics — multi-target
    "compute_multitarget_metrics",
    "aggregate_multitarget_metrics",
    "MultiTargetMetricResults",

    # Acquisition functions
    "get_acquisition_function",
    "list_acquisition_functions",
    "ACQUISITION_FUNCTIONS",
    "AcquisitionFunction",
    "RandomAcquisition",
    "ThompsonSamplingAcquisition",
    "ProbabilityHitAcquisition",
    "ThompsonHitAcquisition",
    "TopKAcquisition",

    # Experiment — single target
    "run_experiment",
    "ExperimentConfig",

    # Experiment — multi-target (naive independent-GP baseline)
    "run_multitarget_experiment",
    "MultiTargetExperimentConfig",
    "MultiTargetExperimentResults",
    "MultiTargetCycleResults",
]

