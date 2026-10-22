"""
Single experiment runner for ablation study.

This module provides the core experiment logic for running a single
configuration of dataset, acquisition method, batch size, and threshold.
"""

import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import warnings

import torch
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.distributions import MultivariateNormal
from gpytorch.mlls import ExactMarginalLogLikelihood

from .datasets import Dataset, generate_dataset, set_seed
from .acquisition import get_acquisition_function, AcquisitionFunction
from .metrics import compute_metrics


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    dataset_name: str
    method_name: str
    batch_size: int
    threshold_pct: float
    seed: int
    n_cycles: int = 10
    n_initial: int = 22
    n_samples: int = 300
    noise_std: float = None
    gp_iterations: int = 100
    gp_lr: float = 0.1


@dataclass
class CycleResults:
    """Results from a single acquisition cycle."""
    cycle: int
    acquired_indices: np.ndarray
    cumulative_acquired: np.ndarray
    metrics: Dict[str, float]
    gp_mean: Optional[np.ndarray] = None
    gp_std: Optional[np.ndarray] = None


@dataclass
class ExperimentResults:
    """Complete results from an experiment."""
    config: ExperimentConfig
    cycles: List[CycleResults]
    final_metrics: Dict[str, float]
    success: bool
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'config': self.config.__dict__,
            'cycles': [
                {
                    'cycle': c.cycle,
                    'acquired_indices': c.acquired_indices.tolist(),
                    'cumulative_acquired': c.cumulative_acquired.tolist(),
                    'metrics': c.metrics
                }
                for c in self.cycles
            ],
            'final_metrics': self.final_metrics,
            'success': self.success,
            'error_message': self.error_message
        }


class ExactGPModel(ExactGP):
    """Exact Gaussian Process model using GPyTorch."""
    
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(RBFKernel())
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


def train_gp(X_train: np.ndarray, y_train: np.ndarray,
             n_iterations: int = 100, lr: float = 0.1) -> Tuple[ExactGPModel, GaussianLikelihood, Dict[str, float]]:
    """Train a Gaussian Process model."""
    y_mean = float(y_train.mean())
    y_std = float(y_train.std())
    if y_std < 1e-8:
        y_std = 1.0
    y_train_norm = (y_train - y_mean) / y_std
    
    train_x = torch.tensor(X_train, dtype=torch.float32)
    train_y = torch.tensor(y_train_norm, dtype=torch.float32)
    
    likelihood = GaussianLikelihood()
    model = ExactGPModel(train_x, train_y, likelihood)
    
    model.covar_module.base_kernel.lengthscale = 0.2
    model.covar_module.outputscale = 1.0
    likelihood.noise_covar.register_constraint('raw_noise', gpytorch.constraints.GreaterThan(1e-4))
    
    model.train()
    likelihood.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    mll = ExactMarginalLogLikelihood(likelihood, model)
    
    for _ in range(n_iterations):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()
    
    norm_params = {'y_mean': y_mean, 'y_std': y_std}
    return model, likelihood, norm_params


def predict_gp(model: ExactGPModel, likelihood: GaussianLikelihood,
               X_test: np.ndarray, norm_params: Optional[Dict[str, float]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Make predictions with a trained GP."""
    model.eval()
    likelihood.eval()
    
    test_x = torch.tensor(X_test, dtype=torch.float32)
    
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(test_x))
        mean = observed_pred.mean.numpy()
        std = observed_pred.stddev.numpy()
    
    if norm_params is not None:
        y_mean = norm_params['y_mean']
        y_std = norm_params['y_std']
        mean = mean * y_std + y_mean
        std = std * y_std
    
    return mean, std


def run_experiment(config: ExperimentConfig, 
                   output_dir: Optional[Path] = None,
                   verbose: bool = False) -> ExperimentResults:
    """
    Run a single ablation experiment.
    
    Args:
        config: Experiment configuration
        output_dir: Directory to save results
        verbose: Whether to print progress
    
    Returns:
        ExperimentResults object
    """
    set_seed(config.seed)
    torch.manual_seed(config.seed)
    
    try:
        if config.noise_std is not None:
            dataset = generate_dataset(
                config.dataset_name, n_samples=config.n_samples,
                noise_std=config.noise_std, seed=config.seed
            )
        else:
            dataset = generate_dataset(
                config.dataset_name, n_samples=config.n_samples, seed=config.seed
            )
        
        X, y = dataset.X, dataset.y
        n_samples = len(y)
        
        threshold = dataset.get_threshold_value(config.threshold_pct)
        true_hit_indices = dataset.get_hits(config.threshold_pct)
        
        acq_func = get_acquisition_function(config.method_name)
        
        all_indices = np.arange(n_samples)
        if config.n_initial > 0:
            acquired_indices = np.random.choice(n_samples, size=config.n_initial, replace=False)
        else:
            acquired_indices = np.array([], dtype=int)
        remaining_indices = np.setdiff1d(all_indices, acquired_indices)
        
        cycles = []
        
        for cycle in range(config.n_cycles):
            if verbose:
                print(f"  Cycle {cycle + 1}/{config.n_cycles}")
            
            if len(acquired_indices) == 0:
                new_acquired = np.random.choice(remaining_indices, 
                                               size=min(config.batch_size, len(remaining_indices)),
                                               replace=False)
                acquired_indices = new_acquired
                remaining_indices = np.setdiff1d(remaining_indices, new_acquired)
                
                metrics = compute_metrics(X, y, acquired_indices, true_hit_indices,
                                          y_pred=None, unseen_indices=remaining_indices)
                
                cycles.append(CycleResults(
                    cycle=cycle + 1, acquired_indices=new_acquired,
                    cumulative_acquired=acquired_indices.copy(), metrics=metrics
                ))
                continue
            
            X_train, y_train = X[acquired_indices], y[acquired_indices]
            model, likelihood, norm_params = train_gp(X_train, y_train, 
                                                       n_iterations=config.gp_iterations, lr=config.gp_lr)
            
            if len(remaining_indices) > 0:
                gp_mean, gp_std = predict_gp(model, likelihood, X[remaining_indices], norm_params)
                local_indices = acq_func.acquire(X[remaining_indices], gp_mean, gp_std,
                                                  batch_size=config.batch_size, threshold=threshold)
                new_acquired = remaining_indices[local_indices]
            else:
                new_acquired = np.array([], dtype=int)
            
            acquired_indices = np.concatenate([acquired_indices, new_acquired])
            remaining_indices = np.setdiff1d(remaining_indices, new_acquired)
            
            gp_mean_all, gp_std_all = predict_gp(model, likelihood, X, norm_params)
            metrics = compute_metrics(X, y, acquired_indices, true_hit_indices,
                                       y_pred=gp_mean_all, unseen_indices=remaining_indices)
            
            cycles.append(CycleResults(
                cycle=cycle, acquired_indices=new_acquired,
                cumulative_acquired=acquired_indices.copy(), metrics=metrics,
                gp_mean=gp_mean_all, gp_std=gp_std_all
            ))
        
        final_metrics = cycles[-1].metrics if cycles else {}
        results = ExperimentResults(config=config, cycles=cycles, 
                                     final_metrics=final_metrics, success=True)
        
    except Exception as e:
        warnings.warn(f"Experiment failed: {str(e)}")
        results = ExperimentResults(config=config, cycles=[], final_metrics={},
                                     success=False, error_message=str(e))
    
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "results.pickle", 'wb') as f:
            pickle.dump(results.to_dict(), f)
    
    return results


if __name__ == "__main__":
    print("Testing single experiment...")
    config = ExperimentConfig(
        dataset_name="sine", method_name="probability_hit",
        batch_size=5, threshold_pct=10, seed=42, n_cycles=3
    )
    results = run_experiment(config, verbose=True)
    print(f"\nSuccess: {results.success}")
    for name, value in results.final_metrics.items():
        print(f"  {name}: {value:.4f}")
