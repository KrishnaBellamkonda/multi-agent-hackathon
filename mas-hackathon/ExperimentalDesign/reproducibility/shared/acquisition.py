"""
Acquisition methods for Bayesian optimization.

This module implements various acquisition functions for hit discovery,
including probability-based, Thompson sampling, and greedy methods.
"""

import numpy as np
from typing import List
from abc import ABC, abstractmethod
from scipy.stats import norm


class AcquisitionFunction(ABC):
    """Abstract base class for acquisition functions."""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def acquire(self, X_candidates: np.ndarray, 
                gp_mean: np.ndarray, gp_std: np.ndarray,
                batch_size: int, threshold: float,
                **kwargs) -> np.ndarray:
        """
        Select points to acquire.
        
        Args:
            X_candidates: Candidate points (n_candidates, n_features)
            gp_mean: GP posterior mean at candidates
            gp_std: GP posterior std at candidates
            batch_size: Number of points to select
            threshold: Hit threshold value
        
        Returns:
            Indices of selected points
        """
        pass


class RandomAcquisition(AcquisitionFunction):
    """Uniform random sampling baseline."""
    
    def __init__(self):
        super().__init__("random")
    
    def acquire(self, X_candidates: np.ndarray,
                gp_mean: np.ndarray, gp_std: np.ndarray,
                batch_size: int, threshold: float, **kwargs) -> np.ndarray:
        n_candidates = len(X_candidates)
        return np.random.choice(n_candidates, size=min(batch_size, n_candidates), 
                               replace=False)


class ProbabilityHitAcquisition(AcquisitionFunction):
    """
    Probability of exceeding threshold.
    Selects points with highest P(y > threshold) from GP posterior.
    """
    
    def __init__(self):
        super().__init__("probability_hit")
    
    def acquire(self, X_candidates: np.ndarray,
                gp_mean: np.ndarray, gp_std: np.ndarray,
                batch_size: int, threshold: float, **kwargs) -> np.ndarray:
        z = (threshold - gp_mean) / (gp_std + 1e-8)
        prob_hit = 1 - norm.cdf(z)
        n_select = min(batch_size, len(prob_hit))
        return np.argsort(prob_hit)[-n_select:][::-1]


class ThompsonHitAcquisition(AcquisitionFunction):
    """
    Thompson sampling with hit indicator.
    Draws ONE posterior sample, identifies predicted hits, selects among them.
    """
    
    def __init__(self):
        super().__init__("thompson_hit")
    
    def acquire(self, X_candidates: np.ndarray,
                gp_mean: np.ndarray, gp_std: np.ndarray,
                batch_size: int, threshold: float, **kwargs) -> np.ndarray:
        samples = gp_mean + gp_std * np.random.randn(len(gp_mean))
        predicted_hits = samples > threshold
        scores = predicted_hits.astype(float) + 0.01 * np.random.rand(len(samples))
        n_select = min(batch_size, len(scores))
        return np.argsort(scores)[-n_select:][::-1]


class ThompsonSamplingAcquisition(AcquisitionFunction):
    """
    Standard Thompson sampling.
    Samples from GP posterior and selects highest sampled values.
    """
    
    def __init__(self):
        super().__init__("thompson_sampling")
    
    def acquire(self, X_candidates: np.ndarray,
                gp_mean: np.ndarray, gp_std: np.ndarray,
                batch_size: int, threshold: float, **kwargs) -> np.ndarray:
        samples = gp_mean + gp_std * np.random.randn(len(gp_mean))
        n_select = min(batch_size, len(samples))
        return np.argsort(samples)[-n_select:][::-1]


class TopKAcquisition(AcquisitionFunction):
    """
    Greedy selection by predicted mean (pure exploitation).
    """
    
    def __init__(self):
        super().__init__("top_k")
    
    def acquire(self, X_candidates: np.ndarray,
                gp_mean: np.ndarray, gp_std: np.ndarray,
                batch_size: int, threshold: float, **kwargs) -> np.ndarray:
        n_select = min(batch_size, len(gp_mean))
        return np.argsort(gp_mean)[-n_select:][::-1]


# Acquisition function registry
ACQUISITION_FUNCTIONS = {
    "random": RandomAcquisition,
    "probability_hit": ProbabilityHitAcquisition,
    "thompson_hit": ThompsonHitAcquisition,
    "thompson_sampling": ThompsonSamplingAcquisition,
    "top_k": TopKAcquisition,
}


def get_acquisition_function(name: str) -> AcquisitionFunction:
    """Get acquisition function by name."""
    if name not in ACQUISITION_FUNCTIONS:
        raise ValueError(f"Unknown acquisition function: {name}. "
                        f"Available: {list(ACQUISITION_FUNCTIONS.keys())}")
    return ACQUISITION_FUNCTIONS[name]()


def list_acquisition_functions() -> List[str]:
    """Get list of all available acquisition function names."""
    return list(ACQUISITION_FUNCTIONS.keys())


if __name__ == "__main__":
    print("Testing acquisition functions...")
    n_candidates = 100
    X = np.random.randn(n_candidates, 2)
    mean = np.random.randn(n_candidates)
    std = np.abs(np.random.randn(n_candidates)) + 0.1
    threshold = np.percentile(mean, 90)
    
    for name in list_acquisition_functions():
        acq = get_acquisition_function(name)
        selected = acq.acquire(X, mean, std, batch_size=5, threshold=threshold)
        print(f"  {name}: selected = {selected}")
