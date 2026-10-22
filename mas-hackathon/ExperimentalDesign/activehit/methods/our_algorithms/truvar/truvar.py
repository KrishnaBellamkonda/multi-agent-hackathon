"""
TruVaR-style Thresholded Bandit Acquisition
--------------------------------------------

Implements a truncated-variance-reduction acquisition function inspired by:
  Bogunovic, Scarlett, Krause, Cevher,
  "Truncated Variance Reduction: A Unified Approach to Bayesian Optimization
  and Level-Set Estimation" (NeurIPS 2016)

The key idea: only invest sampling budget on points whose classification
(above/below threshold τ) remains *uncertain*.  A point is "classified"
with confidence when |μ_n(x) - τ| > β·σ_n(x), meaning the posterior
mean is far enough from the threshold relative to the posterior uncertainty.

For the remaining uncertain points, select those with the *highest posterior
variance* (i.e., where reducing uncertainty has the most value for correctly
classifying the point relative to τ).

Acquisition score:

    a(x) = σ_n(x)   if |μ_n(x) - τ| < β·σ_n(x)   (uncertain about classification)
    a(x) = 0         otherwise                        (already classified)

This focuses sampling on the "ambiguity band" around the threshold,
which is precisely the behaviour of the Truncated Variance Reduction
algorithm (TruVaR).
"""

from typing import AnyStr, List, Optional
import numpy as np
from slingpy import AbstractDataSource
from slingpy.models.abstract_base_model import AbstractBaseModel
from genedisco.active_learning_methods.acquisition_functions.base_acquisition_function import (
    BaseBatchAcquisitionFunction,
)


class TruVaRAcquisition(BaseBatchAcquisitionFunction):
    """TruVaR-style thresholded bandit acquisition.

    Parameters
    ----------
    beta : float
        Classification confidence parameter.  A point is considered
        "classified" when |μ(x) - τ| > β·σ(x).  Default 1.96.
    """

    def __init__(self, beta: float = 1.96):
        self.beta = beta

    def _get_posterior_mean_and_std(self, model, X_avail):
        """Extract posterior mean and std from model predictions."""
        mu_n = None
        sigma_n = None

        # Path 1: GeneDisco models
        if hasattr(model, "get_model_prediction"):
            try:
                num_mc_samples = 50
                pred = model.get_model_prediction(
                    X_avail, return_multiple_preds=True, num_target_samples=num_mc_samples
                )
                if isinstance(pred, (tuple, list)) and len(pred) > 0:
                    samples = pred[0]
                    if hasattr(samples, "detach"):
                        samples = samples.detach().cpu().numpy()
                    else:
                        samples = np.asarray(samples)
                    if samples.ndim == 1:
                        samples = samples.reshape(-1, 1)
                    elif samples.shape[0] == num_mc_samples:
                        samples = samples.T
                    mu_n = samples.mean(axis=1).flatten()
                    sigma_n = np.sqrt(np.clip(samples.var(axis=1).flatten(), 1e-10, None))
            except Exception:
                pass

        # Path 2: Toy GP models
        if mu_n is None:
            try:
                X_for_model = X_avail.get_data()[0] if hasattr(X_avail, "get_data") else np.asarray(X_avail)
                pred = model.predict(X_for_model, return_std_and_margin=True)
                if isinstance(pred, (tuple, list)) and len(pred) >= 2:
                    mu_n = np.asarray(pred[0]).reshape(-1)
                    second = np.asarray(pred[1])
                    if second.ndim == 0:
                        sigma_n = np.full(len(mu_n), np.sqrt(max(float(second), 1e-10)))
                    elif second.ndim == 1:
                        sigma_n = np.sqrt(np.clip(second.reshape(-1), 1e-10, None))
                    elif second.ndim == 2:
                        sigma_n = np.sqrt(np.clip(np.diag(second), 1e-10, None))
                    else:
                        sigma_n = np.ones_like(mu_n) * 1e-5
                else:
                    mu_n = np.asarray(pred).reshape(-1)
                    sigma_n = np.ones_like(mu_n) * 1e-5
            except Exception:
                pass

        # Path 3: Simple fallback
        if mu_n is None:
            try:
                X_for_model = X_avail.get_data()[0] if hasattr(X_avail, "get_data") else np.asarray(X_avail)
                pred = model.predict(X_for_model)
                if isinstance(pred, (tuple, list)):
                    mu_n = np.asarray(pred[0]).reshape(-1)
                else:
                    mu_n = np.asarray(pred).reshape(-1)
                sigma_n = np.ones_like(mu_n) * 1e-5
            except Exception as e:
                raise ValueError(
                    f"TruVaRAcquisition: could not get predictions. Error: {e}"
                )

        return mu_n, sigma_n

    def __call__(
        self,
        dataset_x,
        select_size: int,
        available_indices: List[AnyStr],
        cumulative_indices: Optional[List[AnyStr]] = None,
        model=None,
        dataset_y=None,
        threshold: Optional[float] = None,
        observation_noise_variance: Optional[float] = None,
        t: int = 1,
    ) -> List[int]:
        if model is None:
            raise ValueError("TruVaRAcquisition requires a trained `model`.")

        is_numpy = isinstance(dataset_x, np.ndarray)
        if is_numpy:
            X_avail = dataset_x[np.asarray(available_indices, dtype=int)]
        else:
            X_avail = dataset_x.subset(available_indices)

        mu_n, sigma_n = self._get_posterior_mean_and_std(model, X_avail)

        tau = 0.0 if threshold is None else float(threshold)

        # TruVaR score: σ(x) where |μ(x) - τ| < β·σ(x), else 0
        # Points already confidently classified are zeroed out.
        uncertain_mask = np.abs(mu_n - tau) < self.beta * sigma_n
        scores = np.where(uncertain_mask, sigma_n, 0.0)

        # If all points are classified, fall back to pure uncertainty sampling
        # (avoid returning nothing)
        if scores.max() == 0.0:
            scores = sigma_n

        # Select top-b candidates with random tie-breaking
        b = int(select_size)
        n_candidates = len(scores)

        if n_candidates <= b:
            chosen_local = list(range(n_candidates))
        else:
            noise = np.random.rand(n_candidates) * 1e-10
            order = np.argsort(-(scores + noise))
            chosen_local = order[:b].tolist()

        if is_numpy:
            available_indices_arr = np.asarray(available_indices, dtype=int)
        else:
            available_indices_arr = np.asarray(available_indices, dtype=object)
        proposal = available_indices_arr[np.asarray(chosen_local, dtype=int)].tolist()

        return list(dict.fromkeys(proposal))[:b]
