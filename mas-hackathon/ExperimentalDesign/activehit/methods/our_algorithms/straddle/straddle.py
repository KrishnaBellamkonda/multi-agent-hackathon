"""
Straddle Heuristic for Level-Set Estimation
--------------------------------------------

Implements the straddle heuristic from:
  Bryan, Schneider, Nichol, Guestrin, "Active exploration for level sets"
  (NIPS 2005 / ICML 2006 workshop)
  Gotovos, Casati, Hitz, Krause, "Active Learning for Level Set Estimation"
  (IJCAI 2013)

The acquisition score is:

    a(x) = β·σ_n(x) - |μ_n(x) - τ|

This selects points near the decision boundary (where |μ - τ| is small)
that also have high posterior uncertainty (σ large). The parameter β
controls the exploration-exploitation trade-off.

Points with the highest straddle score are selected; this concentrates
samples where the GP is most uncertain about whether y > τ.
"""

from typing import AnyStr, List, Optional, Tuple
import numpy as np
from scipy.stats import norm
from slingpy import AbstractDataSource
from slingpy.models.abstract_base_model import AbstractBaseModel
from genedisco.active_learning_methods.acquisition_functions.base_acquisition_function import (
    BaseBatchAcquisitionFunction,
)


class StraddleAcquisition(BaseBatchAcquisitionFunction):
    """Straddle heuristic for threshold-aware active learning.

    Selects points that straddle the decision boundary τ with high uncertainty.

    Parameters
    ----------
    beta : float
        Exploration weight. Higher β favours uncertainty; lower β targets
        points closer to the threshold.  Default 1.96 (≈ 95% CI width).
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
                    f"StraddleAcquisition: could not get predictions. Error: {e}"
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
            raise ValueError("StraddleAcquisition requires a trained `model`.")

        is_numpy = isinstance(dataset_x, np.ndarray)
        if is_numpy:
            X_avail = dataset_x[np.asarray(available_indices, dtype=int)]
        else:
            X_avail = dataset_x.subset(available_indices)

        mu_n, sigma_n = self._get_posterior_mean_and_std(model, X_avail)

        tau = 0.0 if threshold is None else float(threshold)

        # Straddle score: β·σ(x) - |μ(x) - τ|
        scores = self.beta * sigma_n - np.abs(mu_n - tau)

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
