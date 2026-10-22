"""
Expected Improvement above Threshold (EI-τ)
---------------------------------------------

A threshold-aware acquisition function based on Expected Improvement (EI)
with respect to a fixed threshold τ rather than the current best observed
value.  This is a natural adaptation of the classic EI criterion
(Jones, Schonlau, Welch, "Efficient Global Optimization of Expensive
Black-Box Functions", J. Global Optim. 1998) to the level-set /
hit-identification setting.

For each candidate x with posterior mean μ(x) and std σ(x):

    EI_τ(x) = (μ(x) - τ) · Φ(z) + σ(x) · φ(z)

where z = (μ(x) - τ) / σ(x), Φ is the standard normal CDF,
and φ is the standard normal PDF.

Points with the highest EI_τ are selected.  This naturally balances
exploitation (preferring points predicted above τ) with exploration
(preferring points with high uncertainty near τ).
"""

from typing import AnyStr, List, Optional, Tuple
import numpy as np
from scipy.stats import norm
from slingpy import AbstractDataSource
from slingpy.models.abstract_base_model import AbstractBaseModel
from genedisco.active_learning_methods.acquisition_functions.base_acquisition_function import (
    BaseBatchAcquisitionFunction,
)


class EITauAcquisition(BaseBatchAcquisitionFunction):
    """Expected Improvement above threshold τ.

    Selects points maximising expected improvement over a fixed
    threshold rather than the current incumbent.
    """

    def _get_posterior_mean_and_std(self, model, X_avail):
        """Extract posterior mean and std from model predictions.
        
        X_avail can be either a numpy array or an AbstractDataSource.
        """
        mu_n = None
        sigma_n = None

        # Resolve X to a numpy array for prediction
        if hasattr(X_avail, 'get_data'):
            X_np = X_avail.get_data()[0]
        else:
            X_np = np.asarray(X_avail)

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
                pred = model.predict(X_np, return_std_and_margin=True)
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
                pred = model.predict(X_np)
                mu_n = np.asarray(pred).reshape(-1)
                sigma_n = np.ones_like(mu_n) * 1e-5
            except Exception:
                pass

        return mu_n, sigma_n

    def __call__(
        self,
        dataset_x: AbstractDataSource,
        batch_size: int,
        available_indices: List[int],
        last_selected_indices: List[int] = None,
        cumulative_indices: List[int] = None,
        model: AbstractBaseModel = None,
        dataset_y: AbstractDataSource = None,
        temp_folder_name: str = "tmp",
        threshold: float = None,
        **kwargs,
    ) -> List[int]:
        available_indices = list(available_indices)
        n_avail = len(available_indices)
        if n_avail == 0:
            return []

        select_size = min(batch_size, n_avail)

        # Build the subset of feature data for available indices
        if hasattr(dataset_x, 'subset'):
            X_avail = dataset_x.subset(available_indices)
        else:
            X_avail = np.asarray(dataset_x)[available_indices]

        if model is None or threshold is None:
            rng = np.random.RandomState(hash(tuple(available_indices)) % (2**31))
            return list(rng.choice(available_indices, size=select_size, replace=False))

        mu_n, sigma_n = self._get_posterior_mean_and_std(model, X_avail)
        if mu_n is None:
            rng = np.random.RandomState(hash(tuple(available_indices)) % (2**31))
            return list(rng.choice(available_indices, size=select_size, replace=False))

        tau = float(threshold)
        # EI_τ(x) = (μ(x) - τ)·Φ(z) + σ(x)·φ(z)  where z = (μ(x) - τ)/σ(x)
        z = (mu_n - tau) / sigma_n
        ei = (mu_n - tau) * norm.cdf(z) + sigma_n * norm.pdf(z)

        # Select top-scoring points
        top_k = min(select_size, len(ei))
        local_indices = np.argsort(-ei)[:top_k]
        return [available_indices[i] for i in local_indices]
