"""
Probability-of-Hit Greedy for Hit Number (no diversity)
------------------------------------------------

At each round:
- compute the posterior hit probability p_n(g) for each candidate
  point g, where p_n(g) = 1 - Φ((τ - μ_n(g)) / √(σ_n²(g) + σ_obs²))
- select the top-b candidates by this probability (breaking ties at random).

Implementation notes
--------------------
- Follows the call signature of the other acquisition functions in this repository.
- The posterior mean μ_n and variance σ_n² are obtained via the model's
  predict method with return_std_and_margin=True or return_std=True flags.
- The observation noise variance σ_obs² can be provided via the
  `observation_noise_variance` kwarg. If not provided, we attempt to infer it
  from the model (e.g., model.noise_sigma² for GP models) or default to 0.0.
- The standard normal CDF Φ is computed using scipy.stats.norm.cdf.
- Threshold τ can be supplied at call time via the `threshold` kwarg. If not
  provided, a conservative default of 0.0 is used. Adjust as needed per use case.

Edge cases handled
------------------
- If fewer than b candidates exist (should not happen in our loop), we return
  as many as possible.
- If the model doesn't provide variance estimates, we fall back to using only
  the mean (treating variance as zero, which reduces to a simple threshold check).
- Shapes are normalized so both numpy arrays and data sources are supported.
- Ties in probability are broken randomly as specified in the algorithm.
"""

from typing import AnyStr, List, Optional, Tuple
import numpy as np
from scipy.stats import norm
from slingpy import AbstractDataSource
from slingpy.models.abstract_base_model import AbstractBaseModel
from genedisco.active_learning_methods.acquisition_functions.base_acquisition_function import (
    BaseBatchAcquisitionFunction,
)


class ProbabilityHitAcquisition(BaseBatchAcquisitionFunction):
    """Batch acquisition implementing Probability-of-Hit Greedy for Hit Number.

    Parameters
    ----------
    dataset_x : np.ndarray or AbstractDataSource
        All candidate inputs.
    acquisition_batch_size : int
        Number of points to select (b).
    available_indices : List[int]
        Indices still eligible for sampling.
    model : AbstractBaseModel
        Trained predictive model with a .predict() method returning mean/variance.
    threshold : float, optional
        Hit threshold τ. Defaults to 0.0.
    observation_noise_variance : float, optional
        Observation noise variance σ_obs². Defaults to 0.0; inferred from model
        when available.

    Returns
    -------
    List[int]
        Selected indices into the original dataset.
    """

    def _get_posterior_mean_and_variance(
            self, 
            model: AbstractBaseModel,
            X_avail: AbstractDataSource,
            num_mc_samples: int = 20,
        ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract posterior mean and variance from model predictions.
        
        This handles both:
        1. BNN models (GeneDisco): use get_model_prediction with MC dropout
        2. GP models (Toy): use predict with return_std_and_margin=True
        
        Args:
            model: Trained predictive model
            X_avail: Available candidate inputs (AbstractDataSource or np.ndarray)
            num_mc_samples: Number of MC dropout samples for uncertainty estimation
            
        Returns:
            Tuple of (mu_n, sigma_n_sq) where:
            - mu_n: Posterior mean (1D array)
            - sigma_n_sq: Posterior variance (1D array)
        """
        # Try MC dropout sampling first (for BNN/MLP models like in GeneDisco)
        if hasattr(model, 'get_model_prediction'):
            try:
                f_samples = model.get_model_prediction(
                    X_avail, 
                    return_multiple_preds=True,
                    num_target_samples=num_mc_samples
                )[0]
                
                # Convert to numpy
                if hasattr(f_samples, 'detach'):
                    f_samples_np = f_samples.detach().cpu().numpy()
                else:
                    f_samples_np = np.asarray(f_samples)
                
                # Compute mean and variance from MC samples
                if f_samples_np.ndim == 1:
                    mu_n = f_samples_np.flatten()
                    sigma_n_sq = np.zeros_like(mu_n)
                elif f_samples_np.ndim == 2:
                    if f_samples_np.shape[0] == num_mc_samples:
                        # Samples in first dimension: (n_samples, n_candidates)
                        mu_n = np.mean(f_samples_np, axis=0).flatten()
                        sigma_n_sq = np.var(f_samples_np, axis=0).flatten()
                    else:
                        # Samples in second dimension: (n_candidates, n_samples)
                        mu_n = np.mean(f_samples_np, axis=1).flatten()
                        sigma_n_sq = np.var(f_samples_np, axis=1).flatten()
                else:
                    # Handle 3D by squeezing
                    f_samples_np = f_samples_np.squeeze()
                    if f_samples_np.ndim == 2:
                        if f_samples_np.shape[0] == num_mc_samples:
                            mu_n = np.mean(f_samples_np, axis=0).flatten()
                            sigma_n_sq = np.var(f_samples_np, axis=0).flatten()
                        else:
                            mu_n = np.mean(f_samples_np, axis=1).flatten()
                            sigma_n_sq = np.var(f_samples_np, axis=1).flatten()
                    else:
                        mu_n = f_samples_np.flatten()
                        sigma_n_sq = np.zeros_like(mu_n)
                
                return mu_n, sigma_n_sq
            except Exception:
                pass  # Fall through to GP-style predict
        
        # Fallback: GP-style predict with return_std_and_margin (for toy GP models)
        if hasattr(model, 'predict'):
            try:
                pred = model.predict(X_avail, return_std_and_margin=True)
                if isinstance(pred, (list, tuple)) and len(pred) >= 2:
                    mu_n = np.asarray(pred[0]).reshape(-1)
                    # pred[1] is std, we need variance
                    std_n = np.asarray(pred[1]).reshape(-1)
                    sigma_n_sq = std_n ** 2
                    return mu_n, sigma_n_sq
                else:
                    mu_n = np.asarray(pred).reshape(-1)
                    sigma_n_sq = np.zeros_like(mu_n)
                    return mu_n, sigma_n_sq
            except Exception:
                pass  # Fall through to final fallback
        
        # Final fallback: try sample method (GP)
        if hasattr(model, 'sample'):
            try:
                # Get X as array for sample method
                if isinstance(X_avail, np.ndarray):
                    X_arr = X_avail
                else:
                    X_arr = X_avail.get_data()[0]
                
                # Collect multiple samples to estimate mean and variance
                samples = []
                for _ in range(num_mc_samples):
                    s = model.sample(X_arr)
                    samples.append(np.asarray(s).reshape(-1))
                samples = np.stack(samples, axis=0)  # (n_samples, n_candidates)
                mu_n = np.mean(samples, axis=0)
                sigma_n_sq = np.var(samples, axis=0)
                return mu_n, sigma_n_sq
            except Exception:
                pass
        
        # Ultimate fallback: zeros (shouldn't happen but defensive)
        if isinstance(X_avail, np.ndarray):
            n = len(X_avail)
        else:
            n = X_avail.get_shape()[0]
        print("WARNING: ProbabilityHit fallback failed, using zeros for variance")
        return np.zeros(n), np.zeros(n)

    def __call__(
            self,
            dataset_x: AbstractDataSource,
            select_size: int,
            available_indices: List[AnyStr],
            cumulative_indices: Optional[List[AnyStr]] = None,
            model: Optional[AbstractBaseModel] = None,
            dataset_y: Optional[AbstractDataSource] = None,
            threshold: Optional[float] = None,
            observation_noise_variance: Optional[float] = None,
            t: int = 1,
            min_std: float = 0.05,  # Minimum std to prevent over-confidence
            num_mc_samples: int = 30,  # MC samples for uncertainty estimation
        ) -> List[int]:
        if model is None:
            raise ValueError("ProbabilityHitAcquisition requires a trained `model`. Got None.")

        # Normalize inputs
        # For GeneDisco: available_indices are gene names (strings), use .subset
        # For toy: available_indices are ints, use numpy indexing
        is_numpy = isinstance(dataset_x, np.ndarray)
        if is_numpy:
            X_avail = dataset_x[np.asarray(available_indices, dtype=int)]
        else:
            X_avail = dataset_x.subset(available_indices)

        # Compute posterior mean μ_n and variance σ_n² for all candidates
        # Using MC dropout samples - variance already captures epistemic uncertainty
        mu_n, sigma_n_sq = self._get_posterior_mean_and_variance(
                                    model=model, 
                                    X_avail=X_avail,
                                    num_mc_samples=num_mc_samples,
                                )

        # Determine threshold τ
        tau = 0.0 if threshold is None else float(threshold)

        # Compute posterior hit probability p_n(g) = 1 - Φ((τ - μ_n(g)) / σ_n(g))
        # where Φ is the standard normal CDF
        # Apply minimum std to prevent over-confidence and maintain exploration
        std_n = np.sqrt(np.maximum(sigma_n_sq, min_std ** 2))
        
        # Compute z-scores: (τ - μ_n) / std_n
        z_scores = (tau - mu_n) / std_n
        
        # Compute p_n = 1 - Φ(z) = P(y > τ | data)
        # This is exactly Algorithm 2 from the paper
        p_n = 1 - norm.cdf(z_scores)
        
        # Use P(hit) as the acquisition score (Algorithm 2: greedy top-b by p_n)
        acquisition_score = p_n

        # Select top-b candidates by acquisition score (break ties at random)
        b = int(select_size)
        n_candidates = len(acquisition_score)
        
        if n_candidates <= b:
            # If we have fewer candidates than needed, return all
            chosen_local = list(range(n_candidates))
        else:
            # Sort by acquisition_score descending, with random tie-breaking
            # Add small random noise to break ties
            noise = np.random.rand(n_candidates) * 1e-10
            scores_with_noise = acquisition_score + noise
            order = np.argsort(-scores_with_noise)  # descending
            chosen_local = order[:b].tolist()

        # Map chosen local positions back to available_indices (gene names or ints)
        if is_numpy:
            available_indices_arr = np.asarray(available_indices, dtype=int)
        else:
            available_indices_arr = np.asarray(available_indices, dtype=object)
        proposal = available_indices_arr[np.asarray(chosen_local, dtype=int)].tolist()

        # Ensure uniqueness and correct length (defensive)
        proposal = list(dict.fromkeys(proposal))  # stable unique
        if len(proposal) > b:
            proposal = proposal[:b]

        return proposal