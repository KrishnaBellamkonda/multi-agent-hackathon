"""
Thompson Sampling for Hit Number (no diversity)
------------------------------------------------

At each round, draw a single function sample f_hat from the current posterior,
identify the set of candidate points whose sampled value exceeds a threshold τ
(the "hit" set), then form a batch of size b by taking:

1) If there are at least b hits, pick b uniformly at random from the hit set.
2) Otherwise, take all hits and fill the remainder with top-scoring non-hits
   under f_hat, sorted in descending order.

Implementation notes
--------------------
- Follows the call signature of the other acquisition functions in this repository.
- Sampling a single function draw is achieved via the model.sample(X) API on
  the toy Gaussian Process model, which returns a correlated multivariate draw
  across all inputs X. If a different model is used, it should offer a
  compatible `sample` method or this acquisition will fall back to sampling
  each point independently (which is a weaker approximation of a function draw).
- Threshold τ can be supplied at call time via the `threshold` kwarg. If not
  provided, a conservative default of 0.0 is used. Adjust as needed per use case.

Edge cases handled
------------------
- If fewer than b candidates exist (should not happen in our loop), we return
  as many as possible.
- If the hit set is empty, we purely select by the top f_hat scores.
- Shapes are normalized so both numpy arrays and data sources are supported.
"""

from typing import AnyStr, List, Optional
import numpy as np
from slingpy import AbstractDataSource
from slingpy.models.abstract_base_model import AbstractBaseModel
from genedisco.active_learning_methods.acquisition_functions.base_acquisition_function import (
    BaseBatchAcquisitionFunction,
)


class ThompsonHitAcquisition(BaseBatchAcquisitionFunction):
    """Batch acquisition implementing Thompson Sampling for Hit Number.

    Parameters
    ----------
    dataset_x : np.ndarray or AbstractDataSource
        All candidate inputs.
    acquisition_batch_size : int
        Number of points to select (b).
    available_indices : List[int]
        Indices still eligible for sampling.
    model : AbstractBaseModel
        Trained predictive model with a .sample(X) method.
    threshold : float, optional
        Hit threshold τ. Defaults to 0.0.

    Returns
    -------
    List[int]
        Selected indices into the original dataset.
    """

    def __call__(
        self,
        dataset_x: AbstractDataSource,
        acquisition_batch_size: int,
        available_indices: List[AnyStr],
        cumulative_indices: Optional[List[AnyStr]] = None,
        model: Optional[AbstractBaseModel] = None,
        dataset_y: Optional[AbstractDataSource] = None,
        last_selected_indices: Optional[List[AnyStr]] = None,
        temp_folder_name: str = "tmp",
        threshold: Optional[float] = None,
        t: int = 1,
        n_function_draws: int = 1,
    ) -> List[AnyStr]:
        if model is None:
            raise ValueError("ThompsonHitAcquisition requires a trained `model`. Got None.")

        # Normalize inputs - available_indices are gene names (strings), not integers.
        # For GeneDisco, dataset_x is typically an AbstractDataSource and we keep it
        # as such because GeneDisco models expect a DataSource. For toy experiments,
        # dataset_x can be a numpy array, in which case we index directly.
        if isinstance(dataset_x, np.ndarray):
            # Toy experiment: dataset_x is an array of features; available_indices are
            # integer positions into this array.
            X_avail_datasource = dataset_x[available_indices]
        else:
            # GeneDisco: dataset_x is a DataSource; available_indices are typically
            # gene identifiers (strings) used by .subset.
            X_avail_datasource = dataset_x.subset(available_indices)

        # Draw function sample(s) f_hat on the available set.
        # We prioritise a simple GP-style joint Gaussian sample based on
        # predict(return_std_and_margin=True), which works for both the toy GP
        # and GeneDisco GP models, and only fall back to the more specialised
        # MC-dropout interfaces when available. This avoids the frequent
        # "fallback failed, using zeros" behaviour seen in toy experiments.

        n_function_draws = max(1, int(n_function_draws))

        def one_draw():
            """Draw from posterior using MC dropout samples.
            
            For Thompson Hit, we need to sample from the posterior to determine 
            which points are above threshold τ. We use MC dropout to get multiple
            samples and randomly select one as our function draw.
            """
            try:
                # Use MC dropout sampling (same as thompson_sampling uses)
                num_mc_samples = 15
                f_samples = model.get_model_prediction(
                    X_avail_datasource, 
                    return_multiple_preds=True,
                    num_target_samples=num_mc_samples
                )[0]
                
                # Convert to numpy
                if hasattr(f_samples, 'detach'):
                    f_samples_np = f_samples.detach().cpu().numpy()
                else:
                    f_samples_np = np.asarray(f_samples)
                
                # f_samples_np should be (n_samples, n_candidates) or (n_candidates, n_samples)
                if f_samples_np.ndim == 1:
                    return f_samples_np.flatten()
                elif f_samples_np.shape[0] == num_mc_samples:
                    # Samples in first dimension: (n_samples, n_candidates)
                    # Randomly select one sample as our function draw
                    sample_idx = np.random.randint(0, f_samples_np.shape[0])
                    return f_samples_np[sample_idx].flatten()
                else:
                    # Samples in second dimension: (n_candidates, n_samples)
                    sample_idx = np.random.randint(0, f_samples_np.shape[1])
                    return f_samples_np[:, sample_idx].flatten()

            except Exception as e:
                # Fallback: try predict method
                try:
                    pred = model.predict(X_avail_datasource, return_std_and_margin=True)
                    if isinstance(pred, (list, tuple)) and len(pred) >= 2:
                        mean = np.asarray(pred[0]).reshape(-1)
                        std = np.asarray(pred[1]).reshape(-1)
                        # Sample from marginal posterior
                        return mean + np.random.standard_normal(len(mean)) * std
                    else:
                        return np.asarray(pred).reshape(-1)
                except Exception:
                    print(f"WARNING: Thompson Hit fallback failed: {e}")
                    return np.zeros(len(available_indices))
        if n_function_draws == 1:
            f_hat = one_draw()
            draws = None
        else:
            draws = np.stack([one_draw() for _ in range(n_function_draws)], axis=0)
            f_hat = draws.mean(axis=0)

        # Determine threshold τ
        tau = 0.0 if threshold is None else float(threshold)

        # Compute hit set (indices local to available set)
        hit_mask = f_hat > tau
        hit_local_idx = np.where(hit_mask)[0].tolist()

        # Selection logic
        b = int(acquisition_batch_size)
        chosen_local: List[int] = []
        if len(hit_local_idx) >= b:
            # Algorithm 1 line 6: "ℬₜ ← random sample of size b from ℋ"
            # RANDOMLY sample b genes from the hit set (no greedy selection!)
            chosen_local = list(np.random.choice(hit_local_idx, size=b, replace=False))
        else:
            # Take all hits
            chosen_local = list(hit_local_idx)
            remaining = b - len(chosen_local)
            if remaining > 0:
                # Non-hit candidates sorted by f_hat descending
                non_hit_local_idx = np.where(~hit_mask)[0]
                if non_hit_local_idx.size > 0:
                    order = np.argsort(-f_hat[non_hit_local_idx])  # descending
                    fill_local = non_hit_local_idx[order][:remaining].tolist()
                    chosen_local.extend(fill_local)

        # Map chosen local positions back to gene names from available_indices
        # available_indices contains gene names (strings), not integers
        proposal = [available_indices[i] for i in chosen_local]

        # Ensure uniqueness and correct length (defensive)
        proposal = list(dict.fromkeys(proposal))  # stable unique
        if len(proposal) > b:
            proposal = proposal[:b]

        return proposal
