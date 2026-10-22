"""
Batched Thompson Sampling with Information-Gain Exploration
------------------------------------------------------------

This module implements Algorithm 1: "Batched Thompson Sampling with 
Information-Gain Exploration" as described in the provided specification.

The algorithm combines:
1. **Thompson Sampling** (early cycles): Sample from posterior, select top-scoring genes
2. **Information Gain** (final cycle): Use exploitation scores (posterior mean)
3. **Batch Selection**: Sequential variance-based selection to encourage diversity

Key Features
------------
- Exploration-exploitation trade-off: Early cycles explore via Thompson sampling,
  final cycle exploits using posterior mean
- Batch diversity: Sequential selection with covariance updates to avoid redundancy
- Threshold-based filtering: Only genes above threshold λ are considered hits
- Hybrid strategy: Combines top-scoring hits with variance-based selection

Algorithm Details (from image)
------------------------------
Input: features {x_i}, threshold λ, prior parameters (λ_0, σ²), batch size B, cycles T

For t = 1 to T:
  1. Compute posterior (μ_n, Σ_n) from D_n
  2. Let U_t be unsampled genes
  
  3. If t < T (exploration):
     - Sample θ̂ ~ N(μ_n, Σ_n)
     - Compute scores s_i = x_i^T θ̂
     - S_t^hit ← top-scoring genes above threshold (max B)
  
  4. Else (final cycle - exploitation):
     - Compute scores s_i = μ_n^T x_i
     - S_t ← top-B genes by s_i
     - Return to line 28
  
  5. Initialize S_t ← S_t^hit
  6. If |S_t| < B:
     - K ← B - |S_t|
     - For k = 1 to K:
       * For i ∈ U_t \ S_t, compute variance v_i = x_i^T Σ_n x_i
       * Select j = arg max_i v_i
       * S_t ← S_t ∪ {j}
       * Update covariance: Σ_n ← Σ_n - (Σ_n x_j x_j^T Σ_n)/(σ² + x_j^T Σ_n x_j)
  
  7. Run experiments on S_t, obtain outcomes {y_i}
  8. Update dataset: D_{n+|S_t|} = D_n ∪ {(x_i, y_i) : i ∈ S_t}

Return: discovered hits and final posterior

Implementation Notes
--------------------
- Compatible with GeneDisco framework using AbstractDataSource
- Works with any model providing predict() with return_std_and_margin=True
- Threshold λ defaults to 0 if not provided
- Handles edge cases (empty hit set, insufficient candidates)
"""

from typing import AnyStr, List, Optional
import numpy as np
from slingpy import AbstractDataSource
from slingpy.models.abstract_base_model import AbstractBaseModel
from genedisco.active_learning_methods.acquisition_functions.base_acquisition_function import (
    BaseBatchAcquisitionFunction,
)


class ThompsonIGAcquisition(BaseBatchAcquisitionFunction):
    """Batched Thompson Sampling with Information-Gain Exploration.

    This acquisition function implements a hybrid strategy:
    - Early cycles: Thompson sampling for exploration
    - Final cycle: Posterior mean for exploitation
    - Batch selection: Variance-based diversity within each batch

    Parameters
    ----------
    threshold : float, optional
        Hit threshold λ. Only genes with scores > λ are considered hits.
        Defaults to 0.0 if not provided.
    
    num_cycles : int, optional
        Total number of active learning cycles T. Used to detect final cycle.
        If not provided, assumes all cycles use Thompson sampling.
    
    current_cycle : int, optional
        Current cycle number (1-indexed). Used to switch between exploration
        and exploitation. Defaults to 1.
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
        num_cycles: Optional[int] = None,
        current_cycle: int = 1,
    ) -> List[AnyStr]:
        """Select batch using Thompson sampling + information gain.

        Parameters
        ----------
        dataset_x : AbstractDataSource
            Feature matrix for all candidates
        acquisition_batch_size : int
            Batch size B
        available_indices : List[AnyStr]
            Gene names still available for selection
        model : AbstractBaseModel
            Trained model providing posterior (mean, std)
        threshold : float, optional
            Hit threshold λ
        num_cycles : int, optional
            Total number of cycles T
        current_cycle : int, optional
            Current cycle t (1-indexed)

        Returns
        -------
        List[AnyStr]
            Selected gene names for this batch
        """
        if model is None:
            raise ValueError("ThompsonIGAcquisition requires a trained model.")

        # Parameters
        tau = 0.0 if threshold is None else float(threshold)
        B = int(acquisition_batch_size)
        t = int(current_cycle)
        T = int(num_cycles) if num_cycles is not None else t + 1  # Assume not final if unknown

        # Get available candidates - handle both numpy arrays (toy) and DataSource (genedisco)
        if isinstance(dataset_x, np.ndarray):
            X_avail = dataset_x[available_indices]
        else:
            X_avail = dataset_x.subset(available_indices)
        n_avail = len(available_indices)

        # Get posterior predictions: mean and std
        try:
            # GeneDisco models typically return [pred, std_or_cov, margin] when return_std_and_margin=True.
            # In toy_experiment, the GP returns a *covariance matrix* as the second output.
            # We normalise both cases to a 1D vector of standard deviations matching mu.
            predictions = model.predict(X_avail, return_std_and_margin=True)
            if isinstance(predictions, (list, tuple)) and len(predictions) >= 2:
                mu = np.asarray(predictions[0]).reshape(-1)
                second = np.asarray(predictions[1])
                # If "second" is a covariance matrix, take its diagonal; otherwise treat as
                # variance / std-like vector and coerce to non-negative then sqrt.
                if second.ndim == 2:
                    var_vec = np.clip(np.diag(second), 0.0, None)
                    std = np.sqrt(var_vec)
                else:
                    second_flat = second.reshape(-1)
                    var_vec = np.clip(second_flat, 0.0, None)
                    std = np.sqrt(var_vec)
            else:
                mu = np.asarray(predictions).reshape(-1)
                std = np.ones_like(mu) * 0.01  # Fallback: small homoscedastic noise
        except Exception as e:
            # Fallback: predict point by point
            mu = []
            std = []
            for idx in available_indices:
                # Handle both numpy arrays (toy) and DataSource (genedisco)
                if isinstance(dataset_x, np.ndarray):
                    single_ds = dataset_x[[idx]]
                else:
                    single_ds = dataset_x.subset([idx])
                try:
                    pred = model.predict(single_ds, return_std_and_margin=True)
                    if isinstance(pred, (list, tuple)) and len(pred) >= 2:
                        mu.append(float(pred[0][0]))
                        std.append(float(pred[1][0]))
                    else:
                        mu.append(float(pred[0]))
                        std.append(0.01)
                except Exception:
                    mu.append(0.0)
                    std.append(0.01)
            mu = np.array(mu)
            std = np.array(std)

        # Step 3-4: Compute scores based on cycle
        if t < T:
            # EXPLORATION: Thompson sampling (sample from posterior)
            # θ̂ ~ N(μ_n, Σ_n) → use one sample per feature dimension
            # Since we don't have full covariance, use independent samples
            theta_hat = mu + np.random.randn(len(mu)) * std
            scores = theta_hat
            is_final_cycle = False
        else:
            # EXPLOITATION: Use posterior mean
            scores = mu
            is_final_cycle = True

        # Identify hit set: genes above threshold
        hit_mask = scores > tau
        hit_local_idx = np.where(hit_mask)[0]

        if is_final_cycle:
            # Final cycle: Just take top-B by exploitation scores
            order = np.argsort(-scores)  # Descending
            chosen_local = order[:B].tolist()
            proposal = [available_indices[i] for i in chosen_local]
            return list(dict.fromkeys(proposal))[:B]

        # Step 5: Initialize S_t with top hits (up to B)
        if len(hit_local_idx) > 0:
            # Sort hits by score descending
            hit_scores = scores[hit_local_idx]
            hit_order = np.argsort(-hit_scores)
            top_hits = hit_local_idx[hit_order][:B]
            S_t = set(top_hits.tolist())
        else:
            S_t = set()

        # Step 6: Fill remaining batch with variance-based selection
        if len(S_t) < B:
            K = B - len(S_t)
            
            # We need feature vectors to compute variance
            # Extract features for all available candidates
            try:
                # Try to get raw features
                X_features = X_avail.get_data()[0]  # [0] gets features (X)
                if hasattr(X_features, 'toarray'):
                    X_features = X_features.toarray()
                X_features = np.asarray(X_features)
            except Exception:
                # Fallback: can't compute variance without features
                # Just select remaining by scores
                remaining_idx = [i for i in range(n_avail) if i not in S_t]
                if len(remaining_idx) > 0:
                    remaining_scores = scores[remaining_idx]
                    order = np.argsort(-remaining_scores)
                    fill = [remaining_idx[order[i]] for i in range(min(K, len(remaining_idx)))]
                    S_t.update(fill)
                chosen_local = list(S_t)[:B]
                proposal = [available_indices[i] for i in chosen_local]
                return list(dict.fromkeys(proposal))[:B]

            # Initialize covariance matrix
            # We approximate the posterior covariance using a simple diagonal structure
            # Σ_n ≈ σ_0² I where σ_0² is the average prediction variance
            d = X_features.shape[1]  # Feature dimension
            avg_var = np.mean(std ** 2)  # Average variance across candidates
            Sigma = np.eye(d) * avg_var  # (d, d) covariance matrix
            sigma_sq = 1.0  # Noise variance σ²

            for k in range(K):
                # Compute variance for unselected candidates
                U_t_minus_St = [i for i in range(n_avail) if i not in S_t]
                if len(U_t_minus_St) == 0:
                    break

                variances = []
                for i in U_t_minus_St:
                    x_i = X_features[i:i+1, :]  # Shape: (1, d)
                    # v_i = x_i^T Σ_n x_i
                    v_i = float(x_i @ Sigma @ x_i.T)
                    variances.append(v_i)

                # Select j = arg max v_i
                j_relative = np.argmax(variances)
                j = U_t_minus_St[j_relative]
                S_t.add(j)

                # Update covariance: Σ_n ← Σ_n - (Σ_n x_j x_j^T Σ_n)/(σ² + x_j^T Σ_n x_j)
                x_j = X_features[j:j+1, :].T  # Shape: (d, 1)
                Sigma_x_j = Sigma @ x_j  # (d, d) @ (d, 1) = (d, 1)
                denominator = sigma_sq + float(x_j.T @ Sigma_x_j)
                
                if denominator > 1e-10:  # Avoid division by zero
                    Sigma = Sigma - (Sigma_x_j @ Sigma_x_j.T) / denominator

        # Convert local indices to gene names
        chosen_local = list(S_t)[:B]
        proposal = [available_indices[i] for i in chosen_local]
        
        # Ensure uniqueness and correct length
        proposal = list(dict.fromkeys(proposal))[:B]
        return proposal
