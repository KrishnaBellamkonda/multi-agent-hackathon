"""
Copyright 2025 Andrea Rubbi, University of Cambridge
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import numpy as np
from typing import AnyStr, List, Optional, Any, Tuple
from slingpy import AbstractDataSource
from slingpy.models.abstract_base_model import AbstractBaseModel
from genedisco.active_learning_methods.acquisition_functions.base_acquisition_function import (
    BaseBatchAcquisitionFunction,
)
from activehit.data import abstract_numpy_class

#TODO: move this to utils later
def _to_numpy_1d(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    x = np.asarray(x)
    return x.reshape(-1)


def _greedy_top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    if np.any(np.isnan(scores)):
        scores = np.where(np.isnan(scores), -np.inf, scores)
    n = scores.shape[0]
    if k <= 0 or n == 0:
        return np.array([], dtype=int)
    k = min(k, n)
    # argpartition for efficiency, then sort the top-k slice by score desc
    top_k_unsorted = np.argpartition(-scores, kth=k - 1)[:k]
    top_k_sorted = top_k_unsorted[np.argsort(-scores[top_k_unsorted], kind="stable")]
    return top_k_sorted


def _as_2d_array(x: Any) -> np.ndarray:
    """Best-effort conversion of a dataset subset to a 2D numpy array (n, d)."""
    if isinstance(x, np.ndarray):
        arr = x
    else:
        arr = np.asarray(x)
    # common pattern in this repo: (n,1,d) containers
    if arr.ndim > 2 and arr.shape[1] == 1:
        arr = np.squeeze(arr, axis=1)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def _farthest_first_k_indices(X: np.ndarray, k: int) -> np.ndarray:
    """Select k indices that maximize spread via farthest-first traversal (k-center).

    Deterministic seeding: first center is the point farthest from the mean.
    """
    X = _as_2d_array(X)
    n = X.shape[0]
    if k <= 0 or n == 0:
        return np.array([], dtype=int)
    k = min(k, n)

    # Start from the point farthest from the mean for determinism
    mean = X.mean(axis=0, keepdims=True)
    d0 = np.linalg.norm(X - mean, axis=1)
    first = int(np.argmax(d0))
    centers = [first]

    # Maintain distances to the nearest selected center
    min_dists = np.linalg.norm(X - X[first], axis=1)
    for _ in range(1, k):
        next_idx = int(np.argmax(min_dists))
        centers.append(next_idx)
        # update min distances
        dist_new_center = np.linalg.norm(X - X[next_idx], axis=1)
        min_dists = np.minimum(min_dists, dist_new_center)
    return np.asarray(centers, dtype=int)


def _compute_uncertainty_scores(model: Any, data: Any) -> Optional[np.ndarray]:
    """Try to compute per-point uncertainty scores using the model.

    Returns a 1D numpy array of uncertainties if possible, otherwise None.
    Tries multiple interfaces for robustness.
    """
    # 1) Multiple predictions -> variance
    if hasattr(model, "get_model_prediction"):
        try:
            out = model.get_model_prediction(data, return_multiple_preds=True)
            # out could be (samples, ...) or a tuple/list; try to normalize
            arr = np.asarray(out)
            if arr.ndim >= 2:
                # assume axis=0 are samples
                unc = np.var(arr, axis=0)
                return unc.reshape(-1)
            if isinstance(out, (tuple, list)) and len(out) > 0:
                arr = np.asarray(out[0])
                # If this still isn't multiple samples, skip
        except Exception:
            pass
        # 2) Std via flag if supported
        try:
            out = model.get_model_prediction(data, return_std_and_margin=True)
            # Expect (y_pred, y_std, y_margin) or similar
            if isinstance(out, (tuple, list)) and len(out) >= 2:
                y_std = out[1]
                return _to_numpy_1d(y_std)
        except Exception:
            pass
    # 3) Predict with std if available
    if hasattr(model, "predict"):
        try:
            out = model.predict(data, return_std=True)
            if isinstance(out, (tuple, list)) and len(out) >= 2:
                y_std = out[1]
                return _to_numpy_1d(y_std)
        except Exception:
            pass
    return None


class TopKAcquisition(BaseBatchAcquisitionFunction):
    def __init__(self, initial_strategy: str = None):
        """
        initial_strategy: one of {"uncertainty", "spread"}.
          - "uncertainty": at the very first selection (no previous picks), choose the
            top-k points with highest uncertainty as reported by the model.
          - "spread": choose k points that maximize coverage (k-center / farthest-first).
        The default can be overridden by environment variable TOPK_INITIAL_STRATEGY.
        """
        super().__init__()
        env_val = os.environ.get("TOPK_INITIAL_STRATEGY", "").strip().lower()
        if initial_strategy is None:
            initial_strategy = env_val if env_val in {"uncertainty", "spread"} else "uncertainty"
        self.initial_strategy = initial_strategy
    def __call__(
        self,
        dataset_x: AbstractDataSource,
        acquisition_batch_size: int,
        available_indices: List[AnyStr],
        last_selected_indices: List[AnyStr],
        model: Any = None,
        cumulative_indices: Optional[List[AnyStr]] = None,
        dataset_y: Optional[AbstractDataSource] = None,
        temp_folder_name: str = "tmp/model",
    ) -> List[AnyStr]:
        # Build the available subset of X
        if isinstance(dataset_x, np.ndarray):
            avail_dataset_x = dataset_x[available_indices]
        else:
            avail_dataset_x = dataset_x.subset(available_indices)

        # Initial step logic: if no items have been selected so far
        is_initial = (not last_selected_indices) and (not cumulative_indices)
        if is_initial:
            if self.initial_strategy == "uncertainty":
                if model is None:
                    # No model -> fall back to spread
                    X = _as_2d_array(avail_dataset_x)
                    idx_local = _farthest_first_k_indices(X, acquisition_batch_size)
                else:
                    unc = _compute_uncertainty_scores(model, avail_dataset_x)
                    if unc is None:
                        # Fall back to spread if uncertainty not available
                        X = _as_2d_array(avail_dataset_x)
                        idx_local = _farthest_first_k_indices(X, acquisition_batch_size)
                    else:
                        # pick top-k uncertainty
                        idx_local = _greedy_top_k_indices(unc, acquisition_batch_size)
            else:  # "spread"
                X = _as_2d_array(avail_dataset_x)
                idx_local = _farthest_first_k_indices(X, acquisition_batch_size)

            proposal = np.asarray(available_indices, dtype=object)[idx_local]
            return proposal.tolist()

        # Subsequent steps: greedy top-k by predictive score (as before)
        if model is None:
            raise ValueError(
                "TopKAcquisition received model=None. Ensure a trained model is passed to the acquisition function."
            )
        if hasattr(model, "get_model_prediction"):
            preds = model.get_model_prediction(
                avail_dataset_x, return_multiple_preds=False
            )[0]
        elif hasattr(model, "predict"):
            pred_out = model.predict(avail_dataset_x)
            preds = pred_out[0] if isinstance(pred_out, tuple) else pred_out
        else:
            raise AttributeError(
                "Model does not support prediction via 'get_model_prediction' or 'predict'."
            )

        scores = _to_numpy_1d(preds)
        best_indices_local = _greedy_top_k_indices(scores, acquisition_batch_size)
        proposal = np.asarray(available_indices, dtype=object)[best_indices_local]
        return proposal.tolist()
