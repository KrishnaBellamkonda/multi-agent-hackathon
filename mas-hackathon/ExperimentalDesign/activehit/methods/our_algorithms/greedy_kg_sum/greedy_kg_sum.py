"""Greedy KG-Sum Acquisition Function
-----------------------------------

Knowledge-gradient style one-step lookahead tailored to the cumulative SUM of
posterior means over the top-H points. For each candidate x, we sample L fantasy
observations, refit a light model, and recompute the sum of the largest H
posterior means. We take the expected gain and greedily pick the best.

Two batch strategies:
    sequential  – pick points one-by-one, refitting after each fantasy step.
    topk-approx – score all candidates once from the base fit, then take the
                   top-k by KG-Sum score (faster, approximate).

Robust interfaces: works with numpy arrays or slingpy AbstractDataSource; and
supports models exposing either predict(...) returning mean + std / covariance
or get_model_prediction(...) for torch-based implementations.

Copyright 2025
Licensed under the Apache License, Version 2.0
"""
from __future__ import annotations

import os
from typing import AnyStr, List, Optional, Union
import numpy as np

from slingpy import AbstractDataSource, AbstractBaseModel
from genedisco.active_learning_methods.acquisition_functions.base_acquisition_function import (
    BaseBatchAcquisitionFunction,
)

# Optional numpy datasource helper from activehit (for torch models expecting DataSource)
try:  # pragma: no cover - defensive import
    from activehit.data import abstract_numpy_class  # type: ignore
    HAS_DSX = True
except Exception:  # pragma: no cover
    HAS_DSX = False

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def _to_np_x(ds):
    return ds if isinstance(ds, np.ndarray) else ds.get_data()[0]

def _subset_x(ds, idxs, ds_mode):
    if ds_mode:
        return ds.subset(idxs).get_data()[0]
    return _to_np_x(ds)[idxs]

def _subset_y(dy, idxs, ds_mode):
    if dy is None:
        return np.empty((0,))
    if isinstance(dy, np.ndarray):
        return np.asarray(dy)[idxs]
    return dy.subset(idxs).get_data()[0] if ds_mode else dy.get_data()[0][idxs]

def _predict_mu_cov(model, x, ds_mode: bool = False, dataset_x=None, indices=None):
    """Return (mu, cov) if available; fall back to (mu, diag(var)).
    
    Parameters
    ----------
    model : model object
        The model to use for prediction.
    x : np.ndarray or DataSource
        Input features. If ds_mode=True, x should be numpy array and dataset_x/indices must be provided.
    ds_mode : bool
        If True, use dataset_x.subset(indices) instead of wrapping x in NumpyDataSource.
    dataset_x : DataSource, optional
        Original dataset (required if ds_mode=True).
    indices : list, optional
        Indices to subset (required if ds_mode=True).
    """
    if hasattr(model, "predict"):
        # In GeneDisco mode, model.predict expects DataSource objects
        # Use dataset_x.subset(indices) instead of wrapping numpy arrays
        if ds_mode and dataset_x is not None and indices is not None:
            x_input = dataset_x.subset(indices)
        else:
            x_input = x
        
        # Try return_std_and_margin first (GeneDisco standard)
        try:
            result = model.predict(x_input, return_std_and_margin=True)
            # GeneDisco returns a list, extract first element
            if isinstance(result, tuple) and len(result) == 2:
                mu, s = result
            elif isinstance(result, list) and len(result) >= 2:
                mu, s = result[0], result[1]
            else:
                raise TypeError("Unexpected return format from model.predict")
        except (TypeError, AttributeError):
            # Fallback to no kwargs (basic predict)
            try:
                result = model.predict(x_input)
                if isinstance(result, tuple) and len(result) == 2:
                    mu, s = result
                elif isinstance(result, list):
                    mu = result[0] if len(result) > 0 else result
                    s = np.zeros_like(mu)
                else:
                    mu = result
                    s = np.zeros_like(mu)
            except Exception as e:
                raise AttributeError(f"model.predict() failed with error: {e}")
        
        mu = np.asarray(mu).reshape(-1)
        s = np.asarray(s)
        if s.ndim == 2:  # full covariance provided
            cov = s
        else:  # std vector
            cov = np.diag(np.square(s.reshape(-1)))
        return mu, cov
    if hasattr(model, "get_model_prediction"):
        try:
            pred = model.get_model_prediction(x, return_multiple_preds=False)[0]
        except Exception:
            if not HAS_DSX:
                raise AttributeError("Need predict() or NumpyDataSource wrapper for get_model_prediction().")
            pred = model.get_model_prediction(
                abstract_numpy_class.NumpyDataSource(x), return_multiple_preds=False
            )[0]
        mu = pred.detach().cpu().numpy().reshape(-1)
        cov = np.diag(np.zeros_like(mu))
        return mu, cov
    raise AttributeError("Model must expose predict(...) or get_model_prediction(...).")

# -----------------------------------------------------------------------------
# Acquisition class
# -----------------------------------------------------------------------------
class GreedyKGSumAcquisition(BaseBatchAcquisitionFunction):
    """One-step KG-like lookahead tailored to cumulative-sum reward.

    For each candidate x, sample its fantasy observation y_hat ~ Normal(mu_x, var_x + noise_var),
    refit quickly with the augmented dataset, and recompute the sum of the top-H posterior means.
    Average this gain over L fantasy samples.

    Parameters
    ----------
    H : int
        Lookahead span (how many top future picks to approximate).
    L : int
        Number of fantasy draws per candidate.
    noise_var : float
        Observation noise variance added to posterior variance when sampling fantasies.
    batch_strategy : {"sequential", "topk-approx"}
        Strategy for batch selection.
    compat_mode : {"auto", "datasource", "toy"}
        Auto infers whether to treat inputs as DataSource objects.
    """

    def __init__(
        self,
        H: int = 5,
        L: int = 8,
        noise_var: float = 1e-6,
        batch_strategy: str = "sequential",
        compat_mode: str = "auto",
    ):
        super().__init__()
        assert batch_strategy in {"sequential", "topk-approx"}, "Invalid batch_strategy"
        self.H = int(H)
        self.L = int(L)
        self.noise_var = float(noise_var)
        self.batch_strategy = batch_strategy
        self.compat_mode = compat_mode

    # ------------------------------------------------------------------
    def _score_once(self, x_avail, idx_choice, base_x, base_y, model, ds_mode, mu_base=None, cov_base=None):
        """Compute expected KG-Sum gain for a single candidate (local index).

        Fast path: if cov_base is a full covariance matrix over x_avail, use
        rank-one analytic fantasy updates without refitting the model.
        Fallback: refit-based simulation as before.
        """
        # If not provided, compute base posterior on available set once
        # Note: This is called from within _pick_one or __call__, which don't have dataset_x in scope
        # So we'll keep x_avail as numpy array and not use ds_mode here
        if mu_base is None or cov_base is None:
            mu_base, cov_base = _predict_mu_cov(model, x_avail, ds_mode=False)

        sumH_base = np.sort(mu_base)[-self.H:].sum()

        # Analytic fantasy update when full covariance is available
        if isinstance(cov_base, np.ndarray) and cov_base.ndim == 2:
            mu_x = float(mu_base[idx_choice])
            v_x = float(cov_base[idx_choice, idx_choice])
            denom = max(v_x + self.noise_var, 1e-12)
            std_x = np.sqrt(denom)
            gains = []
            c_col = cov_base[:, idx_choice]
            for _ in range(self.L):
                y_hat = np.random.normal(mu_x, std_x)
                # Posterior mean update at all avail points
                mu_new = mu_base + c_col * ((y_hat - mu_x) / denom)
                sumH_new = np.sort(mu_new)[-self.H:].sum()
                gains.append(sumH_new - sumH_base)
            return float(np.mean(gains))

        # Fallback: refit after appending fantasy observation
        mu_x = float(mu_base[idx_choice])
        var_x = float(cov_base[idx_choice]) if np.ndim(cov_base) == 1 else 0.0
        std_x = np.sqrt(max(var_x + self.noise_var, 0.0))
        gains = []
        for _ in range(self.L):
            y_hat = np.random.normal(mu_x, std_x)
            x_new = x_avail[idx_choice:idx_choice + 1]
            
            # Handle empty base_x case
            is_base_empty = False
            try:
                if hasattr(base_x, 'size'):
                    is_base_empty = (base_x.size == 0)
                elif hasattr(base_x, 'shape'):
                    is_base_empty = (base_x.shape[0] == 0)
                elif hasattr(base_x, '__len__'):
                    is_base_empty = (len(base_x) == 0)
            except:
                is_base_empty = True
            
            if is_base_empty:
                x_fit = x_new
                y_fit = np.array([y_hat])
            else:
                x_fit = np.concatenate([base_x, x_new], axis=0)
                y_fit = np.concatenate([np.asarray(base_y).reshape(-1), np.array([y_hat])], axis=0)

            # Refit
            try:
                if ds_mode and HAS_DSX:
                    nx = abstract_numpy_class.NumpyDataSource(x_fit)
                    ny = abstract_numpy_class.NumpyDataSource(y_fit)
                    model.fit(nx, ny)
                else:
                    model.fit(x_fit, y_fit)
            except (IndexError, ValueError) as e:
                # If fitting fails, skip this fantasy simulation
                print(f"WARNING: model.fit failed in _score_once with error: {e}. Skipping this sample.")
                continue

            mu_new, _ = _predict_mu_cov(model, x_avail, ds_mode=False)
            sumH_new = np.sort(mu_new)[-self.H:].sum()
            gains.append(sumH_new - sumH_base)
        return float(np.mean(gains))

    # ------------------------------------------------------------------
    def _pick_one(
        self,
        dataset_x,
        available_indices,
        cumulative_indices,
        model,
        dataset_y,
        ds_mode,
        last_selected_indices=None,
    ):
        # Guard: no candidates left
        if not available_indices or len(available_indices) == 0:
            raise ValueError("GreedyKGSumAcquisition._pick_one: available_indices is empty.")
        
        x_full = _to_np_x(dataset_x)
        x_avail = _subset_x(dataset_x, available_indices, ds_mode)

        # Use only cumulative_indices from PREVIOUS cycles (genes with observed y-values)
        # Never use last_selected_indices - those are genes selected in THIS batch,
        # which don't have observed y-values yet
        effective_cum = cumulative_indices if (cumulative_indices is not None and len(cumulative_indices) > 0) else []
        
        if effective_cum is None or len(effective_cum) == 0:
            base_x = np.empty((0, x_full.shape[1])) if x_full.ndim > 1 else np.empty((0,))
            base_y = np.empty((0,))
            is_empty = True
        else:
            # When ds_mode=True, we need DataSource objects, not numpy arrays
            if ds_mode:
                base_x = dataset_x.subset(effective_cum)
                base_y = dataset_y.subset(effective_cum)
            else:
                base_x = _subset_x(dataset_x, effective_cum, ds_mode)
                base_y = _subset_y(dataset_y, effective_cum, ds_mode)
            is_empty = False

        # Base fit - skip if base is empty (no data to fit on yet)
        if not is_empty:
            try:
                # In ds_mode, base_x and base_y are already DataSource objects
                model.fit(base_x, base_y)
            except (IndexError, ValueError) as e:
                # If fitting fails (e.g., due to empty data issues), skip refitting
                # and use the pre-trained model as-is
                print(f"WARNING: model.fit failed in _pick_one with error: {e}. Using pre-trained model.")
                pass
        # If base is empty, model remains at its current state (pretrained from previous cycle)

        # Compute base posterior once
        # In ds_mode, pass dataset_x and available_indices so model.predict gets DataSource
        mu_base, cov_base = _predict_mu_cov(
            model, x_avail, ds_mode=ds_mode, dataset_x=dataset_x, indices=available_indices
        )

        scores = []
        for j in range(len(available_indices)):
            s = self._score_once(
                x_avail, j, base_x, base_y, model, ds_mode, mu_base=mu_base, cov_base=cov_base
            )
            scores.append(s)
        # np.alltrue was removed in NumPy 2; rely on np.all and guard
        # against degenerate all-equal score vectors by falling back to
        # the first candidate (agent-agnostic tie-breaking).
        scores_arr = np.asarray(scores, dtype=float)
        if scores_arr.size == 0:
            raise ValueError("GreedyKGSumAcquisition: no available candidates to score.")

        max_val = float(np.max(scores_arr))
        # Determine if all scores are (numerically) identical
        all_equal = np.all(np.isclose(scores_arr, max_val, rtol=1e-8, atol=1e-12))
        if all_equal:
            best_local = 0
        else:
            best_local = int(np.argmax(scores_arr))
        
        # Guard: ensure best_local is in bounds
        if best_local < 0 or best_local >= len(available_indices):
            raise IndexError(f"GreedyKGSumAcquisition: best_local={best_local} out of range for available_indices length={len(available_indices)}")
        
        return available_indices[best_local]

    # ------------------------------------------------------------------
    def __call__(
        self,
        dataset_x: AbstractDataSource,
        acquisition_batch_size: int,
        available_indices: List[AnyStr],
        last_selected_indices: List[AnyStr],
        cumulative_indices: List[AnyStr] | None = None,
        model: AbstractBaseModel | None = None,
        dataset_y: AbstractDataSource | None = None,
        temp_folder_name: str = "tmp/model/model.pt",
        H: Optional[int] = None,  # Allow overriding H at call time
        **kwargs,  # Accept additional kwargs for compatibility
    ) -> List[AnyStr]:
        if model is None:
            raise ValueError("GreedyKGSumAcquisition requires a predictive `model`.")
        
        # Guard: no candidates left
        if not available_indices or len(available_indices) == 0:
            return []
        
        # Allow overriding H based on dataset size
        if H is not None:
            self.H = int(H)
            print(f"INFO: GreedyKGSumAcquisition using H={self.H} (overridden at call time)")
        
        # Cap batch size to available candidates
        batch_size = min(acquisition_batch_size, len(available_indices))
        
        if self.compat_mode == "auto":
            ds_mode = not isinstance(dataset_x, np.ndarray)
        else:
            ds_mode = self.compat_mode == "datasource"

        if batch_size == 1 or self.batch_strategy == "sequential":
            chosen = []
            # cum is only for genes with observed y-values (from previous cycles)
            # Don't add newly selected genes to cum - they don't have y-values yet!
            cum = [] if cumulative_indices is None else list(cumulative_indices)
            avail = list(available_indices)
            for _ in range(batch_size):
                if not avail:
                    break
                pick = self._pick_one(
                    dataset_x,
                    avail,
                    cum,  # Only pass genes from previous cycles
                    model,
                    dataset_y,
                    ds_mode,
                    last_selected_indices=last_selected_indices,
                )
                chosen.append(pick)
                # DO NOT add pick to cum - it doesn't have a y-value yet!
                # cum.append(pick)  # <-- This was the bug
                try:
                    avail.remove(pick)
                except ValueError:
                    pass
            return chosen

        # topk-approx path: score each candidate once using the base fit, choose top-k
        x_full = _to_np_x(dataset_x)
        # Use only cumulative_indices from PREVIOUS cycles (genes with observed y-values)
        effective_cum = cumulative_indices if (cumulative_indices is not None and len(cumulative_indices) > 0) else []
        
        is_empty = (effective_cum is None or len(effective_cum) == 0)
        
        if is_empty:
            base_x = np.empty((0, x_full.shape[1])) if x_full.ndim > 1 else np.empty((0,))
            base_y = np.empty((0,))
        else:
            # When ds_mode=True, use DataSource objects directly (like sequential path does)
            if ds_mode:
                base_x = dataset_x.subset(effective_cum)
                base_y = dataset_y.subset(effective_cum)
            else:
                base_x = _subset_x(dataset_x, effective_cum, ds_mode)
                base_y = _subset_y(dataset_y, effective_cum, ds_mode)

        # Only fit if we have training data
        if not is_empty:
            # In ds_mode, base_x and base_y are already DataSource objects
            model.fit(base_x, base_y)

        x_avail = _subset_x(dataset_x, available_indices, ds_mode)
        mu_base, cov_base = _predict_mu_cov(
            model, x_avail, ds_mode=ds_mode, dataset_x=dataset_x, indices=available_indices
        )
        scores = []
        for j in range(len(available_indices)):
            s = self._score_once(
                x_avail, j, base_x, base_y, model, ds_mode, mu_base=mu_base, cov_base=cov_base
            )
            scores.append(s)
        
        # Guard empty scores (should not happen if available_indices is non-empty, but defensive)
        if not scores:
            return []
        
        idxs = np.argsort(scores)[-batch_size:]
        return list(np.asarray(available_indices, dtype=object)[idxs])

__all__ = ["GreedyKGSumAcquisition"]
