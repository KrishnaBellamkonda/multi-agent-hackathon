"""QThompsonFantasyBatchAcquisition
---------------------------------

Batch Thompson Sampling with sequential fantasy conditioning.

At each step:
  - draw one function sample on the CURRENT available set,
  - pick the argmax,
  - add a fantasy observation (either the sampled value, or believer mean),
  - condition the model (true fantasy if available; else believer re-fit),
  - continue until batch is full.

This implementation is compatible with numpy arrays and slingpy AbstractDataSource
inputs, and supports models exposing any of the following:
  - model.sample(X)
  - model.predict(X, return_std/return_std_and_margin)
  - model.get_model_prediction(X, return_multiple_preds=...)

It follows the call signature used by other acquisitions in this repo.
"""
from __future__ import annotations

from typing import AnyStr, List, Optional
import os
import numpy as np

from slingpy import AbstractDataSource, AbstractBaseModel
from genedisco.active_learning_methods.acquisition_functions.base_acquisition_function import (
    BaseBatchAcquisitionFunction,
)

# Optional: discobax numpy datasource wrapper for models expecting a DataSource
try:  # pragma: no cover - defensive import
    from activehit.data import abstract_numpy_class  # type: ignore
    HAS_DSX = True
except Exception:  # pragma: no cover
    HAS_DSX = False

# -----------------------------------------------------------------------------
# Small utilities (shared behavior with other acquisitions)
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

def _predict_mu_sigma(model, x: np.ndarray):
    """Return (mu, sigma_vector). If covariance is provided, take its diagonal sqrt."""
    if hasattr(model, "predict"):
        try:
            mu, s = model.predict(x, return_std_and_margin=True)
        except TypeError:
            mu, s = model.predict(x, return_std=True)
        mu = np.asarray(mu).reshape(-1)
        s = np.asarray(s)
        if s.ndim == 2:
            # covariance provided
            sig = np.sqrt(np.clip(np.diag(s), 0.0, None))
        else:
            sig = s.reshape(-1)
        return mu, sig
    if hasattr(model, "get_model_prediction"):
        try:
            out = model.get_model_prediction(x, return_std_and_margin=True)
            if isinstance(out, (tuple, list)) and len(out) >= 2:
                mu, s = out[0], out[1]
                mu = np.asarray(mu).reshape(-1)
                s = np.asarray(s).reshape(-1)
                return mu, s
        except Exception:
            pass
        # Fallback: single deterministic head, zero-uncertainty
        try:
            pred = model.get_model_prediction(x, return_multiple_preds=False)[0]
        except Exception:
            if not HAS_DSX:
                raise AttributeError("Need predict() or DSX wrapper for get_model_prediction(...).")
            pred = model.get_model_prediction(abstract_numpy_class.NumpyDataSource(x),
                                              return_multiple_preds=False)[0]
        mu = pred.detach().cpu().numpy().reshape(-1)
        sig = np.zeros_like(mu)
        return mu, sig
    raise AttributeError("Model must expose predict(...) or get_model_prediction(...).")

def _refit(model, x_fit: np.ndarray, y_fit: np.ndarray, ds_mode: bool):
    if ds_mode and HAS_DSX:
        model.fit(abstract_numpy_class.NumpyDataSource(x_fit),
                  abstract_numpy_class.NumpyDataSource(y_fit))
    else:
        model.fit(x_fit, y_fit)

def _try_save(model, path: str) -> bool:
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    except Exception:
        pass
    if hasattr(model, "save_folder"):
        try:
            model.save_folder(path)
            return True
        except Exception:
            return False
    if hasattr(model, "save"):
        try:
            model.save()
            return True
        except Exception:
            return False
    return False

def _try_load(model, path: str) -> bool:
    if hasattr(model, "load_folder"):
        try:
            model.load_folder(path)
            return True
        except Exception:
            return False
    if hasattr(model, "load"):
        try:
            model.load()
            return True
        except Exception:
            return False
    return False

# -----------------------------------------------------------------------------
# Acquisition
# -----------------------------------------------------------------------------
class QThompsonFantasyBatchAcquisition(BaseBatchAcquisitionFunction):
    """
    Batch Thompson Sampling with sequential fantasy conditioning.
    At each step:
      - draw one function sample on the CURRENT available set,
      - pick the argmax,
      - add a fantasy observation (either the sampled value, or believer mean),
      - condition the model (true fantasy if available; else believer re-fit),
      - continue until batch is full.
    """

    def __init__(self, use_sampled_value_as_fantasy: bool = False, compat_mode: str = "auto", n_function_draws: int = 1, tie_eps: float = 1e-8):
        """
        Args:
            use_sampled_value_as_fantasy: If True, fantasy adds sampled value; else believer mean.
            compat_mode: 'auto'|'datasource' controls dataset handling.
            n_function_draws: Average this many function samples per step before selecting argmax.
            tie_eps: When two candidates are within tie_eps on the averaged draw, break ties by posterior mean.
        """
        self.use_sampled_value_as_fantasy = bool(use_sampled_value_as_fantasy)
        self.compat_mode = compat_mode
        self.n_function_draws = max(1, int(n_function_draws))
        self.tie_eps = float(tie_eps)

    def _sample_function_values(self, model, x: np.ndarray, n_samples: int = 1) -> np.ndarray:
        # native sampler
        if hasattr(model, "sample"):
            draws = [np.asarray(model.sample(x)).reshape(-1) for _ in range(n_samples)]
            return np.stack(draws, axis=0)
        # use (mu, cov)
        if hasattr(model, "predict"):
            try:
                mu, s = model.predict(x, return_std_and_margin=True)
            except TypeError:
                mu, s = model.predict(x, return_std=True)
            mu = np.asarray(mu).reshape(-1)
            s = np.asarray(s)
            cov = s if s.ndim == 2 else np.diag(np.square(s.reshape(-1)))
            return np.random.multivariate_normal(mu, cov, size=n_samples)
        # stochastic head
        if hasattr(model, "get_model_prediction"):
            try:
                fx = model.get_model_prediction(
                    x,
                    return_multiple_preds=True,
                    num_target_samples=n_samples,
                )[0]
            except Exception:
                if not HAS_DSX:
                    raise AttributeError(
                        "Need predict(...) or DSX wrapper for get_model_prediction(...)."
                    )
                fx = model.get_model_prediction(
                    abstract_numpy_class.NumpyDataSource(x),
                    return_multiple_preds=True,
                    num_target_samples=n_samples,
                )[0]
            f = fx.detach().cpu()
            if f.ndim == 1:
                f = f.view(-1, n_samples)
            elif f.shape[0] != x.shape[0]:
                f = f.view(x.shape[0], n_samples)
            return f.T.numpy()
        raise AttributeError("Model does not support sampling/predictive distribution.")

    def _fantasy_update(
        self,
        model,
        x_new: np.ndarray,
        y_new: float,
        base_x: np.ndarray,
        base_y: np.ndarray,
        ds_mode: bool,
    ):
        # try true fantasy first
        if hasattr(model, "fantasy"):
            try:
                return model.fantasy(x_new, np.array([y_new]))
            except Exception:
                pass
        inner = getattr(model, "model", None) or getattr(model, "_model", None)
        if inner is not None and hasattr(inner, "get_fantasy_model"):
            try:
                import torch
                X_t = torch.as_tensor(x_new).float()
                y_t = torch.as_tensor([y_new]).float()
                fmod = inner.get_fantasy_model(X_t, y_t)
                if hasattr(model, "copy_with_new_inner"):
                    return model.copy_with_new_inner(fmod)
            except Exception:
                pass
        # believer
        x_fit = np.concatenate([base_x, x_new], axis=0)
        y_fit = np.concatenate([np.asarray(base_y).reshape(-1), np.array([y_new])], axis=0)
        _refit(model, x_fit, y_fit, ds_mode)
        return model

    def __call__(
        self,
        dataset_x: AbstractDataSource,
        acquisition_batch_size: int,
        available_indices: List[AnyStr],
        last_selected_indices: List[AnyStr],
        cumulative_indices: List[AnyStr] = None,
        model: AbstractBaseModel = None,
        dataset_y: AbstractDataSource = None,
        temp_folder_name: str = "tmp/model/model.pt",
    ) -> List[AnyStr]:
        if model is None:
            raise ValueError("QThompsonFantasyBatchAcquisition requires a predictive `model`.")
        ds_mode = (
            self.compat_mode == "datasource"
            if self.compat_mode != "auto"
            else not isinstance(dataset_x, np.ndarray)
        )
        x_full = _to_np_x(dataset_x)
        x_avail = _subset_x(dataset_x, available_indices, ds_mode)

        if cumulative_indices is None or len(cumulative_indices) == 0:
            base_x = np.empty((0, x_full.shape[1])) if x_full.ndim > 1 else np.empty((0,))
            base_y = np.empty((0,))
        else:
            base_x = _subset_x(dataset_x, cumulative_indices, ds_mode)
            base_y = _subset_y(dataset_y, cumulative_indices, ds_mode)

        # base fit
        _refit(model, base_x, base_y, ds_mode)

        chosen_local: List[int] = []
        avail_mask = np.ones(len(available_indices), dtype=bool)

        saved = _try_save(model, "tmp/qts_model.pt")

        for _ in range(acquisition_batch_size):
            x_cur = x_avail[avail_mask]
            # average over n function draws for robustness
            draws = self._sample_function_values(model, x_cur, n_samples=self.n_function_draws)
            draw_mean = draws.mean(axis=0)
            # tie-break using posterior mean if needed
            j_local = int(np.argmax(draw_mean))
            # check for near-ties
            if self.tie_eps > 0.0:
                near = np.where(np.abs(draw_mean - draw_mean[j_local]) <= self.tie_eps)[0]
                if near.size > 1:
                    mu_cur, _ = _predict_mu_sigma(model, x_cur)
                    j_local = int(near[np.argmax(mu_cur[near])])
            # map to local-global within avail set
            rem = np.flatnonzero(avail_mask)
            j_global_local = rem[j_local]
            chosen_local.append(j_global_local)
            avail_mask[j_global_local] = False

            # fantasy add picked point
            if self.use_sampled_value_as_fantasy:
                y_fant = float(draw_mean[j_local])
            else:
                mu_cur, _ = _predict_mu_sigma(model, x_cur)
                y_fant = float(mu_cur[j_local])
            x_new = x_cur[j_local : j_local + 1]
            model = self._fantasy_update(model, x_new, y_fant, base_x, base_y, ds_mode)

            # also update our base containers (for believer case)
            base_x = np.concatenate([base_x, x_new], axis=0)
            base_y = np.concatenate([base_y, np.array([y_fant])], axis=0)

        if saved:
            _try_load(model, "tmp/qts_model.pt")
        return list(np.asarray(available_indices, dtype=object)[chosen_local])

__all__ = ["QThompsonFantasyBatchAcquisition"]
