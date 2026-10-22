"""Thresholded q-Thompson Sampling with sequential fantasy conditioning.

Variant of q-Thompson fantasy batch acquisition that respects a threshold τ when
choosing each point. At every inner step j of the batch construction:
  1. Sample one function draw over the remaining candidates.
  2. Form the hit set H = {g : f_draw(g) > τ}.
  3. If H non-empty select argmax within H; else select global argmax.
  4. Create a fantasy observation at the chosen point using either the sampled
     value (sampled fantasy) or the posterior mean (believer fantasy).
  5. Condition posterior on this fantasy (true fantasy model if supported,
     otherwise believer refit) and continue.

The final batch is returned; real observations are incorporated by the outer
loop of the application afterward.
"""
from __future__ import annotations

from typing import AnyStr, List, Optional
import numpy as np

from slingpy import AbstractDataSource, AbstractBaseModel
from genedisco.active_learning_methods.acquisition_functions.base_acquisition_function import (
    BaseBatchAcquisitionFunction,
)

try:  # optional DataSource wrapper
    from activehit.data import abstract_numpy_class  # type: ignore
    HAS_DSX = True
except Exception:
    HAS_DSX = False

# -----------------------------------------------------------------------------
# Utilities reused from other methods (minimal duplication for isolation)
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
    if hasattr(model, "predict"):
        try:
            mu, s = model.predict(x, return_std_and_margin=True)
        except TypeError:
            mu, s = model.predict(x, return_std=True)
        mu = np.asarray(mu).reshape(-1)
        s = np.asarray(s)
        if s.ndim == 2:
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
        # deterministic fallback
        try:
            pred = model.get_model_prediction(x, return_multiple_preds=False)[0]
        except Exception:
            if not HAS_DSX:
                raise AttributeError("Need predict() or DSX wrapper for get_model_prediction().")
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

# -----------------------------------------------------------------------------
# Acquisition class
# -----------------------------------------------------------------------------
class TQThompsonSamplingBatchAcquisition(BaseBatchAcquisitionFunction):
    def __init__(
        self,
        threshold: float = 0.0,
        use_sampled_value_as_fantasy: bool = True,
        compat_mode: str = "auto",
    ):
        self.threshold = float(threshold)
        self.use_sampled_value_as_fantasy = bool(use_sampled_value_as_fantasy)
        self.compat_mode = compat_mode

    def _sample_function_values(self, model, x: np.ndarray, n_samples: int = 1) -> np.ndarray:
        if hasattr(model, "sample"):
            draws = [np.asarray(model.sample(x)).reshape(-1) for _ in range(n_samples)]
            return np.stack(draws, axis=0)
        if hasattr(model, "predict"):
            try:
                mu, s = model.predict(x, return_std_and_margin=True)
            except TypeError:
                mu, s = model.predict(x, return_std=True)
            mu = np.asarray(mu).reshape(-1)
            s = np.asarray(s)
            cov = s if s.ndim == 2 else np.diag(np.square(s.reshape(-1)))
            return np.random.multivariate_normal(mu, cov, size=n_samples)
        if hasattr(model, "get_model_prediction"):
            try:
                fx = model.get_model_prediction(
                    x,
                    return_multiple_preds=True,
                    num_target_samples=n_samples,
                )[0]
            except Exception:
                if not HAS_DSX:
                    raise AttributeError("Need predict() or DSX wrapper for get_model_prediction(...).")
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

    def _fantasy_update(self, model, x_new: np.ndarray, y_new: float,
                        base_x: np.ndarray, base_y: np.ndarray, ds_mode: bool):
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
        # believer fallback
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
            raise ValueError("TQThompsonSamplingBatchAcquisition requires a predictive `model`.")
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

        _refit(model, base_x, base_y, ds_mode)

        chosen_local: List[int] = []
        avail_mask = np.ones(len(available_indices), dtype=bool)

        tau = self.threshold
        for _ in range(acquisition_batch_size):
            x_cur = x_avail[avail_mask]
            draw = self._sample_function_values(model, x_cur, n_samples=1)[0]
            hits_local = np.where(draw > tau)[0]
            if hits_local.size > 0:
                j_local = int(hits_local[np.argmax(draw[hits_local])])
            else:
                j_local = int(np.argmax(draw))  # fallback to global best
            rem = np.flatnonzero(avail_mask)
            j_global_local = rem[j_local]
            chosen_local.append(j_global_local)
            avail_mask[j_global_local] = False

            if self.use_sampled_value_as_fantasy:
                y_fant = float(draw[j_local])
            else:
                mu_cur, _ = _predict_mu_sigma(model, x_cur)
                y_fant = float(mu_cur[j_local])
            x_new = x_cur[j_local : j_local + 1]
            model = self._fantasy_update(model, x_new, y_fant, base_x, base_y, ds_mode)
            base_x = np.concatenate([base_x, x_new], axis=0)
            base_y = np.concatenate([base_y, np.array([y_fant])], axis=0)

        return list(np.asarray(available_indices, dtype=object)[chosen_local])

__all__ = ["TQThompsonSamplingBatchAcquisition"]
