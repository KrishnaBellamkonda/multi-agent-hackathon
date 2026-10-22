"""
Copyright 2022 Clare Lyle, University of Oxford
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
import numpy as np
from typing import AnyStr, List
from slingpy import AbstractDataSource
from slingpy.models.abstract_base_model import AbstractBaseModel
from genedisco.active_learning_methods.acquisition_functions.base_acquisition_function import (
    BaseBatchAcquisitionFunction,
)
from activehit.methods.bax_acquisition import bax_sampling_toy_experiment

top_k_idx = bax_sampling_toy_experiment.top_k_idx


class ThompsonSamplingcquisition(BaseBatchAcquisitionFunction):
    def __call__(
        self,
        dataset_x: AbstractDataSource,
        select_size: int,
        available_indices: List[AnyStr],
        cumulative_indices: List[AnyStr] = None,
        model: AbstractBaseModel = None,
        dataset_y: AbstractDataSource = None,
        alpha=lambda t: 10.0,
        t=1,
    ) -> List:
        """Sample a joint GP posterior over the available points and pick the top draws."""
        if model is None:
            raise ValueError("Thompson sampling requires a trained model with posterior covariance.")

        if isinstance(dataset_x, np.ndarray):
            avail_dataset_x = dataset_x[available_indices]
        else:
            avail_dataset_x = dataset_x.subset(available_indices)

        try:
            # Request full covariance for correlated posterior samples
            pred = model.predict(avail_dataset_x, return_std_and_margin=True, return_full_cov=True)
        except TypeError:
            # Old API without return_full_cov parameter
            pred = model.predict(avail_dataset_x, return_std_and_margin=True)
        except Exception as exc:
            raise RuntimeError("GP model failed to provide joint posterior for Thompson sampling") from exc

        pred_mean = np.asarray(pred[0]).reshape(-1)
        pred_cov = np.asarray(pred[1])
        if pred_cov.ndim == 1:
            pred_cov = np.diag(pred_cov)

        # Symmetrize and add jitter to guard against numerical issues
        pred_cov = 0.5 * (pred_cov + pred_cov.T)
        jitter = 1e-9 * np.eye(pred_cov.shape[0])
        pred_cov = pred_cov + jitter

        try:
            sample = np.random.multivariate_normal(pred_mean, pred_cov)
        except np.linalg.LinAlgError:
            # Fallback: ensure positive definiteness with larger jitter, or sample independently
            try:
                jitter = 1e-6 * np.eye(pred_cov.shape[0])
                sample = np.random.multivariate_normal(pred_mean, pred_cov + jitter)
            except np.linalg.LinAlgError:
                # Final fallback: independent sampling using diagonal variances
                diag = np.clip(np.diag(pred_cov), a_min=1e-9, a_max=None)
                sample = pred_mean + np.random.standard_normal(len(pred_mean)) * np.sqrt(diag)

        best_indices = top_k_idx(sample.reshape(-1), select_size)
        proposal = np.asarray(available_indices)[best_indices]
        return proposal
