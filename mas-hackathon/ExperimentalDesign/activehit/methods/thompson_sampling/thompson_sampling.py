"""
Copyright 2022 Pascal Notin, University of Oxford
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
from activehit.methods.bax_acquisition import bax_sampling
from activehit.data import abstract_numpy_class

top_k_idx = bax_sampling.top_k_idx


class ThompsonSamplingcquisition(BaseBatchAcquisitionFunction):
    def __call__(
        self,
        dataset_x: AbstractDataSource,
        acquisition_batch_size: int,
        available_indices: List[AnyStr],
        last_selected_indices: List[AnyStr],
        cumulative_indices: List[AnyStr] = None,
        model: AbstractBaseModel = None,
        dataset_y: AbstractDataSource = None,
        temp_folder_name: str = "tmp/model",
    ) -> List:
        if type(dataset_x) == np.ndarray:
            avail_dataset_x = dataset_x[available_indices]
        else:
            avail_dataset_x = dataset_x.subset(available_indices)

        # FIXED: Sample from posterior via MC dropout
        # Use multiple dropout samples (10-20) and pick one to get a good posterior sample
        # More samples = better approximation of posterior uncertainty
        num_mc_samples = 15  # Typical choice for MC dropout
        f_samples = model.get_model_prediction(
            avail_dataset_x, 
            return_multiple_preds=True,
            num_target_samples=num_mc_samples
        )[0]

        # f_samples has shape (n_candidates, num_mc_samples) or similar
        # For Thompson sampling, we want to sample ONE function from the posterior
        # Pick a random column (one complete function draw across all candidates)
        if hasattr(f_samples, 'detach'):
            f_samples_np = f_samples.detach().cpu().numpy()
        else:
            f_samples_np = np.asarray(f_samples)
        
        # Reshape to (n_candidates, num_samples)
        if f_samples_np.ndim == 1:
            f_samples_np = f_samples_np.reshape(-1, 1)
        elif f_samples_np.shape[0] == num_mc_samples:
            # Transpose if samples are in first dimension
            f_samples_np = f_samples_np.T
        
        # Randomly select one of the MC dropout samples as our function draw
        sample_idx = np.random.randint(0, f_samples_np.shape[1])
        f = f_samples_np[:, sample_idx].flatten()

        best_indices = top_k_idx(f, acquisition_batch_size)
        proposal = np.asarray(available_indices)[best_indices]
        return proposal
