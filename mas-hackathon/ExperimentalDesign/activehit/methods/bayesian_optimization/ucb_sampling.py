"""
Copyright 2022 Clare Lyle, University of Oxford; Pascal Notin, University of Oxford
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
from slingpy import AbstractDataSource, AbstractBaseModel
from genedisco.active_learning_methods.acquisition_functions.base_acquisition_function import BaseBatchAcquisitionFunction

class UCBAcquisition(BaseBatchAcquisitionFunction):
    def __call__(self,
                dataset_x: AbstractDataSource,
                acquisition_batch_size: int,
                available_indices: List[AnyStr],
                last_selected_indices: List[AnyStr], 
                cumulative_indices: List[AnyStr] = None,
                model: AbstractBaseModel = None,
                dataset_y: AbstractDataSource = None,
                temp_folder_name: str = 'tmp/model',
                alpha = lambda t : 10.0, 
                t = 1,
                ) -> List:
        if type(dataset_x) == np.ndarray:
            avail_dataset_x = dataset_x[available_indices]
        else:
            avail_dataset_x = dataset_x.subset(available_indices)

        if model is None:
            raise ValueError("UCBAcquisition received model=None. Ensure a trained model with predict() is provided.")

        # Try with named kwarg first; if unsupported, fall back
        try:
            model_predictions = model.predict(avail_dataset_x, return_std_and_margin=True)
        except TypeError:
            model_predictions = model.predict(avail_dataset_x)
        # Support models that return (mean, var) or (mean, std/var, margins)
        if isinstance(model_predictions, tuple):
            if len(model_predictions) == 3:
                pred_mean, pred_uncertainties, pred_margins = model_predictions
            elif len(model_predictions) == 2:
                pred_mean, pred_uncertainties = model_predictions
                pred_margins = None
            elif len(model_predictions) == 1:
                pred_mean = model_predictions[0]
                pred_uncertainties = np.zeros_like(pred_mean)
                pred_margins = None
            else:
                # Unexpected; fall back defensively
                pred_mean = model_predictions[0]
                pred_uncertainties = np.zeros_like(pred_mean)
                pred_margins = None
        else:
            pred_mean = model_predictions
            pred_uncertainties = np.zeros_like(pred_mean)
            pred_margins = None

        pred_uncertainties = np.asarray(pred_uncertainties)
        assert len(pred_uncertainties.shape) in [1,2], "prediction uncertainties have unexpected shape"
        if len(pred_uncertainties.shape)==1:
            #pred_uncertainties already a 1D vector aligned with model_predictions
            pred_ubs = pred_mean + alpha(t) * pred_uncertainties
        elif len(pred_uncertainties.shape)==2:
            #pred_uncertainties is a 2D matrix that we extract the diagonal off
            uncs = np.diag(pred_uncertainties)
            pred_ubs = pred_mean + alpha(t) * uncs
        
        if len(pred_mean) < acquisition_batch_size:
            raise ValueError("The number of query samples exceeds"
                             "the size of the available data.")
        
        numerical_selected_indices = np.flip(
            np.argsort(pred_ubs)
        )[:acquisition_batch_size]
        selected_indices = [available_indices[i] for i in numerical_selected_indices]
        return selected_indices
