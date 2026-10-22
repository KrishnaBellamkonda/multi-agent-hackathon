"""
Copyright 2022 Arash Mehrjou, GlaxoSmithKline plc
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


import six
import os
import numpy as np
import sklearn.datasets as dt
from abc import ABCMeta, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, AnyStr
from slingpy.utils.path_tools import PathTools
from activehit.data.abstract_data_synthesizer import AbastractDataSynthesizer
import pickle
import json
from activehit.data.utils.mog import mog_1d
from activehit.data.utils.sin1 import sine_1d
from activehit.data.utils.sin_2d import sine_2d


class GTDataSynthesizer(AbastractDataSynthesizer):
    """
    Class to generate synthetic datasets for benchmarking discobax methods
    """

    def __init__(
        self,
        mode: AnyStr = "sine1d",  # [sine1d|sine2d|sinc1d|sinc2d|mog1d|mog2d]
        n_samples: int = 100,
        n_clusters: Optional[int] = 2,  # only applicable to mog mode
        noise_std: Optional[float] = 0.0,
        seed: Optional[int] = 2022,
        saved_directory: Optional[AnyStr] = None,
        **kwargs,
    ):
        """
        Initialize the data generator class with groundtruth analytical functions.


        Args:
            mode (str): [sine|sinc|mog].
            n_samples (int):
            n_features (int): The number of features which determines the dimension of the data.
            n_clusters (int): The number of clusters (components) in the MoG mode.
            noise_std (float): The standard deviation of the gaussian noise applied to the output.
            seed (int): The random seed. Keep it fixed to generate the same datasets.
        """
        n_features = 1 if mode in set(["sine1d", "sinc1d", "mog1d"]) else 2
        super(GTDataSynthesizer, self).__init__(
            mode, n_samples, n_features, 1, noise_std, seed, saved_directory
        )
        self.kwargs = kwargs
        self.n_clusters = n_clusters

    def _generate_data(self):
        if self.mode == "mog1d":
            # 1D mixture of Gaussians with seed-driven variability and optional complexity.
            x, y, gt_function = mog_1d(
                n_samples=self.n_samples,
                n_components=self.n_clusters,
                seed=self.seed,
                distribution_pattern=self.kwargs.get("mog_distribution_pattern", "balanced"),
                background_weight=self.kwargs.get("mog_background_weight", 0.0),
                tail=self.kwargs.get("mog_tail", "gaussian"),
                student_df=self.kwargs.get("mog_student_df", 3),
            )
            self.dataset = (x, y)
            self.gt_function = gt_function
        elif self.mode == "sine1d":
            x, y, gt_function = sine_1d(n_samples=self.n_samples, seed=self.seed)
            self.dataset = (x, y)
            self.gt_function = gt_function
        elif self.mode == "sine2d":
            x, y, gt_function = sine_2d(n_samples=self.n_samples, seed=self.seed)
            self.dataset = (x, y)
            self.gt_function = gt_function
        else:
            return NotImplementedError()

        if self.saved_directory:
            PathTools.mkdir_if_not_exists(self.saved_directory)
            datadict = {
                "metadata": self.get_dataset_metadata(),
                "data": self.dataset,
            }
            filepath = os.path.join(self.saved_directory, "datadict.pkl")
            with open(filepath, "wb") as fp:
                pickle.dump(datadict, fp, pickle.HIGHEST_PROTOCOL)

            # If our ground-truth function is annotated (e.g., MoG), persist its parameters for audit
            try:
                params = {}
                for key in [
                    "means",
                    "stds",
                    "weights",
                    "background_weight",
                    "pattern",
                    "heavy_tail_mask",
                    "tail",
                    "student_df",
                    "domain",
                ]:
                    if hasattr(self.gt_function, key):
                        val = getattr(self.gt_function, key)
                        # Convert numpy types to lists
                        if hasattr(val, "tolist"):
                            val = val.tolist()
                        params[key] = val
                if params:
                    with open(os.path.join(self.saved_directory, "gt_params.json"), "w") as jf:
                        json.dump(params, jf, indent=2)
            except Exception:
                pass

    def get_groundtruth_function(self):
        return self.gt_function


if __name__ == "__main__":
    # Example syntehtic dataset generation and ploting
    params = {
        "mode": "mog1d",
        "n_samples": 100,
    }
    synth = GTDataSynthesizer(**params)
    synth._generate_data()
    print(synth.get_groundtruth_function())
