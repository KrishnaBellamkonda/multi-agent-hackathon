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

******************
Contains the utility functions for creating 1 and 2 dimensional MoG functions
and samples from these functions.
"""

import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, AnyStr, Sequence
import scipy.stats as ss
import functools
import itertools 
import pdb


def cartesian_product(arrays):
    la = len(arrays)
    print(la)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)

def product_itertools(arrays):
    return np.array(list(itertools.product(*arrays)))

def sine_2d(
    n_samples: Optional[int] = 16 ** 2,
    limits: Tuple[float, float] = [-2 * np.pi, 2 * np.pi],
    seed: Optional[int] = None,
):
    """Create a 2D sinusoid dataset with seed-controlled variability.

    Adds mild randomness to the weight matrix and phases so different seeds
    produce distinct yet comparable landscapes. Keeps domain fixed.
    """
    rng = np.random.default_rng(seed)
    base_weights = np.asarray([[0.25, -1 / np.pi], [0.1, 0.02]])
    jitter = rng.normal(scale=0.05, size=base_weights.shape)
    weights = base_weights + jitter
    phases = rng.uniform(-np.pi, np.pi, size=(weights.shape[0],))

    def sin_fn(x) -> Callable:
        # weights @ x.T -> (components, N); add phases per component then average
        raw = weights @ x.T + phases[:, None]
        return np.mean(np.sin(raw), axis=0)

    side = int(n_samples ** (1 / 2))
    linspace = np.linspace(-np.pi, np.pi, side)
    x_samples = product_itertools([linspace] * 2)
    y_samples = sin_fn(x_samples)

    return x_samples, y_samples, sin_fn

if __name__ == "__main__":
    # Quick manual sanity run
    xs, ys, f = sine_2d(seed=0)
    print("Generated 2D sine with", xs.shape[0], "points; y range:", (float(np.min(ys)), float(np.max(ys))))