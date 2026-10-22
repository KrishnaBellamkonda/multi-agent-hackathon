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
import matplotlib.pyplot as plt

def sine_1d(
    n_samples: Optional[int] = 100,
    limits: Tuple[float, float] = [-2 * np.pi, 2 * np.pi],
    seed: Optional[int] = None,
):
    """Create a 1D sinusoid dataset with seed-controlled variability.

    The original implementation was fully deterministic, which made runs look
    identical across seeds. Here, we introduce small, seed-driven variations in
    phase and amplitudes while keeping the overall shape comparable.
    """

    rng = np.random.default_rng(seed)

    # Base frequencies (kept stable) and seed-dependent amplitudes/phases
    w1 = 2.5 / np.pi
    w2 = 2 * np.pi / 3
    w3 = 10.0
    a1 = 1.0 * (1.0 + 0.10 * rng.normal())
    a2 = 0.95 * (1.0 + 0.10 * rng.normal())
    a3 = 0.25 * (1.0 + 0.10 * rng.normal())
    p1 = rng.uniform(-np.pi, np.pi)
    p2 = rng.uniform(-np.pi, np.pi)
    p3 = rng.uniform(-np.pi, np.pi)

    def sin_fn(x) -> Callable:
        return (
            a1 * np.sin(w1 * x + p1)
            + a2 * np.cos(w2 * x + p2)
            + a3 * np.cos(w3 * x + p3)
        )

    x_samples = np.linspace(limits[0], limits[1], n_samples)
    y_samples = sin_fn(x_samples)

    return x_samples, y_samples, sin_fn

if __name__ == "__main__":
    xs, ys, f = sine_1d()
    plt.plot(xs, ys)
    plt.savefig("sin.png")