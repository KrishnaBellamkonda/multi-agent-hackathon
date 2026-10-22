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


def mog_1d(
    n_samples: Optional[int] = 100,
    limits: Tuple[float, float] = (-1.0, 1.0),
    means: Optional[Sequence[float]] = None,
    stds: Optional[Sequence[float]] = None,
    weights: Optional[Sequence[float]] = None,
    n_components: Optional[int] = None,
    seed: Optional[int] = None,
    distribution_pattern: Optional[str] = "balanced",
    background_weight: float = 0.0,
    tail: str = "gaussian",  # or 'student'
    student_df: int = 3,
):
    """Create a 1D Mixture-of-Gaussians ground-truth function with optional complexity.

    Complexity controls (set via arguments):
    - distribution_pattern:
        * 'balanced'   : Components similar scale and weights (default).
        * 'unbalanced' : Sparse weight vector (few dominant components).
        * 'multiscale' : Mix of narrow and wide components.
        * 'clustered'  : Parent cluster centers with jittered sub-components.
        * 'heavy_tail' : Some components use Student-t kernels (via 'tail').
    - background_weight: Adds a uniform background mass over the domain.
    - tail: 'gaussian' or 'student' (heavy-tailed). If 'heavy_tail' pattern is
      selected, half the components are Student-t regardless of tail arg.
    - student_df: Degrees of freedom for Student-t heavy-tailed components.

    If means/stds/weights supplied, they override automatic generation for deterministic behavior.
    Returned pdf object is annotated with parameters (means, stds, weights, pattern, tail, background_weight).
    """

    rng = np.random.default_rng(seed)

    # Determine number of components
    if means is not None:
        n_comp = len(means)
    elif n_components is not None:
        n_comp = int(n_components)
    else:
        n_comp = 2

    low, high = float(limits[0]), float(limits[1])
    width = high - low

    pattern = (distribution_pattern or "balanced").lower()

    # Generate means if not supplied
    if means is None:
        if pattern == "clustered" and n_comp >= 4:
            n_parent = max(2, n_comp // 3)
            parent_centers = rng.uniform(low + 0.2 * width, high - 0.2 * width, size=n_parent)
            means_list = []
            for i in range(n_comp):
                parent = parent_centers[i % n_parent]
                jitter_scale = 0.05 * width
                means_list.append(parent + rng.normal(scale=jitter_scale))
            means = np.sort(np.clip(means_list, low + 0.05 * width, high - 0.05 * width))
        else:
            means = np.sort(rng.uniform(low + 0.1 * width, high - 0.1 * width, size=n_comp))
    means = np.asarray(means, dtype=float)

    # Generate stds
    if stds is None:
        if pattern == "multiscale" and n_comp >= 2:
            narrow_ct = n_comp // 2
            wide_ct = n_comp - narrow_ct
            narrow_stds = rng.uniform(0.02 * width, 0.08 * width, size=narrow_ct)
            wide_stds = rng.uniform(0.12 * width, 0.30 * width, size=wide_ct)
            stds = np.concatenate([narrow_stds, wide_stds])
            rng.shuffle(stds)
        else:
            stds = rng.uniform(0.05 * width, 0.2 * width, size=n_comp)
    stds = np.asarray(stds, dtype=float)

    # Weights
    if weights is None:
        if pattern == "unbalanced":
            # Sparse concentration -> a few large components
            alpha = np.full(n_comp, 0.5)
            weights = rng.dirichlet(alpha=alpha)
        else:
            weights = rng.dirichlet(alpha=np.ones(n_comp))
    weights = np.asarray(weights, dtype=float)
    weights = weights / np.sum(weights)

    # Optional uniform background component (not part of weights sum; separate additive mass)
    bg_w = max(0.0, float(background_weight))
    if bg_w > 0.5:
        bg_w = 0.5  # cap to keep dominated by mixture

    # Determine which components are heavy-tailed (Student-t)
    heavy_tail_mask = np.zeros(n_comp, dtype=bool)
    if pattern == "heavy_tail":
        # Half heavy-tailed
        heavy_idx = rng.choice(n_comp, size=max(1, n_comp // 2), replace=False)
        heavy_tail_mask[heavy_idx] = True
    elif tail == "student":
        # All components heavy-tailed when tail param explicitly student
        heavy_tail_mask[:] = True

    def mog_pdf(
        means: Optional[Sequence[float]] = means,
        stds: Optional[Sequence[float]] = stds,
        weights: Optional[Sequence[float]] = weights,
    ) -> Callable:
        component_pdfs = []
        n_components_local = len(means)
        for comp_index in range(n_components_local):
            loc = means[comp_index]
            scale = stds[comp_index]
            if heavy_tail_mask[comp_index]:
                # Student-t component
                component_pdfs.append(
                    functools.partial(ss.t.pdf, df=student_df, loc=loc, scale=scale)
                )
            else:
                component_pdfs.append(
                    functools.partial(ss.norm.pdf, loc=loc, scale=scale)
                )

        def pdf_func(x):
            x_arr = np.asarray(x, dtype=float)
            out = 0.0
            for comp_index in range(n_components_local):
                out += weights[comp_index] * component_pdfs[comp_index](x_arr)
            if bg_w > 0.0:
                # Uniform background over domain
                out += bg_w * (1.0 / width)
            return out

        # Annotate for downstream inspection/debugging
        pdf_func.means = means
        pdf_func.stds = stds
        pdf_func.weights = weights
        pdf_func.background_weight = bg_w
        pdf_func.pattern = pattern
        pdf_func.heavy_tail_mask = heavy_tail_mask
        pdf_func.tail = tail
        pdf_func.student_df = student_df
        pdf_func.domain = (low, high)
        return pdf_func

    pdf = mog_pdf()
    x_samples = np.linspace(low, high, n_samples)
    y_samples = pdf(x_samples)

    return x_samples, y_samples, pdf

    def mog_pdf(
        means: Optional[Sequence[float]] = [-0.5, 0.5],
        stds: Optional[Sequence[float]] = [0.1, 0.1],
        weights: Optional[Sequence[float]] = [0.3, 0.7],
    ) -> Callable:
        component_pdfs = []
        n_components = len(means)
        for comp_index in range(n_components):
            component_pdfs.append(
                functools.partial(
                    ss.norm.pdf, loc=means[comp_index], scale=stds[comp_index]
                )
            )

        def pdf_func(x):
            out = 0
            for comp_index in range(n_components):
                out += weights[comp_index] * component_pdfs[comp_index](x)
            return out

        return pdf_func

    pdf = mog_pdf(means, stds, weights)
    x_samples = np.linspace(low, high, n_samples)
    y_samples = pdf(x_samples)

    return x_samples, y_samples, pdf
