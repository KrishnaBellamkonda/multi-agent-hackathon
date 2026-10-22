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
import torch
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.distributions import MultivariateNormal

from slingpy import AbstractDataSource, AbstractBaseModel
from genedisco.active_learning_methods.acquisition_functions.base_acquisition_function import (
    BaseBatchAcquisitionFunction,
)
from activehit.data import abstract_numpy_class
from activehit.models import toy_experiment_models


"""BAX acquisition function for top k estimation """


class BaxAcquisition(BaseBatchAcquisitionFunction):
    def __init__(
        self,
        objective_function: str = "topk",  # BAX objective function [topk,level_set,DiscoBAX]
        k: int = 3,  # k in topk (Top-k BAX only)
        c: float = 1.0,  # level value in level-set (Level-set BAX only)
        subset_size: int = 5,  # subset size in subset select (DiscoBAX only)
        noise_type: str = "additive",  # noise type in subset select [additive|multiplicative] (DiscoBAX only)
        noise_lengthscale: float = 1.0,  # lengthscale param for noise computation (DiscoBAX only)
        noise_outputscale: float = 1.0,  # output scale param for noise computation (DiscoBAX only)
        num_samples_EIG: int = 20,  # number of samples to compute EIG (All BAX)
        num_samples_entropy: int = 20,  # number of samples for entropy computation (All BAX)
        entropy_average_mode: str = "harmonic",  # type of average across samples in entropy computation [harmonic|arithmetic] (All BAX)
        batch_selection_mode: str = "topk_EIG",  # method to select all points in acquisition batch [topk_EIG, best_subset]. best_subset only valid if subset_size = batch size (All BAX)
        compat_mode: str = "datasource",  # 'datasource' (original), 'toy' (numpy), or 'auto'
    ):
        self.objective_function = objective_function
        self.k = k
        self.c = c
        self.noise_type = noise_type
        self.noise_lengthscale = noise_lengthscale
        self.noise_outputscale = noise_outputscale
        self.subset_size = subset_size
        self.num_samples_EIG = num_samples_EIG
        self.num_samples_entropy = num_samples_entropy
        self.batch_selection_mode = batch_selection_mode
        self.entropy_average_mode = entropy_average_mode
        self.compat_mode = compat_mode

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
    ) -> List:
        # Decide operating mode
        if self.compat_mode == "auto":
            ds_mode = not isinstance(dataset_x, np.ndarray)
        else:
            ds_mode = (self.compat_mode == "datasource")

        # Subset of available candidates
        dataset_x_avail = (
            dataset_x.subset(available_indices) if ds_mode else dataset_x[available_indices]
        )

        hxs = []
        outputs = []

        # Save initial model state for reloading in MC loop
        if ds_mode and hasattr(model, "save_folder"):
            model.save_folder(temp_folder_name)
        elif (not ds_mode) and hasattr(model, "save"):
            model.save()

        # We obtain several MC samples to estimate the second term in Equation 1
        for j in range(self.num_samples_EIG):
            # Reload base model state
            if ds_mode and hasattr(model, "load_folder"):
                model.load_folder(temp_folder_name)
            elif (not ds_mode) and hasattr(model, "load"):
                model.load()

            # Predict f on full X and keep numpy view of X
            if ds_mode:
                x_full = dataset_x.get_data()[0]
                if hasattr(model, "get_model_prediction"):
                    f = (
                        model.get_model_prediction(dataset_x, return_multiple_preds=False)[0]
                        .flatten()
                        .detach()
                        .numpy()
                    )
                elif hasattr(model, "predict"):
                    mu, _ = model.predict(x_full, return_std_and_margin=True)
                    f = np.asarray(mu).reshape(-1)
                else:
                    raise AttributeError(
                        "Model does not support prediction interfaces expected by BAX acquisition."
                    )
            else:
                x_full = dataset_x if isinstance(dataset_x, np.ndarray) else dataset_x.get_data()[0]
                if hasattr(model, "predict"):
                    mu, _ = model.predict(x_full, return_std_and_margin=True)
                    f = np.asarray(mu).reshape(-1)
                elif hasattr(model, "get_model_prediction"):
                    f = (
                        model.get_model_prediction(dataset_x, return_multiple_preds=False)[0]
                        .flatten()
                        .detach()
                        .numpy()
                    )
                else:
                    raise AttributeError(
                        "Model does not support prediction interfaces expected by BAX acquisition."
                    )

            # Compute S_j based on the sampled (f_ip)_j
            if self.objective_function == "topk":
                out = top_k_idx(f, self.k)
            elif self.objective_function == "level_set":
                out = level_set(f, self.c)
            elif self.objective_function == "discobax":
                out = subset_select(
                    f,
                    noise_subset_select(
                        self.noise_type,
                        x_full,
                        self.noise_lengthscale,
                        self.noise_outputscale,
                    ),
                    self.subset_size,
                )
            elif self.objective_function == "jepig":
                out = np.random.choice(len(f), self.subset_size, replace=False)
            outputs.append(out)

            # Prepare concatenated training data (previous selections + newly selected)
            if cumulative_indices is None:
                prev_x = np.empty((0,) + x_full.shape[1:]) if x_full.ndim > 1 else np.empty((0,))
                prev_y = np.empty((0,))
            else:
                if ds_mode:
                    prev_x = dataset_x.subset(cumulative_indices).get_data()[0]
                    prev_y = (
                        np.empty((0,)) if dataset_y is None else dataset_y.subset(cumulative_indices).get_data()[0]
                    )
                else:
                    prev_x = x_full[cumulative_indices]
                    if dataset_y is None:
                        prev_y = np.empty((0,))
                    elif isinstance(dataset_y, np.ndarray):
                        prev_y = np.asarray(dataset_y)[cumulative_indices]
                    else:
                        prev_y = dataset_y.subset(cumulative_indices).get_data()[0]

            cat_x = np.concatenate([prev_x, x_full[out]])
            cat_y = np.concatenate([np.asarray(prev_y).reshape(-1), f[out].reshape(-1)])

            # Fit a new model
            if ds_mode:
                new_x = abstract_numpy_class.NumpyDataSource(cat_x)
                new_y = abstract_numpy_class.NumpyDataSource(cat_y)
                model.fit(new_x, new_y)
            else:
                model.fit(cat_x, cat_y)

            # Get entropy values for all points that can be selected in the next cycle
            new_post_entropy = [
                list(
                    entropy(
                        dataset_x_avail,
                        model,
                        self.num_samples_entropy,
                        avg_mode=self.entropy_average_mode,
                    )
                )
            ]
            hxs.append(new_post_entropy)

        # Compute information gain
        hxs = np.array(hxs).reshape(self.num_samples_EIG, len(available_indices))

        # Reload base model and compute current entropy
        if ds_mode and hasattr(model, "load_folder"):
            model.load_folder(temp_folder_name)
        elif (not ds_mode) and hasattr(model, "load"):
            model.load()

        hx = entropy(
            dataset_x_avail,
            model,
            self.num_samples_entropy,
            avg_mode=self.entropy_average_mode,
        )  # Outside of for loop to make the most of GPU parallelism

        eigs = []
        for i in range(len(available_indices)):
            eigs.append(hx[i] - np.mean(hxs[:, i]))

        if acquisition_batch_size == 1:
            best_indices = [random_argmax(eigs)]
        elif self.batch_selection_mode == "topk_EIG":
            best_indices = top_k_idx(eigs, acquisition_batch_size)
        elif self.batch_selection_mode == "best_subset":
            assert (
                acquisition_batch_size == self.subset_size
            ), "best_subset selection only implemented for when batch size = subset_size"
            best_indices = None
            best_item = random_argmax(eigs)
            for i in range(self.num_samples_EIG):
                if best_item in outputs[i]:
                    best_indices = outputs[i]
                    break
            if best_indices is None:
                best_indices = top_k_idx(eigs, acquisition_batch_size)

        proposal = np.array(available_indices)[best_indices]
        return proposal


def noise_subset_select(noise_type, x, lengthscale=1.0, outputscale=1.0):
    if noise_type == "additive":
        return lambda f: gaussian_noise_sampler(x, f, lengthscale, outputscale)
    elif noise_type == "multiplicative":
        return lambda f: bernoulli_noise_sampler(x, f, lengthscale, outputscale)


def entropy(
    dataset_x, model, num_samples_entropy=20, precision=1.0, avg_mode="arithmetic"
):
    with torch.no_grad():
        if hasattr(model, "get_model_prediction"):
            f = (
                model.get_model_prediction(
                    dataset_x,
                    return_multiple_preds=True,
                    num_target_samples=num_samples_entropy,
                )[0]
                .detach()
                .view(-1, num_samples_entropy)
            )
        else:
            # Fallback path for models without get_model_prediction (e.g., simple GP)
            # Build x as numpy array for sampling
            if isinstance(dataset_x, np.ndarray):
                x_full = dataset_x
            else:
                x_full = dataset_x.get_data()[0]
            samples = []
            for _ in range(num_samples_entropy):
                if hasattr(model, "sample"):
                    f_np = np.asarray(model.sample(x_full)).reshape(-1)
                elif hasattr(model, "predict"):
                    mu, cov = model.predict(x_full, return_std_and_margin=True)
                    # Draw one sample from N(mu, cov)
                    f_np = np.random.multivariate_normal(np.asarray(mu).reshape(-1), np.asarray(cov))
                else:
                    raise AttributeError("Model does not support sampling or predictive distribution for entropy computation.")
                samples.append(torch.tensor(f_np, dtype=torch.float32))
            f = torch.stack(samples, dim=1)  # shape (n, num_samples_entropy)
    # As per equation (22) in Gal & Ghahramani, "Dropout as a Bayesian Approximation:Representing Model Uncertainty in Deep Learning"
    E = 0
    for sample_index in range(num_samples_entropy):
        yi = f[:, sample_index].view(-1, 1)
        if avg_mode == "arithmetic":
            E += torch.logsumexp(-precision / 2 * (f - yi) ** 2, dim=1)
        elif avg_mode == "harmonic":
            E -= torch.logsumexp(precision / 2 * (f - yi) ** 2, dim=1)
    E = (
        E / num_samples_entropy
        - np.log(num_samples_entropy)
        - 0.5 * np.log(2 * np.pi)
        - 0.5 * np.log(1 / precision)
    )
    return -E.numpy()


def top_k_idx(v, k):
    idxes = np.argsort(v)[-k:]
    return idxes


def level_set(v, c):
    idxes = np.where(v > c)
    return idxes


def subset_select(v, h_sampler, subset_size, budget=20):
    # for moment, just do monte carlo estimate
    # h_sampler : v -> h(v, eta), with eta sampled from some distribution
    # out_fn = either gaussian additive noise or multiplicative bernoulli sampled from GP classifier
    values = np.asarray([h_sampler(v) for _ in range(budget)])
    mx = random_argmax(np.mean(values, axis=0))
    idxes = [mx]
    n = len(v)
    for i in range(subset_size - 1):
        e_vals = np.zeros(n)
        for j in range(len(v)):
            test_idxes = idxes
            if j not in idxes:
                test_idxes = idxes + [j]
                test_idxes = np.asarray(test_idxes)
                e_vals[j] = np.mean(np.max(values[:, test_idxes], axis=-1))
        idxes.append(random_argmax(e_vals))
    return idxes


def gaussian_noise_sampler(x, fx, lengthscale=1.0, outputscale=1.0, seed=None):
    """Additive Gaussian process noise sampler with CPU/GPU auto-selection."""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    with torch.no_grad():
        kernel = ScaleKernel(
            RBFKernel(lengthscale=lengthscale), outputscale=outputscale
        ).to(device)
        x_t = torch.tensor(x).float().to(device)
        cov = kernel(x_t)
        mean = torch.zeros(fx.shape).float().to(device)
        eta = MultivariateNormal(mean, cov).sample().detach().cpu().numpy()
    return np.maximum(0, fx + eta)


def bernoulli_noise_sampler(x, fx, lengthscale=1.0, outputscale=1.0, seed=None):
    """Multiplicative Bernoulli noise with logits drawn from a GP; CPU/GPU auto-selection."""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    with torch.no_grad():
        kernel = ScaleKernel(
            RBFKernel(lengthscale=lengthscale), outputscale=outputscale
        ).to(device)
        x_t = torch.tensor(x).float().to(device)
        cov = kernel(x_t)
        mean = torch.zeros(fx.shape).float().to(device)
        l = MultivariateNormal(mean, cov).sample().detach().cpu().numpy()
    p = 1 / (1 + np.exp(-l))
    eta = np.random.binomial(1, p)
    return np.maximum(0, fx * eta)


def random_argmax(vals):
    max_val = np.max(vals)
    idxes = np.where(vals == max_val)[0]
    return np.random.choice(idxes)
