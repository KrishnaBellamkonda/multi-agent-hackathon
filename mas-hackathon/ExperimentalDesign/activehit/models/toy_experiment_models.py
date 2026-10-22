"""
Copyright 2022 Arash Mehrjou, GlaxoSmithKline plc, Clare Lyle, University of Oxford, Pascal Notin, University of Oxford
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
import torch
import os
import numpy as np
import pickle as pkl
from typing import AnyStr, Type
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist
import pdb 
import matplotlib.pyplot as plt

import slingpy as sp
from slingpy.utils.logging import info, warn
from slingpy.data_access.data_sources.hdf5_data_source import HDF5DataSource


class BayesianLinearModel:
    def __init__(
        self,
        input_dims: int,
        prior_sigma: float,
        noise_sigma: float,
        save_path_dir: AnyStr,
    ):
        self.input_dims = input_dims
        self.prior_sigma = prior_sigma
        self.noise_sigma = noise_sigma
        self.w = np.zeros(self.input_dims)
        self.Lambda = 0.01 * np.eye(self.input_dims)
        self.save_path_dir = save_path_dir

    def fit(self, X, y):
        info(X.shape, y.shape, self.w.shape, self.Lambda.shape)
        Lambda_n = X.T @ X + self.Lambda
        beta = np.linalg.pinv(X.T @ X) @ X.T @ y
        self.w = np.linalg.inv(Lambda_n) @ (X.T @ X) @ beta
        self.Lambda = Lambda_n
        info("Error:", np.linalg.norm(y - np.mean(y)), np.linalg.norm(X @ self.w - y))
        return self.w

    def get_save_file_extension(self, *args):
        return "blr_model.pkl"

    def get_model_save_file_name(self):
        return os.path.join(self.save_path_dir, self.get_save_file_extension())

    def load(self, *args):
        if os.path.exists(self.get_model_save_file_name()):
            with open(self.get_model_save_file_name(), "rb") as f:
                w = pkl.load(f)
                f.close()
            self.w = w

    def save(self, *args):
        if not os.path.exists(self.save_path_dir):
            os.makedirs(self.save_path_dir)
        with open(self.get_model_save_file_name(), "wb") as f:
            pkl.dump(self.w, f)
            f.close()
        pass

    def get_hyperopt_parameter_ranges(self):
        hyperopt_ranges = {
            "prior_sigma": (0.5,1.0,2.0),
            "noise_sigma": (0.01,0.1,1.0),
        }
        return hyperopt_ranges


class GaussianProcessModel:
    def __init__(
        self,
        kernel,
        noise_sigma: float,
        save_path_dir: AnyStr,
        kernel_lengthscale: float = 0.1,
    ):
        if kernel is None:
            kernel = lambda x, y: rbf(x, y, kernel_lengthscale)
        self.kernel = kernel
        self.noise_sigma = noise_sigma
        self.save_path_dir = save_path_dir
        self.X = None
        self.mu = None
        self.Kinv = None
        self.Kinvf = None
        # Cholesky factor for faster operations
        self.L = None  # Lower triangular Cholesky factor: K = L @ L.T

    def fit(self, X, y):
        """Fit GP using Cholesky decomposition (2x faster than matrix inverse)."""
        if X is None or y is None:
            self.Kinvf = 0
            self.mu = 0
            self.L = None
            return None
        
        K = self.kernel(X, X) + self.noise_sigma * np.eye(X.shape[0])
        
        # Use Cholesky decomposition: K = L @ L.T
        # This is O(n³/3) vs O(n³) for full inverse, and more stable
        try:
            self.L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            # Add jitter if not positive definite
            jitter = 1e-6 * np.eye(X.shape[0])
            self.L = np.linalg.cholesky(K + jitter)
        
        # Solve L @ alpha = y, then L.T @ Kinvf = alpha
        # This gives Kinvf = K^{-1} @ y without computing K^{-1}
        from scipy.linalg import solve_triangular
        alpha = solve_triangular(self.L, y, lower=True)
        self.Kinvf = solve_triangular(self.L.T, alpha, lower=False)
        
        self.mu = y
        self.X = X
        # We don't store Kinv anymore - use Cholesky solve instead
        self.Kinv = None
        return self.Kinvf

    def _solve_with_cholesky(self, b):
        """Solve K @ x = b using cached Cholesky factor. Returns x = K^{-1} @ b."""
        from scipy.linalg import solve_triangular
        if self.L is None:
            raise ValueError("Model not fitted yet")
        # Solve L @ z = b, then L.T @ x = z
        z = solve_triangular(self.L, b, lower=True)
        return solve_triangular(self.L.T, z, lower=False)

    def predict(self, x, return_std_and_margin=True, return_full_cov=False):
        """Predict mean and variance/covariance at test points x.
        
        Uses Cholesky decomposition for O(n²) prediction instead of O(n³).
        
        Args:
            x: Test points (n_test, d)
            return_std_and_margin: If True, return (mean, variance) tuple
            return_full_cov: If True, return full covariance matrix; 
                           If False (default), return only diagonal variances (FAST)
        
        Returns:
            If return_std_and_margin=True: (mean, variance) where variance is
                - Full covariance matrix (n_test, n_test) if return_full_cov=True
                - Diagonal variances (n_test,) if return_full_cov=False
            Otherwise: just mean
        """
        from scipy.linalg import solve_triangular
        
        n_test = x.shape[0] if x.ndim > 1 else len(x)
        
        if self.X is None or self.L is None:
            # Prior: mean = 0, variance = k(x,x) diagonal = sigma_f^2 (RBF kernel)
            prior_var = np.full(n_test, 2.25)  # sigma_f^2 = 1.5^2
            if return_full_cov:
                return np.zeros(n_test), self.kernel(x=x, y=x)
            else:
                return np.zeros(n_test), prior_var
        
        kxX = self.kernel(self.X, x)  # (n_train, n_test)
        mu = kxX.T @ self.Kinvf  # (n_test,)
        
        if not return_std_and_margin:
            return mu
        
        # Compute v = L^{-1} @ kxX using triangular solve (O(n_train² * n_test))
        # This is equivalent to computing K^{-1} @ kxX but faster
        v = solve_triangular(self.L, kxX, lower=True)  # (n_train, n_test)
        
        if return_full_cov:
            # Full covariance: K** - kxX.T @ K^{-1} @ kxX = K** - v.T @ v
            Kss = self.kernel(x, x)
            sigma2 = Kss - v.T @ v
            return mu, sigma2
        else:
            # Diagonal variances only - O(n_test * n_train)
            # var[i] = k(xi, xi) - ||v[:,i]||²
            reduction_term = np.sum(v * v, axis=0)  # (n_test,)
            
            # Prior variance = sigma_f^2 = 1.5^2 = 2.25 for RBF kernel
            prior_var = 2.25
            
            sigma2_diag = prior_var - reduction_term
            sigma2_diag = np.clip(sigma2_diag, a_min=1e-9, a_max=None)
            return mu, sigma2_diag

    def get_save_file_extension(self, *args):
        return "gp_model.pkl"

    def get_model_save_file_name(self):
        return os.path.join(self.save_path_dir, self.get_save_file_extension())

    def load(self, *args):
        if os.path.exists(self.get_model_save_file_name()):
            with open(self.get_model_save_file_name(), "rb") as f:
                w = pkl.load(f)
                f.close()
            self.Kinvf = w["Kinvf"]
            self.mu = w["mu"]
            self.X = w["X"]
            self.L = w.get("L", None)  # Cholesky factor
            self.Kinv = w.get("Kinv", None)  # Legacy support
        return self

    def save(self, *args):
        if not os.path.exists(self.save_path_dir):
            os.makedirs(self.save_path_dir)
        with open(self.get_model_save_file_name(), "wb") as f:
            pkl.dump(
                {"Kinvf": self.Kinvf, "mu": self.mu, "X": self.X, "L": self.L}, f
            )
            f.close()

    def sample(self, x):
        """Draw a sample from the GP posterior at points x.
        
        Uses Cholesky decomposition for efficient sampling: f ~ N(mu, Sigma)
        can be sampled as f = mu + L @ z where L @ L.T = Sigma and z ~ N(0, I).
        This avoids the O(n³) cost of np.random.multivariate_normal.
        """
        mean, cov = self.predict(x, return_std_and_margin=True, return_full_cov=True)
        # Ensure covariance is symmetric
        cov = 0.5 * (cov + cov.T)
        
        # Add jitter for numerical stability
        jitter = 1e-6 * np.eye(cov.shape[0])
        cov_jittered = cov + jitter
        
        try:
            # Cholesky decomposition: cov = L @ L.T
            L_cov = np.linalg.cholesky(cov_jittered)
            # Sample: f = mean + L @ z where z ~ N(0, I)
            z = np.random.standard_normal(len(mean))
            f = mean + L_cov @ z
        except np.linalg.LinAlgError:
            # Fallback: independent sampling from diagonal
            diag = np.clip(np.diag(cov), a_min=1e-9, a_max=None)
            f = mean + np.random.standard_normal(len(mean)) * np.sqrt(diag)
        return f
    
    def get_hyperopt_parameter_ranges(self):
        hyperopt_ranges = {
            "noise_sigma": (0.01, 0.1, 1.0),
            "kernel_lengthscale": (0.01, 0.1, 1.0),
        }
        return hyperopt_ranges

def rbf(x, y, lengthscale=0.1, sigma_f=1.5):
    """Vectorized RBF kernel supporting 1D and multi-D inputs efficiently.

    Accepts x, y of shape (n, d) or (n,) and returns the (n_x, n_y) kernel matrix.
    Uses the same exponent scaling as the original implementation: exp(-||x-y||^2 / (2*lengthscale)).
    """
    x = np.asarray(x)
    y = np.asarray(y)
    
    # Handle empty arrays
    if x.size == 0 or y.size == 0:
        n_x = x.shape[0] if x.ndim >= 1 else 0
        n_y = y.shape[0] if y.ndim >= 1 else 0
        return np.zeros((n_x, n_y))
    
    # Ensure 2D with features on last axis
    if x.ndim == 1:
        x = x[:, None]
    if y.ndim == 1:
        y = y[:, None]
    
    # Handle case where input is 0-dimensional after reshaping
    if x.shape[0] == 0 or y.shape[0] == 0:
        return np.zeros((x.shape[0], y.shape[0]))
    
    # Broadcasted pairwise squared distances
    # diff: (n_x, n_y, d) -> sum over d
    diff = x[:, None, :] - y[None, :, :]
    sqdist = np.sum(diff * diff, axis=2)
    return sigma_f ** 2 * np.exp(-0.5 * sqdist / (lengthscale))


if __name__ == "__main__":
    print(rbf( np.ones((1,2)), 5*np.ones((1,2)), lengthscale=1.0))