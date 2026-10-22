"""
Multi-output Gaussian Process for multi-target hit discovery.

Uses independent GPs with shared observations (one GP per target).
This is the simplest multi-output GP — each target has its own
hyperparameters but all share the same observed gene set.

For the paper, "shared observations" is the key: querying gene i
reveals y_{i,1}, ..., y_{i,K} for all K targets simultaneously.
The GP gives posterior mean μ_k(x) and variance σ²_k(x) for each target.

Acquisition functions then combine these K posteriors.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist


class RBFKernel:
    """Radial Basis Function (squared exponential) kernel.

    k(x, x') = amplitude² · exp(-||x-x'||² / (2·lengthscale²))
    """

    def __init__(self, lengthscale: float = 0.3, amplitude: float = 1.0):
        self.lengthscale = lengthscale
        self.amplitude = amplitude

    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        sq_dist = cdist(X1, X2, metric="sqeuclidean")
        return self.amplitude ** 2 * np.exp(-sq_dist / (2 * self.lengthscale ** 2))


class SingleOutputGP:
    """Single-output GP with RBF kernel.

    Provides posterior mean, variance, and sampling for one target.
    """

    def __init__(
        self,
        lengthscale: float = 0.3,
        amplitude: float = 1.0,
        noise: float = 0.01,
    ):
        self.kernel = RBFKernel(lengthscale, amplitude)
        self.noise = noise
        self._X_train = None
        self._y_train = None
        self._L = None  # Cholesky of K + σ²I
        self._alpha = None  # (K + σ²I)^{-1} y

    def _log_marginal_likelihood(self, X, y, lengthscale, amplitude, noise):
        """Compute log marginal likelihood for hyperparameter evaluation.

        log p(y|X,θ) = -½ y^T K^{-1} y - ½ log|K| - n/2 log(2π)
        """
        n = len(X)
        kern = RBFKernel(lengthscale, amplitude)
        K = kern(X, X) + noise * np.eye(n) + 1e-6 * np.eye(n)
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            return -np.inf
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
        lml = -0.5 * y @ alpha - np.sum(np.log(np.diag(L))) - 0.5 * n * np.log(2 * np.pi)
        return lml

    def fit(self, X: np.ndarray, y: np.ndarray, optimize: bool = False):
        """Fit GP to observed data.

        Parameters
        ----------
        optimize : if True, tune lengthscale via marginal likelihood
        """
        self._X_train = X.copy()
        self._y_train = y.copy()

        if optimize and len(X) >= 10:
            # Grid search over lengthscale candidates
            best_lml = -np.inf
            best_ls = self.kernel.lengthscale
            amp = self.kernel.amplitude
            for ls in [0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0]:
                lml = self._log_marginal_likelihood(X, y, ls, amp, self.noise)
                if lml > best_lml:
                    best_lml = lml
                    best_ls = ls
            self.kernel.lengthscale = best_ls

        n = len(X)
        K = self.kernel(X, X) + self.noise * np.eye(n)
        self._L = np.linalg.cholesky(K + 1e-6 * np.eye(n))
        self._alpha = np.linalg.solve(
            self._L.T, np.linalg.solve(self._L, y)
        )

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Posterior mean and standard deviation at X.

        Returns
        -------
        mu : (n,) posterior mean
        sigma : (n,) posterior standard deviation
        """
        K_star = self.kernel(X, self._X_train)
        mu = K_star @ self._alpha

        v = np.linalg.solve(self._L, K_star.T)
        # Prior variance on diagonal: k(x,x) = amplitude^2 for RBF
        k_diag = self.kernel.amplitude ** 2 * np.ones(len(X))
        var = k_diag - np.sum(v ** 2, axis=0)
        sigma = np.sqrt(np.maximum(var, 1e-8))
        return mu, sigma

    def sample_posterior(
        self, X: np.ndarray, n_samples: int = 1, rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Draw posterior samples at X.

        Returns
        -------
        samples : (n_samples, n_points) array
        """
        if rng is None:
            rng = np.random.default_rng()
        mu, sigma = self.predict(X)
        # Diagonal approximation for speed (sufficient for Thompson sampling)
        eps = rng.standard_normal((n_samples, len(X)))
        return mu[None, :] + sigma[None, :] * eps


class MultiOutputGP:
    """Multi-output GP: K independent GPs sharing observations.

    Each target k has its own GP with its own hyperparameters.
    All GPs see the SAME set of observed genes (shared query budget).

    This is the appropriate model when:
    - Targets share the same input space (gene features)
    - Querying a gene reveals ALL target responses
    - Targets may have different smoothness/noise characteristics
    """

    def __init__(
        self,
        K: int,
        lengthscale: float | list[float] = 0.3,
        amplitude: float | list[float] = 1.0,
        noise: float | list[float] = 0.01,
    ):
        """
        Parameters
        ----------
        K : number of targets
        lengthscale, amplitude, noise : per-target or shared hyperparameters
        """
        self.K = K

        def _expand(val, name):
            if isinstance(val, (int, float)):
                return [float(val)] * K
            if len(val) != K:
                raise ValueError(f"{name} must be scalar or length K={K}")
            return list(val)

        ls_list = _expand(lengthscale, "lengthscale")
        amp_list = _expand(amplitude, "amplitude")
        noise_list = _expand(noise, "noise")

        self.gps = [
            SingleOutputGP(ls_list[k], amp_list[k], noise_list[k])
            for k in range(K)
        ]

    def fit(self, X: np.ndarray, Y: np.ndarray, optimize: bool = False):
        """Fit all K GPs on shared observed data.

        Parameters
        ----------
        X : (n_obs, d) features of observed genes
        Y : (n_obs, K) responses for each target
        optimize : if True, tune per-target lengthscale via marginal likelihood
        """
        for k in range(self.K):
            self.gps[k].fit(X, Y[:, k], optimize=optimize)

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Posterior mean and std for all targets.

        Returns
        -------
        mu : (n, K) posterior means
        sigma : (n, K) posterior standard deviations
        """
        n = len(X)
        mu = np.zeros((n, self.K))
        sigma = np.zeros((n, self.K))
        for k in range(self.K):
            mu[:, k], sigma[:, k] = self.gps[k].predict(X)
        return mu, sigma

    def predict_phit(
        self, X: np.ndarray, thresholds: np.ndarray,
    ) -> np.ndarray:
        r"""Probability of hit for each target.

        P(hit_k | x) = P(f_k(x) > τ_k) = 1 - Φ((τ_k - μ_k(x)) / σ_k(x))

        Returns
        -------
        p_hit : (n, K) hit probabilities
        """
        from scipy.stats import norm
        mu, sigma = self.predict(X)
        p_hit = np.zeros_like(mu)
        for k in range(self.K):
            s = np.maximum(sigma[:, k], 1e-6)
            p_hit[:, k] = 1.0 - norm.cdf((thresholds[k] - mu[:, k]) / s)
        return p_hit

    def sample_posterior(
        self, X: np.ndarray, n_samples: int = 1,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Draw posterior samples for all targets.

        Returns
        -------
        samples : (n_samples, n_points, K)
        """
        if rng is None:
            rng = np.random.default_rng()
        n = len(X)
        samples = np.zeros((n_samples, n, self.K))
        for k in range(self.K):
            samples[:, :, k] = self.gps[k].sample_posterior(X, n_samples, rng)
        return samples


class CorrelatedMultiOutputGP:
    """Multi-output GP with Intrinsic Coregionalization Model (ICM).

    Joint kernel: K_full((x,k), (x',k')) = B[k,k'] · k_rbf(x,x')

    where B ∈ ℝ^{K×K} is a positive semi-definite coregionalization
    matrix encoding cross-target correlations.

    Key advantages over independent GPs:
    - Observing gene i on target 1 gives information about gene i on target 2
    - When targets are correlated, this is a free lunch: more effective training data
    - The GP can borrow statistical strength from easy targets to help hard ones

    Complexity: O((nK)³) for Cholesky — acceptable for n ≤ 500, K ≤ 5.
    Falls back to independent GPs if n·K > 2000 for speed.
    """

    def __init__(
        self,
        K: int,
        lengthscale: float = 0.3,
        amplitude: float = 1.0,
        noise: float = 0.01,
    ):
        self.K = K
        self.kernel = RBFKernel(lengthscale, amplitude)
        self.noise = noise
        self.B = np.eye(K)  # coregionalization matrix (learned from data)
        self._fallback = MultiOutputGP(K, lengthscale, amplitude, noise)
        self._use_fallback = False
        self._X_train = None
        self._Y_train = None
        self._L = None
        self._alpha = None

    def _estimate_B(self, Y: np.ndarray) -> np.ndarray:
        """Estimate coregionalization matrix from observed responses."""
        n, K = Y.shape
        if n < K + 2:
            return np.eye(K)
        # Sample correlation → coregionalization
        C = np.corrcoef(Y.T)
        C = np.nan_to_num(C, nan=0.0)
        np.fill_diagonal(C, 1.0)
        # Ensure PSD via eigendecomposition
        eigvals, eigvecs = np.linalg.eigh(C)
        eigvals = np.maximum(eigvals, 0.01)
        B = eigvecs @ np.diag(eigvals) @ eigvecs.T
        return B

    def fit(self, X: np.ndarray, Y: np.ndarray, optimize: bool = False):
        """Fit the correlated multi-output GP.

        Falls back to independent GPs if n·K > 2000.
        """
        n, K = Y.shape

        if n * K > 2000:
            self._use_fallback = True
            self._fallback.fit(X, Y, optimize=optimize)
            return

        self._use_fallback = False
        self._X_train = X.copy()
        self._Y_train = Y.copy()

        # Estimate cross-target coregionalization
        self.B = self._estimate_B(Y)

        # Optimize lengthscale via marginal likelihood (simple grid)
        if optimize and n >= 10:
            best_lml = -np.inf
            best_ls = self.kernel.lengthscale
            for ls in [0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0]:
                lml = self._compute_lml(X, Y, ls)
                if lml > best_lml:
                    best_lml = lml
                    best_ls = ls
            self.kernel.lengthscale = best_ls

        # Build and factorize the full (nK × nK) kernel matrix
        K_x = self.kernel(X, X)  # (n, n)
        # Full kernel: K_full = B ⊗ K_x (Kronecker product)
        nK = n * K
        K_full = np.kron(self.B, K_x)
        K_full += self.noise * np.eye(nK) + 1e-6 * np.eye(nK)

        self._L = np.linalg.cholesky(K_full)

        # Stack Y column-major: [y_1; y_2; ...; y_K] each (n,)
        y_stacked = Y.T.ravel()  # (nK,)
        self._alpha = np.linalg.solve(
            self._L.T, np.linalg.solve(self._L, y_stacked)
        )

    def _compute_lml(self, X, Y, lengthscale):
        """Log marginal likelihood for hyperparameter grid search."""
        n, K = Y.shape
        kern = RBFKernel(lengthscale, self.kernel.amplitude)
        K_x = kern(X, X)
        nK = n * K
        K_full = np.kron(self.B, K_x)
        K_full += self.noise * np.eye(nK) + 1e-6 * np.eye(nK)
        try:
            L = np.linalg.cholesky(K_full)
        except np.linalg.LinAlgError:
            return -np.inf
        y_stacked = Y.T.ravel()
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_stacked))
        lml = (-0.5 * y_stacked @ alpha
               - np.sum(np.log(np.diag(L)))
               - 0.5 * nK * np.log(2 * np.pi))
        return lml

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Posterior mean and std for all targets at X.

        Returns mu (n_test, K), sigma (n_test, K).
        """
        if self._use_fallback:
            return self._fallback.predict(X)

        n_test = len(X)
        n_train = len(self._X_train)
        K = self.K

        # K_star: (n_test·K, n_train·K) = B ⊗ k(X*, X_train)
        k_star_x = self.kernel(X, self._X_train)  # (n_test, n_train)

        # Posterior mean for each (test_point, target) pair
        mu = np.zeros((n_test, K))
        sigma = np.zeros((n_test, K))

        for k in range(K):
            # Row block k of K_star: B[k,:] ⊗ k_star_x
            # = sum_j B[k,j] * k(x*, x_train) @ alpha_j
            for j in range(K):
                alpha_j = self._alpha[j * n_train:(j + 1) * n_train]
                mu[:, k] += self.B[k, j] * (k_star_x @ alpha_j)

        # Posterior variance (diagonal only for efficiency)
        # var_k(x*) = B[k,k] * k(x*,x*) - sum over blocks
        for k in range(K):
            # Prior variance
            prior_var = self.B[k, k] * self.kernel.amplitude ** 2

            # Posterior reduction: v = L^{-1} K_star_row
            # K_star_row for point i, target k: B[k,:] ⊗ k(x_i*, X_train)
            for i in range(n_test):
                k_row = np.zeros(n_train * K)
                for j in range(K):
                    k_row[j * n_train:(j + 1) * n_train] = (
                        self.B[k, j] * k_star_x[i, :]
                    )
                v = np.linalg.solve(self._L, k_row)
                var_i = prior_var - v @ v
                sigma[i, k] = np.sqrt(max(var_i, 1e-8))

        return mu, sigma

    def predict_phit(
        self, X: np.ndarray, thresholds: np.ndarray,
    ) -> np.ndarray:
        """P(hit_k | x) for each target."""
        from scipy.stats import norm
        mu, sigma = self.predict(X)
        p_hit = np.zeros_like(mu)
        for k in range(self.K):
            s = np.maximum(sigma[:, k], 1e-6)
            p_hit[:, k] = 1.0 - norm.cdf((thresholds[k] - mu[:, k]) / s)
        return p_hit

    def sample_posterior(
        self, X: np.ndarray, n_samples: int = 1,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Draw posterior samples (diagonal approximation for speed)."""
        if rng is None:
            rng = np.random.default_rng()
        mu, sigma = self.predict(X)
        n = len(X)
        samples = np.zeros((n_samples, n, self.K))
        for k in range(self.K):
            eps = rng.standard_normal((n_samples, n))
            samples[:, :, k] = mu[None, :, k] + sigma[None, :, k] * eps
        return samples
