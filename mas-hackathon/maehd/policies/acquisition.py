"""
Mathematical acquisition functions for multi-target hit discovery.

All policies are closed-form, interpretable, and based on standard
Bayesian decision theory. No neural networks or LLMs.

For each candidate gene x, given K targets with GP posteriors:
    μ_k(x), σ_k(x)  for k = 1, ..., K

We define several acquisition strategies that combine these K posteriors
into a single score for ranking candidates.

References
----------
[1] Bryan, Schneider, Nichol, et al. (2006). Active learning for identifying
    function threshold boundaries. NeurIPS.
[2] Srinivas et al. (2010). Gaussian Process Optimization in the Bandit
    Setting. ICML.
[3] Chapelle & Li (2011). An Empirical Evaluation of Thompson Sampling.
    NeurIPS.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

# --------------------------------------------------------------------------- #
#                        Single-target acquisition                             #
# --------------------------------------------------------------------------- #


def probability_of_hit(mu: np.ndarray, sigma: np.ndarray, tau: float) -> np.ndarray:
    r"""Probability that f(x) > τ.

    .. math::

        \text{PoH}(x) = 1 - \Phi\!\left(\frac{\tau - \mu(x)}{\sigma(x)}\right)

    Parameters
    ----------
    mu : (n,) posterior means
    sigma : (n,) posterior standard deviations
    tau : hit threshold

    Returns
    -------
    poh : (n,) probabilities
    """
    s = np.maximum(sigma, 1e-8)
    return 1.0 - norm.cdf((tau - mu) / s)


def ucb(mu: np.ndarray, sigma: np.ndarray, beta: float = 2.0) -> np.ndarray:
    r"""Upper Confidence Bound.

    .. math::

        \text{UCB}(x) = \mu(x) + \beta \, \sigma(x)

    Balances exploitation (high μ) with exploration (high σ).
    """
    return mu + beta * sigma


def thompson_sample(
    mu: np.ndarray, sigma: np.ndarray,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    r"""Thompson sampling: draw f ~ posterior, return f(x).

    Uses diagonal approximation (independent draws at each x).
    """
    if rng is None:
        rng = np.random.default_rng()
    return mu + sigma * rng.standard_normal(mu.shape)


def expected_improvement_tau(
    mu: np.ndarray, sigma: np.ndarray, tau: float,
) -> np.ndarray:
    r"""Expected Improvement above threshold τ.

    .. math::

        \text{EI}_\tau(x) = \sigma(x)\bigl[\,z\,\Phi(z) + \phi(z)\bigr],
        \quad z = \frac{\mu(x) - \tau}{\sigma(x)}
    """
    s = np.maximum(sigma, 1e-8)
    z = (mu - tau) / s
    return s * (z * norm.cdf(z) + norm.pdf(z))


# --------------------------------------------------------------------------- #
#                        Multi-target scoring                                  #
# --------------------------------------------------------------------------- #


def max_poh(
    mu: np.ndarray, sigma: np.ndarray, thresholds: np.ndarray,
) -> np.ndarray:
    r"""Max-PoH: probability of being a hit for ANY target.

    .. math::

        \alpha(x) = \max_k \, \text{PoH}_k(x)

    Favours genes that are strong candidates for at least one target.
    Good when each query reveals all targets (bandwidth not wasted).

    Parameters
    ----------
    mu : (n, K) posterior means
    sigma : (n, K) posterior stds
    thresholds : (K,) per-target thresholds

    Returns
    -------
    scores : (n,)
    """
    K = mu.shape[1]
    poh_all = np.column_stack([
        probability_of_hit(mu[:, k], sigma[:, k], thresholds[k])
        for k in range(K)
    ])
    return np.max(poh_all, axis=1)


def mean_poh(
    mu: np.ndarray, sigma: np.ndarray, thresholds: np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    r"""Weighted-mean PoH across targets.

    .. math::

        \alpha(x) = \sum_k w_k \, \text{PoH}_k(x)

    With uniform weights, this is the expected number of targets
    for which x is a hit.

    Parameters
    ----------
    weights : (K,) per-target weights, default uniform
    """
    K = mu.shape[1]
    poh_all = np.column_stack([
        probability_of_hit(mu[:, k], sigma[:, k], thresholds[k])
        for k in range(K)
    ])
    if weights is None:
        weights = np.ones(K) / K
    return poh_all @ weights


def joint_poh(
    mu: np.ndarray, sigma: np.ndarray, thresholds: np.ndarray,
) -> np.ndarray:
    r"""Joint-PoH: probability of being a hit for ALL targets.

    .. math::

        \alpha(x) = \prod_k \, \text{PoH}_k(x)

    Under GP independence assumption (our multi-output model).
    Useful for finding pan-target hits (shared mechanisms).
    """
    K = mu.shape[1]
    poh_all = np.column_stack([
        probability_of_hit(mu[:, k], sigma[:, k], thresholds[k])
        for k in range(K)
    ])
    return np.prod(poh_all, axis=1)


def multi_target_ucb(
    mu: np.ndarray, sigma: np.ndarray, beta: float = 2.0,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    r"""Multi-target UCB: weighted sum of per-target UCBs.

    .. math::

        \alpha(x) = \sum_k w_k \bigl[\mu_k(x) + \beta \, \sigma_k(x)\bigr]
    """
    K = mu.shape[1]
    if weights is None:
        weights = np.ones(K) / K
    ucb_all = mu + beta * sigma  # (n, K)
    return ucb_all @ weights


def multi_target_thompson(
    mu: np.ndarray, sigma: np.ndarray, thresholds: np.ndarray,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    r"""Multi-target Thompson: sample each target, score by hit count.

    Draw f_k ~ GP_k for each target k.
    Score(x) = number of targets for which f_k(x) > τ_k.
    """
    if rng is None:
        rng = np.random.default_rng()
    K = mu.shape[1]
    scores = np.zeros(len(mu))
    for k in range(K):
        fk = thompson_sample(mu[:, k], sigma[:, k], rng)
        scores += (fk > thresholds[k]).astype(float)
    return scores


def information_gain(
    sigma: np.ndarray, weights: np.ndarray | None = None,
) -> np.ndarray:
    r"""Total information gain: weighted sum of per-target variances.

    .. math::

        \alpha(x) = \sum_k w_k \, \sigma_k^2(x)

    Selects genes where querying would maximally reduce posterior
    uncertainty. Pure exploration — useful when budget is limited
    and we want to learn the response surface efficiently.
    """
    K = sigma.shape[1]
    if weights is None:
        weights = np.ones(K) / K
    return (sigma ** 2) @ weights


def coverage_poh(
    mu: np.ndarray,
    sigma: np.ndarray,
    thresholds: np.ndarray,
    recall_so_far: np.ndarray,
) -> np.ndarray:
    r"""Coverage-aware PoH: boost under-explored targets dynamically.

    .. math::

        \alpha(x) = \sum_k \underbrace{(1 - r_k)^2}_{\text{need}_k}
                     \cdot \text{PoH}_k(x)

    where :math:`r_k` is the current recall for target k.

    Targets with low recall get quadratically more weight — forces the
    system to shift attention to hard/neglected targets (e.g., NK).

    When recall is uniform, this reduces to mean_poh.
    When one target is saturated, focus shifts entirely to the others.

    Parameters
    ----------
    recall_so_far : (K,) current recall per target (0 to 1)
    """
    K = mu.shape[1]
    poh_all = np.column_stack([
        probability_of_hit(mu[:, k], sigma[:, k], thresholds[k])
        for k in range(K)
    ])
    # Quadratic need — targets with low recall get much more weight
    need = (1.0 - np.clip(recall_so_far, 0, 1)) ** 2
    need = np.maximum(need, 0.01)
    need = need / need.sum()
    return poh_all @ need


# --------------------------------------------------------------------------- #
#                            Registry                                          #
# --------------------------------------------------------------------------- #


# Single-target policies (operate on one target at a time)
SINGLE_TARGET_POLICIES = {
    "poh": probability_of_hit,
    "ucb": ucb,
    "thompson": thompson_sample,
    "ei_tau": expected_improvement_tau,
}

# Multi-target policies (combine across K targets)
MULTI_TARGET_POLICIES = {
    "max_poh": max_poh,
    "mean_poh": mean_poh,
    "joint_poh": joint_poh,
    "mt_ucb": multi_target_ucb,
    "mt_thompson": multi_target_thompson,
    "info_gain": information_gain,
    "coverage_poh": coverage_poh,
}


# --------------------------------------------------------------------------- #
#                      Batch diversification                                   #
# --------------------------------------------------------------------------- #


def diverse_batch_select(
    scores: np.ndarray,
    X_pool: np.ndarray,
    batch_size: int,
    lam: float = 1.0,
) -> np.ndarray:
    r"""Select a diverse batch by penalizing feature-space proximity.

    Greedy sequential selection:
    1. Pick the gene with highest score
    2. For remaining candidates, discount score by proximity to already-selected:

    .. math::

        \text{score}'_i = \text{score}_i - \lambda \max_{j \in S}
            \exp\!\bigl(-\|x_i - x_j\|^2 / (2 h^2)\bigr)

    where h = median pairwise distance (auto-scaled), S = selected set.

    This prevents selecting genes clustered in the same feature-space region.
    Runs in O(batch_size × n_pool) — cheap for batch_size ≪ n_pool.

    Parameters
    ----------
    scores : (n_pool,) acquisition scores
    X_pool : (n_pool, d) features of candidates
    batch_size : how many to select
    lam : penalty strength (1.0 = balanced quality/diversity)

    Returns
    -------
    selected : (batch_size,) indices into the pool
    """
    n = len(scores)
    if batch_size >= n:
        return np.arange(n)

    # Normalize scores to [0, 1] for calibrated penalty
    s_min, s_max = scores.min(), scores.max()
    if s_max - s_min < 1e-12:
        # All scores equal — just pick randomly
        return np.arange(batch_size)
    s_norm = (scores - s_min) / (s_max - s_min)

    # Auto-calibrate bandwidth: use squared median distance among
    # top candidates (cheaper than full pairwise)
    top_k = min(max(batch_size * 5, 100), n)
    top_idx = np.argpartition(scores, -top_k)[-top_k:]
    from scipy.spatial.distance import pdist
    dists = pdist(X_pool[top_idx], metric="sqeuclidean")
    h_sq = max(np.median(dists), 1e-8) * 2  # 2h² in the denominator

    selected = []
    penalty = np.zeros(n)

    for _ in range(batch_size):
        effective = s_norm - lam * penalty
        # Mask already selected
        for idx in selected:
            effective[idx] = -np.inf
        best = np.argmax(effective)
        selected.append(best)

        # Update penalty: proximity to the newly selected gene
        diff_sq = np.sum((X_pool - X_pool[best]) ** 2, axis=1)
        proximity = np.exp(-diff_sq / h_sq)
        penalty = np.maximum(penalty, proximity)

    return np.array(selected)
