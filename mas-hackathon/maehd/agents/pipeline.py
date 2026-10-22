"""
Mathematical multi-agent framework for multi-target hit discovery.

The framework has four agents that communicate through NUMBERS, not text:

1. **FeatureAgent** (NEW — target-aware feature refinement)
   - Uses observed (X, Y) pairs to learn which feature dimensions
     are relevant for predicting target responses
   - Outputs feature weights w ∈ ℝ^d that re-scale the input space
   - Math: per-feature absolute Spearman correlation with each target,
     then max across targets, normalized

2. **CorrelationAgent** (cross-target structure)
   - Estimates cross-target correlation from observed data
   - Outputs a (K, K) correlation matrix and per-target weights
   - Math: running sample correlation on observed (gene, target) pairs

3. **AllocationAgent** (bandit-based policy selection)
   - Uses a UCB1 bandit to track which acquisition policy works best
   - Selects the policy that maximizes estimated hit rate + exploration bonus
   - Math: standard UCB1 multi-armed bandit on policy arms

4. **AcquisitionAgent** (candidate scoring)
   - Scores and ranks candidate genes using the GP posteriors
   - Outputs: ranked list of genes to query
   - Math: multi-target acquisition function (from policies/acquisition.py)

All agents are stateless functions that take data → produce decisions.
No LLMs, no neural networks, no black boxes.

The paper story:
    "Four interpretable mathematical agents collaborate to share
     a fixed experimental budget across K phenotypic targets."
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import spearmanr

from maehd.models.gp import MultiOutputGP
from maehd.policies import acquisition as acq


# --------------------------------------------------------------------------- #
#                            Data containers                                   #
# --------------------------------------------------------------------------- #


@dataclass
class CycleState:
    """All information available at the start of a cycle."""

    X_pool: np.ndarray          # (n_pool, d) features of unqueried genes
    X_obs: np.ndarray           # (n_obs, d) features of observed genes
    Y_obs: np.ndarray           # (n_obs, K) observed responses
    pool_indices: np.ndarray    # (n_pool,) original indices into full dataset
    obs_indices: np.ndarray     # (n_obs,) original indices into full dataset
    thresholds: np.ndarray      # (K,) per-target hit thresholds
    cycle: int                  # current cycle number
    budget_remaining: int       # queries left in total budget
    gp: MultiOutputGP | None = None  # fitted GP (set by experiment loop)
    recall_so_far: np.ndarray | None = None  # (K,) for coverage-aware policies


@dataclass
class CycleDecision:
    """Output of the agent pipeline for one cycle."""

    query_indices: np.ndarray   # indices into pool_indices to query
    policy_name: str            # which policy was used
    scores: np.ndarray          # (n_pool,) acquisition scores
    weights: np.ndarray         # (K,) per-target weights used
    correlation_matrix: np.ndarray  # (K, K) estimated cross-target correlation
    feature_weights: np.ndarray | None = None  # (d,) feature relevance weights


# --------------------------------------------------------------------------- #
#                      0. Feature Refinement Agent                             #
# --------------------------------------------------------------------------- #


def feature_agent(
    X_obs: np.ndarray,
    Y_obs: np.ndarray,
    min_obs: int = 15,
) -> np.ndarray:
    r"""Learn per-feature importance from observed (X, Y) pairs.

    Uses supervised feature scoring: for each feature dimension j,
    compute the coefficient of determination (R²) from a simple
    linear regression of Y_{:,k} on X_{:,j} for each target k.
    Take the max R² across targets.

    This is more robust than Spearman correlation for small samples
    and naturally captures linear associations.

    .. math::

        w_j = \max_k R^2(X_{:,j}, Y_{:,k})

    Returns
    -------
    feature_weights : (d,) non-negative weights, mean=1
    """
    n_obs, d = X_obs.shape
    K = Y_obs.shape[1]

    if n_obs < min_obs:
        return np.ones(d)

    w = np.zeros(d)
    for j in range(d):
        xj = X_obs[:, j]
        std_x = xj.std()
        if std_x < 1e-10:
            continue
        # Center x
        xj_c = xj - xj.mean()
        for k in range(K):
            yk = Y_obs[:, k]
            std_y = yk.std()
            if std_y < 1e-10:
                continue
            # Pearson R²
            r = np.corrcoef(xj, yk)[0, 1]
            if np.isnan(r):
                r = 0.0
            w[j] = max(w[j], r ** 2)

    # Soft thresholding: amplify differences
    w = w ** 1.5  # less aggressive than squaring for small R²

    # Floor to prevent zeroing out
    w = np.maximum(w, 0.01)

    # Normalize so mean = 1
    w = w * d / w.sum()

    return w


def supervised_projection(
    X_obs: np.ndarray,
    Y_obs: np.ndarray,
    X_full: np.ndarray,
    n_components: int = 10,
    min_obs: int = 30,
) -> np.ndarray | None:
    r"""Supervised dimensionality reduction via target-informed PCA.

    Finds the directions in feature space that best predict target
    responses using CCA-like approach:

    1. Compute W = X_obs^T Y_obs  (d × K cross-covariance)
    2. SVD of W → U S V^T
    3. Project onto top-n_components columns of U

    This finds feature combinations maximally correlated with targets.
    Falls back to None if not enough observations.

    Parameters
    ----------
    X_obs : (n_obs, d) observed features
    Y_obs : (n_obs, K) observed responses
    X_full : (n_total, d) full feature matrix
    n_components : output dimensionality
    min_obs : minimum observations needed

    Returns
    -------
    X_proj : (n_total, n_components) projected features, or None
    """
    n_obs, d = X_obs.shape
    K = Y_obs.shape[1]

    if n_obs < min_obs or n_components >= d:
        return None

    # Center
    X_c = X_obs - X_obs.mean(axis=0)
    Y_c = Y_obs - Y_obs.mean(axis=0)

    # Cross-covariance
    W = X_c.T @ Y_c / (n_obs - 1)  # (d, K)

    # SVD — first n_components left singular vectors
    U, S, Vt = np.linalg.svd(W, full_matrices=False)
    n_comp = min(n_components, len(S))
    P = U[:, :n_comp]  # (d, n_comp) projection matrix

    # Project full dataset
    X_centered_full = X_full - X_obs.mean(axis=0)  # use training mean
    X_proj = X_centered_full @ P

    # Normalize columns to unit variance for GP
    stds = X_proj.std(axis=0)
    stds[stds < 1e-10] = 1.0
    X_proj = X_proj / stds

    return X_proj


def apply_feature_weights(
    X: np.ndarray, feature_weights: np.ndarray,
) -> np.ndarray:
    """Scale features by sqrt(weights) — equivalent to ARD kernel."""
    return X * np.sqrt(feature_weights)[None, :]


# --------------------------------------------------------------------------- #
#                          1. Correlation Agent                                #
# --------------------------------------------------------------------------- #


def correlation_agent(state: CycleState) -> tuple[np.ndarray, np.ndarray]:
    r"""Estimate cross-target structure from observed data.

    Given observed responses Y ∈ ℝ^{n × K}, compute:
    1. Sample correlation matrix C ∈ ℝ^{K × K}
    2. Per-target weights w_k proportional to remaining discovery potential

    The weights encode: "how much attention should we pay to target k?"
    - If target k has high remaining variance → high weight (more to learn)
    - If target k is highly correlated with others → lower weight (redundant)

    Returns
    -------
    corr_matrix : (K, K) Pearson correlation matrix
    weights : (K,) normalized per-target weights
    """
    Y = state.Y_obs
    n, K = Y.shape

    if n < 3:
        # Not enough data to estimate correlation
        return np.eye(K), np.ones(K) / K

    # 1. Sample correlation matrix
    corr_matrix = np.corrcoef(Y.T)  # (K, K)
    # Ensure valid (can get NaN if a target is constant)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
    np.fill_diagonal(corr_matrix, 1.0)

    # 2. Per-target weights based on "uniqueness" and "difficulty"
    #    Uniqueness: 1 - mean |correlation| with other targets
    #    Difficulty: fraction of observed genes that are NOT hits
    off_diag_corr = np.abs(corr_matrix) - np.eye(K)
    uniqueness = 1.0 - off_diag_corr.sum(axis=1) / max(K - 1, 1)

    hit_rates = np.zeros(K)
    for k in range(K):
        hit_rates[k] = np.mean(Y[:, k] > state.thresholds[k])
    # Harder targets (low hit rate) deserve more budget
    difficulty = 1.0 - hit_rates

    raw_weights = uniqueness * difficulty
    raw_weights = np.maximum(raw_weights, 0.01)  # floor
    weights = raw_weights / raw_weights.sum()

    return corr_matrix, weights


# --------------------------------------------------------------------------- #
#                          2. Allocation Agent (Bandit)                         #
# --------------------------------------------------------------------------- #


_MT_POLICY_ARMS = ["max_poh", "mean_poh", "coverage_poh", "mt_thompson"]


class PolicyBandit:
    """Thompson Sampling bandit for adaptive policy selection.

    Each arm is an acquisition policy. We model the reward (hit rate)
    with a Beta distribution per arm: Beta(α, β).

    Selection: sample θ_a ~ Beta(α_a, β_a) for each arm, pick argmax.
    Update: on reward r (hit rate), α += r, β += (1 - r).

    Warm-start: prior beliefs from domain knowledge.
    - max_poh/coverage_poh get α=2, β=3 (expected ~0.40, slight optimism)
    - mean_poh gets α=2, β=3 (same prior)
    - mt_thompson gets α=1, β=3 (expected ~0.25, less trusted)

    Thompson Sampling is more exploration-efficient than UCB1 —
    it doesn't need to play every arm, and adapts faster.
    """

    def __init__(self, arms: list[str] = None):
        self.arms = arms or list(_MT_POLICY_ARMS)
        n = len(self.arms)
        # Warm-start priors (αlpha, βeta)
        self.alpha = np.ones(n) * 2.0
        self.beta = np.ones(n) * 3.0
        # Give exploration-heavy policies weaker prior
        for i, arm in enumerate(self.arms):
            if arm in ("mt_ucb", "mt_thompson"):
                self.alpha[i] = 1.0
                self.beta[i] = 3.0
        self.counts = np.zeros(n)

    def select(self, rng: np.random.Generator | None = None) -> str:
        """Select arm via Thompson Sampling."""
        if rng is None:
            rng = np.random.default_rng()
        # Sample from each arm's Beta posterior
        samples = rng.beta(self.alpha, self.beta)
        idx = np.argmax(samples)
        return self.arms[idx]

    def update(self, arm_name: str, reward: float):
        """Update Beta posterior with observed reward.

        reward is the hit rate (0 to 1) for this cycle.
        We scale to pseudo-counts: α += 10*reward, β += 10*(1-reward)
        This gives ~10 observations' worth of evidence per cycle.
        """
        idx = self.arms.index(arm_name)
        self.alpha[idx] += 10 * reward
        self.beta[idx] += 10 * (1.0 - reward)
        self.counts[idx] += 1

    def best_arm(self) -> str:
        """Return the arm with highest posterior mean."""
        means = self.alpha / (self.alpha + self.beta)
        return self.arms[np.argmax(means)]


def allocation_agent(
    state: CycleState,
    weights: np.ndarray,
    batch_size: int = 20,
    policy_name: str | None = None,
    bandit: PolicyBandit | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[str, int]:
    r"""Decide acquisition policy and batch size for this cycle.

    If policy_name is given, use it directly.
    If a bandit is provided, use UCB1 to select.
    Otherwise, fall back to the simple schedule.

    Returns
    -------
    policy : name of multi-target acquisition function
    batch : number of genes to query this cycle
    """
    actual_batch = min(batch_size, state.budget_remaining, len(state.X_pool))

    if policy_name is not None:
        return policy_name, actual_batch

    if bandit is not None:
        chosen = bandit.select(rng)
        return chosen, actual_batch

    # Fallback: simple schedule
    n_obs = len(state.X_obs) if state.X_obs is not None else 0
    K = len(state.thresholds)
    if n_obs < 3 * K:
        chosen = "mt_ucb"
    elif state.budget_remaining <= 2 * batch_size:
        chosen = "max_poh"
    else:
        chosen = "mean_poh"

    return chosen, actual_batch


# --------------------------------------------------------------------------- #
#                        3. Acquisition Agent                                  #
# --------------------------------------------------------------------------- #


def acquisition_agent(
    state: CycleState,
    policy_name: str,
    weights: np.ndarray,
    batch_size: int,
    rng: np.random.Generator | None = None,
    recall_so_far: np.ndarray | None = None,
    use_diversity: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Score and rank candidates, return top batch_size.

    Parameters
    ----------
    state : current cycle state (must have gp fitted)
    policy_name : which multi-target policy to use
    weights : (K,) per-target relevance weights from correlation agent
    batch_size : how many genes to select
    rng : random generator (for Thompson-based policies)
    recall_so_far : (K,) current recall per target (for coverage_poh)
    use_diversity : if True, use diverse batch selection (recommended)

    Returns
    -------
    selected : (batch_size,) indices into state.pool_indices
    scores : (n_pool,) acquisition scores for all candidates
    """
    if rng is None:
        rng = np.random.default_rng()

    mu, sigma = state.gp.predict(state.X_pool)
    thresholds = state.thresholds

    # Dispatch to the appropriate policy
    if policy_name == "max_poh":
        scores = acq.max_poh(mu, sigma, thresholds)
    elif policy_name == "mean_poh":
        scores = acq.mean_poh(mu, sigma, thresholds, weights)
    elif policy_name == "joint_poh":
        scores = acq.joint_poh(mu, sigma, thresholds)
    elif policy_name == "mt_ucb":
        scores = acq.multi_target_ucb(mu, sigma, weights=weights)
    elif policy_name == "mt_thompson":
        scores = acq.multi_target_thompson(mu, sigma, thresholds, rng)
    elif policy_name == "info_gain":
        scores = acq.information_gain(sigma, weights)
    elif policy_name == "coverage_poh":
        r = recall_so_far if recall_so_far is not None else np.zeros(len(thresholds))
        scores = acq.coverage_poh(mu, sigma, thresholds, r)
    else:
        raise ValueError(f"Unknown policy: {policy_name}")

    # Batch selection: diverse or greedy
    if use_diversity and batch_size > 1:
        selected = acq.diverse_batch_select(
            scores, state.X_pool, batch_size, lam=1.0,
        )
    else:
        selected = np.argsort(scores)[::-1][:batch_size]

    return selected, scores


# --------------------------------------------------------------------------- #
#                       Full agent pipeline                                    #
# --------------------------------------------------------------------------- #


def run_agent_pipeline(
    state: CycleState,
    batch_size: int = 20,
    policy_name: str | None = None,
    rng: np.random.Generator | None = None,
    bandit: PolicyBandit | None = None,
    use_feature_agent: bool = False,
    recall_so_far: np.ndarray | None = None,
    use_diversity: bool = True,
) -> CycleDecision:
    """Execute the agent pipeline for one cycle.

    1. (optional) FeatureAgent → learn feature importance weights
    2. CorrelationAgent → estimates cross-target weights
    3. AllocationAgent → selects policy and batch size (with bandit)
    4. AcquisitionAgent → scores candidates and selects diverse batch

    Parameters
    ----------
    state : current cycle state with fitted GP
    batch_size : default batch size per cycle
    policy_name : override policy selection (None = adaptive)
    rng : random generator
    bandit : PolicyBandit for adaptive allocation (None = schedule)
    use_feature_agent : whether to apply feature re-weighting
    recall_so_far : (K,) current recall per target (for coverage_poh)
    use_diversity : whether to use diverse batch selection

    Returns
    -------
    CycleDecision with selected genes, scores, weights, correlations
    """
    feat_weights = None
    if use_feature_agent:
        feat_weights = feature_agent(state.X_obs, state.Y_obs)

    # 1. Correlation Agent
    corr_matrix, weights = correlation_agent(state)

    # 2. Allocation Agent (with bandit if provided)
    chosen_policy, actual_batch = allocation_agent(
        state, weights, batch_size, policy_name, bandit, rng
    )

    # 3. Acquisition Agent (with batch diversity)
    selected, scores = acquisition_agent(
        state, chosen_policy, weights, actual_batch, rng,
        recall_so_far=recall_so_far,
        use_diversity=use_diversity,
    )

    return CycleDecision(
        query_indices=selected,
        policy_name=chosen_policy,
        scores=scores,
        weights=weights,
        correlation_matrix=corr_matrix,
        feature_weights=feat_weights,
    )
