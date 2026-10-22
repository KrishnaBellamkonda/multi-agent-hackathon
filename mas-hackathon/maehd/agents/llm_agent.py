"""
LLM-powered Experiment Agent for multi-target hit discovery.

This is the real agentic component. Each cycle, the LLM:
1. Observes the full experiment state (recall, GP uncertainty, correlations)
2. Reasons about which acquisition policy to use
3. Decides batch size and target weighting
4. Optionally recommends early stopping

The LLM communicates via structured JSON decisions.
Falls back to a fixed heuristic if the LLM is unavailable or returns garbage.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from maehd.agents.llm_client import OllamaClient, parse_json_from_text

logger = logging.getLogger("maehd.agent")

# Available policies the agent can choose from
AVAILABLE_POLICIES = ["max_poh", "mean_poh", "coverage_poh", "mt_thompson"]


@dataclass
class AgentDecision:
    """Structured decision output from the LLM agent."""

    policy: str                    # acquisition policy to use
    target_weights: np.ndarray     # (K,) relative weight per target
    batch_size: int                # genes to query this cycle
    reasoning: str                 # LLM's chain-of-thought explanation
    raw_response: str              # full LLM output (for logging)
    used_llm: bool                 # True if LLM was actually used


_SYSTEM_PROMPT_TEMPLATE = """\
You are an expert experiment designer for multi-target genetic perturbation screens.

CONTEXT: We are running an active learning loop to discover "hit" genes — genes \
whose perturbation causes a strong phenotypic effect — across K biological targets \
simultaneously. Each cycle, we query a batch of genes and observe their effect on \
ALL K targets at once.

BIOLOGY CONTEXT:
{biology_context}

FEATURES: The GP model uses data-driven features (PCA of gene dependency/interaction \
data) PLUS LLM-derived biology prior features (known regulators, pathway membership, \
confidence scores you provided earlier). These priors give the GP a warm start.

YOUR ROLE: Each cycle, observe the experiment state and decide:
1. Which acquisition POLICY to use
2. How to WEIGHT the K targets (focus on harder/lagging ones)
3. The BATCH SIZE for this cycle

AVAILABLE POLICIES:
- "max_poh": STRONG DEFAULT. Selects genes with highest probability of being a hit \
for ANY target. Works well from cycle 0 onward. Ignores target weights.
- "mean_poh": Balanced. Weighted average of per-target hit probabilities. Use to \
direct effort toward specific lagging targets via target_weights.
- "coverage_poh": Equity-focused. Automatically boosts targets with lowest recall. \
Use when one target is significantly behind (>50% gap in recall).
- "mt_thompson": Stochastic exploration via GP posterior sampling. ONLY use when \
GP is truly uninformative (mean_GP_σ > 0.3). Sacrifices exploitation.

STRATEGY GUIDELINES:
- max_poh is generally best from cycle 0. Use it as default unless there's a reason not to.
- If one target lags significantly: use mean_poh with high weight on the lagging target, \
or coverage_poh.
- mt_thompson: use RARELY. Only if GP sigma is very high (model is uninformative).
- batch_size_multiplier: keep at 1.0 for consistent budget spending. Only increase \
if hit rate is very high (>0.15).
- Think about the BIOLOGY: are the targets correlated? If so, finding hits for one \
target likely helps the other.

OUTPUT FORMAT — respond with EXACTLY this JSON:
```json
{{
  "policy": "<one of: max_poh, mean_poh, coverage_poh, mt_thompson>",
  "target_weights": [<w1>, <w2>, ...],
  "batch_size_multiplier": <float from 0.5 to 2.0>,
  "reasoning": "<one paragraph explaining your decision>"
}}
```

RULES:
- target_weights must have exactly K entries, all positive, summing to 1.0
- batch_size_multiplier scales the default batch size (1.0 = normal)
- ALWAYS explain your reasoning — relate it to recall, GP quality, and biology
"""


def _build_system_prompt(biology_context: str) -> str:
    """Build system prompt with biology context."""
    return _SYSTEM_PROMPT_TEMPLATE.format(biology_context=biology_context)


def _format_state_for_llm(
    cycle: int,
    n_cycles: int,
    n_observed: int,
    n_total: int,
    K: int,
    screen_names: list[str] | None,
    per_target_recall: np.ndarray,
    per_target_hits: np.ndarray,
    per_target_total_hits: np.ndarray,
    thresholds: np.ndarray,
    gp_mean_sigma: np.ndarray,
    target_correlation: np.ndarray,
    last_cycle_hit_rate: float,
    default_batch_size: int,
    budget_remaining: int,
) -> str:
    """Format experiment state as a human-readable prompt for the LLM."""
    target_names = screen_names or [f"Target_{k}" for k in range(K)]

    # Per-target status
    target_lines = []
    for k in range(K):
        name = target_names[k]
        r = per_target_recall[k]
        h = per_target_hits[k]
        total = per_target_total_hits[k]
        sigma_k = gp_mean_sigma[k]
        target_lines.append(
            f"  {name}: recall={r:.3f} ({int(h)}/{int(total)} hits found), "
            f"mean_GP_σ={sigma_k:.3f}"
        )
    target_block = "\n".join(target_lines)

    # Correlation
    corr_lines = []
    for i in range(K):
        row = ", ".join(f"{target_correlation[i,j]:.2f}" for j in range(K))
        corr_lines.append(f"  [{row}]")
    corr_block = "\n".join(corr_lines)

    return f"""\
EXPERIMENT STATE — Cycle {cycle}/{n_cycles}
Observed: {n_observed}/{n_total} genes ({100*n_observed/n_total:.1f}%)
Budget remaining: {budget_remaining} queries
Default batch size: {default_batch_size}
Last cycle hit rate: {last_cycle_hit_rate:.3f}

PER-TARGET STATUS (K={K}):
{target_block}

TARGET CORRELATION MATRIX:
{corr_block}

DECISION NEEDED: What policy, weights, and batch size for cycle {cycle}?"""


def _parse_decision(
    text: str, K: int, default_batch: int,
) -> AgentDecision | None:
    """Parse the LLM's JSON response into an AgentDecision."""
    obj = parse_json_from_text(text)
    if obj is None:
        return None

    policy = obj.get("policy", "")
    if policy not in AVAILABLE_POLICIES:
        return None

    weights_raw = obj.get("target_weights", [])
    if len(weights_raw) != K:
        return None
    try:
        weights = np.array([float(w) for w in weights_raw])
    except (ValueError, TypeError):
        return None
    if np.any(weights <= 0) or np.isnan(weights).any():
        return None
    weights = weights / weights.sum()  # normalize

    multiplier = float(obj.get("batch_size_multiplier", 1.0))
    multiplier = np.clip(multiplier, 0.5, 2.0)
    batch = max(1, int(round(default_batch * multiplier)))

    reasoning = obj.get("reasoning", "(no reasoning)")

    return AgentDecision(
        policy=policy,
        target_weights=weights,
        batch_size=batch,
        reasoning=reasoning,
        raw_response=text,
        used_llm=True,
    )


def _fallback_decision(
    cycle: int,
    n_cycles: int,
    K: int,
    per_target_recall: np.ndarray,
    default_batch: int,
) -> AgentDecision:
    """Deterministic fallback when LLM is unavailable or returns garbage."""
    # Heuristic: max_poh is generally best, switch only if needed
    if cycle < 2:
        policy = "mean_poh"  # balanced early to learn about all targets
    elif cycle < n_cycles - 3:
        # Use coverage_poh if recall is imbalanced, else mean_poh
        recall_range = per_target_recall.max() - per_target_recall.min()
        policy = "coverage_poh" if recall_range > 0.02 else "mean_poh"
    else:
        policy = "max_poh"

    # Weight by inverse recall (harder targets get more)
    raw_w = 1.0 / (per_target_recall + 0.01)
    weights = raw_w / raw_w.sum()

    return AgentDecision(
        policy=policy,
        target_weights=weights,
        batch_size=default_batch,
        reasoning=f"Fallback heuristic: cycle {cycle}/{n_cycles}, policy={policy}",
        raw_response="",
        used_llm=False,
    )


class ExperimentAgent:
    """LLM-powered experiment agent that makes per-cycle decisions.

    Uses Ollama to reason about the experiment state and choose
    the acquisition policy, target weights, and batch size.

    Falls back to a deterministic heuristic if the LLM is unavailable.
    """

    def __init__(
        self,
        client: OllamaClient | None = None,
        screen_names: list[str] | None = None,
        biology_context: str | None = None,
    ):
        self.client = client
        self.screen_names = screen_names
        self.biology_context = biology_context or "No biology context available."
        self.system_prompt = _build_system_prompt(self.biology_context)
        self.history: list[AgentDecision] = []
        self._llm_available: bool | None = None  # lazy check

    def _check_llm(self) -> bool:
        """Check LLM availability (cached)."""
        if self._llm_available is None:
            if self.client is None:
                self._llm_available = False
            else:
                self._llm_available = self.client.is_available()
                if self._llm_available:
                    logger.info("LLM agent connected to Ollama")
                else:
                    logger.warning("Ollama not reachable — using fallback heuristic")
        return self._llm_available

    def decide(
        self,
        cycle: int,
        n_cycles: int,
        n_observed: int,
        n_total: int,
        K: int,
        per_target_recall: np.ndarray,
        per_target_hits: np.ndarray,
        per_target_total_hits: np.ndarray,
        thresholds: np.ndarray,
        gp_mean_sigma: np.ndarray,
        target_correlation: np.ndarray,
        last_cycle_hit_rate: float,
        default_batch_size: int,
        budget_remaining: int,
    ) -> AgentDecision:
        """Get a decision from the LLM agent for this cycle.

        Parameters
        ----------
        cycle : current cycle number
        n_cycles : total cycles
        n_observed : genes observed so far
        n_total : total genes in pool
        K : number of targets
        per_target_recall : (K,) recall per target
        per_target_hits : (K,) hits found per target
        per_target_total_hits : (K,) total hits per target
        thresholds : (K,) hit thresholds
        gp_mean_sigma : (K,) mean posterior std per target
        target_correlation : (K, K) sample correlation matrix
        last_cycle_hit_rate : hit rate in previous cycle
        default_batch_size : default batch size
        budget_remaining : queries remaining

        Returns
        -------
        AgentDecision with policy, weights, batch_size, reasoning
        """
        if not self._check_llm():
            decision = _fallback_decision(
                cycle, n_cycles, K, per_target_recall, default_batch_size,
            )
            self.history.append(decision)
            return decision

        user_msg = _format_state_for_llm(
            cycle=cycle,
            n_cycles=n_cycles,
            n_observed=n_observed,
            n_total=n_total,
            K=K,
            screen_names=self.screen_names,
            per_target_recall=per_target_recall,
            per_target_hits=per_target_hits,
            per_target_total_hits=per_target_total_hits,
            thresholds=thresholds,
            gp_mean_sigma=gp_mean_sigma,
            target_correlation=target_correlation,
            last_cycle_hit_rate=last_cycle_hit_rate,
            default_batch_size=default_batch_size,
            budget_remaining=budget_remaining,
        )

        try:
            response = self.client.chat(
                system=self.system_prompt,
                user=user_msg,
            )
            logger.info(f"LLM response (cycle {cycle}): {response[:200]}...")

            decision = _parse_decision(response, K, default_batch_size)
            if decision is not None:
                self.history.append(decision)
                return decision

            # Parse failed — fall back
            logger.warning(f"Failed to parse LLM response, falling back")
        except Exception as e:
            logger.warning(f"LLM call failed: {e}")

        decision = _fallback_decision(
            cycle, n_cycles, K, per_target_recall, default_batch_size,
        )
        self.history.append(decision)
        return decision
