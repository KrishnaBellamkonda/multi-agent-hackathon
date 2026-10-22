"""
LLM-as-Policy Agent — the LLM directly selects which genes to query.

Instead of the LLM choosing which *formula* to run (max_poh, mean_poh, etc.),
the LLM here IS the acquisition function. Each cycle:

1. GP computes posteriors (μ, σ) for all pool genes
2. We pre-filter to top-N candidates by max_poh (shortlist)
3. LLM sees gene names + GP stats + biology context for the shortlist
4. LLM picks which genes to query, using biology reasoning to break ties

This is where LLM knowledge should actually help: when two genes have
similar GP scores, knowing "JAK1 is a core immune kinase" lets the LLM
make a better selection than pure math.

The LLM can also RERANK: it sees genes the GP thinks are mediocre but
that biology says are important, and can promote them.
"""
from __future__ import annotations

import logging

import numpy as np
from scipy.stats import norm

from maehd.agents.llm_client import OllamaClient, parse_json_from_text

logger = logging.getLogger("maehd.llm_policy")


_LLM_POLICY_SYSTEM = """\
You are a molecular biology expert helping design CRISPR perturbation experiments.

BIOLOGY CONTEXT:
{biology_context}

You are selecting which genes to experimentally perturb next. Each gene you pick \
will be knocked out and the effect on ALL targets will be measured simultaneously.

YOUR GOAL: Maximize the number of "hit" genes discovered — genes whose knockout \
causes a strong phenotypic effect on at least one target. Use both the statistical \
scores AND your biology knowledge to make optimal selections.

STRATEGY:
- The GP model provides statistical scores (probability of hit, uncertainty).
- YOUR value-add is biological reasoning: knowing that certain genes are in key \
pathways, are known regulators, or function in biology relevant to the targets.
- When two genes have similar GP scores, PREFER the one with stronger biological \
rationale for affecting the phenotype.
- Also consider EXPLORATION: genes with high uncertainty AND biological relevance \
are valuable — querying them teaches us the most.
- Think about gene FUNCTION: kinases, transcription factors, and signaling molecules \
are more likely to be hits than housekeeping genes.
- Consider pathway relationships: if you already queried a gene in pathway X and \
it was a hit, other genes in that pathway are likely hits too.
"""

_LLM_POLICY_PROMPT = """\
CYCLE {cycle}/{n_cycles} — Select {batch_size} genes to query next.

EXPERIMENT STATE:
- Observed: {n_observed}/{n_total} genes
- Budget remaining: {budget_remaining} queries
- Last cycle hit rate: {last_hit_rate:.3f}

PER-TARGET STATUS:
{target_status}

CANDIDATE GENES (top {n_candidates} by GP statistical score):
{gene_table}

From these {n_candidates} candidates, select EXACTLY {batch_size} genes to query.

Consider:
1. GP probability of hit (poh) — statistical evidence
2. GP uncertainty (sigma) — exploration value  
3. Gene biology — your knowledge of gene function and pathway membership
4. Target coverage — are certain targets lagging behind?

Respond with EXACTLY this JSON:
```json
{{
  "selected_genes": ["GENE1", "GENE2", "GENE3", ...],
  "reasoning": "brief explanation of your selection strategy"
}}
```

RULES:
- Select EXACTLY {batch_size} genes from the candidate list
- Use official HUGO gene symbols exactly as shown
- Prefer genes where biology AND statistics both support selection
- Include some high-uncertainty genes for exploration (at least 20%)
"""


class LLMPolicyAgent:
    """LLM that directly selects genes to query (agent IS the policy).

    Parameters
    ----------
    client : OllamaClient for LLM calls
    biology_context : string describing the screens/phenotypes
    n_candidates : how many top GP candidates to show the LLM (default 100)
    """

    def __init__(
        self,
        client: OllamaClient,
        biology_context: str = "",
        screen_names: list[str] | None = None,
        n_candidates: int = 100,
    ):
        self.client = client
        self.biology_context = biology_context or "No biology context provided."
        self.screen_names = screen_names
        self.n_candidates = n_candidates
        self.system_prompt = _LLM_POLICY_SYSTEM.format(
            biology_context=self.biology_context,
        )
        self.history: list[dict] = []

    def select_batch(
        self,
        gene_names: np.ndarray,
        mu_pool: np.ndarray,
        sigma_pool: np.ndarray,
        thresholds: np.ndarray,
        pool_indices: np.ndarray,
        batch_size: int,
        cycle: int,
        n_cycles: int,
        n_observed: int,
        n_total: int,
        budget_remaining: int,
        per_target_recall: np.ndarray,
        last_hit_rate: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Select genes to query using LLM reasoning.

        Parameters
        ----------
        gene_names : (n_pool,) gene names for pool candidates
        mu_pool : (n_pool, K) GP posterior means
        sigma_pool : (n_pool, K) GP posterior stds
        thresholds : (K,) per-target hit thresholds
        pool_indices : (n_pool,) global indices of pool genes
        batch_size : number of genes to select
        ... other state info for the prompt

        Returns
        -------
        selected_pool_idx : (batch_size,) indices INTO pool_indices
        """
        if rng is None:
            rng = np.random.default_rng()

        n_pool, K = mu_pool.shape

        # Compute per-gene stats
        poh_per_target = np.column_stack([
            1.0 - norm.cdf((thresholds[k] - mu_pool[:, k]) / np.maximum(sigma_pool[:, k], 1e-8))
            for k in range(K)
        ])  # (n_pool, K)
        max_poh = poh_per_target.max(axis=1)  # (n_pool,)
        mean_sigma = sigma_pool.mean(axis=1)  # (n_pool,)

        # Pre-filter: top N candidates by a combined score
        # 70% exploitation (max_poh) + 30% exploration (sigma)
        combined = 0.7 * max_poh + 0.3 * (mean_sigma / max(mean_sigma.max(), 1e-8))
        n_show = min(self.n_candidates, n_pool)
        top_idx = np.argsort(combined)[::-1][:n_show]

        # Build gene table for the LLM
        target_names = self.screen_names or [f"Target_{k}" for k in range(K)]
        short_names = [n.split("_")[-1][:8] for n in target_names]

        # Header
        header = f"{'Gene':<12}"
        for sn in short_names:
            header += f" poh_{sn:>6} σ_{sn:>6}"
        header += f" {'max_poh':>8} {'mean_σ':>8}"

        lines = [header]
        candidate_genes = []
        for rank, idx in enumerate(top_idx):
            gname = gene_names[idx] if idx < len(gene_names) else f"gene_{idx}"
            candidate_genes.append(gname)
            row = f"{gname:<12}"
            for k in range(K):
                row += f" {poh_per_target[idx, k]:>8.3f} {sigma_pool[idx, k]:>8.3f}"
            row += f" {max_poh[idx]:>8.3f} {mean_sigma[idx]:>8.3f}"
            lines.append(row)

        gene_table = "\n".join(lines)

        # Build target status
        target_lines = []
        for k in range(K):
            r = per_target_recall[k]
            target_lines.append(f"  {target_names[k]}: recall={r:.3f}, threshold={thresholds[k]:.3f}")
        target_status = "\n".join(target_lines)

        # Build prompt
        prompt = _LLM_POLICY_PROMPT.format(
            cycle=cycle,
            n_cycles=n_cycles,
            batch_size=batch_size,
            n_observed=n_observed,
            n_total=n_total,
            budget_remaining=budget_remaining,
            last_hit_rate=last_hit_rate,
            target_status=target_status,
            n_candidates=n_show,
            gene_table=gene_table,
        )

        # Query LLM
        selected_names = None
        reasoning = ""
        try:
            response = self.client.chat(
                system=self.system_prompt,
                user=prompt,
            )
            result = parse_json_from_text(response)
            if result and "selected_genes" in result:
                selected_names = result["selected_genes"]
                reasoning = result.get("reasoning", "")
        except Exception as e:
            logger.warning(f"LLM policy call failed cycle {cycle}: {e}")

        # Map LLM selections back to pool indices
        if selected_names is not None:
            gene_to_pool = {gene_names[i].upper(): i for i in top_idx}
            llm_selected = []
            for g in selected_names:
                pidx = gene_to_pool.get(g.upper().strip())
                if pidx is not None and pidx not in llm_selected:
                    llm_selected.append(pidx)

            if len(llm_selected) >= batch_size * 0.5:
                # LLM returned enough valid genes — use them
                if len(llm_selected) > batch_size:
                    llm_selected = llm_selected[:batch_size]
                elif len(llm_selected) < batch_size:
                    # Fill remaining from top GP candidates not already selected
                    remaining = [i for i in top_idx if i not in llm_selected]
                    need = batch_size - len(llm_selected)
                    llm_selected.extend(remaining[:need])

                self.history.append({
                    "cycle": cycle,
                    "used_llm": True,
                    "n_llm_valid": len([g for g in selected_names if g.upper().strip() in gene_to_pool]),
                    "reasoning": reasoning[:200],
                })
                return np.array(llm_selected[:batch_size])

        # Fallback: use pure max_poh ranking
        fallback_selected = top_idx[:batch_size]
        self.history.append({
            "cycle": cycle,
            "used_llm": False,
            "reasoning": "Fallback: LLM did not return valid selections",
        })
        return fallback_selected
