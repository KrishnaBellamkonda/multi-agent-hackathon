"""
Hybrid LLM Policy — GP picks the exploitation batch, LLM picks exploration.

v7 showed that full LLM reranking hurts on same-biology setups because
the LLM overrides GP-optimal rankings with vague biology hunches.

v7b fix: **bounded hybrid**
- GP picks top `exploit_frac` of the batch (e.g., 70%) by pure max_poh
- LLM picks the remaining `explore_frac` (30%) from a **tiebreaker pool**:
  genes ranked 31-100 by GP where scores are close, so biology can break ties
- LLM gets **cycle memory**: which gene families produced hits so far

This bounds downside risk (the exploit slots are never worse than classic)
while giving the LLM a high-value exploration niche.
"""
from __future__ import annotations

import logging

import numpy as np
from scipy.stats import norm

from maehd.agents.llm_client import OllamaClient, parse_json_from_text

logger = logging.getLogger("maehd.llm_policy_hybrid")


_HYBRID_SYSTEM = """\
You are a molecular biology expert selecting genes for CRISPR perturbation.

BIOLOGY CONTEXT:
{biology_context}

The GP model already selected the most statistically promising genes. YOUR job \
is to fill the EXPLORATION SLOTS — picking from a pool of borderline candidates \
where statistics alone can't distinguish them. This is WHERE your biology \
knowledge adds the most value.

STRATEGY:
- All candidates in your pool have SIMILAR GP scores — statistics can't tell \
them apart, so biology reasoning is the tiebreaker.
- PREFER genes in pathways relevant to the targets: signaling cascades, TFs, \
receptors, kinases that affect the phenotype.
- AVOID housekeeping genes (ribosomal proteins, basic metabolic enzymes) \
unless you see specific evidence they're relevant.
- CONSIDER diversity: pick genes from different pathways to spread exploration \
rather than clustering in one family.
"""

_HYBRID_PROMPT = """\
CYCLE {cycle}/{n_cycles} — Fill {n_explore} EXPLORATION SLOTS.

The GP model already locked in {n_exploit} genes (the statistically best ones). \
You need to pick {n_explore} more from the borderline candidates below.

EXPERIMENT STATE:
- Observed: {n_observed}/{n_total} genes
- Budget remaining: {budget_remaining} queries
- Last cycle hit rate: {last_hit_rate:.3f}

{cycle_memory}

PER-TARGET STATUS:
{target_status}

EXPLORATION CANDIDATES (GP ranks {rank_start}-{rank_end}, similar scores):
{gene_table}

Select EXACTLY {n_explore} genes from these candidates.

Respond with EXACTLY this JSON:
```json
{{
  "selected_genes": ["GENE1", "GENE2", ...],
  "reasoning": "brief biology-based justification"
}}
```
"""


class HybridLLMPolicy:
    """GP handles exploitation, LLM handles exploration slots.

    Parameters
    ----------
    client : OllamaClient
    biology_context : string describing screens/phenotypes
    screen_names : list of screen identifiers
    exploit_frac : fraction of batch selected by pure GP (default 0.7)
    tiebreaker_pool_size : how many borderline candidates to show LLM (default 60)
    """

    def __init__(
        self,
        client: OllamaClient,
        biology_context: str = "",
        screen_names: list[str] | None = None,
        exploit_frac: float = 0.7,
        tiebreaker_pool_size: int = 60,
    ):
        self.client = client
        self.biology_context = biology_context or "No biology context provided."
        self.screen_names = screen_names
        self.exploit_frac = exploit_frac
        self.tiebreaker_pool_size = tiebreaker_pool_size
        self.system_prompt = _HYBRID_SYSTEM.format(
            biology_context=self.biology_context,
        )
        self.history: list[dict] = []
        # Cycle memory: track which gene picks were hits
        self._hit_genes: list[str] = []
        self._miss_genes: list[str] = []

    def update_cycle_outcomes(
        self,
        gene_names: list[str],
        was_hit: np.ndarray,
    ):
        """Feed back which genes from the last cycle were hits.

        Parameters
        ----------
        gene_names : names of genes that were queried last cycle
        was_hit : (n,) boolean — True if gene was a hit on any target
        """
        for name, hit in zip(gene_names, was_hit):
            if hit:
                self._hit_genes.append(name)
            else:
                self._miss_genes.append(name)

    def _build_cycle_memory(self) -> str:
        """Build a summary of past outcomes for the LLM."""
        if not self._hit_genes and not self._miss_genes:
            return "CYCLE MEMORY: No previous cycles yet."

        lines = ["CYCLE MEMORY (what worked so far):"]
        if self._hit_genes:
            # Show last 30 hits
            recent_hits = self._hit_genes[-30:]
            lines.append(f"  Recent HITS ({len(self._hit_genes)} total): {', '.join(recent_hits)}")
        if self._miss_genes:
            n_miss = len(self._miss_genes)
            lines.append(f"  Misses: {n_miss} genes queried with no hit")
        total = len(self._hit_genes) + len(self._miss_genes)
        if total > 0:
            rate = len(self._hit_genes) / total
            lines.append(f"  Overall hit rate: {rate:.3f}")
        return "\n".join(lines)

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
        """Select genes: GP picks exploit slots, LLM picks explore slots.

        Returns
        -------
        selected_pool_idx : (batch_size,) indices into the pool
        """
        if rng is None:
            rng = np.random.default_rng()

        n_pool, K = mu_pool.shape

        # Compute max PoH per gene
        poh_per_target = np.column_stack([
            1.0 - norm.cdf(
                (thresholds[k] - mu_pool[:, k]) / np.maximum(sigma_pool[:, k], 1e-8)
            )
            for k in range(K)
        ])
        max_poh = poh_per_target.max(axis=1)
        mean_sigma = sigma_pool.mean(axis=1)

        # Rank by max_poh
        ranked = np.argsort(max_poh)[::-1]

        # Split batch: exploit (GP) + explore (LLM)
        n_exploit = max(1, int(batch_size * self.exploit_frac))
        n_explore = batch_size - n_exploit

        # GP exploitation: take top n_exploit by pure max_poh
        exploit_idx = ranked[:n_exploit]

        # Tiebreaker pool for LLM: genes ranked just below the exploit cutoff
        pool_start = n_exploit
        pool_end = min(n_pool, pool_start + self.tiebreaker_pool_size)
        tiebreaker_idx = ranked[pool_start:pool_end]

        if len(tiebreaker_idx) == 0 or n_explore == 0:
            # No room for LLM exploration, pure GP
            selected = ranked[:batch_size]
            self.history.append({
                "cycle": cycle, "used_llm": False, "n_exploit": batch_size,
                "n_explore": 0, "reasoning": "No tiebreaker pool available",
            })
            return selected

        # Build gene table for tiebreaker pool
        target_names = self.screen_names or [f"Target_{k}" for k in range(K)]
        short_names = [n.split("_")[-1][:8] for n in target_names]

        header = f"{'Gene':<12} {'max_poh':>8}"
        for sn in short_names:
            header += f"  poh_{sn:>6}"
        header += f"  {'mean_σ':>8}"

        lines = [header]
        for idx in tiebreaker_idx:
            gname = gene_names[idx] if idx < len(gene_names) else f"gene_{idx}"
            row = f"{gname:<12} {max_poh[idx]:>8.3f}"
            for k in range(K):
                row += f"  {poh_per_target[idx, k]:>8.3f}"
            row += f"  {mean_sigma[idx]:>8.3f}"
            lines.append(row)

        gene_table = "\n".join(lines)

        # Target status
        target_lines = []
        for k in range(K):
            target_lines.append(
                f"  {target_names[k]}: recall={per_target_recall[k]:.3f}, "
                f"threshold={thresholds[k]:.3f}"
            )
        target_status = "\n".join(target_lines)

        # Cycle memory
        cycle_memory = self._build_cycle_memory()

        # Prompt
        prompt = _HYBRID_PROMPT.format(
            cycle=cycle,
            n_cycles=n_cycles,
            n_exploit=n_exploit,
            n_explore=n_explore,
            n_observed=n_observed,
            n_total=n_total,
            budget_remaining=budget_remaining,
            last_hit_rate=last_hit_rate,
            cycle_memory=cycle_memory,
            target_status=target_status,
            rank_start=pool_start + 1,
            rank_end=pool_end,
            gene_table=gene_table,
        )

        # Query LLM
        explore_idx = None
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

                # Map names to tiebreaker indices
                gene_to_idx = {}
                for idx in tiebreaker_idx:
                    gname = gene_names[idx] if idx < len(gene_names) else f"gene_{idx}"
                    gene_to_idx[gname.upper()] = idx

                llm_picks = []
                for g in selected_names:
                    pidx = gene_to_idx.get(g.upper().strip())
                    if pidx is not None and pidx not in llm_picks:
                        llm_picks.append(pidx)

                if len(llm_picks) >= n_explore * 0.5:
                    explore_idx = np.array(llm_picks[:n_explore])
        except Exception as e:
            logger.warning(f"Hybrid LLM call failed cycle {cycle}: {e}")

        # Fallback: if LLM failed, take next n_explore by GP ranking
        used_llm = explore_idx is not None
        if explore_idx is None:
            explore_idx = tiebreaker_idx[:n_explore]

        # Fill if LLM returned fewer than needed
        if len(explore_idx) < n_explore:
            remaining = [i for i in tiebreaker_idx if i not in set(explore_idx)]
            need = n_explore - len(explore_idx)
            explore_idx = np.concatenate([
                explore_idx, np.array(remaining[:need])
            ])

        selected = np.concatenate([exploit_idx, explore_idx[:n_explore]]).astype(int)

        self.history.append({
            "cycle": cycle,
            "used_llm": used_llm,
            "n_exploit": n_exploit,
            "n_explore": n_explore,
            "n_llm_valid": int(len(explore_idx)) if used_llm else 0,
            "reasoning": reasoning[:200],
        })

        return selected[:batch_size]
