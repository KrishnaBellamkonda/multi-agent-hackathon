"""
Biology Knowledge Agent — uses LLM to generate prior features from domain knowledge.

The key insight: an LLM trained on biological literature knows which genes
are likely regulators of specific phenotypes. We extract this knowledge
as features that augment the data-driven PCA features, giving the GP
a strong prior signal especially in early cycles when data is scarce.

This is the main advantage of an agentic framework over pure math:
the LLM contributes pre-existing knowledge of biology, not just
decisions about hyperparameters.

Two levels of biology features:
  Level 1 — "Seed knowledge": known regulators, pathways, confidence (~50-100 genes)
  Level 2 — "Batch scoring": ALL genes scored on relevance by the LLM in batches
             of 100. This is the key innovation — dense annotations for each gene.

For each phenotypic target, the agent generates 6 features per gene:
  [0] is_positive_regulator (0 or 1)
  [1] is_negative_regulator (0 or 1)
  [2] pathway_membership_score (0.0 to 1.0)
  [3] confidence_prior (0.0 to 1.0)
  [4] llm_relevance_score (0.0 to 1.0) — from batch scoring
  [5] functional_category (0-N encoded) — which functional class LLM assigns
"""
from __future__ import annotations

import logging
import time

import numpy as np

from maehd.agents.llm_client import OllamaClient, parse_json_from_text

logger = logging.getLogger("maehd.biology_agent")


# ── Phenotype and cell-type context for each GeneDisco screen ──

SCREEN_BIOLOGY = {
    "schmidt_2021_ifng": {
        "phenotype": "IFNγ (interferon-gamma) cytokine production",
        "cell_type": "primary human CD4+ T cells",
        "context": (
            "Genome-wide CRISPR knockout screen in primary human CD4+ T cells. "
            "Cells were activated with anti-CD3/CD28 beads and IFNγ secretion "
            "was measured by intracellular staining + FACS. Hit genes are those "
            "whose knockout significantly changes IFNγ production."
        ),
    },
    "schmidt_2021_il2": {
        "phenotype": "IL-2 (interleukin-2) cytokine production",
        "cell_type": "primary human CD4+ T cells",
        "context": (
            "Genome-wide CRISPR knockout screen in primary human CD4+ T cells. "
            "Cells were activated with anti-CD3/CD28 beads and IL-2 secretion "
            "was measured. Hit genes are those whose knockout significantly "
            "changes IL-2 production."
        ),
    },
    "zhuang_2019_nk": {
        "phenotype": "NK cell-mediated cytotoxicity against tumor cells",
        "cell_type": "human natural killer (NK) cells",
        "context": (
            "CRISPR screen in NK cells co-cultured with tumor cell lines. "
            "The readout is tumor cell killing efficiency. Hit genes affect "
            "NK cell cytotoxic function or tumor cell susceptibility to killing."
        ),
    },
    "sanchez_2021_tau": {
        "phenotype": "Tau protein aggregation and levels",
        "cell_type": "human iPSC-derived cortical neurons",
        "context": (
            "CRISPR screen in iPSC-derived neurons measuring Tau protein levels. "
            "Relevant to Alzheimer's disease and tauopathies. Hit genes change "
            "Tau protein expression, phosphorylation, or aggregation."
        ),
    },
    "zhu_2021_sarscov2": {
        "phenotype": "SARS-CoV-2 viral entry and replication",
        "cell_type": "human lung epithelial cells (A549-ACE2)",
        "context": (
            "CRISPR screen for host factors required for SARS-CoV-2 infection. "
            "A549 cells expressing ACE2 were infected and surviving cells were "
            "sequenced. Hit genes are host factors essential for viral entry, "
            "replication, or virus-induced cell death."
        ),
    },
}

# ── Prompt template for extracting biology knowledge ──

_KNOWLEDGE_PROMPT = """\
You are a molecular biology and genomics expert. For the following CRISPR \
perturbation screen, provide comprehensive biology knowledge.

SCREEN: {context}
PHENOTYPE: {phenotype}
CELL TYPE: {cell_type}

Think carefully about:
- Genes in the core signaling pathway controlling this phenotype
- Transcription factors that directly regulate it
- Upstream regulators, kinases, phosphatases
- Known positive regulators (KO of these genes DECREASES the phenotype)
- Known negative regulators (KO of these genes INCREASES the phenotype)
- Genes from relevant KEGG/Reactome/GO biological process terms
- Chromatin remodelers, epigenetic regulators
- Metabolic enzymes that influence this phenotype
- Surface receptors and ligands
- Ubiquitin/proteasome pathway members
- RNA-binding proteins and splicing factors relevant to this biology

Respond with EXACTLY this JSON:
```json
{{
  "positive_regulators": ["GENE1", "GENE2", ...],
  "negative_regulators": ["GENE1", "GENE2", ...],
  "key_pathways": {{
    "pathway_name": ["GENE1", "GENE2", ...],
    "another_pathway": ["GENE1", "GENE2", ...]
  }},
  "confidence_genes": {{
    "GENE1": 0.95,
    "GENE2": 0.8,
    "GENE3": 0.6
  }},
  "functional_categories": {{
    "category_name": ["GENE1", "GENE2", ...],
    "another_category": ["GENE3", "GENE4", ...]
  }},
  "reasoning": "brief explanation of your choices"
}}
```

RULES:
- Use official HUGO gene symbols (e.g., IFNG, JAK1, STAT1, TP53)
- List at LEAST 50 positive and 30 negative regulators — be comprehensive
- confidence_genes: gene → confidence (0.0 to 1.0) for your top 100+ most important genes
- functional_categories: group genes into 8-15 functional categories relevant to this phenotype
- Include canonical well-known AND less-obvious regulators (reach for 100+ total genes)
- Genes must be human genes present in a genome-wide CRISPR library (~18000 genes)
"""

# ── Batch scoring prompt — scores ALL genes, not just known ones ──

_BATCH_SCORE_PROMPT = """\
You are a molecular biology expert. For the following CRISPR screen phenotype, \
score each gene in the list below on how likely its knockout is to affect the phenotype.

PHENOTYPE: {phenotype}
CELL TYPE: {cell_type}
SCREEN: {context}

GENE LIST (score each one):
{gene_list}

For each gene, provide a relevance score from 0.0 to 1.0:
- 1.0 = gene is a known direct regulator of this phenotype
- 0.7-0.9 = gene is in a relevant pathway or has strong indirect evidence
- 0.3-0.6 = gene has weak/indirect connection to this biology
- 0.1-0.2 = gene has minimal known connection but is biologically active
- 0.0 = housekeeping gene or no known connection to this phenotype

Respond with EXACTLY this JSON (no other text):
```json
{{
  "scores": {{
    "GENE1": 0.8,
    "GENE2": 0.1,
    "GENE3": 0.6
  }}
}}
```

RULES:
- Score EVERY gene in the list — do not skip any
- Use your knowledge of gene function, pathway membership, and biological role
- Be calibrated: most genes should score 0.0-0.2 (not relevant), only ~10% should be >0.5
"""

# ── Feature-set recommendation prompt ──

_FEATURE_SELECT_PROMPT = """\
We are studying the following phenotypes in a CRISPR screen:
{target_descriptions}

Available gene feature sets:
- "achilles": Gene dependency scores from DepMap/Achilles (808 dimensions). \
Measures which genes are essential in various cancer cell lines.
- "string": Protein-protein interaction network embeddings from STRING database \
(800 dimensions). Captures network neighborhood and interaction partners.

Which feature set is more likely to be informative for predicting hits in these \
specific screens? Consider:
- What the features actually measure vs what the screens measure
- Whether gene dependency (achilles) or network structure (string) is more \
relevant to the phenotype
- Whether the cell types in DepMap overlap with the screen cell types

Respond with EXACTLY this JSON:
```json
{{
  "recommended_feature": "<achilles or string>",
  "reasoning": "one paragraph explanation",
  "combine_both": <true or false>,
  "feature_weights": {{"achilles": <0-1>, "string": <0-1>}}
}}
```
"""


class BiologyKnowledgeAgent:
    """Extracts biology knowledge from an LLM to create prior features.

    Two-stage approach:
      Stage 1: Seed knowledge — known regulators, pathways, categories (1 call per target)
      Stage 2: Batch scoring — score ALL genes on relevance (many calls per target)

    This produces DENSE features for every gene, not just sparse annotations
    on the ~60 genes the LLM remembers by name.
    """

    def __init__(self, client: OllamaClient, batch_size: int = 100):
        self.client = client
        self.batch_size = batch_size
        self._cache: dict[str, dict] = {}

    def query_knowledge(self, screen_name: str) -> dict:
        """Query LLM for seed biology knowledge about a specific screen.

        Returns dict with keys: positive_regulators, negative_regulators,
        key_pathways, confidence_genes, functional_categories, reasoning.
        """
        if screen_name in self._cache:
            return self._cache[screen_name]

        bio = SCREEN_BIOLOGY.get(screen_name, {})
        if not bio:
            logger.warning(f"No biology context for screen: {screen_name}")
            empty = _empty_knowledge("unknown screen")
            self._cache[screen_name] = empty
            return empty

        prompt = _KNOWLEDGE_PROMPT.format(
            context=bio["context"],
            phenotype=bio["phenotype"],
            cell_type=bio["cell_type"],
        )

        try:
            response = self.client.chat(
                system=(
                    "You are a world-class molecular biologist with deep expertise "
                    "in genomics, CRISPR screens, immunology, neuroscience, and virology. "
                    "Provide accurate gene annotations using official HUGO symbols. "
                    "Be comprehensive — list as many genes as you confidently can."
                ),
                user=prompt,
            )
            knowledge = parse_json_from_text(response)
            if knowledge is None:
                logger.warning(f"Failed to parse biology knowledge for {screen_name}")
                knowledge = _empty_knowledge("parse_failed")
            else:
                logger.info(
                    f"Biology knowledge for {screen_name}: "
                    f"{len(knowledge.get('positive_regulators', []))} pos, "
                    f"{len(knowledge.get('negative_regulators', []))} neg regulators, "
                    f"{len(knowledge.get('confidence_genes', {}))} confidence scores, "
                    f"{len(knowledge.get('functional_categories', {}))} categories"
                )
        except Exception as e:
            logger.warning(f"Biology query failed for {screen_name}: {e}")
            knowledge = _empty_knowledge(f"error: {e}")

        self._cache[screen_name] = knowledge
        return knowledge

    def batch_score_genes(
        self,
        gene_names: list[str],
        screen_name: str,
        verbose: bool = True,
    ) -> np.ndarray:
        """Score ALL genes on relevance to a screen using LLM batch queries.

        Splits gene list into batches, asks the LLM to score each batch.
        Returns (n_genes,) array of relevance scores in [0, 1].
        """
        bio = SCREEN_BIOLOGY.get(screen_name, {})
        if not bio:
            return np.zeros(len(gene_names), dtype=np.float32)

        n = len(gene_names)
        scores = np.full(n, 0.05, dtype=np.float32)  # default low prior
        gene_lookup = {g.upper(): i for i, g in enumerate(gene_names)}

        n_batches = (n + self.batch_size - 1) // self.batch_size
        scored_count = 0
        failed_batches = 0

        for batch_idx in range(n_batches):
            start = batch_idx * self.batch_size
            end = min(start + self.batch_size, n)
            batch_genes = gene_names[start:end]
            gene_list_str = ", ".join(batch_genes)

            prompt = _BATCH_SCORE_PROMPT.format(
                phenotype=bio["phenotype"],
                cell_type=bio["cell_type"],
                context=bio["context"],
                gene_list=gene_list_str,
            )

            try:
                response = self.client.chat(
                    system=(
                        "You are a molecular biology expert. Score each gene's "
                        "relevance to the given phenotype. Be calibrated: most "
                        "genes should score 0.0-0.2. Respond with JSON only."
                    ),
                    user=prompt,
                )
                result = parse_json_from_text(response)
                if result and "scores" in result:
                    batch_scores = result["scores"]
                    for gene, score in batch_scores.items():
                        idx = gene_lookup.get(gene.upper().strip())
                        if idx is not None:
                            try:
                                scores[idx] = np.clip(float(score), 0.0, 1.0)
                                scored_count += 1
                            except (ValueError, TypeError):
                                pass
                else:
                    failed_batches += 1
            except Exception as e:
                failed_batches += 1
                logger.warning(f"Batch {batch_idx} failed for {screen_name}: {e}")

            # Brief pause between batches to avoid overwhelming Ollama
            if batch_idx < n_batches - 1:
                time.sleep(0.1)

        if verbose:
            pheno = bio.get("phenotype", screen_name)
            print(
                f"    Batch scoring ({pheno}): "
                f"{scored_count}/{n} genes scored, "
                f"{failed_batches}/{n_batches} batches failed, "
                f"mean_score={scores.mean():.3f}, "
                f"genes>0.5: {(scores > 0.5).sum()}"
            )

        return scores

    def recommend_features(self, screen_names: list[str]) -> dict:
        """Ask LLM which feature set is best for these screens."""
        descriptions = []
        for sname in screen_names:
            bio = SCREEN_BIOLOGY.get(sname, {})
            if bio:
                descriptions.append(
                    f"- {bio['phenotype']} in {bio['cell_type']}"
                )
            else:
                descriptions.append(f"- {sname}")

        prompt = _FEATURE_SELECT_PROMPT.format(
            target_descriptions="\n".join(descriptions),
        )

        try:
            response = self.client.chat(
                system="You are a computational biology expert.",
                user=prompt,
            )
            result = parse_json_from_text(response)
            if result is not None:
                return result
        except Exception as e:
            logger.warning(f"Feature recommendation failed: {e}")

        return {"recommended_feature": "achilles", "reasoning": "default", "combine_both": False}

    def create_prior_features(
        self,
        gene_names: list[str],
        screen_names: list[str],
        do_batch_scoring: bool = True,
        verbose: bool = True,
    ) -> np.ndarray:
        """Create LLM-derived prior features for all genes.

        For each target k, generates 6 features per gene:
            [0] is_positive_regulator: 1 if LLM names this gene as positive regulator
            [1] is_negative_regulator: 1 if LLM names as negative regulator
            [2] pathway_score: fraction of relevant pathways gene appears in
            [3] confidence_prior: LLM's named confidence (0-1), from seed knowledge
            [4] llm_relevance: dense batch-scored relevance (0-1) for ALL genes
            [5] category_encoded: which functional category the gene belongs to

        Parameters
        ----------
        gene_names : list of HUGO gene symbols from the dataset
        screen_names : list of screen identifiers
        do_batch_scoring : whether to score ALL genes (expensive, ~175 LLM calls per target)

        Returns
        -------
        features : (n_genes, 6*K) array of prior features
        """
        n = len(gene_names)
        K = len(screen_names)
        gene_lookup = {g.upper(): i for i, g in enumerate(gene_names)}

        features = np.zeros((n, 6 * K), dtype=np.float32)

        for k, sname in enumerate(screen_names):
            knowledge = self.query_knowledge(sname)
            pos_ct = neg_ct = path_ct = conf_ct = cat_ct = 0
            offset = 6 * k

            # [0] Positive regulators
            for gene in knowledge.get("positive_regulators", []):
                idx = gene_lookup.get(gene.upper().strip())
                if idx is not None:
                    features[idx, offset] = 1.0
                    pos_ct += 1

            # [1] Negative regulators
            for gene in knowledge.get("negative_regulators", []):
                idx = gene_lookup.get(gene.upper().strip())
                if idx is not None:
                    features[idx, offset + 1] = 1.0
                    neg_ct += 1

            # [2] Pathway membership scores
            pathways = knowledge.get("key_pathways", {})
            if isinstance(pathways, dict) and pathways:
                n_pathways = len(pathways)
                for _pname, pgenes in pathways.items():
                    if isinstance(pgenes, list):
                        for gene in pgenes:
                            idx = gene_lookup.get(gene.upper().strip())
                            if idx is not None:
                                features[idx, offset + 2] += 1.0 / n_pathways
                                path_ct += 1

            # [3] Confidence scores (seed)
            confidence = knowledge.get("confidence_genes", {})
            if isinstance(confidence, dict):
                for gene, conf in confidence.items():
                    idx = gene_lookup.get(gene.upper().strip())
                    if idx is not None:
                        try:
                            features[idx, offset + 3] = np.clip(float(conf), 0.0, 1.0)
                            conf_ct += 1
                        except (ValueError, TypeError):
                            pass

            # [4] Dense batch scoring — relevance for ALL genes
            if do_batch_scoring:
                if verbose:
                    print(f"  Batch scoring all genes for target {k}...")
                relevance = self.batch_score_genes(gene_names, sname, verbose=verbose)
                features[:, offset + 4] = relevance

            # [5] Functional category features
            categories = knowledge.get("functional_categories", {})
            if isinstance(categories, dict) and categories:
                cat_names = sorted(categories.keys())
                for cat_idx, cat_name in enumerate(cat_names):
                    cat_genes = categories[cat_name]
                    if isinstance(cat_genes, list):
                        # Encode as (cat_idx + 1) / n_categories
                        val = (cat_idx + 1) / len(cat_names)
                        for gene in cat_genes:
                            idx = gene_lookup.get(gene.upper().strip())
                            if idx is not None:
                                features[idx, offset + 5] = val
                                cat_ct += 1

            if verbose:
                bio = SCREEN_BIOLOGY.get(sname, {})
                pheno = bio.get("phenotype", sname)
                print(
                    f"  Target {k} ({pheno}): "
                    f"{pos_ct} pos, {neg_ct} neg regulators, "
                    f"{path_ct} pathway, {conf_ct} confidence, "
                    f"{cat_ct} categorized"
                )
                reasoning = knowledge.get("reasoning", "")
                if reasoning and reasoning not in ("parse_failed", "unknown screen"):
                    print(f"    LLM: {reasoning[:150]}...")

        # Normalize pathway scores to [0, 1]
        for k in range(K):
            col = 6 * k + 2
            mx = features[:, col].max()
            if mx > 0:
                features[:, col] /= mx

        return features

    def get_phenotype_context(self, screen_names: list[str]) -> str:
        """Human-readable description of the screens (for per-cycle agent prompt)."""
        lines = []
        for k, sname in enumerate(screen_names):
            bio = SCREEN_BIOLOGY.get(sname, {})
            if bio:
                lines.append(
                    f"Target {k}: {bio['phenotype']} in {bio['cell_type']}"
                )
            else:
                lines.append(f"Target {k}: {sname}")
        return "\n".join(lines)


def _empty_knowledge(reason: str) -> dict:
    return {
        "positive_regulators": [],
        "negative_regulators": [],
        "key_pathways": {},
        "confidence_genes": {},
        "functional_categories": {},
        "reasoning": reason,
    }
