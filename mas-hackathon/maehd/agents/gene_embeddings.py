"""
LLM-based gene functional embeddings.

Instead of scoring each gene on a single relevance axis, we score genes
along MULTIPLE biological function axes simultaneously. This creates
a multi-dimensional "LLM functional embedding" for each gene.

Axes (target-independent):
  0: immune_signaling — role in immune signaling pathways
  1: transcription_regulation — acts as transcription factor or cofactor
  2: kinase_phosphatase — kinase, phosphatase, or signaling transducer
  3: cell_surface — surface receptor, ion channel, or transporter
  4: metabolism — metabolic enzyme or pathway component
  5: chromatin_epigenetic — chromatin remodeling or epigenetic function
  6: protein_turnover — ubiquitin-proteasome or autophagy pathway
  7: rna_processing — RNA binding, splicing, or translation regulation

These 8 axes are scored once (target-independent) and concatenated
with the data-driven features, giving the GP a biologically informed
prior on gene function.
"""
from __future__ import annotations

import logging

import numpy as np

from maehd.agents.llm_client import OllamaClient, parse_json_from_text

logger = logging.getLogger("maehd.gene_embeddings")

FUNCTIONAL_AXES = [
    "immune_signaling",
    "transcription_regulation",
    "kinase_phosphatase",
    "cell_surface",
    "metabolism",
    "chromatin_epigenetic",
    "protein_turnover",
    "rna_processing",
]

_MULTI_AXIS_PROMPT = """\
You are a molecular biology expert. For each gene below, score it on 8 \
functional axes from 0.0 to 1.0.

AXES:
- immune_signaling: role in immune signaling (cytokines, TCR/BCR, interleukins)
- transcription_regulation: transcription factor, coactivator, or corepressor
- kinase_phosphatase: kinase, phosphatase, or direct signaling transducer
- cell_surface: surface receptor, ion channel, transporter, or adhesion molecule
- metabolism: metabolic enzyme, metabolite transporter, or metabolic regulator
- chromatin_epigenetic: chromatin remodeler, histone modifier, DNA methylation
- protein_turnover: ubiquitin ligase, proteasome, autophagy, or degradation pathway
- rna_processing: RNA binding, splicing factor, ribosome, or translation regulator

GENE LIST:
{gene_list}

For each gene, provide scores on ALL 8 axes (0.0 = not at all, 1.0 = core function).
Most genes will score high on 1-2 axes and low on others.

Respond with EXACTLY this JSON:
```json
{{
  "scores": {{
    "GENE1": [0.1, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0],
    "GENE2": [0.9, 0.0, 0.3, 0.7, 0.0, 0.0, 0.0, 0.0]
  }}
}}
```

RULES:
- Score EVERY gene — do not skip any
- Each gene gets exactly 8 scores in the order listed above
- Be calibrated: most scores should be 0.0-0.2 (irrelevant axis)
- Use your knowledge of gene function, protein domains, and pathway membership
"""


def compute_llm_gene_embeddings(
    gene_names: list[str],
    client: OllamaClient,
    batch_size: int = 80,
    verbose: bool = True,
) -> np.ndarray:
    """Score all genes on 8 functional axes using LLM.

    Parameters
    ----------
    gene_names : list of HUGO gene symbols
    client : OllamaClient instance
    batch_size : genes per LLM call (fewer than relevance-only since 8 scores per gene)

    Returns
    -------
    embeddings : (n_genes, 8) array of functional scores
    """
    n = len(gene_names)
    embeddings = np.full((n, len(FUNCTIONAL_AXES)), 0.05, dtype=np.float32)
    gene_lookup = {g.upper(): i for i, g in enumerate(gene_names)}

    n_batches = (n + batch_size - 1) // batch_size
    scored = 0
    failed = 0

    if verbose:
        print(f"  Computing LLM gene embeddings: {n} genes, {n_batches} batches, "
              f"{len(FUNCTIONAL_AXES)} axes")

    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, n)
        batch_genes = gene_names[start:end]

        prompt = _MULTI_AXIS_PROMPT.format(gene_list=", ".join(batch_genes))

        try:
            response = client.chat(
                system=(
                    "You are a molecular biology expert. Score each gene on "
                    "8 functional axes. Respond with JSON only."
                ),
                user=prompt,
            )
            result = parse_json_from_text(response)
            if result and "scores" in result:
                for gene, scores in result["scores"].items():
                    idx = gene_lookup.get(gene.upper().strip())
                    if idx is not None and isinstance(scores, list):
                        n_axes = min(len(scores), len(FUNCTIONAL_AXES))
                        for j in range(n_axes):
                            try:
                                embeddings[idx, j] = np.clip(float(scores[j]), 0.0, 1.0)
                            except (ValueError, TypeError):
                                pass
                        scored += 1
            else:
                failed += 1
        except Exception as e:
            failed += 1
            logger.warning(f"Gene embedding batch {batch_idx} failed: {e}")

        if verbose and (batch_idx + 1) % 25 == 0:
            print(f"    Batch {batch_idx+1}/{n_batches}: {scored} genes scored so far")

    if verbose:
        print(f"  Gene embeddings complete: {scored}/{n} scored, "
              f"{failed}/{n_batches} batches failed")
        for j, axis in enumerate(FUNCTIONAL_AXES):
            col = embeddings[:, j]
            print(f"    {axis}: mean={col.mean():.3f}, >0.5: {(col > 0.5).sum()}")

    return embeddings
