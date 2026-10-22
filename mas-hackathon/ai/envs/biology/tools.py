"""Tool functions for the biology gene-perturbation environment.

Two action vocabularies are defined here:

  BIOLOGY (orchestrator / single-agent):
    select_genes   — final gene selection submitted to the environment
    select_indices — same, by numeric index
    finish         — early-stop the episode

  BIOLOGY_NOMINATE (specialist agents):
    nominate_genes — propose genes with biology reasoning; the orchestrator
                     aggregates all nominations before calling select_genes
    finish         — signal that the specialist has no further nominations
"""

from __future__ import annotations


# ── Orchestrator / single-agent tools ────────────────────────────────


def select_genes(genes: list[str]) -> None:
    """Submit the final gene selection for this acquisition cycle.

    Called by the orchestrator (or a single BiologyAgent) once the batch
    has been decided.  Gene names must match entries in the unobserved pool
    exactly.  The environment will reveal true responses for every selected
    gene and return an updated observation.

    Args:
        genes: Exact gene names to query (e.g. ["JAK1", "STAT3"]).
    """


def select_indices(indices: list[int]) -> None:
    """Submit the final gene selection by numeric pool index.

    Alternative to select_genes when working with integer indices directly.
    Each index must refer to an unobserved gene.

    Args:
        indices: Integer gene indices from the candidate pool.
    """


def finish(summary: str) -> None:
    """End the experiment episode before the cycle limit is reached.

    Use when further cycles are unlikely to improve recall, or when the
    budget is nearly exhausted.

    Args:
        summary: Brief explanation of findings and rationale for stopping.
    """


# ── Specialist-agent tool ─────────────────────────────────────────────


def nominate_genes(genes: list[str], reasoning: str) -> None:
    """Nominate candidate genes for the next query batch.

    Called by a specialist agent (one per phenotype) to propose genes that
    are biologically relevant to *its* target.  Nominations from all
    specialists are collected by the orchestrator, which then decides the
    final batch via select_genes().

    You are NOT limited to genes listed in the pool sample — you may
    nominate any gene from your biology knowledge.  The orchestrator will
    validate availability before the final selection.

    Args:
        genes: Gene names you recommend querying (HUGO symbols preferred,
               e.g. ["JAK1", "STAT3", "TYK2"]).
        reasoning: One-paragraph biology rationale explaining why these
                   specific genes are most likely to be hits for your target.
    """
