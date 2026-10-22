"""Typed hint dataclasses for environment observations.

Each dataclass groups hints by semantic concern.  Environments populate
the relevant hints, and prompt elements / logging consume them via typed
attribute access instead of stringly-typed ``dict.get()`` calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ── Browser environment hints ─────────────────────────────────────


@dataclass
class TabsHints:
    """Browser tab / page state.

    Consumed by: ``Tabs`` prompt element, experiment loop logging.
    """

    url: str = ""
    open_pages_urls: list[str] = field(default_factory=list)
    open_pages_titles: list[str] = field(default_factory=list)
    active_page_index: int | None = None


@dataclass
class ErrorHints:
    """Last-action error state.

    Consumed by: ``Error`` prompt element, experiment loop logging.
    """

    last_action: str = ""
    last_action_error: str = ""


@dataclass
class FocusedElementHints:
    """Currently focused element.

    Consumed by: ``FocusedElement`` prompt element, experiment loop logging.
    """

    focused_element_bid: str | None = None


@dataclass
class ScreenshotSOMHints:
    """Set-of-Marks annotated screenshot.

    Consumed by: ``ObservationPrompt.to_user_message``.
    """

    screenshot_som: bytes | None = None


@dataclass
class ChatHints:
    """Chat message history from BrowserGym.

    Consumed by: experiment loop logging / termination diagnostics.
    """

    chat_messages: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class BrowserInternalHints:
    """Internal BrowserGym data stored for preprocessing but not
    directly consumed by prompt elements.
    """

    extra_element_properties: dict | None = None
    elapsed_time: float | None = None


# ── OSWorld environment hints ─────────────────────────────────────


@dataclass
class OSWorldTaskHints:
    """OSWorld task metadata."""

    domain: str = ""
    task_id: str = ""
    fail: bool = False
    done: bool = False


# ── Biology / gene-perturbation experiment hints ──────────────────────


@dataclass
class ExperimentHints:
    """High-level experiment progress state.

    Consumed by: ExperimentObservationPrompt, OrchestratorDiscussionPrompt.
    """

    cycle: int = 0
    n_cycles: int = 0
    n_observed: int = 0
    n_total: int = 0
    budget_remaining: int = 0
    batch_size: int = 0
    last_hit_rate: float = 0.0
    # Full list of already-observed gene names — needed by the orchestrator
    # to validate specialist nominations before submitting select_genes().
    observed_gene_names: list[str] = field(default_factory=list)


@dataclass
class TargetMetricsHints:
    """Per-target hit-discovery metrics.

    All array fields use plain Python lists so this module stays
    free of numpy dependencies.

    Consumed by: ExperimentObservationPrompt.
    """

    target_names: list[str] = field(default_factory=list)
    per_target_recall: list[float] = field(default_factory=list)   # (K,)
    per_target_hits: list[int] = field(default_factory=list)        # (K,)
    total_hits: int = 0
    thresholds: list[float] = field(default_factory=list)           # (K,)
    mean_gp_sigma: float = 0.0
    correlation_matrix: list[list[float]] | None = None             # (K, K)


@dataclass
class CandidateSummary:
    """Summary statistics for a single candidate gene.

    Consumed by: ExperimentObservationPrompt (candidate pool table).
    """

    gene_name: str = ""
    gene_index: int = 0
    max_poh: float = 0.0        # max P(hit) across all targets
    mean_sigma: float = 0.0     # mean GP posterior std across targets
    per_target_poh: list[float] = field(default_factory=list)  # (K,)


@dataclass
class LastBatchHints:
    """Outcomes of the most recently executed query batch.

    Consumed by: ExperimentObservationPrompt.
    """

    selected_genes: list[str] = field(default_factory=list)
    selected_indices: list[int] = field(default_factory=list)
    was_hit_any: bool = False
    per_target_hits: list[int] = field(default_factory=list)  # (K,) counts


@dataclass
class SpecialistViewHints:
    """Single-target view built for one specialist agent.

    Each specialist agent receives an Observation containing exactly one
    SpecialistViewHints (for its phenotype).  It has no visibility into
    other targets' outcomes, mirroring a realistic experimental setting
    where different assay teams operate independently.

    Consumed by: SpecialistObservationPrompt.
    """

    # Identity of this target
    target_name: str = ""       # screen identifier, e.g. "schmidt_2021_ifng"
    phenotype: str = ""         # human-readable label, e.g. "IFNγ production"
    cell_type: str = ""         # biological context

    # Cumulative progress for this target
    recall: float = 0.0         # fraction of true hits found so far
    n_hits_found: int = 0       # absolute count of confirmed hits
    n_total_hits: int = 0       # total hits defined by threshold (ground truth)

    # Gene-level history for this target
    confirmed_hit_genes: list[str] = field(default_factory=list)
    # Hits and misses from the MOST RECENT batch only (for recency reasoning)
    last_batch_hits: list[str] = field(default_factory=list)
    last_batch_misses: list[str] = field(default_factory=list)

    # A random sample of the unobserved gene pool (shown so the specialist
    # can reference real gene names, though it may nominate any gene it knows)
    pool_sample: list[str] = field(default_factory=list)
    n_pool_total: int = 0       # total unobserved genes (context for scale)
