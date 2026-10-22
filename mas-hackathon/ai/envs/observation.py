"""Environment-agnostic observation representation."""

from __future__ import annotations

import base64
from dataclasses import dataclass

from acua.envs.hints import (
    BrowserInternalHints,
    CandidateSummary,
    ChatHints,
    ErrorHints,
    ExperimentHints,
    FocusedElementHints,
    LastBatchHints,
    OSWorldTaskHints,
    ScreenshotSOMHints,
    SpecialistViewHints,
    TabsHints,
    TargetMetricsHints,
)
from acua.providers.util import MessageBuilder


@dataclass
class Observation:
    """Environment-agnostic observation that can represent desktop, browser, or terminal state.

    All fields are optional — each environment populates only what it provides.

    Typed hints (``tabs``, ``error``, ``focused_element``, etc.) replace
    the old ``hints: dict`` with statically-typed dataclasses grouped by
    semantic concern.  Each environment fills in the hint groups it supports.
    """

    # Visual observation (PNG screenshot as raw bytes)
    screenshot: bytes | None = None

    # Structured DOM representation (full or partial HTML)
    html_dom: str | None = None

    # Accessibility tree (text representation of UI elements)
    accessibility_tree: str | None = None

    # The goal in a universal MessageBuilder format w.r.t. the provider
    goal: MessageBuilder | None = None

    # ── Typed hint groups (all optional) ───────────────────────
    tabs: TabsHints | None = None
    error: ErrorHints | None = None
    focused_element: FocusedElementHints | None = None
    screenshot_som: ScreenshotSOMHints | None = None
    chat: ChatHints | None = None
    browser_internal: BrowserInternalHints | None = None
    osworld_task: OSWorldTaskHints | None = None

    # Current step and max steps: (current_step, max_steps)
    step_info: tuple[int, int] | None = None

    # ── Biology / gene-perturbation experiment fields (all optional) ───

    # Shared view: cycle/budget state, per-target recall summary, GP-ranked pool.
    # Used by single BiologyAgent and by the orchestrator's moderation step.
    experiment: ExperimentHints | None = None
    target_metrics: TargetMetricsHints | None = None
    candidates: list[CandidateSummary] | None = None
    last_batch: LastBatchHints | None = None

    # Per-target specialist views: one entry per target, each showing only
    # that target's hit history and a pool sample.  Populated by BiologyEnv
    # when use_gp=False (multi-agent mode); None in single-agent GP mode.
    specialist_views: list[SpecialistViewHints] | None = None

    # ── Convenience properties ─────────────────────────────────

    @property
    def has_screenshot(self) -> bool:
        return self.screenshot is not None

    @property
    def has_html_dom(self) -> bool:
        return self.html_dom is not None

    @property
    def has_accessibility_tree(self) -> bool:
        return self.accessibility_tree is not None

    def screenshot_base64(self) -> str | None:
        """Return the screenshot as a base64-encoded string, or None."""
        if self.screenshot is None:
            return None
        return base64.b64encode(self.screenshot).decode()  # decode bytes -> str
