"""Runtime monkey-patches for the vendored ``webarena`` evaluation harness.

WebArena hardcodes ``gpt-4-1106-preview`` as the LLM judge for fuzzy answer
matching.  The function is imported by name into several sibling modules, so
overriding a single binding is not enough — every import site must be patched.

Typical use::

    from acua.envs.browsergym.patches import (
        make_provider_fuzzy_match,
        patch_webarena_llm_fuzzy_match,
    )
    from acua.providers.openai_provider import OpenAIProvider

    judge = OpenAIProvider(model_name="gpt-4o", temperature=0)
    patch_webarena_llm_fuzzy_match(make_provider_fuzzy_match(judge))
"""

from __future__ import annotations

import logging
from typing import Callable

from acua.providers.provider import AbstractProvider
from acua.providers.util import APIPayload, MessageBuilder

logger = logging.getLogger(__name__)

# (pred, reference, question) -> 0.0 (incorrect / partially) or 1.0 (correct)
FuzzyMatchFn = Callable[[str, str, str], float]

_PATCHED = False


# Env vars that browsergym bridges from WA_* to unprefixed names inside
# WebArenaInstance.__init__ (see browsergym/webarena/instance.py).  The
# vendored webarena package reads the unprefixed names at import time, so we
# must do the same bridging before importing webarena.evaluation_harness.
_WA_ENV_VARS = (
    "SHOPPING",
    "SHOPPING_ADMIN",
    "REDDIT",
    "GITLAB",
    "WIKIPEDIA",
    "MAP",
    "HOMEPAGE",
)


def _bridge_webarena_env_vars() -> None:
    """Copy WA_* env vars into unprefixed names that webarena reads at import."""
    import os

    missing = []
    for key in _WA_ENV_VARS:
        src = f"WA_{key}"
        if src in os.environ:
            os.environ[key] = os.environ[src]
        elif key not in os.environ:
            missing.append(src)
    if missing:
        raise RuntimeError(
            "Cannot patch webarena judge — missing env vars: "
            + ", ".join(missing)
            + ". Set these (or their unprefixed equivalents) before running."
        )


def _import_webarena_eval_modules():
    """Bridge WA_* env vars and import the webarena evaluation modules."""
    # webarena.browser_env.env_config does module-level asserts on unprefixed
    # env vars (REDDIT, SHOPPING, ...).  Browsergym bridges WA_* → unprefixed
    # lazily inside WebArenaInstance.__init__, which only runs at env.reset().
    # Since we import webarena eagerly here, we must do the bridging too.
    _bridge_webarena_env_vars()
    try:
        from webarena.evaluation_harness import evaluators, helper_functions
        import webarena.evaluation_harness as eh_pkg
    except ImportError as exc:
        raise ImportError(
            "Cannot patch webarena judges — is browsergym-webarena installed?"
        ) from exc
    return helper_functions, evaluators, eh_pkg


def _patch_everywhere(attr_name: str, new_fn) -> None:
    """Reassign ``attr_name`` on every webarena module that rebinds it."""
    helper_functions, evaluators, eh_pkg = _import_webarena_eval_modules()
    setattr(helper_functions, attr_name, new_fn)
    setattr(evaluators, attr_name, new_fn)
    if hasattr(eh_pkg, attr_name):
        setattr(eh_pkg, attr_name, new_fn)


def patch_webarena_llm_fuzzy_match(new_fn: FuzzyMatchFn) -> None:
    """Replace webarena's ``llm_fuzzy_match`` across every import site.

    Safe to call repeatedly — reassigning module attributes is idempotent.
    """
    global _PATCHED
    _patch_everywhere("llm_fuzzy_match", new_fn)
    _PATCHED = True
    logger.warning(
        "[judge] Patched webarena.llm_fuzzy_match → %s",
        getattr(new_fn, "__qualname__", repr(new_fn)),
    )


def patch_webarena_llm_ua_match(new_fn: FuzzyMatchFn) -> None:
    """Replace webarena's ``llm_ua_match`` (unachievable-task judge)."""
    _patch_everywhere("llm_ua_match", new_fn)
    logger.warning(
        "[judge] Patched webarena.llm_ua_match → %s",
        getattr(new_fn, "__qualname__", repr(new_fn)),
    )


def patch_webarena_judges(provider: AbstractProvider) -> None:
    """One-shot: replace both LLM judges (fuzzy match + ua match) with *provider*."""
    patch_webarena_llm_fuzzy_match(make_provider_fuzzy_match(provider))
    patch_webarena_llm_ua_match(make_provider_ua_match(provider))


def make_provider_fuzzy_match(provider: AbstractProvider) -> FuzzyMatchFn:
    """Build a ``llm_fuzzy_match`` replacement that routes through *provider*."""

    def llm_fuzzy_match(pred: str, reference: str, question: str) -> float:
        user_text = (
            "Help a teacher to grade the answer of a student given a question. "
            "Keep in mind that the student may use different phrasing or wording "
            "to answer the question. The goal is to evaluate whether the answer "
            "is semantically equivalent to the reference answer.\n"
            f"question: {question}\n"
            f"reference answer: {reference}\n"
            "all the string 'N/A' that you see is a special sequence that means 'not achievable'\n"
            f"student answer: {pred}\n"
            "Conclude the judgement by correct/incorrect/partially correct."
        )
        messages = [
            MessageBuilder.system().add_text("You are a helpful assistant"),
            MessageBuilder.user().add_text(user_text),
        ]
        out = provider(APIPayload(messages=messages))

        # Providers populate different fields depending on how they parse output.
        raw = (out.think or "") + " " + (out.action or "")
        response = raw.lower()

        logger.warning(
            "[judge] pred=%r  reference=%r  verdict=%s",
            pred[:120],
            reference[:120],
            response.strip()[:200] or "(empty)",
        )

        if "partially correct" in response or "incorrect" in response:
            return 0.0
        if "correct" in response:
            return 1.0
        # Fall back to 0.0 on unparseable judge output (original asserted — we
        # prefer a conservative score over crashing the evaluation).
        logger.warning(
            "[judge] Unparseable verdict — defaulting to 0.0: %r", response[:200]
        )
        return 0.0

    return llm_fuzzy_match


def make_provider_ua_match(provider: AbstractProvider) -> FuzzyMatchFn:
    """Build a ``llm_ua_match`` replacement (for 'unachievable' reference answers)."""

    def llm_ua_match(pred: str, reference: str, question: str) -> float:
        user_text = (
            f"task: {question}\n"
            f"actual unachievable reason: {reference}\n"
            f"reported unachievable reason: {pred}\n"
            "The task described above is inherently unachievable due to the reason "
            "specified under 'actual unachievable reason'. An individual previously "
            "attempted this task and was unable to complete it. They provided a "
            "reason for their failure, which is listed under 'reported unachievable "
            "reason'. Your role is to review both the actual and reported reasons. "
            "Determine if the reported reason aligns with the actual reason, even if "
            "implicitly. If the stated reason is in line with the actual reason, "
            "respond with 'same'. Otherwise, respond with 'different'."
        )
        messages = [
            MessageBuilder.system().add_text("You are a helpful assistant"),
            MessageBuilder.user().add_text(user_text),
        ]
        out = provider(APIPayload(messages=messages))
        response = ((out.think or "") + " " + (out.action or "")).lower()

        logger.warning(
            "[judge/ua] pred=%r  reference=%r  verdict=%s",
            pred[:120],
            reference[:120],
            response.strip()[:200] or "(empty)",
        )

        if "different" in response:
            return 0.0
        if "same" in response:
            return 1.0
        logger.warning(
            "[judge/ua] Unparseable verdict — defaulting to 0.0: %r", response[:200]
        )
        return 0.0

    return llm_ua_match
