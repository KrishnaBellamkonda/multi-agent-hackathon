"""BrowserGym environment wrapper."""

from __future__ import annotations

import io
import logging
from typing import Any

import gymnasium as gym
import numpy as np
from PIL import Image

from acua.providers.util import MessageBuilder

from acua.envs.env import AbstractEnv
from acua.envs.hints import (
    BrowserInternalHints,
    ChatHints,
    ErrorHints,
    FocusedElementHints,
    TabsHints,
)
from acua.envs.observation import Observation

logger = logging.getLogger(__name__)

"""
BrowserGym OpenAI Gym environment observation dictionary format
  ┌──────────────────────────┬───────────────────┬───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │           Key            │       Type        │                                                            Description                                                            │
  ├──────────────────────────┼───────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ chat_messages            │ tuple[dict]       │ List of {"role": str, "timestamp": float, "message": str}. Includes the goal, agent messages, user messages, and infeasible       │
  │                          │                   │ reports.                                                                                                                          │
  ├──────────────────────────┼───────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ goal                     │ str               │ Deprecated. Text-only version of the goal (legacy format).                                                                        │
  ├──────────────────────────┼───────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ goal_object              │ tuple[dict]       │ Goal in OpenAI message format: [{"type": "text", "text": "..."}, {"type": "image_url", "image_url": "..."}]                       │
  ├──────────────────────────┼───────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ open_pages_urls          │ tuple[str]        │ URLs of all open tabs.                                                                                                            │
  ├──────────────────────────┼───────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ open_pages_titles        │ tuple[str]        │ Titles of all open tabs.                                                                                                          │
  ├──────────────────────────┼───────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ active_page_index        │ np.array([int])   │ Index of the active tab in the above tuples.                                                                                      │
  ├──────────────────────────┼───────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ url                      │ str               │ URL of the active page (redundant with above two).                                                                                │
  ├──────────────────────────┼───────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ screenshot               │ np.array (H, W,   │ RGB screenshot of the active page, uint8. Captured via CDP.                                                                       │
  │                          │ 3)                │                                                                                                                                   │
  ├──────────────────────────┼───────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ dom_object               │ dict              │ Full DOM snapshot via CDP's DOMSnapshot.captureSnapshot. Raw Chrome DevTools Protocol format — contains documents, strings, nodes │
  │                          │                   │  with attributes, layout rects, etc.                                                                                              │
  ├──────────────────────────┼───────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ axtree_object            │ dict              │ Merged accessibility tree across all frames. Tree of nodes with roles, names, and BIDs (browsergym element IDs).                  │
  ├──────────────────────────┼───────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ extra_element_properties │ dict[str, dict]   │ Per-BID metadata: {"bid": {"visibility": float, "bbox": [x, y, w, h], "clickable": bool, "set_of_marks": bool}}                   │
  ├──────────────────────────┼───────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ focused_element_bid      │ str               │ BID of the currently focused element.                                                                                             │
  ├──────────────────────────┼───────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ last_action              │ str               │ The raw action string from the previous step.                                                                                     │
  ├──────────────────────────┼───────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ last_action_error        │ str               │ Error message if the last action failed, empty string otherwise.                                                                  │
  ├──────────────────────────┼───────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ elapsed_time             │ np.array([float]) │ Seconds elapsed since reset().                                                                                                    │
  └──────────────────────────┴───────────────────┴───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
"""


def _screenshot_to_png_bytes(screenshot: np.ndarray) -> bytes:
    """Convert an HxWx3 uint8 numpy screenshot to PNG bytes."""
    image = Image.fromarray(screenshot)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _goal_object_to_message_builder(goal_object: tuple[dict]) -> MessageBuilder:
    """Convert a BrowserGym goal_object (OpenAI message format) to a MessageBuilder."""
    msg = MessageBuilder.user()
    for part in goal_object:
        part_type = part.get("type")
        if part_type == "text":
            msg.add_text(part["text"])
        elif part_type == "image_url":
            msg.add_image(part["image_url"])
    return msg


class BrowserGymEnv(AbstractEnv):
    """Wraps a BrowserGym gymnasium environment into the unified AbstractEnv interface."""

    def __init__(self, env: gym.Env):
        super().__init__(env)

    def to_agent_observation(self, obs: Any, info: dict) -> Observation:
        """Convert a BrowserGym observation dict to an environment-agnostic Observation."""
        screenshot = None
        if obs.get("screenshot") is not None:
            screenshot = _screenshot_to_png_bytes(obs["screenshot"])

        # Convert axtree object to text representation
        accessibility_tree = None
        if obs.get("axtree_object"):
            try:
                from browsergym.utils.obs import flatten_axtree_to_str

                accessibility_tree = flatten_axtree_to_str(
                    obs["axtree_object"],
                    extra_properties=obs.get("extra_element_properties"),
                    with_clickable=True,
                    filter_visible_only=True,
                    filter_with_bid_only=True,
                )
            except Exception:
                logger.debug(
                    "Failed to flatten axtree, using raw object", exc_info=True
                )

        # Convert DOM snapshot to text representation
        html_dom = None
        if obs.get("dom_object"):
            try:
                from browsergym.utils.obs import flatten_dom_to_str

                html_dom = flatten_dom_to_str(
                    obs["dom_object"],
                    extra_properties=obs.get("extra_element_properties"),
                    with_clickable=True,
                    filter_visible_only=True,
                    filter_with_bid_only=True,
                )
            except Exception:
                logger.debug("Failed to flatten DOM, using raw object", exc_info=True)
                html_dom = obs.get("dom_object")

        goal = None
        goal_object = obs.get("goal_object")
        if goal_object:
            goal = _goal_object_to_message_builder(goal_object)

        # ── Typed hints ──────────────────────────────────────────
        tabs = TabsHints(
            url=obs.get("url", ""),
            open_pages_urls=list(obs.get("open_pages_urls", ())),
            open_pages_titles=list(obs.get("open_pages_titles", ())),
            active_page_index=(
                int(obs["active_page_index"].item())
                if obs.get("active_page_index") is not None
                else None
            ),
        )

        error = ErrorHints(
            last_action=obs.get("last_action", ""),
            last_action_error=obs.get("last_action_error", ""),
        )

        focused = FocusedElementHints(
            focused_element_bid=obs.get("focused_element_bid") or None,
        )

        chat = ChatHints(
            chat_messages=list(obs.get("chat_messages", ())),
        )

        browser_internal = BrowserInternalHints(
            extra_element_properties=obs.get("extra_element_properties"),
            elapsed_time=(
                float(obs["elapsed_time"].item())
                if obs.get("elapsed_time") is not None
                else None
            ),
        )

        return Observation(
            screenshot=screenshot,
            html_dom=html_dom,
            accessibility_tree=accessibility_tree,
            goal=goal,
            tabs=tabs,
            error=error,
            focused_element=focused,
            chat=chat,
            browser_internal=browser_internal,
        )

    def python_to_env_action(self, action: str) -> str:
        """Pass the action string through to BrowserGym as-is.

        BrowserGym expects Python-style action strings (e.g. 'click("a51")')
        which is already the format our agents produce.
        """
        return action
