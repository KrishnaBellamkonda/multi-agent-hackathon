"""Lightweight HTML-tag parsing utilities for LLM output.

Ported from agentlab.llm.llm_utils to avoid the external dependency.
"""

from __future__ import annotations

import re


class ParseError(Exception):
    pass


def extract_html_tags(
    text: str, keys: tuple[str, ...] | list[str]
) -> dict[str, list[str]]:
    """Extract content within HTML tags for a list of keys.

    Args:
        text: The input string containing the HTML tags.
        keys: The HTML tags to extract the content from.

    Returns:
        A dictionary mapping each key to a list of matches in *text*.
    """
    content_dict: dict[str, list[str]] = {}
    for key in keys:
        pattern = f"<{key}>(.*?)</{key}>"
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            content_dict[key] = [match.strip() for match in matches]
    return content_dict


def parse_html_tags(
    text: str,
    keys: tuple[str, ...] = (),
    optional_keys: tuple[str, ...] = (),
    merge_multiple: bool = False,
) -> tuple[dict[str, str], bool, str]:
    """Extract one match per key and validate that all required keys are present.

    Returns:
        (content_dict, valid, retry_message)
    """
    all_keys = tuple(keys) + tuple(optional_keys)
    content_dict = extract_html_tags(text, all_keys)
    retry_messages: list[str] = []

    for key in all_keys:
        if key not in content_dict:
            if key not in optional_keys:
                retry_messages.append(f"Missing the key <{key}> in the answer.")
        else:
            val = content_dict[key]
            content_dict[key] = val[0]
            if len(val) > 1:
                if not merge_multiple:
                    retry_messages.append(
                        f"Found multiple instances of the key {key}. You should have only one of them."
                    )
                else:
                    content_dict[key] = "\n".join(val)

    valid = len(retry_messages) == 0
    retry_message = "\n".join(retry_messages)
    return content_dict, valid, retry_message


def parse_html_tags_raise(
    text: str,
    keys: tuple[str, ...] = (),
    optional_keys: tuple[str, ...] = (),
    merge_multiple: bool = False,
) -> dict[str, str]:
    """Like :func:`parse_html_tags` but raises :class:`ParseError` on failure."""
    content_dict, valid, retry_message = parse_html_tags(
        text, keys, optional_keys, merge_multiple=merge_multiple
    )
    if not valid:
        raise ParseError(retry_message)
    return content_dict


def extract_code_blocks(text: str) -> list[tuple[str, str]]:
    """Extract fenced code blocks from *text*.

    Returns a list of ``(language, code)`` tuples.
    """
    pattern = re.compile(r"```(\w*\n)?(.*?)```", re.DOTALL)
    matches = pattern.findall(text)
    return [(match[0].strip(), match[1].strip()) for match in matches]
