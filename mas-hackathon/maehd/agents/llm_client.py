"""
Lightweight LLM client for Ollama (OpenAI-compatible API).

Sends structured prompts, parses JSON decisions.
No external dependencies beyond `requests` (already in stdlib-adjacent).
"""
from __future__ import annotations

import json
import logging
import re
import time

import requests

logger = logging.getLogger("maehd.llm")


class OllamaClient:
    """Client for Ollama's OpenAI-compatible /v1/chat/completions endpoint."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434/v1",
        model: str = "llama3.1:8b",
        temperature: float = 0.3,
        max_tokens: int = 2048,
        timeout: float = 120.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    def chat(
        self,
        system: str,
        user: str,
        retries: int = 2,
    ) -> str:
        """Send a chat completion request and return the text response.

        Parameters
        ----------
        system : system prompt
        user : user message
        retries : number of retries on failure

        Returns
        -------
        text : the model's response text
        """
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        url = f"{self.base_url}/chat/completions"

        for attempt in range(retries + 1):
            try:
                resp = requests.post(
                    url, json=payload, timeout=self.timeout,
                )
                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"].get("content", "")
                return content
            except (requests.RequestException, KeyError, IndexError) as e:
                logger.warning(f"LLM call attempt {attempt+1} failed: {e}")
                if attempt < retries:
                    time.sleep(2 ** attempt)
                else:
                    raise RuntimeError(
                        f"LLM call failed after {retries+1} attempts: {e}"
                    ) from e
        return ""  # unreachable

    def is_available(self) -> bool:
        """Check if the Ollama server is reachable."""
        try:
            resp = requests.get(
                self.base_url.replace("/v1", "/api/tags"),
                timeout=5.0,
            )
            return resp.status_code == 200
        except requests.RequestException:
            return False


def parse_json_from_text(text: str) -> dict | None:
    """Extract the first JSON object from LLM output text.

    Handles:
    - ```json ... ``` blocks
    - Bare JSON objects
    - JSON with surrounding text
    """
    # Try ```json blocks first
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # Try finding a bare JSON object
    m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass

    # Try more aggressively — nested braces
    depth = 0
    start = None
    for i, c in enumerate(text):
        if c == "{":
            if depth == 0:
                start = i
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    start = None

    return None
