"""Abstract environment wrapper around OpenAI Gym environments."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from acua.envs.observation import Observation


class AbstractEnv(ABC):
    """Base class that wraps an environment (gymnasium or otherwise).

    Subclasses must implement:
        - to_agent_observation: convert raw env obs to an Observation
        - python_to_env_action: convert a Python action string to an env action

    The default reset() and step() assume a standard gymnasium interface.
    Environments with non-standard interfaces (e.g. OSWorld) should override
    these methods while still implementing the abstract conversion methods.
    """

    def __init__(self, env: Any):
        self.env = env

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> Observation:
        """Reset the environment and return the initial observation."""
        obs, info = self.env.reset(seed=seed, options=options)
        return self.to_agent_observation(obs, info)

    def step(self, action: str) -> tuple[Observation, float, bool, dict]:
        """Execute an action and return (observation, reward, done, info).

        The gym's ``terminated`` and ``truncated`` flags are merged into a
        single ``done`` boolean.  The raw signals are preserved in
        ``info["terminated"]`` and ``info["truncated"]`` for callers that
        need to distinguish task completion from step-limit truncation.
        """
        env_action = self.python_to_env_action(action)
        obs, reward, terminated, truncated, info = self.env.step(env_action)
        info["terminated"] = terminated
        info["truncated"] = truncated
        done = terminated or truncated
        return self.to_agent_observation(obs, info), float(reward), done, info

    def close(self):
        """Close the underlying gym environment."""
        self.env.close()

    @abstractmethod
    def to_agent_observation(self, obs: Any, info: dict) -> Observation:
        """Convert a raw gym observation to an environment-agnostic Observation."""
        ...

    @abstractmethod
    def python_to_env_action(self, action: str) -> Any:
        """Convert a Python action string (from the agent) to a gym-compatible action."""
        ...
