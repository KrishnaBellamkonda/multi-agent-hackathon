from __future__ import annotations

from abc import ABC, abstractmethod

from acua.envs.action_sets import ActionSet
from acua.envs.observation import Observation
from acua.providers.provider import AbstractProvider


class AbstractAgent(ABC):
    provider: AbstractProvider
    action_subsets: ActionSet
    goal_txt: str | None = None
    goal_img: str | None = None
    task_hint: str | None = None
    keep_last_n_obs: int | None = None

    @abstractmethod
    def get_python_action(self, obs: Observation) -> str:
        """Given an observation, return a Python action string to execute."""
        ...

    def reset(self):
        """Reset agent state for a new episode."""
        ...
