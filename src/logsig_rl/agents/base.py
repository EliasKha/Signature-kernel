"Common RL agent interface so training scripts can be generic."
from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
import gymnasium as gym


class Agent(ABC):
    """Minimal interface every agent must implement."""
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def train(self, env: gym.Env): ...

    @abstractmethod
    def evaluate(self, env: gym.Env) -> np.ndarray:
        "Return a cumulative-return path."
