# src/logsig_rl/agents/base.py
"""
Common Agent base class (SB3-friendly, no progress bars).

All concrete agents should:
- set `name` (str)
- set `self.model` (stable_baselines3 BaseAlgorithm) in `train()`
- implement `train()` and `evaluate()`.

This keeps a stable API across GenericRLAgent, SignatureKernelRLAgent,
RandomFourierRLAgent, and UnifiedAgent.
"""
from __future__ import annotations

import abc
import random
from typing import Optional

import numpy as np
import gymnasium as gym

try:
    # Type only; SB3 is required by concrete subclasses at runtime.
    from stable_baselines3.common.base_class import BaseAlgorithm
    from stable_baselines3.common.callbacks import BaseCallback
except Exception:  # pragma: no cover
    BaseAlgorithm = object  # type: ignore
    BaseCallback = object  # type: ignore


class Agent(abc.ABC):
    """Abstract base class for RL agents used in this project."""

    #: Human-readable name used in logs/metadata
    name: str = "BaseAgent"

    def __init__(self) -> None:
        # Subclasses set this after constructing the SB3 model in `train()`.
        self.model: Optional[BaseAlgorithm] = None

    # --------------------------------------------------------------------- API
    @abc.abstractmethod
    def train(
        self,
        env: gym.Env,
        *,
        total_timesteps: int = 100,
        callback: Optional[BaseCallback] = None,
    ) -> None:
        """
        Fit the agent on `env` for `total_timesteps`. Must set `self.model`.
        Implementations should create the SB3 model with `verbose=0`.
        """
        raise NotImplementedError

    def fit(self, env, steps: int, callback=None) -> None:
        self.train(env, total_timesteps=steps, callback=callback)

    @abc.abstractmethod
    def evaluate(self, env: gym.Env, *, render: bool = False) -> np.ndarray:
        """
        Run a deterministic rollout and return the path of rewards as a
        1D NumPy array. Subclasses may wrap `env` (e.g., with feature
        embeddings) before the rollout.
        """
        raise NotImplementedError

    # --------------------------------------------------------------- utilities
    def act(self, obs: np.ndarray, *, deterministic: bool = True):
        """One-step policy action using the trained model."""
        if self.model is None:
            raise RuntimeError("You must call .train() before calling .act().")
        action, _ = self.model.predict(obs, deterministic=deterministic)  # type: ignore[attr-defined]
        return action

    def ensure_trained(self) -> None:
        """Raise a clear error if the agent hasn't been trained yet."""
        if self.model is None:
            raise RuntimeError("Model is not trained. Call .train() first.")

    # Optional helpers for users/tests ----------------------------------------
    def save_model(self, path: str) -> None:
        """Save underlying SB3 model to `path`."""
        self.ensure_trained()
        assert self.model is not None
        self.model.save(path)  # type: ignore[attr-defined]

    def load_model_into(self, path: str) -> None:
        """
        Load weights into an already-constructed model (advanced use).
        Subclasses must have already created `self.model` with same arch.
        """
        if self.model is None:
            raise RuntimeError(
                "load_model_into() requires an existing model. "
                "Construct it (e.g. by calling train() with 0 steps) first."
            )
        assert hasattr(self.model, "load")
        # SB3 models typically expose a classmethod .load; we can emulate by
        # using the underlying policy parameters loader if available.
        loaded = self.model.__class__.load(path)  # type: ignore[attr-defined]
        # Transfer parameters
        self.model.policy.load_state_dict(loaded.policy.state_dict())  # type: ignore[attr-defined]

    # Reproducibility ----------------------------------------------------------
    @staticmethod
    def seed_everything(seed: int) -> None:
        """Best-effort seeding of Python, NumPy, and (optionally) PyTorch."""
        random.seed(seed)
        np.random.seed(seed)
        try:  # optional torch seeding
            import torch

            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except Exception:
            pass  # torch not installed or no CUDA

    # Niceties -----------------------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover
        return f"{self.__class__.__name__}(name={self.name!r}, trained={self.model is not None})"
    
    


__all__ = ["Agent"]
