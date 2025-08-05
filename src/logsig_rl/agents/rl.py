"""
Generic RL Agent (bar‑free, algorithm‑agnostic)
==============================================
A thin wrapper that plugs any Stable‑Baselines3 algorithm into an application‑
level “Agent” interface – now without tqdm progress bars.
"""
from __future__ import annotations

from typing import Mapping, Any, Type

import gymnasium as gym
import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3 import SAC  # default algorithm if none provided
from stable_baselines3.common.callbacks import BaseCallback

from .base import Agent  # project‑local base class

###############################################################################
# Generic RL Agent ############################################################
###############################################################################


class GenericRLAgent(Agent):
    """Algorithm‑agnostic reinforcement‑learning agent (bar‑free)."""

    name = "Generic‑RL"

    def __init__(
        self,
        *,
        algorithm_cls: Type[BaseAlgorithm] = SAC,
        policy: str = "MlpPolicy",
        algo_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        self.algorithm_cls = algorithm_cls
        self.policy = policy
        self.algo_kwargs: dict[str, Any] = dict(algo_kwargs or {})
        self.model: BaseAlgorithm | None = None

    # ------------------------------------------------------------------
    # Training & evaluation
    # ------------------------------------------------------------------
    def train(  # noqa: D401
        self,
        env: gym.Env,
        *,
        total_timesteps: int = 10_000,
        callback: BaseCallback | None = None,
    ) -> None:
        """Train silently for *total_timesteps* interactions."""
        self.model = self.algorithm_cls(
            self.policy,
            env,
            verbose=0,
            **self.algo_kwargs,
        )
        self.model.learn(total_timesteps=total_timesteps, callback=callback)

    def evaluate(self, env: gym.Env, *, render: bool = False):
        """Run one deterministic episode; return cumulative‑reward path."""
        if self.model is None:
            raise RuntimeError("You must call .train() before .evaluate().")

        obs, _ = env.reset()
        cum_reward = 0.0
        rewards_path: list[float] = []
        done = False

        while not done:
            action, _ = self.model.predict(obs, deterministic=True)  # type: ignore[arg-type]
            obs, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
            cum_reward += float(reward)
            rewards_path.append(reward)
            if render:
                env.render()

        return np.asarray(rewards_path, dtype=float)
