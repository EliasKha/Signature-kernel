"""
Signature‑Kernel RL Agent (bar‑free)
===================================
Embeds environments with a signature‑kernel representation of recent
raw‑return paths and trains any Stable‑Baselines3 algorithm—now with
*no* tqdm progress bars.
"""
from __future__ import annotations

from typing import Sequence, Type, Mapping, Any
import collections

import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback

from .base import Agent
from ..signature import sig_dim, sig

###############################################################################
# Signature‑kernel embedding wrapper ##########################################
###############################################################################


class SignatureKernelEnv(gym.Wrapper):
    """Wrap any Gymnasium env with a signature‑kernel observation."""

    def __init__(
        self,
        env: gym.Env,
        prototypes: Sequence[np.ndarray],
        window_length: int,
        truncation_level: int,
    ) -> None:
        super().__init__(env)

        self.T = window_length
        self.N = truncation_level

        d = env.ret.shape[1] if hasattr(env, "ret") else env.observation_space.shape[0]
        self.M = sig_dim(d, self.N)  # signature dimensionality

        # Pre‑compute prototype signatures and whitening matrix
        proto_sigs = np.stack([sig(p, self.N) for p in prototypes], axis=0)  # (m, M)
        K_pp = proto_sigs @ proto_sigs.T
        e, U = np.linalg.eigh(K_pp + 1e-6 * np.eye(K_pp.shape[0]))
        self.W = U @ np.diag(1.0 / np.sqrt(e)) @ U.T           # K^{-1/2}
        self.W_proto = self.W @ proto_sigs                     # (m, M)

        # Spaces
        m = proto_sigs.shape[0]
        self.observation_space = Box(-np.inf, np.inf, (m,), np.float32)
        self.action_space = env.action_space

        # History buffer
        self.history: collections.deque[np.ndarray] = collections.deque(maxlen=self.T)

    # ------------------------------------------------------------------ helpers
    def _get_raw(self, obs: np.ndarray) -> np.ndarray:
        if hasattr(self.env, "ret") and hasattr(self.env, "ptr"):
            idx = max(0, self.env.ptr - 1)
            return self.env.ret[idx]
        return obs

    def _gram_row(self) -> np.ndarray:
        arr = np.stack(self.history, axis=0)
        if arr.shape[0] < self.T:
            pad = np.zeros((self.T - arr.shape[0], arr.shape[1]), dtype=arr.dtype)
            arr = np.vstack([pad, arr])
        sig_new = sig(arr, self.N)                              # (M,)
        return (self.W_proto @ sig_new).astype(np.float32)      # (m,)

    # ------------------------------------------------------------------ Gym API
    def reset(self, **kwargs):  # type: ignore[override]
        obs, info = self.env.reset(**kwargs)
        self.history.clear()
        self.history.append(self._get_raw(obs).astype(float))
        return self._gram_row(), info

    def step(self, action):  # type: ignore[override]
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.history.append(self._get_raw(obs).astype(float))
        return self._gram_row(), reward, terminated, truncated, info


###############################################################################
# Signature‑kernel RL agent ###################################################
###############################################################################


class SignatureKernelRLAgent(Agent):
    """Signature‑kernel RL agent usable with *any* Stable‑Baselines3 algorithm."""

    name = "SigKernel‑RL"

    def __init__(
        self,
        prototypes: Sequence[np.ndarray],
        window_length: int = 50,
        truncation_level: int = 3,
        *,
        algorithm_cls: Type[BaseAlgorithm],
        policy: str = "MlpPolicy",
        algo_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        self.prototypes = prototypes
        self.window_length = window_length
        self.truncation_level = truncation_level
        self.algorithm_cls = algorithm_cls
        self.policy = policy
        self.algo_kwargs: dict[str, Any] = dict(algo_kwargs or {})
        self.model: BaseAlgorithm | None = None

    # ------------------------------------------------------------------ methods
    def train(
        self,
        env: gym.Env,
        *,
        total_timesteps: int = 10_000,
        callback: BaseCallback | None = None,
    ) -> None:
        """Train silently for ``total_timesteps`` interactions."""
        wrapped = SignatureKernelEnv(
            env,
            self.prototypes,
            self.window_length,
            self.truncation_level,
        )
        self.model = self.algorithm_cls(
            self.policy,
            wrapped,
            verbose=0,
            **self.algo_kwargs,
        )
        self.model.learn(total_timesteps=total_timesteps, callback=callback)

    def evaluate(self, env: gym.Env, *, render: bool = False):
        """Deterministic evaluation; returns cumulative‑reward path."""
        if self.model is None:
            raise RuntimeError("Call .train() before .evaluate().")

        wrapped = SignatureKernelEnv(
            env,
            self.prototypes,
            self.window_length,
            self.truncation_level,
        )
        obs, _ = wrapped.reset()
        cum_reward = 0.0
        rewards: list[float] = []
        done = False

        while not done:
            action, _ = self.model.predict(obs, deterministic=True)  # type: ignore[arg-type]
            obs, reward, terminated, truncated, _ = wrapped.step(action)
            done = bool(terminated or truncated)
            cum_reward += float(reward)
            rewards.append(reward)
            if render:
                wrapped.render()

        return np.asarray(rewards, dtype=float)
