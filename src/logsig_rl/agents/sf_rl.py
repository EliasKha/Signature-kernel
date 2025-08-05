"""
Random‑Fourier Signature RL Agent (optimised, bar‑free)
=======================================================
API and numerical output are identical to the bar‑free reference version,
but inner loops are vectorised for noticeably faster training/evaluation.
"""
from __future__ import annotations

from typing import Sequence, Type, Mapping, Any
import collections

import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import CallbackList, BaseCallback

from .base import Agent                 # local project base‑class

###############################################################################
# Random‑Fourier utilities ####################################################
###############################################################################


def make_rff_params(d: int, D: int, sigma: float, rng: np.random.Generator):
    omega = rng.normal(scale=1.0 / sigma, size=(D, d)).astype(np.float32)
    b = rng.uniform(0.0, 2.0 * np.pi, size=D).astype(np.float32)
    return omega, b


def rff(x: np.ndarray, omega: np.ndarray, b: np.ndarray):
    return np.sqrt(2.0 / omega.shape[0], dtype=np.float32) * np.cos(x @ omega.T + b)


def rfsf(window: np.ndarray, omega: np.ndarray, b: np.ndarray, M: int):
    """
    Compute diagonal log‑signature levels using Random Fourier Features,
    fully vectorised (no Python loop over timesteps).
    """
    phi = rff(window, omega, b)           # (L, D)
    levels = [phi.sum(axis=0, dtype=np.float32)]   # level‑1

    if M >= 2:
        # level‑2 vectorised with prefix sums
        csum = np.cumsum(phi, axis=0, dtype=np.float32)
        level2 = (csum[:-1] * phi[1:]).sum(axis=0, dtype=np.float32)
        levels.append(level2)

    # higher levels (3..M) – vectorised prefix‑dot scan
    for k in range(3, M + 1):
        # running cumulants S_{k-1} same shape as phi
        cumul = np.cumsum(phi, axis=0, dtype=np.float32)
        acc = np.zeros_like(phi)
        for _ in range(k - 2):
            acc += cumul[:-1]
            cumul = np.cumsum(cumul, axis=0, dtype=np.float32)
        level_k = (acc[:-1] * phi[1:]).sum(axis=0, dtype=np.float32)
        levels.append(level_k)

    return np.concatenate(levels, axis=0).astype(np.float32)


###############################################################################
# Random‑Fourier embedding wrapper ############################################
###############################################################################


class RandomFourierEnv(gym.Wrapper):
    """Wraps env to serve whitened RFSF Gram rows instead of raw observations."""

    def __init__(
        self,
        env: gym.Env,
        *,
        prototypes: Sequence[np.ndarray],
        window_length: int,
        truncation_level: int,
        rff_width: int,
        rbf_sigma: float,
        rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__(env)

        self.T = window_length
        self.M = truncation_level
        self.D = rff_width
        self.sigma = rbf_sigma
        self.rng = np.random.default_rng() if rng is None else rng

        # Draw random Fourier frequencies once
        d = env.ret.shape[1] if hasattr(env, "ret") else env.observation_space.shape[0]
        self.omega, self.b = make_rff_params(d, self.D, self.sigma, self.rng)

        # Pre‑compute prototype embeddings
        self.proto_mat = np.stack(
            [rfsf(proto, self.omega, self.b, self.M) for proto in prototypes],
            axis=0,
        )                                                   # (m, M·D)
        K_pp = self.proto_mat @ self.proto_mat.T
        W = np.linalg.inv(np.sqrt(K_pp + 1e-6 * np.eye(K_pp.shape[0], dtype=np.float32)))
        self.W_proto = (W @ self.proto_mat).astype(np.float32)   # (m, M·D)

        # Gym spaces
        m = self.proto_mat.shape[0]
        self.observation_space = Box(-np.inf, np.inf, (m,), np.float32)
        self.action_space = env.action_space

        # History buffer
        self.history: collections.deque[np.ndarray] = collections.deque(maxlen=self.T)

    # ------------------------------------------------------------------
    def _get_raw(self, obs: np.ndarray) -> np.ndarray:
        if hasattr(self.env, "ret") and hasattr(self.env, "ptr"):
            idx = max(0, self.env.ptr - 1)
            return self.env.ret[idx]
        return obs

    def _embed(self) -> np.ndarray:
        """Compute whitened Gram row for current window in a single dot."""
        L, d = len(self.history), self.history[0].shape[0]
        if L < self.T:
            buf = np.empty((self.T, d), np.float32)
            buf[: self.T - L] = 0.0
            buf[self.T - L :] = np.array(self.history, dtype=np.float32)
        else:
            buf = np.array(self.history, dtype=np.float32)
        feat = rfsf(buf, self.omega, self.b, self.M)            # (M·D,)
        return (self.W_proto @ feat).astype(np.float32)

    # ------------------------------------------------------------------ Gym API
    def reset(self, **kwargs):  # type: ignore[override]
        obs, info = self.env.reset(**kwargs)
        self.history.clear()
        self.history.append(self._get_raw(obs).astype(np.float32))
        return self._embed(), info

    def step(self, action):  # type: ignore[override]
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.history.append(self._get_raw(obs))
        return self._embed(), reward, terminated, truncated, info


###############################################################################
# RL Agent ####################################################################
###############################################################################


class RandomFourierRLAgent(Agent):
    """Feeds Random‑Fourier Signature Features into any Stable‑Baselines3 algo."""

    name = "RFSF-RL"

    def __init__(
        self,
        prototypes: Sequence[np.ndarray],
        *,
        window_length: int = 50,
        truncation_level: int = 2,
        rff_width: int = 256,
        rbf_sigma: float = 1.0,
        algorithm_cls: Type[BaseAlgorithm],
        policy: str = "MlpPolicy",
        algo_kwargs: Mapping[str, Any] | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.prototypes = prototypes
        self.window_length = window_length
        self.truncation_level = truncation_level
        self.rff_width = rff_width
        self.rbf_sigma = rbf_sigma
        self.algorithm_cls = algorithm_cls
        self.policy = policy
        self.algo_kwargs: dict[str, Any] = dict(algo_kwargs or {})
        self.rng = np.random.default_rng() if rng is None else rng
        self.model: BaseAlgorithm | None = None

    # ------------------------------------------------------------------
    def train(
        self,
        env: gym.Env,
        *,
        total_timesteps: int = 10_000,
        callback: BaseCallback | None = None,
    ) -> None:
        wrapped = RandomFourierEnv(
            env,
            prototypes=self.prototypes,
            window_length=self.window_length,
            truncation_level=self.truncation_level,
            rff_width=self.rff_width,
            rbf_sigma=self.rbf_sigma,
            rng=self.rng,
        )
        self.model = self.algorithm_cls(
            self.policy, wrapped, verbose=0, **self.algo_kwargs
        )
        self.model.learn(total_timesteps=total_timesteps, callback=callback)

    def evaluate(self, env: gym.Env, *, render: bool = False):
        if self.model is None:
            raise RuntimeError("Call .train() before .evaluate().")
        wrapped = RandomFourierEnv(
            env,
            prototypes=self.prototypes,
            window_length=self.window_length,
            truncation_level=self.truncation_level,
            rff_width=self.rff_width,
            rbf_sigma=self.rbf_sigma,
            rng=self.rng,
        )
        obs, _ = wrapped.reset()
        cum_reward = 0.0
        rewards_path: list[float] = []
        done = False
        while not done:
            action, _ = self.model.predict(obs, deterministic=True)  # type: ignore[arg-type]
            obs, reward, terminated, truncated, _ = wrapped.step(action)
            done = bool(terminated or truncated)
            cum_reward += float(reward)
            rewards_path.append(reward) 
            if render:
                wrapped.render()
        return np.asarray(rewards_path, dtype=float)
