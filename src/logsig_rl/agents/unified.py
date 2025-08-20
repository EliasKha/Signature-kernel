# src/logsig_rl/agents/unified.py
"""
Unified RL Agent (bar-free) with pluggable embeddings
=====================================================

family:
  - "rl": vanilla observations (no embedding wrapper)
  - "sk": signature-kernel embedding (requires iisignature)
  - "sf": random-fourier signature features (RFSF)

algo:
  - "sac", "ppo", "a2c", "td3", "dqn" (Stable-Baselines3)

This agent is concrete (not abstract) and implements the base Agent API:
- train(env, total_timesteps, callback)
- evaluate(env, render=False) -> np.ndarray (reward path)
- fit(...) as an alias to train

It prints minimal, helpful diagnostics and avoids tqdm bars (verbose=0).
"""
from __future__ import annotations

from typing import Mapping, Any, Sequence, Tuple, Optional, Dict, Type
import collections
import math

import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import SAC, PPO, A2C, TD3, DQN

from .base import Agent  # your project-local abstract base with abstract train/evaluate
from ..signature import sig_dim, sig  # iisignature-backed helpers


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

_SB3_ALGOS: Dict[str, Type[BaseAlgorithm]] = {
    "sac": SAC,
    "ppo": PPO,
    "a2c": A2C,
    "td3": TD3,
    "dqn": DQN,
}


def parse_agent_name(name: str) -> Tuple[str, str]:
    """
    Parse strings like 'sf_sac', 'sk_ppo', 'rl_dqn' -> (family, algo).
    Kept for backward compatibility with callers that still import this symbol.
    """
    s = name.strip().lower()
    if "_" not in s:
        raise ValueError(f"Agent name '{name}' must look like '<family>_<algo>' e.g. 'sf_sac'.")
    family, algo = s.split("_", 1)
    return family, algo


# -----------------------------------------------------------------------------
# Random-Fourier (RFSF) helpers (no external deps beyond numpy)
# -----------------------------------------------------------------------------

def _make_rff_params(d: int, D: int, sigma: float, rng: np.random.Generator):
    omega = rng.normal(loc=0.0, scale=1.0 / float(sigma), size=(D, d)).astype(np.float32)
    b = rng.uniform(0.0, 2.0 * np.pi, size=D).astype(np.float32)
    return omega, b


def _rff(x: np.ndarray, omega: np.ndarray, b: np.ndarray) -> np.ndarray:
    z = x @ omega.T + b
    # sqrt(2/D) scaling without dtype mishaps on scalars
    phi = np.cos(z).astype(np.float32)
    phi *= np.float32(np.sqrt(2.0 / omega.shape[0]))
    return phi


def _rfsf(window: np.ndarray, omega: np.ndarray, b: np.ndarray, M: int) -> np.ndarray:
    """
    Random-Fourier Signature Features for a window (T, d) using
    elementwise elementary symmetric sums over time.

    For vector features phi_t in R^D:
      level-1:  sum_t phi_t
      level-2:  sum_{t1<t2} phi_{t1} ⊙ phi_{t2}
      ...
      level-k:  sum_{t1<...<tk} ⊙_{j=1..k} phi_{tj}
    """
    phi = _rff(window, omega, b)  # (T, D)
    T, D = phi.shape

    # E[k] holds the order-k elementary symmetric sum (elementwise), k=0..M.
    # Initialize E[0] = 1 (identity for elementwise products), others 0.
    E = [np.ones(D, dtype=np.float32)] + [np.zeros(D, dtype=np.float32) for _ in range(M)]

    # One pass over time; update from high k down to 1 to avoid reuse within step
    for t in range(T):
        x = phi[t]  # (D,)
        for k in range(M, 0, -1):
            E[k] += E[k - 1] * x  # elementwise

    levels = [E[k] for k in range(1, M + 1)]  # E[1]..E[M]
    return np.concatenate(levels, axis=0).astype(np.float32)


# -----------------------------------------------------------------------------
# Observation wrappers
# -----------------------------------------------------------------------------

class _HistoryMixin:
    """Shared helpers to build windows from env.raw returns/obs."""
    T: int  # window length

    def _get_raw(self, obs: np.ndarray) -> np.ndarray:
        # Prefer env.ret[ptr-1] if your MarketEnv exposes it (faster and exact).
        if hasattr(self.env, "ret") and hasattr(self.env, "ptr"):
            idx = max(0, int(getattr(self.env, "ptr")) - 1)
            return np.asarray(self.env.ret[idx], dtype=np.float32)
        return np.asarray(obs, dtype=np.float32)

    def _window_array(self) -> np.ndarray:
        """Return an array of shape (T, d), zero-padded on the left if needed."""
        L = len(self.history)
        d = self.history[0].shape[0]
        if L < self.T:
            buf = np.zeros((self.T, d), dtype=np.float32)
            # right-align the observed history at the end of the buffer
            buf[self.T - L :] = np.asarray(self.history, dtype=np.float32)
        else:
            buf = np.asarray(self.history, dtype=np.float32)
        return buf


class SignatureKernelEnv(gym.Wrapper, _HistoryMixin):
    """
    Wrap a gym env and expose signature-kernel Gram rows against m prototypes.
    """
    def __init__(self, env: gym.Env, *, prototypes: Sequence[np.ndarray], window_length: int, truncation_level: int) -> None:
        super().__init__(env)
        self.T = int(window_length)
        self.N = int(truncation_level)

        # Dimension of raw stream
        d = env.ret.shape[1] if hasattr(env, "ret") else int(env.observation_space.shape[0])
        M = sig_dim(d, self.N)

        # Precompute prototype signatures and whitening
        proto_sigs = np.stack([sig(p, self.N) for p in prototypes], axis=0).astype(np.float32)  # (m, M)
        Kpp = proto_sigs @ proto_sigs.T
        # symmetric PD regularisation and inverse sqrt
        e, U = np.linalg.eigh(Kpp + 1e-6 * np.eye(Kpp.shape[0], dtype=np.float32))
        self.W_proto = (U @ np.diag(1.0 / np.sqrt(np.maximum(e, 1e-12))) @ U.T @ proto_sigs).astype(np.float32)  # (m, M)

        m = proto_sigs.shape[0]
        self.observation_space = Box(-np.inf, np.inf, shape=(m,), dtype=np.float32)
        self.action_space = env.action_space

        self.history: collections.deque[np.ndarray] = collections.deque(maxlen=self.T)

    def reset(self, **kwargs):  # type: ignore[override]
        obs, info = self.env.reset(**kwargs)
        self.history.clear()
        self.history.append(self._get_raw(obs))
        return self._gram_row(), info

    def step(self, action):  # type: ignore[override]
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.history.append(self._get_raw(obs))
        return self._gram_row(), reward, terminated, truncated, info

    def _gram_row(self) -> np.ndarray:
        window = self._window_array()           # (T, d)
        s = sig(window, self.N)                 # (M,)
        return (self.W_proto @ s).astype(np.float32)  # (m,)


class RandomFourierEnv(gym.Wrapper, _HistoryMixin):
    """
    Wrap a gym env and expose whitened Random-Fourier Signature Features Gram rows.
    """
    def __init__(
        self,
        env: gym.Env,
        *,
        prototypes: Sequence[np.ndarray],
        window_length: int,
        truncation_level: int,
        rff_width: int,
        rbf_sigma: float,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        super().__init__(env)
        self.T = int(window_length)
        self.M = int(truncation_level)
        self.D = int(rff_width)
        self.sigma = float(rbf_sigma)
        self.rng = np.random.default_rng() if rng is None else rng

        # Dimensionality of raw stream
        d = env.ret.shape[1] if hasattr(env, "ret") else int(env.observation_space.shape[0])
        self.omega, self.b = _make_rff_params(d, self.D, self.sigma, self.rng)

        # Precompute prototype embeddings and whitening
        proto_feats = np.stack([_rfsf(p, self.omega, self.b, self.M) for p in prototypes], axis=0).astype(np.float32)  # (m, M*D)
        Kpp = proto_feats @ proto_feats.T
        e, U = np.linalg.eigh(Kpp + 1e-6 * np.eye(Kpp.shape[0], dtype=np.float32))
        self.W_proto = (U @ np.diag(1.0 / np.sqrt(np.maximum(e, 1e-12))) @ U.T @ proto_feats).astype(np.float32)  # (m, M*D)

        m = proto_feats.shape[0]
        self.observation_space = Box(-np.inf, np.inf, shape=(m,), dtype=np.float32)
        self.action_space = env.action_space

        self.history: collections.deque[np.ndarray] = collections.deque(maxlen=self.T)

    def reset(self, **kwargs):  # type: ignore[override]
        obs, info = self.env.reset(**kwargs)
        self.history.clear()
        self.history.append(self._get_raw(obs))
        return self._embed(), info

    def step(self, action):  # type: ignore[override]
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.history.append(self._get_raw(obs))
        return self._embed(), reward, terminated, truncated, info

    def _embed(self) -> np.ndarray:
        window = self._window_array()                   # (T, d)
        feat = _rfsf(window, self.omega, self.b, self.M)   # (M*D,)
        return (self.W_proto @ feat).astype(np.float32)    # (m,)


# -----------------------------------------------------------------------------
# Unified Agent
# -----------------------------------------------------------------------------

class UnifiedAgent(Agent):
    """
    One agent that selects the embedding family ('rl' | 'sk' | 'sf')
    and plugs a Stable-Baselines3 algorithm.

    Parameters
    ----------
    family : str
        'rl' (raw), 'sk' (signature-kernel), or 'sf' (random fourier sig features).
    algo : str
        'sac', 'ppo', 'a2c', 'td3', or 'dqn'.
    lam : float
        Kept for CLI compatibility (ignored for RL runs).
    algo_kwargs : Mapping[str, Any]
        Passed to the SB3 algorithm constructor.
    # Embedding options (used for 'sk' and 'sf'):
    window_length : int
    truncation_level : int
    n_prototypes : int
    rff_width : int
    rbf_sigma : float
    rng : np.random.Generator | None
    """

    name = "Unified"

    def __init__(
        self,
        *,
        family: str,
        algo: str = "sac",
        lam: float = 0.0,  # accepted but not used (for CLI compatibility)
        algo_kwargs: Optional[Mapping[str, Any]] = None,
        # embedding options
        window_length: int = 50,
        truncation_level: int = 2,
        n_prototypes: int = 32,
        rff_width: int = 256,
        rbf_sigma: float = 1.0,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.family = family.lower().strip()
        self.algo_name = algo.lower().strip()
        self.lam = float(lam)
        self.algo_kwargs: Dict[str, Any] = dict(algo_kwargs or {})

        if self.algo_name not in _SB3_ALGOS:
            raise ValueError(f"Unknown algo '{algo}'. Choose from {list(_SB3_ALGOS)}.")
        self.algorithm_cls: Type[BaseAlgorithm] = _SB3_ALGOS[self.algo_name]

        if self.family not in {"rl", "sk", "sf"}:
            raise ValueError("family must be one of {'rl','sk','sf'}.")

        # embedding params
        self.window_length = int(window_length)
        self.truncation_level = int(truncation_level)
        self.n_prototypes = int(n_prototypes)
        self.rff_width = int(rff_width)
        self.rbf_sigma = float(rbf_sigma)
        self.rng = np.random.default_rng() if rng is None else rng

        # runtime holders
        self.model: Optional[BaseAlgorithm] = None
        self._prototypes: Optional[np.ndarray] = None  # shape (m, T, d)

        print(
            f"[UnifiedAgent] family={self.family} algo={self.algo_name} "
            f"(lam={self.lam} ignored for RL)"
        )

    # ------------------------------------------------------------------ public API
    def train(
        self,
        env: gym.Env,
        *,
        total_timesteps: int = 100,
        callback: Optional[BaseCallback] = None,
    ) -> None:
        """Train silently for *total_timesteps* interactions."""
        wrapped = self._maybe_wrap_env(env, phase="train")
        print(f"[UnifiedAgent] Starting learn: total_timesteps={total_timesteps} verbose=0")
        self.model = self.algorithm_cls(
            "MlpPolicy", wrapped, verbose=0, **self.algo_kwargs
        )
        self.model.learn(total_timesteps=total_timesteps, callback=callback)

    # alias used by your runner
    def fit(self, env: gym.Env, *, steps: int, callback: Optional[BaseCallback] = None) -> None:  # type: ignore[override]
        return self.train(env, total_timesteps=int(steps), callback=callback)

    def evaluate(self, env: gym.Env, *, render: bool = False):
        """Deterministic episode; returns the reward path as a numpy array."""
        if self.model is None:
            raise RuntimeError("Call .train() before .evaluate().")

        wrapped = self._maybe_wrap_env(env, phase="eval")
        obs, _ = wrapped.reset()
        rewards: list[float] = []
        done = False
        while not done:
            action, _ = self.model.predict(obs, deterministic=True)  # type: ignore[arg-type]
            obs, r, terminated, truncated, _ = wrapped.step(action)
            rewards.append(float(r))
            done = bool(terminated or truncated)
            if render:
                wrapped.render()
        return np.asarray(rewards, dtype=float)

    # ------------------------------------------------------------------ internals
    def _maybe_wrap_env(self, env: gym.Env, *, phase: str) -> gym.Env:
        """Build prototypes (if needed) and wrap env according to family."""
        if self.family == "rl":
            print(f"[UnifiedAgent] ({phase}) vanilla RL: no embedding wrapper")
            return env

        # ensure prototypes exist
        if self._prototypes is None:
            self._prototypes = self._make_prototypes(env)
            print(f"[UnifiedAgent] Built {self._prototypes.shape[0]} prototypes of shape "
                  f"{self._prototypes.shape[1:]} for {self.family.upper()}")

        prototypes = [p for p in self._prototypes]  # Sequence[np.ndarray]

        if self.family == "sk":
            return SignatureKernelEnv(
                env,
                prototypes=prototypes,
                window_length=self.window_length,
                truncation_level=self.truncation_level,
            )
        elif self.family == "sf":
            return RandomFourierEnv(
                env,
                prototypes=prototypes,
                window_length=self.window_length,
                truncation_level=self.truncation_level,
                rff_width=self.rff_width,
                rbf_sigma=self.rbf_sigma,
                rng=self.rng,
            )
        else:
            # Should never get here due to validation in __init__
            raise RuntimeError(f"Unsupported family '{self.family}'.")

    def _make_prototypes(self, env: gym.Env) -> np.ndarray:
        """
        Build m prototypes by sampling rolling windows from env.ret if available,
        otherwise by using zeros with tiny jitter (last resort).
        Returns array of shape (m, T, d).
        """
        m = self.n_prototypes
        T = self.window_length

        # Try to use full return matrix if provided by your MarketEnv
        if hasattr(env, "ret"):
            ret = np.asarray(env.ret, dtype=np.float32)
            if ret.ndim != 2:
                raise ValueError(f"env.ret must be (N, d); got {ret.shape}")
            N, d = ret.shape
            if N < T + 1:
                raise ValueError(f"env.ret has only {N} rows; need at least T={T}+1.")

            idxs = self.rng.integers(low=T, high=N, size=m)
            protos = np.stack([ret[i - T : i] for i in idxs], axis=0)  # (m, T, d)
            return protos.astype(np.float32)

        # Fallback: infer d from observation space and create zero prototypes
        d = int(env.observation_space.shape[0])
        protos = np.zeros((m, T, d), dtype=np.float32)
        # tiny jitter to avoid degenerate Gram (still deterministic-ish)
        protos += 1e-4 * self.rng.normal(size=protos.shape).astype(np.float32)
        print("[UnifiedAgent][WARN] env.ret not found; using zero-jitter prototypes.")
        return protos
