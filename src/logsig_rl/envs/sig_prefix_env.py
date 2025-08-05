"""Gymnasium environment that returns *prefix* log-signatures as
observations and the portfolio return as reward.
"""
from __future__ import annotations
from typing import Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch

from ..models import LogSigLayer
from ..signature import sharpe  # may be handy for debugging
from .. import config


class SigPrefixEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, returns: np.ndarray, depth: int = config.LOGSIG_DEPTH, lam: float = 1.0):
        super().__init__()
        self.ret = returns.astype(np.float32)
        self.T, self.N = self.ret.shape
        self.ptr = 1

        self.lam = lam
        self._logsig = LogSigLayer(self.N, depth).eval()

        with torch.no_grad():
            dummy = torch.zeros(1, 1, self.N)
            self.obs_dim = self._logsig(dummy).shape[-1]

        self.observation_space = spaces.Box(-np.inf, np.inf, (self.obs_dim,), np.float32)
        self.action_space = spaces.Box(0, 1, (self.N,), np.float32)

    def reset(self, *, seed=None, options=None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.ptr = 1
        obs = self._obs()
        return obs, {}  # <-- crucial fix: return (obs, info) as per Gymnasium API

    def step(self, w: np.ndarray):
        w = np.clip(w, 0, 1)
        w /= w.sum() + 1e-12

        cov = np.cov(self.ret.T, ddof=1)
        mean_term = float(self.ret[self.ptr] @ w)
        var_term = float(w @ cov @ w)
        reward = mean_term - self.lam * var_term

        self.ptr += 1
        done = self.ptr >= self.T
        obs = np.zeros(self.obs_dim, np.float32) if done else self._obs()

        return obs, mean_term, done, False, {}

    def _obs(self) -> np.ndarray:
        with torch.no_grad():
            path = torch.from_numpy(self.ret[: self.ptr]).unsqueeze(0)
            logsig = self._logsig(path).squeeze(0).cpu().numpy().astype(np.float32)
        return logsig
