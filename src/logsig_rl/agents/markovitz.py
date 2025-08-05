"""
Markowitz-MPC agent
===================
Classic mean–variance objective computed from **sample averages**, no
signature coordinates involved.

Cost(a) = −mean(a) + λ · variance(a)
"""
from __future__ import annotations
from contextlib import nullcontext
from typing import Tuple
import numpy as np

from .. import config
from .base import Agent

# ────────── optional progress bar ──────────
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# ────────── optional Numba JIT  ────────────
try:
    from numba import njit, prange
except ImportError:                     # graceful fallback
    def njit(**_):
        def _decor(fn):
            return fn
        return _decor
    def prange(x):                      # type: ignore
        return range(x)

# ═════════════ hot kernel: Σr, Σr² ═════════
@njit(cache=True, fastmath=True, nogil=True, parallel=True)
def _rollout_batch(
    ret:  np.ndarray,            # (T, d)
    ptr0: int,
    seqs: np.ndarray,           # (N, H)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    N, H    = seqs.shape
    sum_r   = np.zeros(N, np.float64)
    sum2_r  = np.zeros(N, np.float64)
    n_steps = np.zeros(N, np.int64)

    for h in prange(H):
        ptr = ptr0 + h
        if ptr >= ret.shape[0]:
            break
        r = ret[ptr, seqs[:, h]]
        sum_r  += r
        sum2_r += r * r
        n_steps += 1
    return sum_r, sum2_r, n_steps


# ═════════════ the agent ═══════════════════
class MarkowitzMPCAgent(Agent):
    """Classic mean–variance MPC (sample statistics)."""

    name = "Markowitz-MPC"

    def __init__(
        self,
        horizon:      int   = config.MPC_HORIZON,
        n_candidates: int   = config.MPC_CANDIDATES,
        var_lambda:   float = 1.0,
        show_progress: bool = True,
    ) -> None:
        self.horizon       = horizon
        self.n_candidates  = n_candidates
        self.var_lambda    = var_lambda
        self.show_progress = show_progress

        # will be set in evaluate()
        self.d         = None
        self._seqs_buf = None
        self._w_buf    = None

    # --------------------------------------------------------------
    def train(self, env):
        pass                                     # planning only

    # --------------------------------------------------------------
    @staticmethod
    def _cost(sum_r, sum2_r, n_steps, lam):
        mean = sum_r / n_steps
        var  = (sum2_r / n_steps) - mean * mean
        return -mean + lam * var

    # --------------------------------------------------------------
    def evaluate(self, env):
        self.d = env.N
        rng = np.random.default_rng()

        # scratch
        self._seqs_buf = np.empty(
            (self.n_candidates, self.horizon), np.int64
        )
        self._w_buf = np.zeros(self.d, np.float64)

        cum = 0.0
        path = []

        env.reset()
        done = False

        total_steps = env.ret.shape[0]
        bar_ctx = tqdm(total=total_steps, desc="Markowitz-MPC") \
            if self.show_progress and tqdm is not None else nullcontext()

        with bar_ctx as pbar:
            while not done:
                t = env.ptr

                # sample action sequences
                self._seqs_buf[:] = rng.integers(
                    0, self.d, size=self._seqs_buf.shape, dtype=np.int64
                )

                # roll-out sample mean / var
                sum_r, sum2_r, n_steps = _rollout_batch(
                    env.ret, t - 1, self._seqs_buf
                )
                costs = self._cost(sum_r, sum2_r, n_steps, self.var_lambda)

                best_idx    = int(costs.argmin())
                best_action = int(self._seqs_buf[best_idx, 0])

                # execute first action
                self._w_buf.fill(0.0)
                self._w_buf[best_action] = 1.0
                _, r, done, *_ = env.step(self._w_buf)

                cum += r
                path.append(cum)
                if self.show_progress and tqdm is not None:
                    pbar.update(1)

        return np.asarray(path, float)
