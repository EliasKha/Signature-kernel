from __future__ import annotations

from contextlib import nullcontext
from typing import Tuple
import numpy as np

from .. import config
from .base import Agent
from ..signature import sig_dim, sig_identity, sig, chen

try:                                    # optional progress bar
    from tqdm import tqdm
except ImportError:
    tqdm = None


# ═════════════════════ helpers ════════════════════════════════════════
def _future_signature(
    seg_sig:   np.ndarray,      # (T, sig_dim)
    start_ptr: int,
    horizon:   int,
    id_sig:    np.ndarray,      # (sig_dim,)
    d:         int,
    depth:     int,
) -> np.ndarray:
    """
    Signature of the *future* segment of fixed length ``horizon``
    starting at ``start_ptr`` (exclusive); independent of the
    candidate action, hence computed once per control step.
    """
    s   = id_sig.copy()
    ptr = start_ptr
    for _ in range(horizon):
        if ptr >= seg_sig.shape[0]:
            break
        s   = chen(s, seg_sig[ptr], d, depth)
        ptr += 1
    return s


def _sig_mean_var_cost(
    full_sig:       np.ndarray,  # (d, sig_dim) ─ each row is the same full signature
    first_actions:  np.ndarray,  # (d,) integers in [0,d)
    d:               int,
    lam:            float,      # weight on diagonal variance
    cross_lambda:   float = 0.0,# weight on off-diagonal covariance
) -> np.ndarray:
    """
    Cost for one-hot action k:
      C(k) = −S¹_k
             + lam·(2·S²_kk − (S¹_k)²)
             + cross_lambda·∑_{j≠k} S²_{kj},
    where S¹ and S² come from full_sig.
    """
    lvl1_start = 1
    lvl2_start = 1 + d

    # indices of first-actions
    k = first_actions.astype(np.int64)
    diag_offset = k * d - (k * (k - 1)) // 2

    idx_lvl1 = lvl1_start + k
    idx_lvl2 = lvl2_start + diag_offset

    # extract S¹_k and S²_kk
    S1 = full_sig[np.arange(d), idx_lvl1]   # (d,)
    S2_diag = full_sig[np.arange(d), idx_lvl2]  # (d,)

    # reconstruct full level-2 triangular block once (rows are identical)
    n2 = d * (d + 1) // 2
    block = full_sig[0, lvl2_start : lvl2_start + n2]
    Σ2 = np.zeros((d, d), dtype=full_sig.dtype)
    idx = 0
    for i in range(d):
        for j in range(i, d):
            Σ2[i, j] = Σ2[j, i] = block[idx]
            idx += 1

    # sum off-diagonals per row
    row_sum = Σ2.sum(axis=1)
    off_diag_sum = row_sum - np.diag(Σ2)  # ∑_{j≠k} S²_{kj}

    # final cost
    return (
        -S1
        + lam * (2.0 * S2_diag - S1 * S1)
        + cross_lambda * off_diag_sum
    )


# ═══════════════════ agent ════════════════════════════════════════════
class SignatureMPCAgent(Agent):
    """
    Signature MPC agent following Ohnishi et al. (2023),
    with full second-degree signature terms (diagonal + off-diagonal).

    Parameters
    ----------
    depth         : int
        Signature truncation level (≥2).
    horizon       : int
        MPC look-ahead length.
    var_lambda    : float
        Weight on diagonal variance terms.
    cross_lambda  : float
        Weight on off-diagonal covariance terms.
    show_progress : bool
        Whether to show a tqdm progress bar.
    """
    name = "Sig-MPC-sigMV-full2"

    def __init__(
        self,
        depth:         int   = config.MPC_DEPTH,
        horizon:       int   = config.MPC_HORIZON,
        var_lambda:    float = 1.0,
        cross_lambda:  float = 1.0,
        show_progress: bool  = True,
    ) -> None:
        self.depth         = depth
        self.horizon       = horizon
        self.var_lambda    = var_lambda
        self.cross_lambda  = cross_lambda
        self.show_progress = show_progress

        # filled in evaluate()
        self.d        = None
        self.sig_dim  = None
        self.id_sig   = None
        self.past_sig = None
        self.seg_sig  = None
        self._w_buf   = None

    def train(self, env):
        # planning-only agent
        pass

    def evaluate(self, env):
        # ─ initialise signatures ─
        self.d        = env.N
        self.sig_dim  = sig_dim(self.d, self.depth)
        self.id_sig   = sig_identity(self.d, self.depth)
        self.past_sig = self.id_sig.copy()

        # precompute one-step segment signatures
        self.seg_sig = np.zeros((env.ret.shape[0], self.sig_dim))
        for t in range(1, env.ret.shape[0]):
            self.seg_sig[t] = sig(env.ret[t-1:t], self.depth)

        # buffer for the single-hot portfolio
        self._w_buf = np.zeros(self.d, np.float64)

        cum_return = 0.0
        path = []

        env.reset()
        done = False
        total_steps = env.ret.shape[0]
        bar_ctx = (
            tqdm(total=total_steps, desc="Sig-MPC")
            if self.show_progress and tqdm is not None
            else nullcontext()
        )

        with bar_ctx as pbar:
            while not done:
                t = env.ptr

                # 1) update past signature
                if t >= 2:
                    self.past_sig = chen(
                        self.past_sig,
                        self.seg_sig[t-1],
                        self.d, self.depth
                    )

                # 2) compute future signature
                fut_sig = _future_signature(
                    self.seg_sig, t-1,
                    self.horizon,
                    self.id_sig,
                    self.d, self.depth
                )

                # 3) full signature over [0, t+horizon]
                full_sig_single = chen(
                    self.past_sig, fut_sig,
                    self.d, self.depth
                )

                # 4) enumerate one-hot actions
                first_actions = np.arange(self.d, dtype=np.int64)
                full_sig = np.broadcast_to(
                    full_sig_single,
                    (self.d, self.sig_dim)
                )

                # 5) score via mean-variance + cross terms
                costs = _sig_mean_var_cost(
                    full_sig,
                    first_actions,
                    self.d,
                    self.var_lambda,
                    self.cross_lambda
                )

                best = int(costs.argmin())

                # 6) execute chosen asset (all‐in one-hot)
                self._w_buf.fill(0.0)
                self._w_buf[best] = 1.0
                _, r, done, *_ = env.step(self._w_buf)

                cum_return += r
                path.append(cum_return)

                if self.show_progress and tqdm:
                    pbar.update(1)

        return np.asarray(path, float)
# 