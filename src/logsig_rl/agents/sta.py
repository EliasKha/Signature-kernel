# src/logsig_rl/agents/sta.py

from __future__ import annotations
import numpy as np
from tqdm import tqdm

from .base import Agent
from .. import config
from ..signature import sig_dim, sig, chen


class SigTradingAgent(Agent):
    """
    Signature-MPC via Sig-Factor (arXiv 2308.15135), implemented with our signature helpers.

    Offline (train):
      • build past signature Z_t = sig(lead–lag(R[0:t]), depth)
      • build future signature S_fut = sig(lead–lag(R[t:t+H]), depth)
      • collect Y_mu[t]  = level-1 coords of S_fut
                Y_var[t] = 2·(level-2 diag) − (level-1)^2
      • solve ℓ^μ, ℓ^var by OLS: Z_data @ ℓ = Y.

    Online (evaluate):
      at each t:
        Z_t  = sig(lead–lag(R[0:t]), depth)
        μ     = Z_t @ ℓ^μ      (shape d)
        var   = Z_t @ ℓ^var    (shape d)
        cost  = −μ + λ·var
      pick argmin cost, allocate 100% to that asset.
    """

    name = "Sig-MPC-SigFactor"

    def __init__(
        self,
        depth:      int   = 4,#config.MPC_DEPTH,
        horizon:    int   = config.MPC_HORIZON,
        var_lambda: float = 1.0,
    ) -> None:
        self.depth      = depth
        self.horizon    = horizon
        self.var_lambda = var_lambda

        # will be set in train()
        self.d        = None       # number of assets
        self.sig_dim  = None       # length of signature vector
        self.L_mu     = None       # (sig_dim, d)
        self.L_var    = None       # (sig_dim, d)

    @staticmethod
    def _lead_lag(X: np.ndarray) -> np.ndarray:
        """Hoff lead–lag on X (T,d) → (2T−1, 2d)."""
        T, d = X.shape
        Z = np.empty((2 * T - 1, 2 * d), dtype=X.dtype)
        for i in range(T):
            Z[2*i,   :d] = X[i]
            Z[2*i,   d:] = X[i]
            if 2*i + 1 < 2*T - 1:
                Z[2*i+1, :d] = X[i]
                Z[2*i+1, d:] = X[i]
        return Z

    def train(self, env):
        R = env.ret           # shape (T, d)
        T, d = R.shape
        H    = self.horizon

        self.d = d

        # Effective path dimension after Hoff lead–lag is 2*d
        D_in = 2 * d
        self.sig_dim = sig_dim(D_in, self.depth)

        # Precompute basis signatures for one-step returns of each asset
        Z_basis = np.zeros((self.sig_dim, d), dtype=np.float64)
        for i in range(d):
            one = np.zeros((1, d), dtype=np.float64)
            one[0, i] = 1.0
            Z_basis[:, i] = sig(self._lead_lag(one), self.depth)

        t_max   = T - H
        n_samps = t_max - 1

        # Containers for past signatures and target moments
        Z_data = np.zeros((n_samps, self.sig_dim), dtype=np.float64)
        Y_mu   = np.zeros((n_samps, d),           dtype=np.float64)
        Y_var  = np.zeros((n_samps, d),           dtype=np.float64)

        for idx, t in enumerate(tqdm(range(1, t_max), desc="Fitting Sig-Factor")):
            # past signature
            Xp    = R[:t]
            Zll   = self._lead_lag(Xp)
            Zt    = sig(Zll, self.depth)
            Z_data[idx] = Zt

            # future signature
            Xf    = R[t : t + H]
            Fll   = self._lead_lag(Xf)
            S_fut = sig(Fll, self.depth)

            # extract level-1 & level-2 diagonal coords
            lvl1_start = 1
            lvl2_start = 1 + d
            k          = np.arange(d)
            diag_off   = k * d - (k * (k - 1)) // 2

            S1 = S_fut[lvl1_start + k]
            S2 = S_fut[lvl2_start + diag_off]

            Y_mu[idx]  = S1
            Y_var[idx] = 2.0 * S2 - S1 * S1

        # (Optional) build shuffle feature matrix G if you need Chen combinations
        # G = np.zeros((n_samps, d * self.sig_dim), dtype=np.float64)
        # for idx in range(n_samps):
        #     for i in range(d):
        #         G[idx, i*self.sig_dim:(i+1)*self.sig_dim] = chen(
        #             Z_data[idx], Z_basis[:, i], D_in, self.depth
        #         )

        # Solve ℓ^μ, ℓ^var via pseudoinverse
        P          = np.linalg.pinv(Z_data)       # (sig_dim, n_samps)
        self.L_mu  = P @ Y_mu                     # (sig_dim, d)
        self.L_var = P @ Y_var                    # (sig_dim, d)

    def evaluate(self, env):
        if self.L_mu is None:
            raise RuntimeError("Sig-MPC not trained")

        w_buf = np.zeros(self.d, float)
        cum_R = 0.0
        path  = []

        _, _ = env.reset()
        done = False

        pbar = tqdm(total=env.T - 1, desc="Evaluating Sig-MPC")

        while not done:
            t   = env.ptr
            Xp  = env.ret[:t]
            Zll = self._lead_lag(Xp)
            Zt  = sig(Zll, self.depth)

            mu   = Zt @ self.L_mu
            var  = Zt @ self.L_var
            cost = -mu + self.var_lambda * var

            k = int(np.argmin(cost))
            w_buf[:] = 0.0
            w_buf[k] = 1.0

            _, r, done, *_ = env.step(w_buf)
            cum_R += r
            path.append(cum_R)
            pbar.update(1)

        pbar.close()
        return np.asarray(path, float)
