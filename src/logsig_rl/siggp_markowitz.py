"""
siggp_markowitz.py  — robust GP-Markowitz *and* classical Markowitz benchmark.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import cholesky, cho_solve
from tqdm import tqdm

from .signature import sig
from .envs import SigPrefixEnv
from . import config

# ───────────────────────── helper utils ──────────────────────────
def _f64(x): return np.asarray(x, dtype=np.float64)

def _clean(a: NDArray) -> NDArray:
    a = _f64(a)
    if not np.isfinite(a).all():
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    return a

def K_sig(x: NDArray, y: NDArray, N: int) -> float:
    return float(_clean(sig(x, N)) @ _clean(sig(y, N)))

# ───────────────────── Gaussian-process layer ─────────────────────
def _Gram(paths, N): return np.array([[K_sig(p, q, N) for q in paths] for p in paths])
def _k_vec(paths, x, N): return np.array([K_sig(p, x, N) for p in paths])[:, None]

@dataclass
class SigGPRegressor:
    order: int
    sigma2: float = 1.0
    tau2:   float = 1e-4
    X_: list[NDArray] | None = None
    Y_: NDArray | None = None
    L_: NDArray | None = None

    def fit(self, paths, targets):
        self.X_, self.Y_ = list(paths), _clean(targets)
        K = self.sigma2 * _Gram(self.X_, self.order)
        np.fill_diagonal(K, K.diagonal() + self.tau2)
        self.L_ = cholesky(K, lower=True); return self

    def moments(self, x):
        kx = self.sigma2 * _k_vec(self.X_, x, self.order)
        mu = (kx.T @ cho_solve((self.L_, True), self.Y_)).ravel()
        v  = np.linalg.solve(self.L_, kx)
        var = max(self.sigma2 * K_sig(x, x, self.order) - float(v.T @ v) + self.tau2, 1e-8)
        return _clean(mu), var * np.eye(self.Y_.shape[1])

# ───────────────────── Markowitz optimiser ───────────────────────
def _markowitz(mu, Sigma, lam):
    invS, ones = np.linalg.inv(Sigma), np.ones_like(mu)
    A, B = invS @ ones, invS @ mu
    alpha = float(ones @ A)
    return A / alpha + (1/lam) * (np.eye(len(mu)) - np.outer(A, ones)/alpha) @ B

# ─────────────────────────── GP agent ────────────────────────────
class SignatureKernelMarkowitzAgent:
    def __init__(self, train_ret, *, window=None, sig_order=config.LOGSIG_DEPTH,
                 sigma2=1.0, tau2=1e-4, lam=3.0):
        self._lam, self._win = lam, window or config.SIG_WINDOW_SIZE
        self._d = train_ret.shape[1]
        paths = [train_ret[i:i+self._win] for i in range(len(train_ret)-self._win)]
        targets = train_ret[self._win:]
        self._gp = SigGPRegressor(sig_order, sigma2, tau2).fit(paths, targets)
        self.name = f"SigGP-Markowitz(λ={lam})"

    def evaluate(self, env: SigPrefixEnv) -> NDArray:
        import collections
        env.reset()
        hist: "collections.deque[np.ndarray]" = collections.deque(maxlen=self._win)
        hist.append(_clean(env.ret[env.ptr-1]))
        equal_w = np.full(self._d, 1/self._d)
        eq=[1.0]; done=False
        while not done:
            w = equal_w
            if len(hist)==self._win:
                mu,Sigma=self._gp.moments(np.stack(hist,0)); w=_markowitz(mu,Sigma,self._lam)
                if not np.isfinite(w).all(): w=equal_w
                w=np.clip(w,-10,10); w/=w.sum()+1e-12
            _,r,term,trunc,_=env.step(w); r=0.0 if not np.isfinite(r) else r
            hist.append(_clean(env.ret[env.ptr-1])); done=term or trunc
            eq.append(eq[-1]*(1+r))
        return np.asarray(eq,float)

# ─────────────── sample (classical) Markowitz helpers ─────────────
def _sample_mu_Sigma(window: NDArray) -> tuple[NDArray, NDArray]:
    mu = window.mean(axis=0)
    Sigma = np.cov(window, rowvar=False, ddof=1)
    # regularise to ensure SPD
    eps = 1e-6*np.trace(Sigma)/Sigma.shape[0]
    Sigma = Sigma + eps*np.eye(Sigma.shape[0])
    return mu, Sigma

def _simulate_constant_w(env: SigPrefixEnv, w: NDArray) -> NDArray:
    env.reset(); eq=[1.0]; done=False
    while not done:
        _, r, term, trunc, _ = env.step(w); r=0.0 if not np.isfinite(r) else r
        done = term or trunc
        eq.append(eq[-1]*(1+r))
    return np.asarray(eq,float)

# ───────────────────── efficient frontier ────────────────────────
def efficient_frontier(mu,Sigma,lambdas=None):
    lambdas=np.logspace(-2,2,200) if lambdas is None else list(lambdas)
    invS,ones=np.linalg.inv(Sigma),np.ones_like(mu)
    A,B=invS@ones,invS@mu; alpha,beta=float(ones@A),float(ones@B)
    C=float(mu@B); delta=alpha*C-beta**2
    sig,ret=[],[]
    for lam in lambdas:
        sig2=alpha/lam**2+delta/alpha
        ret_l=beta/lam+beta*delta/alpha**2
        sig.append(math.sqrt(sig2)); ret.append(ret_l)
    return np.asarray(sig),np.asarray(ret)

# ══════════════════════ script entry point ═══════════════════════
if __name__=="__main__":
    import argparse, matplotlib.pyplot as plt
    from .cli import _load_prices
    p=argparse.ArgumentParser(); p.add_argument("--years",type=int,default=1)
    p.add_argument("--lam",type=float,default=1.0); args=p.parse_args()

    prices=_load_prices(config.ASSETS,args.years)
    rets=prices.pct_change().dropna().values.astype(np.float32)
    cut=int(len(rets)*config.SPLIT)

    # GP agent
    gp_agent=SignatureKernelMarkowitzAgent(rets[:cut],lam=args.lam)
    gp_eq=gp_agent.evaluate(SigPrefixEnv(rets[cut:]))
    print(f"GP growth: {gp_eq[-1]:.2f}×")

    # classical benchmark
    last_win=rets[cut-gp_agent._win:cut]
    mu_s,Sigma_s=_sample_mu_Sigma(last_win)
    w_s=_markowitz(mu_s,Sigma_s,args.lam)
    bench_eq=_simulate_constant_w(SigPrefixEnv(rets[cut:]),w_s)
    print(f"Sample-Markowitz growth: {bench_eq[-1]:.2f}×")

    # plot efficient frontiers
    mu_gp,Sig_gp=gp_agent._gp.moments(last_win)
    sig_gp,ret_gp=efficient_frontier(mu_gp,Sig_gp)
    sig_s, ret_s = efficient_frontier(mu_s,Sigma_s)

    plt.figure(figsize=(6,4))
    plt.plot(sig_gp,ret_gp,label="Posterior Frontier",lw=2)
    plt.plot(sig_s,ret_s,label="Sample Frontier",lw=2,ls="--")
    plt.xlabel("σ  (volatility)"); plt.ylabel("μ  (expected return)")
    plt.title("Efficient Frontiers (GP vs. Sample)"); plt.legend()
    plt.tight_layout(); plt.show()
