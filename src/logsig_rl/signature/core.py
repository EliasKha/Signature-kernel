"""Vectorised helpers around `iisignature`.

This version adds *log‑signature* support and a minimal BCH (Baker–Campbell–Hausdorff)
composition implemented via iisignature’s ``exp``/``log`` maps.  All public
symbols are gathered in ``__all__`` for clean wildcard imports.
"""
from __future__ import annotations

from typing import Callable, Any
import numpy as np
import iisignature as _iis

__all__ = [
    # statistics
    "sharpe",
    # signature helpers
    "sig_dim",
    "sig_identity",
    "sig",
    "chen",
    # log‑signature helpers
    "logsig_dim",
    "logsig_identity",
    "logsig",
    "bch",
]

# --------------------------------------------------------------------------- #
# Basic statistics                                                            #
# --------------------------------------------------------------------------- #

def sharpe(x: np.ndarray) -> float:
    """Return the (numerically‑stabilised) Sharpe ratio of *x*."""
    x = np.asarray(x, dtype=float)
    return float(x.mean() / (x.std() + 1e-12))

# --------------------------------------------------------------------------- #
# Tensor signature utilities                                                  #
# --------------------------------------------------------------------------- #

def sig_dim(d: int, m: int) -> int:
    """Total dimension of a depth‑*m* *signature* in *d* dimensions."""
    return _iis.siglength(d, m)


def sig(path: np.ndarray, depth: int) -> np.ndarray:
    """Signature of a **two‑row** array segment (used in Chen combos)."""
    return _iis.sig(path, depth)


def chen(a: np.ndarray, b: np.ndarray, d: int, m: int) -> np.ndarray:
    """Chen’s identity: combine two signatures of successive pieces."""
    return _iis.sigcombine(a, b, d, m)


def sig_identity(d: int, m: int) -> np.ndarray:
    """Identity element under signature concatenation (1, 0, 0, …)."""
    out = np.zeros(_iis.siglength(d, m), dtype=np.float64)
    out[0] = 1.0
    return out

# --------------------------------------------------------------------------- #
# Log‑signature utilities                                                     #
# --------------------------------------------------------------------------- #

def logsig_dim(d: int, m: int) -> int:
    """Total dimension of a depth‑*m* **log‑signature** in *d* dimensions."""
    return _iis.logsiglength(d, m)


def logsig(path: np.ndarray, depth: int) -> np.ndarray:
    """Log‑signature of a **two‑row** array segment.

    Notes
    -----
    ``iisignature.logsig`` expects a *path*, not a pre‑computed signature, so we
    call it directly here for raw segments.
    """
    return _iis.logsig(path, depth)


def logsig_identity(d: int, m: int) -> np.ndarray:
    """Additive identity in log‑signature space (all zeros)."""
    return np.zeros(_iis.logsiglength(d, m), dtype=np.float64)


# --------------------------------------------------------------------------- #
# BCH composition (log‑sig concatenation)                                     #
# --------------------------------------------------------------------------- #

def _logsig_to_sig(logsig: np.ndarray, d: int, m: int) -> np.ndarray:
    """Helper: map a log‑signature to its tensor‑signature via *exp*.

    ``iisignature`` calls this `exp` from v0.25 onward; for earlier versions we
    try a few reasonable fallbacks.
    """
    if hasattr(_iis, "exp"):
        return _iis.exp(logsig, d, m)  # type: ignore[arg-type]
    for name in ("logsig2sig", "logsig_to_sig"):
        if hasattr(_iis, name):
            return getattr(_iis, name)(logsig, d, m)  # type: ignore[misc]
    raise RuntimeError("Your iisignature build lacks log‑sig → sig conversion.")


def _sig_to_logsig(sig: np.ndarray, d: int, m: int) -> np.ndarray:
    """Helper: map a tensor‑signature to its log‑signature via *log*.

    From v0.25 `iisignature.log` is provided; earlier versions have
    `sig2logsig`/`sig_to_logsig`.
    """
    if hasattr(_iis, "log"):
        return _iis.log(sig, d, m)  # type: ignore[arg-type]
    for name in ("sig2logsig", "sig_to_logsig"):
        if hasattr(_iis, name):
            return getattr(_iis, name)(sig, d, m)  # type: ignore[misc]
    raise RuntimeError("Your iisignature build lacks sig → log‑sig conversion.")


def bch(a: np.ndarray, b: np.ndarray, d: int, m: int) -> np.ndarray:
    """Baker–Campbell–Hausdorff composition of two log‑signatures.

    It is implemented by lifting to the tensor algebra, applying Chen’s
    identity, then projecting back.  This costs one `exp` + one `log` per call
    but keeps the public API tiny and avoids a heavyweight pure‑Python BCH
    implementation.
    """
    sig_a = _logsig_to_sig(a, d, m)
    sig_b = _logsig_to_sig(b, d, m)
    sig_ab = _iis.sigcombine(sig_a, sig_b, d, m)
    return _sig_to_logsig(sig_ab, d, m)
