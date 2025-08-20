# src/logsig_rl/signature.py
"""
Thin iisignature wrapper used by the unified agent.

Provides:
- sig_dim(d, m): total signature dimension up to level m
- sig(path, m): signature vector for a (T, d) path up to level m

Requires `iisignature`. Install:  pip install iisignature
"""
from __future__ import annotations

from functools import lru_cache
import numpy as np

try:
    import iisignature as _iisig
    _IISIG_ERR: Exception | None = None
except Exception as e:  # pragma: no cover
    _iisig = None
    _IISIG_ERR = e


def _ensure_iisig() -> None:
    if _iisig is None:
        raise ImportError(
            "iisignature is required for signature features. "
            "Install with: pip install iisignature"
        ) from _IISIG_ERR


@lru_cache(maxsize=None)
def _prepared(d: int, m: int):
    """Cache iisignature.prepare(d, m) for speed on repeated calls."""
    _ensure_iisig()
    return _iisig.prepare(d, m)


def sig_dim(d: int, m: int) -> int:
    """Return signature dimension for streams of dimension d up to level m."""
    if _iisig is not None:
        return int(_iisig.siglength(d, m))
    # Fallback (slow but correct) if iisignature isn't importable at import time.
    return sum(d ** k for k in range(1, m + 1))


def sig(path: np.ndarray, m: int) -> np.ndarray:
    """
    Compute the (Chen) signature up to level m.

    Parameters
    ----------
    path : array-like, shape (T, d)
        Time-ordered observations/returns.
    m : int
        Truncation level.

    Returns
    -------
    np.ndarray, shape (sig_dim(d, m),), dtype float32
    """
    _ensure_iisig()
    arr = np.asarray(path, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"sig expects a 2D array (T, d); got shape {arr.shape}")
    T, d = arr.shape
    if T == 0 or d == 0:
        # Return a zero vector with correct length
        return np.zeros(sig_dim(d, m), dtype=np.float32)

    prep = _prepared(d, m)
    out = _iisig.sig(arr, prep)  # float64
    return np.asarray(out, dtype=np.float32)
