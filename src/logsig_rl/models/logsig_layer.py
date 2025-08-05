"""A thin torch wrapper that caches the iisignature ‘handle’ so we avoid
the ‘input not from prepare()’ runtime error with iisignature≥0.24.
"""
from __future__ import annotations
from typing import Sequence

import torch
import iisignature as _iis
import numpy as np


class LogSigLayer(torch.nn.Module):
    """
    Convert a batch of *paths* with shape (B, L, D) into log-signature vectors
    of depth `depth`.
    """
    def __init__(self, dim: int, depth: int):
        super().__init__()
        self.dim    = dim
        self.depth  = depth
        self.handle = _iis.prepare(dim, depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)  ⇒ list[np.ndarray] ⇒ stack ⇒ tensor
        logsigs = [_iis.logsig(sample.cpu().numpy(), self.handle) for sample in x]
        return torch.from_numpy(np.stack(logsigs))  # float64
