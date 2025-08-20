from collections import deque
import numpy as np

class IdentityAdapter:
    def __init__(self): pass
    def reset(self): pass
    def transform(self, obs: np.ndarray) -> np.ndarray:
        return np.asarray(obs, dtype=np.float32)

class StateFeaturesAdapter:
    """
    Simple engineered features: augment raw features with squares and diffs.
    """
    def __init__(self, history: int = 5):
        self.buf = deque(maxlen=history)
    def reset(self):
        self.buf.clear()
    def transform(self, obs: np.ndarray) -> np.ndarray:
        x = np.asarray(obs, dtype=np.float32).ravel()
        if len(self.buf) == 0:
            dx = np.zeros_like(x)
        else:
            dx = x - self.buf[-1]
        self.buf.append(x)
        feats = np.concatenate([x, x * x, dx], axis=0)
        return feats.astype(np.float32)

class SigKernelAdapter:
    """
    Signature features from the last W observations using signatory.
    Depth grows with lam.
    """
    def __init__(self, lam: float = 0.0, window: int = None):
        try:
            import signatory  # noqa: F401
        except Exception as e:
            raise RuntimeError(
                "SigKernelAdapter requires the 'signatory' package. "
                "Install it (pip install signatory) to use sk_* agents."
            ) from e
        self.lam = float(lam)
        depth = max(2, int(round(2 + self.lam)))   # 2..7 for lam 0..5
        self.depth = int(depth)
        self.W = int(window) if window is not None else int(20 + 4 * self.depth)
        self.buf = deque(maxlen=self.W)

    def reset(self):
        self.buf.clear()

    def _signature(self, path_np: np.ndarray) -> np.ndarray:
        import torch, signatory
        device = "cpu"
        path = torch.tensor(path_np[None, :, :], dtype=torch.float32, device=device)
        sig = signatory.signature(path, depth=self.depth)  # shape (1, feature_dim)
        return sig.cpu().numpy().ravel().astype(np.float32)

    def transform(self, obs: np.ndarray) -> np.ndarray:
        x = np.asarray(obs, dtype=np.float32).ravel()
        self.buf.append(x)
        path = np.stack(self.buf, axis=0)  # (T, d)
        sig = self._signature(path)
        return sig
