from typing import Tuple, Optional
from pathlib import Path
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces


# ---------- local CSV path (inside repo ./data) ----------

def _resolve_csv(ticker: str) -> Path:
    return Path("data") / f"market_{ticker.upper()}.csv"


# ---------- robust CSV -> returns-only ----------

def _pick_close_column(df: pd.DataFrame) -> str:
    for cand in ("Close", "Adj Close", "close", "adj close"):
        if cand in df.columns:
            return cand
    for col in df.columns:
        if "close" in str(col).lower():
            return col
    raise ValueError(f"No close-like column found. Columns: {list(df.columns)}")


def _parse_date_index(df: pd.DataFrame) -> pd.DataFrame:
    """Parse index with strict format first, then per-row fallback without warnings."""
    idx_str = df.index.astype(str)
    idx = pd.to_datetime(idx_str, format="%Y-%m-%d", errors="coerce")
    if idx.isna().any():
        from dateutil import parser as dtparser
        filled = []
        for i, s in zip(idx, idx_str):
            if pd.notna(i):
                filled.append(i)
            else:
                try:
                    filled.append(dtparser.parse(s))
                except Exception:
                    filled.append(pd.NaT)
        idx = pd.DatetimeIndex(filled)
    out = df.copy()
    out.index = idx
    out = out[~out.index.isna()]
    return out


def _returns_only_from_csv(df: pd.DataFrame) -> pd.DataFrame:
    col = _pick_close_column(df)
    px_series = pd.to_numeric(df[col], errors="coerce").dropna()
    px = px_series.to_numpy(dtype=float)
    if px.size < 2:
        raise ValueError("Not enough rows to compute returns.")
    ret = np.diff(px, prepend=px[0]) / (px + 1e-12)  # simple return approx
    out = pd.DataFrame({"ret": ret.astype(np.float32)}, index=px_series.index)
    return out


def load_market_data(ticker: str, years: int) -> pd.DataFrame:
    path = _resolve_csv(ticker)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found.\n"
            f"Download locally with:\n"
            f"  python -m logsig_rl.envs.download --ticker '{ticker}' --years {years}\n"
            f"(this writes to ./data/)"
        )
    df = pd.read_csv(path, index_col=0)  # "Date" index
    df = _parse_date_index(df)
    feats = _returns_only_from_csv(df).dropna().astype(np.float32)
    return feats


def split_dataframe(
    df: pd.DataFrame, split: float, val_split: float
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assert 0.0 < split < 1.0 and 0.0 <= val_split < 1.0 and split + val_split < 1.0
    n = len(df)
    n_tr = int(n * split)
    n_val = int(n * val_split)
    df_tr = df.iloc[:n_tr].copy()
    df_val = df.iloc[n_tr : n_tr + n_val].copy()
    df_te = df.iloc[n_tr + n_val :].copy()
    return df_tr, df_val, df_te


class MarketEnv(gym.Env):
    """
    Continuous trading environment with target weight action in [-1, 1].
    Observation: returns only (shape = (1,)).
    Reward: position_{t-1} * ret_t - tcost * |pos_t - pos_{t-1}|
    """

    metadata = {"render_modes": []}

    def __init__(self, df: pd.DataFrame, tcost: float = 0.0001, seed: Optional[int] = None):
        super().__init__()
        if "ret" not in df.columns or df.shape[1] != 1:
            raise ValueError("Expected a DataFrame with a single column 'ret'.")
        self.df = df.reset_index(drop=True)
        self.returns = self.df["ret"].to_numpy(dtype=np.float32)
        self.obs_mat = self.returns.reshape(-1, 1).astype(np.float32)
        self.tcost = float(tcost)

        self.t = 0
        self.pos = 0.0

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    def reset(self, *, seed: Optional[int] = None, options=None):
        self.t = 0
        self.pos = 0.0
        obs = self.obs_mat[self.t]
        info = {}
        return obs, info

    def step(self, action):
        a = float(np.clip(action[0], -1.0, 1.0))
        prev_pos = self.pos
        self.pos = a

        self.t += 1
        terminated = self.t >= (len(self.obs_mat) - 1)
        truncated = False

        ret_t = float(self.returns[self.t])
        reward = prev_pos * ret_t - self.tcost * abs(self.pos - prev_pos)
        obs = self.obs_mat[self.t]
        info = {"position": self.pos, "ret_t": ret_t}
        return obs, float(reward), terminated, truncated, info

    def render(self):
        pass
