import json
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import enum
import numpy as np


# Keep only these keys in the saved "params"
_ALLOWED_PARAM_KEYS = {
    "learning_rate",
    "gamma",
    "buffer_size",
    "batch_size",
    "tau",
    "ent_coef",
    "train_freq",
}

def _to_python_scalar(x):
    # numpy -> python
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    return x

def _canonicalize_train_freq(v):
    """
    Convert SB3 train_freq variants to a single int frequency.
    Accepts:
      - SB3 TrainFrequency-like object (has .frequency, .unit)
      - tuple/list like [n, "step"] or (n, "step")
      - plain int
    """
    if hasattr(v, "frequency") and hasattr(v, "unit"):
        # SB3 TrainFrequency
        try:
            return int(v.frequency)
        except Exception:
            return int(_to_python_scalar(v.frequency))
    if isinstance(v, (list, tuple)) and len(v) >= 1:
        return int(_to_python_scalar(v[0]))
    return int(_to_python_scalar(v))

def _select_and_canonicalize_params(params: dict) -> dict:
    """
    Keep only the allowed keys and make sure values are basic JSON types.
    Special handling: train_freq -> int.
    """
    out = {}
    for k in _ALLOWED_PARAM_KEYS:
        if k not in params:
            continue
        v = params[k]
        if k == "train_freq":
            try:
                v = _canonicalize_train_freq(v)
            except Exception:
                # If something weird slips through, fall back to 1
                v = 1
        else:
            v = _to_python_scalar(v)
        out[k] = v
    return out


def _lam_tag(lam: float) -> str:
    return f"lam_{lam:.4f}"

def _json_default(o):
    """
    Make trials JSON-serializable:
    - numpy scalars/arrays -> Python scalars/lists
    - Enum (incl. SB3 TrainFrequencyUnit) -> .value
    - SB3 TrainFrequency-like objects (have .frequency and .unit) -> dict
    - fallback: str(o)
    """
    # numpy scalars
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()

    # numpy arrays
    if isinstance(o, np.ndarray):
        return o.tolist()

    # enums (e.g., TrainFrequencyUnit)
    if isinstance(o, enum.Enum):
        return o.value

    # SB3 TrainFrequency-like objects: have .frequency and .unit
    if hasattr(o, "frequency") and hasattr(o, "unit"):
        unit = getattr(o.unit, "value", str(o.unit))
        try:
            freq = int(getattr(o, "frequency"))
        except Exception:
            freq = getattr(o, "frequency")
        return {"frequency": freq, "unit": unit}

    # last resort
    return str(o)



class ResultsDirWriter:
    """
    Expected on disk (per λ):
      results/<family>/<algo>/<TICKER>/lam_0.0000.csv             # BEST trial time-series
      results/<family>/<algo>/<TICKER>/lam_0.0000_hparams.json    # JSON list of trials:
         [
           {
             "trial": int,
             "score": float,               # used for model selection
             "params": {...},              # model hyperparams (subset)
             "val_returns": [float, ...],  # validation incremental returns
             "test_returns": [float, ...]  # test incremental returns (agent)
           },
           ...
         ]

    We DO NOT create any *_trials.csv files.
    """

    def __init__(self, base_dir: str, family: str, algo: str, ticker: str, lam: float):
        self.family = str(family)
        self.algo = str(algo)
        self.ticker = str(ticker)
        self.lam = float(lam)

        self.dir = Path(base_dir) / self.family / self.algo / self.ticker
        self.dir.mkdir(parents=True, exist_ok=True)

        lam_tag = _lam_tag(self.lam)
        self.series_path = self.dir / f"{lam_tag}.csv"
        self.json_path = self.dir / f"{lam_tag}_hparams.json"



    # ---------- JSON trials helpers ----------

    def _load_trials(self) -> List[Dict[str, Any]]:
        if not self.json_path.exists():
            return []
        try:
            with self.json_path.open("r", encoding="utf-8") as fh:
                trials = json.load(fh)
            if not isinstance(trials, list):
                return []
            # sanity pass
            cleaned: List[Dict[str, Any]] = []
            for r in trials:
                try:
                    t = int(r["trial"])
                    s = float(r["score"])
                    p = dict(r.get("params", {}))
                    v = list(r.get("val_returns", []))
                    u = list(r.get("test_returns", []))
                    cleaned.append({"trial": t, "score": s, "params": p, "val_returns": v, "test_returns": u})
                except Exception as e:
                    print(f"[ResultWriter][WARN] Skipping malformed JSON trial row: {r} -> {e}")
            return cleaned
        except Exception as e:
            print(f"[ResultWriter][ERROR] Could not read {self.json_path.name}: {e}")
            return []
        
    def _save_trials(self, trials: List[Dict[str, Any]]) -> None:
        self.json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.json_path, "w", encoding="utf-8") as fh:
            json.dump(trials, fh, indent=2, default=_json_default)  # <-- add default


    def completed_trials(self) -> Sequence[int]:
        ids = [r["trial"] for r in self._load_trials()]
        return ids

    def trial_exists(self, trial: int) -> bool:
        exists = int(trial) in set(self.completed_trials())
        return exists

    def _best_from(self, trials: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not trials:
            return None
        # Best = max score; tie-breaker = smaller trial id
        best = None
        for r in trials:
            if best is None or (r["score"] > best["score"]) or (r["score"] == best["score"] and r["trial"] < best["trial"]):
                best = r
        return best

    def _write_best_series(
        self,
        agent_returns: List[float],
        benchmark_returns: Optional[List[float]],
    ) -> None:
        col_agent = f"{self.family}_{self.algo}_return"
        with self.series_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if benchmark_returns is not None:
                n = min(len(agent_returns), len(benchmark_returns))
                w.writerow([col_agent, "benchmark_return"])
                for i in range(n):
                    w.writerow([float(agent_returns[i]), float(benchmark_returns[i])])
            else:
                n = len(agent_returns)
                w.writerow([col_agent])
                for i in range(n):
                    w.writerow([float(agent_returns[i])])

    # ---------- public API ----------

    def append_trial(
        self,
        trial: int,
        score: float,
        params: Dict[str, Any],
        val_returns: List[float],
        test_returns: List[float],
        test_benchmark_returns: Optional[List[float]] = None,
    ) -> None:
        """
        Append a trial to the JSON list if missing, then refresh BEST series CSV
        *only if this trial becomes the best* (so we don't need to store benchmark
        for all non-best trials).
        """
        trials = self._load_trials()

        if any(r["trial"] == int(trial) for r in trials):
            return
        else:
            record = {
                "trial": int(trial),
                "score": float(score),
                "params": _select_and_canonicalize_params(dict(params)),  # <--- changed
                "val_returns": [float(x) for x in val_returns],
                "test_returns": [float(x) for x in test_returns],
            }
            trials.append(record)
            self._save_trials(trials)

        # Recompute best and update series if THIS trial is now best
        best = self._best_from(trials)
        if best and best["trial"] == int(trial):
            self._write_best_series(agent_returns=test_returns, benchmark_returns=test_benchmark_returns)
