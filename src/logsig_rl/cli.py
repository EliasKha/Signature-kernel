# ─────────────────────────── src/logsig_rl/cli.py ───────────────────────────
from __future__ import annotations

# ╭────────────────────────────── Std-lib ───────────────────────────────╮
import argparse
import datetime as dt
import json
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any, Mapping, Sequence

# ╭──────────────────── Third-party scientific stack ────────────────────╮
import numpy as np
import optuna
import pandas as pd
import yfinance as yf
from tqdm import tqdm
from stable_baselines3 import SAC, PPO, A2C, DQN, DDPG, TD3

# ╭────────────────────────────── Project ───────────────────────────────╮
from . import config
from .envs import SigPrefixEnv
from .agents import (
    SignatureKernelRLAgent,
    RandomFourierRLAgent,
    GenericRLAgent,
)
from .signature import sharpe

# ╭──────────────────────────── Static paths ─────────────────────────────╮
DATA_DIR     = Path("data")
CACHE_INDEX  = DATA_DIR / "cache_index.json"
RESULTS_ROOT = Path("results")
DATA_DIR.mkdir(exist_ok=True)

# ╭──────────────────────────── SB-3 mapping ─────────────────────────────╮
ALG_MAP = {
    "sac": SAC,
    "td3": TD3,
    "ddpg": DDPG,
    "dqn": DQN,
    "ppo": PPO,
    "a2c": A2C,
}

# ╭─────────────────────── Market-data loader + cache ────────────────────╮
def _load_prices(tickers: Sequence[str] | str, years: int) -> pd.DataFrame:
    if isinstance(tickers, str):
        tickers = [tickers]

    end   = dt.datetime.utcnow().replace(tzinfo=None)
    start = end - dt.timedelta(days=365 * years)

    cache: dict[str, dict[str, str]] = (
        json.loads(CACHE_INDEX.read_text()) if CACHE_INDEX.exists() else {}
    )
    series: list[pd.Series] = []

    for tk in tickers:
        csv_path = DATA_DIR / f"{tk.replace('^', '')}.csv"
        meta      = cache.get(tk, {})
        have_from = dt.datetime.fromisoformat(meta.get("from", "1900-01-01"))
        have_to   = dt.datetime.fromisoformat(meta.get("to",   "1900-01-01"))

        if not csv_path.exists() or have_from > start or have_to < end:
            df = yf.download(tk, start=start, end=end, progress=False)
            if df.empty:
                raise RuntimeError(f"No data for {tk}")
            df.to_csv(csv_path)
            cache[tk] = {
                "from": df.index.min().date().isoformat(),
                "to":   df.index.max().date().isoformat(),
            }
        else:
            df = pd.read_csv(csv_path, index_col=0, parse_dates=True)

        ser = df["Close"].astype(np.float32)
        ser.name = tk
        series.append(ser)

    CACHE_INDEX.write_text(json.dumps(cache, indent=2))
    return (
        pd.concat(series, axis=1)
          .loc[start:end]
          .dropna()
          .astype(np.float32)
    )

# ╭──────────────────── Hyper-parameter search spaces ────────────────────╮
def _suggest_hyperparams(trial: optuna.Trial, algo: str) -> dict[str, Any]:
    hp: dict[str, Any] = {}

    if algo in {"sac", "td3", "ddpg", "dqn"}:
        hp["learning_rate"] = trial.suggest_float("learning_rate", 1e-5, 3e-3, log=True)
        hp["gamma"]         = trial.suggest_float("gamma", 0.9, 0.9999, log=True)
        hp["buffer_size"]   = int(trial.suggest_float("buffer_size", 2e5, 1e6, log=True))
        hp["batch_size"]    = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
        hp["tau"]           = trial.suggest_float("tau", 1e-4, 0.05, log=True)

    if algo == "sac":
        hp["ent_coef"]  = trial.suggest_float("ent_coef", 1e-5, 0.2, log=True)
        hp["train_freq"] = trial.suggest_categorical("train_freq", [1, 2, 4, 8, 16])

    if algo in {"ppo", "a2c"}:
        hp["learning_rate"] = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        hp["gamma"]         = trial.suggest_float("gamma", 0.5, 0.9999, log=True)
        hp["n_steps"]       = trial.suggest_categorical("n_steps", [128, 256, 512, 1024])
        hp["gae_lambda"]    = trial.suggest_float("gae_lambda", 0.8, 0.99)
        hp["ent_coef"]      = trial.suggest_float("ent_coef", 1e-5, 0.5, log=True)

    if algo == "dqn":
        hp["exploration_fraction"] = trial.suggest_float("exploration_fraction", 0.05, 0.3)
        hp["exploration_final_eps"] = trial.suggest_float("exploration_final_eps", 0.01, 0.1)
    return hp

# ╭──────────────────────── Agent factory helpers ─────────────────────────╮
def _make_agent(name: str, train_ret: np.ndarray,
                kwargs: Mapping[str, Any] | None = None):
    kwargs = dict(kwargs or {})
    base   = name[3:] if name.startswith(("sk_", "sf_")) else name

    if name.startswith("sk_"):
        return SignatureKernelRLAgent(
            prototypes=[train_ret],
            window_length=config.SIG_WINDOW_SIZE,
            truncation_level=config.LOGSIG_DEPTH,
            algorithm_cls=ALG_MAP[base],
            policy="MlpPolicy",
            algo_kwargs=kwargs,
        )

    if name.startswith("sf_"):
        return RandomFourierRLAgent(
            prototypes=[train_ret],
            window_length=config.SIG_WINDOW_SIZE,
            truncation_level=config.LOGSIG_DEPTH,
            rff_width=getattr(config, "RFF_WIDTH", 256),
            rbf_sigma=getattr(config, "RBF_SIGMA", 1.0),
            algorithm_cls=ALG_MAP[base],
            policy="MlpPolicy",
            algo_kwargs=kwargs,
        )

    return GenericRLAgent(
        algorithm_cls=ALG_MAP[name],
        policy="MlpPolicy",
        algo_kwargs=kwargs,
    )

# ╭────────────────────── One complete (algo, λ) job ──────────────────────╮
def _train_one_job(
    algo_name: str,
    lam: float,
    train_ret: np.ndarray,
    val_ret: np.ndarray,
    test_ret: np.ndarray,
    bench_ret: np.ndarray,
    bench_ticker: str | Sequence[str],
    trials: int,
    steps: int,
    root: str,
) -> None:

    optuna.logging.set_verbosity(logging.WARNING)
    bench_tag = "-".join(bench_ticker) if isinstance(bench_ticker, (list, tuple)) else bench_ticker

    method    = algo_name[:2] if algo_name.startswith(("sk_", "sf_")) else "vanilla"
    base_algo = algo_name[3:] if method != "vanilla" else algo_name
    out_dir   = Path(root) / method / base_algo / bench_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    lam_tag     = f"lam_{lam:.4f}"
    csv_path    = out_dir / f"{lam_tag}.csv"
    hparam_path = out_dir / f"{lam_tag}_hparams.json"

    # ──────────────── load previous Optuna trials (if any) ───────────────
    past_trials: list[dict[str, Any]] = []
    best_score, best_params = -np.inf, {}
    best_test_returns: list[float] = []

    if hparam_path.exists():
        with hparam_path.open() as fh:
            past_trials = json.load(fh) or []
        if past_trials:
            best_idx         = int(np.argmax([t["score"] for t in past_trials]))
            best_trial       = past_trials[best_idx]
            best_score       = best_trial["score"]
            best_params      = best_trial["params"]
            best_test_returns = best_trial["test_returns"]

    # ------------------------------------------------------------------
    remaining = max(trials - len(past_trials), 0)

    # ───────────────── run any missing Optuna trials ────────────────────
    if remaining > 0:
        env_train = SigPrefixEnv(train_ret, lam=lam)
        env_valid = SigPrefixEnv(val_ret,   lam=lam)
        env_test  = SigPrefixEnv(test_ret,  lam=lam)
        trial_log: list[dict[str, Any]] = []

        def objective(trial: optuna.Trial) -> float:
            nonlocal best_score, best_params, best_test_returns
            params = _suggest_hyperparams(trial, base_algo)
            agent  = _make_agent(algo_name, train_ret, params)
            agent.train(env_train, total_timesteps=steps)

            val_returns  = agent.evaluate(env_valid).astype(np.float32)
            score        = sharpe(val_returns)
            test_returns = agent.evaluate(env_test).astype(np.float32)

            trial_log.append({
                "trial":         len(past_trials) + trial.number,
                "score":         float(score),
                "params":        params,
                "val_returns":   val_returns.tolist(),
                "test_returns":  test_returns.tolist(),
            })

            if score > best_score:
                best_score, best_params, best_test_returns = score, params, test_returns.tolist()
            return score

        optuna.create_study(direction="maximize").optimize(
            objective, n_trials=remaining, show_progress_bar=False
        )
        past_trials.extend(trial_log)
        with hparam_path.open("w") as fh:
            json.dump(past_trials, fh, indent=2)

    # ──────────────────── early exit if all done before ──────────────────
    if remaining == 0 and best_params and best_test_returns:
        if not csv_path.exists():
            pd.DataFrame({
                f"{algo_name}_return": best_test_returns,
                "benchmark_return":    bench_ret[: len(best_test_returns)],
            }).to_csv(csv_path, index=False)
        return

    # ──────────────────── final training with best hp ────────────────────
    env_full = SigPrefixEnv(np.concatenate([train_ret, val_ret]), lam=lam)
    env_test = SigPrefixEnv(test_ret,  lam=lam)
    agent    = _make_agent(algo_name, np.concatenate([train_ret, val_ret]), best_params)
    agent.train(env_full, total_timesteps=steps)

    test_returns = agent.evaluate(env_test).astype(np.float32)
    pd.DataFrame({
        f"{algo_name}_return": test_returns,
        "benchmark_return":    bench_ret[: len(test_returns)],
    }).to_csv(csv_path, index=False)

# ╭───────────────────────────── CLI entry ────────────────────────────────╮
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    base = list(ALG_MAP.keys())
    parser.add_argument("--train", nargs="+",
                        choices=base + [f"sk_{a}" for a in base] + [f"sf_{a}" for a in base],
                        default=["sac"])
    parser.add_argument("--years",  type=int,   default=config.YEARS)
    parser.add_argument("--split",  type=float, default=config.SPLIT,
                        help="fraction of samples used for *training*")
    parser.add_argument("--val-split", type=float, default=getattr(config, "VAL_SPLIT", 0.20),
                        help="additional fraction (after train) for *validation*")
    parser.add_argument("--trials", type=int,   default=30)
    parser.add_argument("--lam",    type=str,   default="1.0",
                        help="λ list e.g. '0:5.1:0.5' or '0.5 1.0'")
    parser.add_argument("--no-optimize", action="store_true",
                        help="skip Optuna and use library defaults")
    parser.add_argument("--steps", type=int, default=10_000,
                        help="environment steps per training run")
    parser.add_argument("--max-procs", type=int, default=min(8, cpu_count()),
                        help="limit parallel workers")
    args = parser.parse_args(argv)

    if args.no_optimize:
        args.trials = 0

    # ─────────── Sanity check on splits ───────────
    if args.split + args.val_split >= 1.0:
        parser.error("--split + --val-split must be < 1.0")

    lam_vals = (
        np.arange(*map(float, args.lam.split(":")))
        if ":" in args.lam else [float(x) for x in args.lam.split()]
    )

    price_df = _load_prices(config.ASSETS, args.years)
    bench_df = _load_prices(config.BENCHMARK, args.years)

    n_total   = len(price_df)
    cut_train = int(n_total * args.split)
    cut_val   = cut_train + int(n_total * args.val_split)

    train_ret = price_df.iloc[:cut_train].pct_change().dropna().values
    val_ret   = price_df.iloc[cut_train:cut_val].pct_change().dropna().values
    test_ret  = price_df.iloc[cut_val:].pct_change().dropna().values

    bench_ret = bench_df.iloc[cut_val:].pct_change().dropna().values.flatten()

    jobs = [
        (algo, lam, train_ret, val_ret, test_ret, bench_ret, config.BENCHMARK,
         args.trials, args.steps, str(RESULTS_ROOT))
        for lam in lam_vals for algo in args.train
    ]

    n_procs = min(args.max_procs, max(1, len(jobs)))
    print(f"🚀 Dispatching {len(jobs)} jobs on {n_procs} cores …")

    with ProcessPoolExecutor(max_workers=n_procs) as pool, \
         tqdm(total=len(jobs), desc="jobs") as bar:
        for fut in as_completed(pool.submit(_train_one_job, *j) for j in jobs):
            fut.result()
            bar.update(1)

if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(line_buffering=True)
    main()
# ──────────────────────────────────────────────────────────────────────────
