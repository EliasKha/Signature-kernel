from typing import Dict, List, Tuple
import numpy as np
import traceback
from tqdm import tqdm

from logsig_rl.agents.unified import UnifiedAgent, parse_agent_name
from logsig_rl.envs.registry import make_env_splits
from logsig_rl.utils.logging import ResultsDirWriter
from logsig_rl.utils.seeding import seed_everything


def _extract_params(model) -> Dict:
    keys = [
        "learning_rate", "gamma", "n_steps", "gae_lambda", "ent_coef",
        "vf_coef", "max_grad_norm", "clip_range", "clip_range_vf",
        "buffer_size", "batch_size", "tau", "train_freq", "gradient_steps",
        "learning_starts", "policy_delay", "target_update_interval",
    ]
    params: Dict = {}
    for k in keys:
        if hasattr(model, k):
            v = getattr(model, k)
            try:
                if callable(v):
                    v = float(v(1))
            except Exception:
                pass
            if k == "train_freq" and not isinstance(v, (int, float, str)):
                try:
                    v = tuple(v)
                except Exception:
                    pass
            params[k] = v
    return params


def _rollout_series(agent: UnifiedAgent, env) -> Tuple[List[float], List[float], float]:
    """
    Roll a full episode and return:
      agent_returns  -> per-step reward
      bench_returns  -> per-step market return (info['ret_t'] if available, else 0.0)
      score          -> mean(agent_returns) for model selection
    """
    obs, _ = env.reset()
    agent_returns: List[float] = []
    bench_returns: List[float] = []
    steps = 0
    while True:
        action, _ = agent.model.predict(obs, deterministic=True)  # type: ignore[attr-defined]
        obs, reward, terminated, truncated, info = env.step(action)
        agent_returns.append(float(reward))
        bench_returns.append(float(info.get("ret_t", 0.0)))
        steps += 1
        if terminated or truncated:
            break
    score = float(np.mean(agent_returns)) if agent_returns else 0.0
    return agent_returns, bench_returns, score


def run_experiments(
    algo_names: List[str],
    years: int,
    lam_values: List[float],
    trials: int,
    split: float,
    val_split: float,
    env_name: str,
    ticker: str,
    steps: int,
    tcost: float,
    skip_existing: bool = True,
):
    """
    For each λ, produce exactly these files:
      results/<family>/<algo>/<TICKER>/lam_xxxxx.csv             # BEST series (agent + benchmark)
      results/<family>/<algo>/<TICKER>/lam_xxxxx_hparams.json    # list of trials (JSON) as per user's format
    """

    with tqdm(total=len(algo_names) * len(lam_values) * trials, desc="Experiments", unit="job") as pbar:
        for name in algo_names:
            family, algo = parse_agent_name(name)

            for lam in lam_values:
                writer = ResultsDirWriter(
                    base_dir="results",
                    family=family,
                    algo=algo,
                    ticker=ticker,
                    lam=lam,
                )

                for trial in range(trials):
                    if skip_existing and writer.trial_exists(trial):
                        msg = f"[Skip] {family}/{algo}/{ticker} λ={lam:.4f} trial={trial}"
                        pbar.update(1)
                        pbar.set_postfix_str(msg)
                        
                        continue

                    try:
                        seed = 10_000 + 97 * trial
                        seed_everything(seed)

                        env_train, env_val, env_test = make_env_splits(
                            env_name=env_name,
                            ticker=ticker,
                            years=years,
                            split=split,
                            val_split=val_split,
                            family=family,
                            lam=lam,
                            tcost=tcost,
                        )

                        agent = UnifiedAgent(family=family, algo=algo, lam=lam)
                        agent.fit(env_train, steps=steps)

                        env_val_wrapped = agent._maybe_wrap_env(env_val, phase="val")
                        val_agent, _val_bench, val_score = _rollout_series(agent, env_val_wrapped)

                        env_test_wrapped = agent._maybe_wrap_env(env_test, phase="test")
                        test_agent, test_bench, test_score = _rollout_series(agent, env_test_wrapped)
                        try:
                            params = _extract_params(agent.model)  # type: ignore[attr-defined]
                        except Exception:
                            params = {}

                        writer.append_trial(
                            trial=trial,
                            score=val_score,
                            params=params,
                            val_returns=val_agent,
                            test_returns=test_agent,
                            test_benchmark_returns=test_bench,  # only used if THIS trial is best
                        )

                        msg = f"[OK] {family}/{algo}/{ticker} λ={lam:.4f} trial={trial} score={val_score:.6g}"
                        pbar.set_postfix_str(msg)
                    except Exception:
                        print(f"[ERROR] Failure at {family}/{algo}/{ticker} λ={lam:.4f} trial={trial}")
                        traceback.print_exc()
                    finally:
                        pbar.update(1)
