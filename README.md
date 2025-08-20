# logsig-rl

Unified training for RL, signature-kernel RL, and state-feature RL across algorithms (SAC, TD3, DDPG, PPO, A2C).
One thin CLI, one unified agent.

## Install (as a library)

### From Git
```bash
pip install "git+https://github.com/EliasKha/Signature-kernel.git"
```

### From local folder
```bash
pip install -e .
```

## Download / prepare environment data
This project uses a simple market environment based on daily OHLCV downloaded via `yfinance`.

```bash
python -m logsig_rl.envs.download --ticker SPY --years 10
```

## Train (your exact command)
This is the command you asked to run (it works with this package layout):
```bash
python -u -m logsig_rl.cli --train rl_sac rl_td3 rl_ddpg rl_ppo rl_a2c sk_sac sk_td3 sk_ddpg sk_ppo sk_a2c sf_sac sf_td3 sf_ddpg sf_ppo sf_a2c --years 10 --lam 0:5.1:0.5 --trials 10 --split 0.60 --val-split 0.10
```
Optional common flags:
- `--ticker SPY` (default)
- `--steps 50000`
- `--tcost 0.0001`

Results are appended to `results.csv` in the working directory.

## Repo layout
```
src/logsig_rl/
  cli.py
  runners/experiment.py
  agents/unified.py
  agents/backends/sb3.py
  features/adapters.py
  features/wrapper.py
  envs/download.py
  envs/market.py
  envs/registry.py
  utils/parsing.py
  utils/logging.py
  utils/seeding.py
```

## License
MIT
