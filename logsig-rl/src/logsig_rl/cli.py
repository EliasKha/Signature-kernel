import argparse
from logsig_rl.runners.experiment import run_experiments
from logsig_rl.utils.parsing import parse_range

def main():
    p = argparse.ArgumentParser("logsig_rl")
    p.add_argument("--train", nargs="+", required=True,
                   help="Algorithms to train, e.g. sac ppo sk_sac sf_td3 ...")
    p.add_argument("--years", type=int, default=10)
    p.add_argument("--lam", type=str, default="0:5.1:0.5", help="start:end:step")
    p.add_argument("--trials", type=int, default=10)
    p.add_argument("--split", type=float, default=0.60, help="train split fraction")
    p.add_argument("--val-split", type=float, default=0.10, help="validation fraction (from remaining)")
    p.add_argument("--ticker", type=str, default="SPY", help="market ticker for env")
    p.add_argument("--steps", type=int, default=50000, help="SB3 training steps per trial")
    p.add_argument("--env", type=str, default="market", help="environment name (market)")
    p.add_argument("--tcost", type=float, default=0.0001, help="transaction cost per unit turnover")

    args = p.parse_args()
    lam_vals = parse_range(args.lam)

    run_experiments(
        algo_names=args.train,
        years=args.years,
        lam_values=lam_vals,
        trials=args.trials,
        split=args.split,
        val_split=args.val_split,
        env_name=args.env,
        ticker=args.ticker,
        steps=args.steps,
        tcost=args.tcost,
    )

if __name__ == "__main__":
    main()
