from logsig_rl.envs.market import load_market_data, split_dataframe, MarketEnv

def make_env_splits(
    env_name: str,
    ticker: str,
    years: int,
    split: float,
    val_split: float,
    family: str,
    lam: float,
    tcost: float,
):
    if env_name != "market":
        raise ValueError("Only env_name='market' is supported in this build.")
    df = load_market_data(ticker=ticker, years=years)
    df_tr, df_val, df_te = split_dataframe(df, split=split, val_split=val_split)
    env_tr  = MarketEnv(df_tr, tcost=tcost, seed=1234)
    env_val = MarketEnv(df_val, tcost=tcost, seed=2345)
    env_te  = MarketEnv(df_te, tcost=tcost, seed=3456)
    return env_tr, env_val, env_te
