"""
Central hyper-parameter configuration.
"""
LOGSIG_DEPTH = 3          # env + RL agent
SIG_WINDOW_SIZE = 50  # sliding window for Sig-Trading


BENCHMARK = ["^DJI"]

ASSETS = [
    "NVDA","MSFT","AAPL","AMZN","JPM","WMT","V","PG","JNJ","HD",
    "KO","UNH","CSCO","IBM","CRM","CVX","DIS","AXP","MCD","GS",
    "MRK","CAT","VZ","BA","AMGN","HON","NKE","SHW","MMM","TRV"
]