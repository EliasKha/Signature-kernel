"""
Central hyper-parameter configuration.

You can later swap this out for Hydra/yaml/argparse if you need
runtime overrides.
"""
# Signature depth for observations and MPC planning
LOGSIG_DEPTH = 2          # env + RL agent
MPC_DEPTH    = 2          # planning controller
SIG_WINDOW_SIZE = 50  # sliding window for Sig-Trading
SIG_MAX_VAR = 5
# RL (A2C) hyper-params
HIDDEN_SIZE   = 16
SIG_EPOCHS    = 50
GAMMA         = 0.99
LR            = 1e-4

# MPC hyper-params
MPC_HORIZON    = 10
MPC_CANDIDATES = 1024
MPC_REG_LAMBDA = 1e-4

BENCHMARK = ["^DJI"]

ASSETS = [
    "NVDA","MSFT","AAPL","AMZN","JPM","WMT","V","PG","JNJ","HD",
    "KO","UNH","CSCO","IBM","CRM","CVX","DIS","AXP","MCD","GS",
    "MRK","CAT","VZ","BA","AMGN","HON","NKE","SHW","MMM","TRV"
]


# ASSETS = [
#     "AAPL", "ABBV", "ABT", "ACN", "ADBE", "AIG", "AMD", "AMGN", "AMT", "AMZN",
#     "AVGO", "AXP", "BA", "BAC", "BIIB", "BK", "BKNG", "BLK", "BMY", "C",
#     "CAT", "CHTR", "CL", "CMCSA", "COF", "COP", "COST", "CRM", "CSCO", "CVS",
#     "CVX", "DHR", "DIS", "DOW", "DUK", "EMR", "EXC", "F", "FDX", "GD",
#     "GE", "GILD", "GM", "GOOG", "GOOGL", "GS", "HD", "HON", "IBM", "INTC",
#     "INTU", "ISRG", "JNJ", "JPM", "KHC", "KMI", "KO", "LIN", "LLY", "LMT",
#     "LOW", "MA", "MCD", "MDLZ", "MDT", "MET", "META", "MMM", "MO", "MRK",
#     "MS", "MSFT", "NEE", "NFLX", "NKE", "NOW", "NVDA", "ORCL", "PEP", "PFE",
#     "PM", "PYPL", "QCOM", "RTX", "SBUX", "SCHW", "SO", "SPG", "T", "TGT",
#     "TMO", "TMUS", "TSLA", "TXN", "UNH", "UNP", "UPS", "USB", "V", "VZ",
#     "WBA", "WFC", "WMT", "XOM"
# ]

# BENCHMARK = ["^OEX"]


YEARS = 5
SPLIT = 0.7     # 70 % train, remainder test
