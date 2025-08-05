"""
Simple pass-through to CLI so users can run `python -m logsig_rl`.
"""
from .cli import main as _main

if __name__ == "__main__":
    _main()
