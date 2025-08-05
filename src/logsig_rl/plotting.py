"Matplotlib helper to compare cumulative-return paths."
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from .signature import sharpe


def plot_paths(results: dict[str, np.ndarray]):
    min_len = min(len(v) for v in results.values())
    t = np.arange(min_len)

    plt.figure(figsize=(12, 6))
    for name, series in results.items():
        plt.plot(t, series[:min_len], lw=2,
                 label=f"{name} (Sharpe {sharpe(np.diff(series[:min_len])):.2f})")
        # plt.plot(t, benchmark[:min_len].cumsum(), lw=2, ls="--",
        #          label=f"{name} Benchmark (Sharpe {sharpe(np.diff(benchmark[:min_len].cumsum())):.2f})")
                 
    plt.xlabel("timestep"); plt.ylabel("cumulative return")
    plt.title("Log-Signature RL vs Signature-MPC")
    plt.legend(); plt.tight_layout(); plt.show()
