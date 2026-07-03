"""plots.py -- regenerate the paper figures into figures/ from the pipeline."""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier

from dcml import directional_changes, build_features
from run_walkforward import (load_prices, walk_forward_predict, evaluate_asset,
                             TRAIN_FRAC, COST_BPS)
from backtest import daily_returns_from_trades, buy_hold_daily

PRICES = load_prices()
COL = {"S&P500": "#1f77b4", "NASDAQ": "#ff7f0e", "WTIoil": "#2ca02c"}
os.makedirs("figures", exist_ok=True)


def fig_scaling():
    sl = pd.read_csv("results/scaling_laws.csv")
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    for name, g in sl.groupby("asset"):
        ax[0].loglog(g["theta"], g["n_events"], "o-", label=name, color=COL[name])
        ax[1].plot(g["theta"] * 100, g["mean_OSV"], "o-", label=name, color=COL[name])
    ax[0].set_xlabel(r"threshold $\theta$"); ax[0].set_ylabel("DC events")
    ax[0].set_title("(a) Event frequency (log-log)"); ax[0].legend(fontsize=8)
    ax[1].set_xlabel(r"threshold $\theta$ (%)"); ax[1].set_ylabel("mean OSV")
    ax[1].set_title("(b) Overshoot vs threshold"); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig("figures/fig1_scaling.pdf"); plt.close(fig)


def fig_importance():
    imp = pd.read_csv("results/feature_importance.csv", index_col=0)["importance"]
    imp = imp.sort_values()
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.barh(imp.index, imp.values, color="#4a6fa5")
    ax.set_xlabel("mean gradient-boosting importance (3 assets, $\\theta$=1%)")
    ax.set_title("Trend-magnitude & overshoot dominate; raw direction ~ 0")
    fig.tight_layout(); fig.savefig("figures/fig3_importance.pdf"); plt.close(fig)


def fig_drawdown():
    wf = pd.read_csv("results/walkforward_theta1.csv")
    piv = wf[wf.strategy.isin(["Buy&Hold", "DC-ML"])].pivot(
        index="asset", columns="strategy", values="max_dd").reindex(
        ["S&P500", "NASDAQ", "WTIoil"])
    fig, ax = plt.subplots(figsize=(6, 3.4))
    x = np.arange(len(piv)); w = 0.38
    ax.bar(x - w/2, -piv["Buy&Hold"]*100, w, label="Buy&Hold", color="#bbbbbb")
    ax.bar(x + w/2, -piv["DC-ML"]*100, w, label="DC-ML", color="#4a6fa5")
    ax.set_xticks(x); ax.set_xticklabels(piv.index)
    ax.set_ylabel("max drawdown (%)"); ax.legend()
    ax.set_title("Out-of-sample max drawdown ($\\theta$=1%)")
    fig.tight_layout(); fig.savefig("figures/fig2_drawdown.pdf"); plt.close(fig)


def fig_equity(theta=0.01):
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for name, p in PRICES.items():
        ev = directional_changes(p, theta); X, y, meta = build_features(ev)
        M = len(X); it = int(TRAIN_FRAC*M); oos = np.arange(it, M)
        s_i, e_i = int(meta.loc[it, "conf_idx"]), int(meta.loc[M-1, "exit_idx"])
        preds, _ = walk_forward_predict(X, y, it); mask = preds == 1
        tr = [(int(meta.loc[k, "conf_idx"]), int(meta.loc[k, "exit_idx"]),
               int(meta.loc[k, "direction"])) for k in oos[mask[oos]]]
        dr = daily_returns_from_trades(p, tr, s_i, e_i, COST_BPS)
        eq = np.cumprod(1 + dr)
        ax.plot(np.linspace(0, 1, len(eq)), eq, label=f"{name} DC-ML", color=COL[name])
    ax.axhline(1, color="k", lw=0.6, ls=":")
    ax.set_xlabel("out-of-sample progress"); ax.set_ylabel("equity (start=1)")
    ax.set_title("DC-ML out-of-sample equity curves ($\\theta$=1%)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig("figures/fig4_equity.pdf"); plt.close(fig)


if __name__ == "__main__":
    fig_scaling(); fig_importance(); fig_drawdown(); fig_equity()
    print("wrote figures/fig1_scaling.pdf .. fig4_equity.pdf")
