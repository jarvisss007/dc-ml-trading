"""
backtest.py -- turn a list of executed DC trades into an honest daily equity
curve and risk metrics, plus a stationary block-bootstrap significance test.

Key discipline (the mistakes we refuse to repeat):
  * Sharpe is annualised from a DAILY return series (x sqrt(252)), never from
    per-trade returns x sqrt(252) -- the latter inflates by ~sqrt(trades/day).
  * Transaction costs are charged on every entry and every exit.
  * All predictions used here are strictly out-of-sample (see run_walkforward).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# metrics from a daily return series
# ---------------------------------------------------------------------------
def curve_metrics(daily_ret):
    """Annualised return (CAGR), annualised Sharpe, and max drawdown."""
    r = np.asarray(daily_ret, dtype=float)
    if len(r) == 0:
        return dict(ann_return=np.nan, sharpe=np.nan, max_dd=np.nan)
    equity = np.cumprod(1.0 + r)
    n = len(r)
    cagr = equity[-1] ** (TRADING_DAYS / n) - 1.0
    sd = r.std(ddof=1)
    sharpe = (r.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else np.nan
    peak = np.maximum.accumulate(equity)
    max_dd = (equity / peak - 1.0).min()
    return dict(ann_return=cagr, sharpe=sharpe, max_dd=max_dd)


# ---------------------------------------------------------------------------
# build a daily strategy return series from executed trades
# ---------------------------------------------------------------------------
def daily_returns_from_trades(prices, trades, start_idx, end_idx,
                              cost_bps_per_side=2.0):
    """Construct daily strategy returns over [start_idx, end_idx].

    prices : full 1-D price array (aligned to integer indices used by events)
    trades : list of (entry_idx, exit_idx, direction)
    Position is `direction` on days entry_idx+1 .. exit_idx (close-to-close),
    flat otherwise. Round-trip cost (2 * cost/side) is charged on the exit day.
    """
    p = np.asarray(prices, dtype=float)
    asset_ret = np.zeros_like(p)
    asset_ret[1:] = p[1:] / p[:-1] - 1.0

    pos = np.zeros_like(p)
    cost = np.zeros_like(p)
    c = 2.0 * cost_bps_per_side * 1e-4
    for e, x, d in trades:
        if x <= e:
            continue
        pos[e + 1:x + 1] = d
        cost[x] += c

    sl = slice(start_idx + 1, end_idx + 1)
    strat = pos[sl] * asset_ret[sl] - cost[sl]
    return strat


def buy_hold_daily(prices, start_idx, end_idx):
    p = np.asarray(prices, dtype=float)
    seg = p[start_idx:end_idx + 1]
    return seg[1:] / seg[:-1] - 1.0


# ---------------------------------------------------------------------------
# stationary block bootstrap (Politis & Romano 1994)
# ---------------------------------------------------------------------------
def stationary_bootstrap_pvalue(x, mean_block_len=5, n_boot=5000, seed=0):
    """One-sided p-value for H0: E[x] <= 0 via the stationary bootstrap.

    Returns the fraction of bootstrap resample means that are <= 0.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 3:
        return np.nan
    rng = np.random.default_rng(seed)
    p_geom = 1.0 / mean_block_len
    obs_mean = x.mean()
    boot_means = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.empty(n, dtype=int)
        i = rng.integers(0, n)
        for k in range(n):
            idx[k] = i
            if rng.random() < p_geom:
                i = rng.integers(0, n)          # start a new block
            else:
                i = (i + 1) % n                 # continue block
        boot_means[b] = x[idx].mean()
    # centre at 0 under H0: p = P(centred boot mean >= observed) equivalently
    centred = boot_means - obs_mean
    p_value = np.mean(centred >= obs_mean)      # P(mean <= 0)-style one-sided
    return float(p_value)
