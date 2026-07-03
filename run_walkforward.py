"""
run_walkforward.py -- reproduces the headline out-of-sample table.

For each asset and threshold theta:
  1. detect DC events, build features/labels
  2. expanding-window walk-forward: train on events [0, i), predict the next
     block of BLOCK events with gradient boosting, advance -- no look-ahead
  3. evaluate three strategies over the identical OOS window:
       Buy&Hold, DC-Trend (trade every event), DC-ML (trade only when the
       classifier predicts a profitable move)
  4. annualise from a DAILY equity curve; bootstrap p-value for DC-ML per-trade

Usage:  python run_walkforward.py            # theta=1%, all assets -> Table 1
        python run_walkforward.py --grid     # full theta grid -> results/
"""
from __future__ import annotations
import argparse, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

import arch.data.sp500, arch.data.nasdaq, arch.data.wti
from dcml import directional_changes, build_features
from backtest import (curve_metrics, daily_returns_from_trades, buy_hold_daily,
                      stationary_bootstrap_pvalue)

TRAIN_FRAC = 0.40      # initial expanding-window training fraction
BLOCK = 30             # predict this many events, then retrain
COST_BPS = 2.0


def load_prices():
    return {
        "S&P500": arch.data.sp500.load()["Adj Close"].dropna().values,
        "NASDAQ": arch.data.nasdaq.load()["Adj Close"].dropna().values,
        "WTIoil": arch.data.wti.load()["DCOILWTICO"].dropna().values,
    }


def walk_forward_predict(X, y, init_train, block=BLOCK, seed=0):
    """Return OOS predicted labels/probabilities for events [init_train:]."""
    Xv, yv = X.values, y.values
    n = len(Xv)
    preds = np.full(n, -1, dtype=int)
    proba = np.full(n, np.nan)
    i = init_train
    while i < n:
        j = min(i + block, n)
        clf = GradientBoostingClassifier(random_state=seed)
        # guard: need both classes present in training window
        if len(np.unique(yv[:i])) < 2:
            preds[i:j] = 1
            proba[i:j] = 0.5
        else:
            clf.fit(Xv[:i], yv[:i])
            preds[i:j] = clf.predict(Xv[i:j])
            proba[i:j] = clf.predict_proba(Xv[i:j])[:, 1]
        i = j
    return preds, proba


def evaluate_asset(name, prices, theta, seed=0):
    events = directional_changes(prices, theta)
    X, y, meta = build_features(events)
    M = len(X)
    if M < 60:
        return None
    init_train = int(TRAIN_FRAC * M)
    preds, proba = walk_forward_predict(X, y, init_train, seed=seed)

    oos = np.arange(init_train, M)
    start_idx = int(meta.loc[init_train, "conf_idx"])
    end_idx = int(meta.loc[M - 1, "exit_idx"])

    def trades_for(mask):
        out = []
        for k in oos[mask[oos]]:
            e = int(meta.loc[k, "conf_idx"]); x = int(meta.loc[k, "exit_idx"])
            d = int(meta.loc[k, "direction"])
            out.append((e, x, d))
        return out

    all_mask = np.ones(M, dtype=bool)
    ml_mask = (preds == 1)

    rows = []
    # Buy & Hold over identical OOS window
    bh = buy_hold_daily(prices, start_idx, end_idx)
    m = curve_metrics(bh)
    rows.append(dict(asset=name, strategy="Buy&Hold", **m, trades=1, pvalue=np.nan))

    # DC-Trend
    tr = trades_for(all_mask)
    dt = daily_returns_from_trades(prices, tr, start_idx, end_idx, COST_BPS)
    m = curve_metrics(dt)
    rows.append(dict(asset=name, strategy="DC-Trend", **m, trades=len(tr), pvalue=np.nan))

    # DC-ML
    trm = trades_for(ml_mask)
    dm = daily_returns_from_trades(prices, trm, start_idx, end_idx, COST_BPS)
    m = curve_metrics(dm)
    # per-trade net returns for bootstrap
    net = (meta.loc[oos[ml_mask[oos]], "trade_ret"].values
           - 2.0 * COST_BPS * 1e-4)
    pval = stationary_bootstrap_pvalue(net, mean_block_len=5, seed=seed)
    hit = float((net > 0).mean()) if len(net) else np.nan
    rows.append(dict(asset=name, strategy="DC-ML", **m, trades=len(trm),
                     pvalue=pval, hit=hit))
    return rows


def fmt(v, pct=True):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "  -  "
    return f"{v*100:+.1f}%" if pct else f"{v:.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--theta", type=float, default=0.01)
    ap.add_argument("--grid", action="store_true")
    args = ap.parse_args()
    prices = load_prices()

    if args.grid:
        thetas = [0.005, 0.0075, 0.01, 0.02, 0.03]
        allrows = []
        for th in thetas:
            for name, p in prices.items():
                r = evaluate_asset(name, p, th)
                if r:
                    for row in r:
                        row["theta"] = th
                        allrows.append(row)
        df = pd.DataFrame(allrows)
        df.to_csv("results/walkforward_full.csv", index=False)
        print("wrote results/walkforward_full.csv", df.shape)
        return

    th = args.theta
    allrows = []
    for name, p in prices.items():
        allrows += evaluate_asset(name, p, th)
    df = pd.DataFrame(allrows)
    df.to_csv(f"results/walkforward_theta{int(th*100)}.csv", index=False)

    print(f"\nTable 1 -- Out-of-sample walk-forward (theta={th*100:.0f}%), "
          f"net of {COST_BPS:.0f}bps/side\n")
    hdr = f"{'Asset':7} {'Strategy':9} {'AnnRet':>8} {'Sharpe':>7} {'MaxDD':>8} {'Trades':>7} {'p':>6}"
    print(hdr); print("-" * len(hdr))
    for _, r in df.iterrows():
        print(f"{r['asset']:7} {r['strategy']:9} {fmt(r['ann_return']):>8} "
              f"{fmt(r['sharpe'],pct=False):>7} {fmt(r['max_dd']):>8} "
              f"{int(r['trades']):>7} {fmt(r['pvalue'],pct=False):>6}")
    print(f"\nwrote results/walkforward_theta{int(th*100)}.csv")


if __name__ == "__main__":
    main()
