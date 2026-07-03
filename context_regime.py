"""
context_regime.py -- reproduces the market-context and regime-conditioning
robustness results (paper section 7.6).

Augments the DC feature vector with four market-state variables computed at each
event's confirmation bar -- 20-day realised volatility, price / 200-day MA,
20-day momentum, drawdown-from-peak -- re-runs the same walk-forward, and splits
the out-of-sample DC-ML trades by prevailing regime (bull = price above its
200-day moving average).

Writes results/context_walkforward.csv and results/regime_split.csv.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
import numpy as np, pandas as pd

from dcml import directional_changes, build_features
from run_walkforward import load_prices, walk_forward_predict, TRAIN_FRAC, COST_BPS
from backtest import daily_returns_from_trades, curve_metrics

PRICES = load_prices()


def market_state(prices):
    """Daily market-state series aligned to integer price index."""
    s = pd.Series(np.asarray(prices, dtype=float))
    ret = s.pct_change()
    vol20 = ret.rolling(20).std()
    ma200 = s.rolling(200).mean()
    p_ma = s / ma200 - 1.0
    mom20 = s / s.shift(20) - 1.0
    dd = s / s.cummax() - 1.0
    return pd.DataFrame({"vol20": vol20, "p_ma200": p_ma,
                         "mom20": mom20, "dd_peak": dd})


def run(theta=0.01):
    ctx_rows, reg_rows = [], []
    for name, p in PRICES.items():
        ev = directional_changes(p, theta)
        X, y, meta = build_features(ev)
        ms = market_state(p)
        # attach market state at each event's confirmation bar
        cx = ms.iloc[meta["conf_idx"].values].reset_index(drop=True)
        Xc = pd.concat([X, cx], axis=1)
        keep = Xc.notna().all(axis=1)
        Xc, yc = Xc[keep].reset_index(drop=True), y[keep].reset_index(drop=True)
        metac = meta[keep].reset_index(drop=True)
        conf_ctx = cx[keep].reset_index(drop=True)

        M = len(Xc); it = int(TRAIN_FRAC * M); oos = np.arange(it, M)
        s_i, e_i = int(metac.loc[it, "conf_idx"]), int(metac.loc[M-1, "exit_idx"])
        preds, _ = walk_forward_predict(Xc, yc, it)
        mask = preds == 1

        trades = [(int(metac.loc[k, "conf_idx"]), int(metac.loc[k, "exit_idx"]),
                   int(metac.loc[k, "direction"])) for k in oos[mask[oos]]]
        dr = daily_returns_from_trades(p, trades, s_i, e_i, COST_BPS)
        m = curve_metrics(dr)
        ctx_rows.append(dict(asset=name, n_trades=len(trades),
                             ann_return=m["ann_return"], sharpe=m["sharpe"],
                             max_dd=m["max_dd"]))

        # regime split of the OOS DC-ML trades: bull = price above 200d MA
        for regime, sel in [("Bull", conf_ctx["p_ma200"] > 0),
                            ("Bear", conf_ctx["p_ma200"] <= 0)]:
            idx = oos[mask[oos] & sel.values[oos]]
            net = metac.loc[idx, "trade_ret"].values - 2.0 * COST_BPS * 1e-4
            if len(net) == 0:
                continue
            # annualise the per-regime daily curve
            rtr = [(int(metac.loc[k, "conf_idx"]), int(metac.loc[k, "exit_idx"]),
                    int(metac.loc[k, "direction"])) for k in idx]
            drr = daily_returns_from_trades(p, rtr, s_i, e_i, COST_BPS)
            mm = curve_metrics(drr)
            reg_rows.append(dict(asset=name, regime=regime, n_trades=len(net),
                                 hit=float((net > 0).mean()),
                                 ann_return=mm["ann_return"], sharpe=mm["sharpe"]))

    ctx = pd.DataFrame(ctx_rows); ctx.to_csv("results/context_walkforward.csv", index=False)
    reg = pd.DataFrame(reg_rows); reg.to_csv("results/regime_split.csv", index=False)
    print("context (+market-state features), theta=1%:\n", ctx.to_string(index=False))
    print("\nregime split (DC-ML +context):\n", reg.to_string(index=False))


if __name__ == "__main__":
    run()
