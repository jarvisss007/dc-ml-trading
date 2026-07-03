"""
extra_analysis.py -- reproduces the supporting results:
  * scaling-law event statistics       -> results/scaling_laws.csv
  * learner comparison (logit/RF/GB)    -> results/model_comparison.csv
  * DC-ML feature importance            -> results/feature_importance.csv
  * cost sensitivity (DC-ML)            -> results/cost_sensitivity.csv
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

import arch.data.sp500, arch.data.nasdaq, arch.data.wti
from dcml import directional_changes, build_features
from run_walkforward import load_prices, walk_forward_predict, TRAIN_FRAC
from backtest import daily_returns_from_trades, curve_metrics

PRICES = load_prices()


def scaling_laws():
    rows = []
    for name, p in PRICES.items():
        for th in [0.005, 0.0075, 0.01, 0.02, 0.03]:
            ev = directional_changes(p, th)
            rows.append(dict(asset=name, theta=th, n_events=len(ev),
                             mean_T=ev["T"].mean(), mean_OSV=ev["OSV"].mean(),
                             mean_TMV=ev["TMV"].mean()))
    df = pd.DataFrame(rows); df.to_csv("results/scaling_laws.csv", index=False)
    print("scaling_laws.csv\n", df.to_string(index=False), "\n")


def model_comparison(theta=0.01):
    rows = []
    models = {
        "Logistic": lambda: LogisticRegression(max_iter=1000),
        "RandomForest": lambda: RandomForestClassifier(n_estimators=200, random_state=0),
        "GradBoost": lambda: GradientBoostingClassifier(random_state=0),
    }
    for name, p in PRICES.items():
        ev = directional_changes(p, theta)
        X, y, meta = build_features(ev)
        M = len(X); it = int(TRAIN_FRAC * M); oos = np.arange(it, M)
        s, e = int(meta.loc[it, "conf_idx"]), int(meta.loc[M-1, "exit_idx"])
        for mname, mk in models.items():
            preds = np.full(M, -1)
            i = it
            while i < M:
                j = min(i+30, M)
                if len(np.unique(y.values[:i])) < 2:
                    preds[i:j] = 1
                else:
                    clf = mk(); clf.fit(X.values[:i], y.values[:i])
                    preds[i:j] = clf.predict(X.values[i:j])
                i = j
            mask = preds == 1
            trades = [(int(meta.loc[k,"conf_idx"]), int(meta.loc[k,"exit_idx"]),
                       int(meta.loc[k,"direction"])) for k in oos[mask[oos]]]
            dr = daily_returns_from_trades(p, trades, s, e)
            m = curve_metrics(dr)
            rows.append(dict(asset=name, model=mname, n_trades=len(trades),
                             ann_return=m["ann_return"], sharpe=m["sharpe"]))
    df = pd.DataFrame(rows); df.to_csv("results/model_comparison.csv", index=False)
    print("model_comparison.csv\n", df.to_string(index=False), "\n")


def feature_importance(theta=0.01):
    imps = []
    for name, p in PRICES.items():
        ev = directional_changes(p, theta)
        X, y, meta = build_features(ev)
        clf = GradientBoostingClassifier(random_state=0).fit(X.values, y.values)
        imps.append(pd.Series(clf.feature_importances_, index=X.columns))
    mean_imp = pd.concat(imps, axis=1).mean(axis=1).sort_values(ascending=False)
    mean_imp.to_csv("results/feature_importance.csv", header=["importance"])
    print("feature_importance.csv (mean across 3 assets, theta=1%)\n",
          mean_imp.to_string(), "\n")


def cost_sensitivity(theta=0.01):
    rows = []
    for name, p in PRICES.items():
        ev = directional_changes(p, theta)
        X, y, meta = build_features(ev)
        M = len(X); it = int(TRAIN_FRAC*M); oos = np.arange(it, M)
        s, e = int(meta.loc[it,"conf_idx"]), int(meta.loc[M-1,"exit_idx"])
        preds, _ = walk_forward_predict(X, y, it)
        mask = preds == 1
        trades = [(int(meta.loc[k,"conf_idx"]), int(meta.loc[k,"exit_idx"]),
                   int(meta.loc[k,"direction"])) for k in oos[mask[oos]]]
        for cbps in [0, 1, 2, 5, 10]:
            dr = daily_returns_from_trades(p, trades, s, e, cost_bps_per_side=cbps)
            rows.append(dict(asset=name, cost_bps_per_side=cbps,
                             ann_return=curve_metrics(dr)["ann_return"]))
    df = pd.DataFrame(rows); df.to_csv("results/cost_sensitivity.csv", index=False)
    print("cost_sensitivity.csv\n", df.pivot(index="asset",
          columns="cost_bps_per_side", values="ann_return").to_string(), "\n")


if __name__ == "__main__":
    scaling_laws()
    feature_importance()
    model_comparison()
    cost_sensitivity()
