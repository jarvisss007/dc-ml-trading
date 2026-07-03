"""
dcml.py -- Directional-Change (DC) event detection, indicators, feature/label
construction, and a simple cost model.

Reference definitions (Tsang 2010; Glattfelder, Dupuis & Olsen 2011):

A DC event is confirmed when price reverses by a fixed fraction ``theta`` from
the most recent local extreme. Between two consecutive extremes the trend is
split into the directional-change part (size ~theta) and a variable overshoot.

For each confirmed event ``k`` we record, normalised by ``theta`` so quantities
are comparable across thresholds:

    TMV_k = |p_ext_k - p_ext_{k-1}| / (p_ext_{k-1} * theta)   total move value
    T_k   = (bars from previous extreme to current extreme)   trend time
    OSV_k = |p_conf_k - p_ext_k| / (p_ext_k * theta)          overshoot value

where ``p_ext_k`` is the extreme (peak/trough) that closed trend ``k`` and
``p_conf_k`` is the price at the confirmation bar.

This module is deliberately dependency-light (numpy/pandas only) so every number
in the paper is reproducible with ``python run_walkforward.py``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# DC event detection
# ---------------------------------------------------------------------------
def directional_changes(prices, theta):
    """Detect directional-change events in a 1-D price series.

    Parameters
    ----------
    prices : array-like of float
    theta  : float, reversal threshold as a fraction (e.g. 0.01 for 1%)

    Returns
    -------
    pandas.DataFrame with one row per confirmed event, columns:
        ext_idx     integer index of the extreme that closed this trend
        ext_price   price at that extreme
        conf_idx    integer index of the confirmation bar
        conf_price  price at the confirmation bar
        direction   +1 if this event confirms an UPTURN (extreme was a trough),
                    -1 if it confirms a DOWNTURN (extreme was a peak)
        TMV, T, OSV normalised indicators for the just-completed trend
    """
    p = np.asarray(prices, dtype=float)
    n = len(p)
    if n < 2:
        return pd.DataFrame(
            columns=["ext_idx", "ext_price", "conf_idx", "conf_price",
                     "direction", "TMV", "T", "OSV"]
        )

    events = []          # confirmed events
    # Initialise mode from the first decisive move away from p[0].
    # mode = +1 : in an uptrend, tracking a running peak, waiting for a -theta
    #             move to confirm a DOWNTURN.
    # mode = -1 : in a downtrend, tracking a running trough, waiting for a
    #             +theta move to confirm an UPTURN.
    mode = 0
    ext_idx = 0
    ext_price = p[0]
    prev_ext_idx = 0
    prev_ext_price = p[0]

    for t in range(1, n):
        price = p[t]

        if mode == 0:
            # Bootstrap: first move of size theta in either direction sets mode.
            if price <= ext_price * (1.0 - theta):
                mode = -1                      # confirmed downturn
                _record(events, ext_idx, ext_price, t, price, -1,
                        prev_ext_idx, prev_ext_price, theta)
                prev_ext_idx, prev_ext_price = ext_idx, ext_price
                ext_idx, ext_price = t, price  # start tracking new trough
            elif price >= ext_price * (1.0 + theta):
                mode = +1                      # confirmed upturn
                _record(events, ext_idx, ext_price, t, price, +1,
                        prev_ext_idx, prev_ext_price, theta)
                prev_ext_idx, prev_ext_price = ext_idx, ext_price
                ext_idx, ext_price = t, price
            else:
                # extend the initial extreme toward whichever way price drifts
                if price > ext_price:
                    ext_idx, ext_price = t, price
                elif price < ext_price:
                    ext_idx, ext_price = t, price

        elif mode == +1:
            # uptrend: update running peak; confirm downturn on -theta reversal
            if price > ext_price:
                ext_idx, ext_price = t, price
            elif price <= ext_price * (1.0 - theta):
                mode = -1
                _record(events, ext_idx, ext_price, t, price, -1,
                        prev_ext_idx, prev_ext_price, theta)
                prev_ext_idx, prev_ext_price = ext_idx, ext_price
                ext_idx, ext_price = t, price

        else:  # mode == -1
            # downtrend: update running trough; confirm upturn on +theta reversal
            if price < ext_price:
                ext_idx, ext_price = t, price
            elif price >= ext_price * (1.0 + theta):
                mode = +1
                _record(events, ext_idx, ext_price, t, price, +1,
                        prev_ext_idx, prev_ext_price, theta)
                prev_ext_idx, prev_ext_price = ext_idx, ext_price
                ext_idx, ext_price = t, price

    df = pd.DataFrame(events)
    # First recorded event has no well-defined previous extreme -> drop NaN TMV.
    if len(df):
        df = df.dropna(subset=["TMV"]).reset_index(drop=True)
    return df


def _record(events, ext_idx, ext_price, conf_idx, conf_price, direction,
            prev_ext_idx, prev_ext_price, theta):
    """Append one confirmed event with normalised indicators."""
    if prev_ext_price and prev_ext_idx != ext_idx:
        tmv = abs(ext_price - prev_ext_price) / (prev_ext_price * theta)
        trend_time = float(ext_idx - prev_ext_idx)  # bars, prev extreme -> extreme
    else:
        tmv = np.nan
        trend_time = np.nan
    osv = abs(conf_price - ext_price) / (ext_price * theta)
    events.append(dict(
        ext_idx=int(ext_idx), ext_price=float(ext_price),
        conf_idx=int(conf_idx), conf_price=float(conf_price),
        direction=int(direction), TMV=tmv, T=trend_time, OSV=osv,
    ))


# ---------------------------------------------------------------------------
# Features and labels
# ---------------------------------------------------------------------------
def build_features(events, n_lags=2):
    """Build the per-event feature matrix and sign label.

    Features (all known at the confirmation bar of event k):
        TMV, OSV, log(T) for the current event and ``n_lags`` prior events,
        a short regime tilt (mean direction of the last few events),
        and the current overshoot (OSV).

    Label:
        sign of the trade return from this confirmation to the NEXT confirmation
        taken in the confirmed direction (+1 profitable, 0 otherwise).
    """
    ev = events.reset_index(drop=True).copy()
    ev["logT"] = np.log(ev["T"].clip(lower=1.0))

    # trade return: enter at conf_price[k] in direction[k], exit at conf_price[k+1]
    fwd = ev["conf_price"].shift(-1) / ev["conf_price"] - 1.0
    ev["trade_ret"] = ev["direction"] * fwd
    ev["label"] = (ev["trade_ret"] > 0).astype(int)
    ev["exit_idx"] = ev["conf_idx"].shift(-1)

    feat = {}
    for lag in range(0, n_lags + 1):
        feat[f"TMV_l{lag}"] = ev["TMV"].shift(lag)
        feat[f"OSV_l{lag}"] = ev["OSV"].shift(lag)
        feat[f"logT_l{lag}"] = ev["logT"].shift(lag)
    X = pd.DataFrame(feat)
    X["regime_tilt"] = ev["direction"].rolling(5, min_periods=1).mean().shift(1)
    X["cur_overshoot"] = ev["OSV"]
    X["direction"] = ev["direction"]  # available; model shows it's ~irrelevant

    keep = X.notna().all(axis=1) & ev["trade_ret"].notna()
    X = X[keep].reset_index(drop=True)
    y = ev.loc[keep, "label"].reset_index(drop=True)
    meta = ev.loc[keep, ["conf_idx", "exit_idx", "conf_price", "direction",
                         "trade_ret"]].reset_index(drop=True)
    meta["exit_idx"] = meta["exit_idx"].astype(int)
    return X, y, meta


FEATURE_COLS = None  # set on first build for reference


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------
def apply_costs(trade_ret, cost_bps_per_side=2.0):
    """Subtract round-trip cost (enter + exit) in basis points per side."""
    return trade_ret - 2.0 * cost_bps_per_side * 1e-4
