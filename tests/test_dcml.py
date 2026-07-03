"""
Smoke tests: the paper's anchor numbers must reproduce exactly from the
bundled data. If any of these fail, the repository no longer backs the paper.

Run: pytest -q
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import arch.data.nasdaq
import arch.data.sp500
import arch.data.wti

from backtest import curve_metrics, stationary_bootstrap_pvalue
from dcml import build_features, directional_changes


def _prices():
    return {
        "sp500": arch.data.sp500.load()["Adj Close"].dropna().values,
        "nasdaq": arch.data.nasdaq.load()["Adj Close"].dropna().values,
        "wti": arch.data.wti.load()["DCOILWTICO"].dropna().values,
    }


def test_event_count_anchors():
    """DC event counts reported in the paper (Table A) reproduce exactly."""
    p = _prices()
    assert len(directional_changes(p["wti"], 0.005)) == 3286
    assert len(directional_changes(p["wti"], 0.03)) == 1184
    assert len(directional_changes(p["nasdaq"], 0.0075)) == 1350
    assert len(directional_changes(p["nasdaq"], 0.03)) == 439
    assert len(directional_changes(p["sp500"], 0.01)) == 968


def test_scaling_law_directions():
    """Frequency falls, duration rises, normalised overshoot falls with theta."""
    p = _prices()["nasdaq"]
    lo, hi = directional_changes(p, 0.0075), directional_changes(p, 0.03)
    assert len(lo) > len(hi)
    assert lo["T"].mean() < hi["T"].mean()
    assert lo["OSV"].mean() > hi["OSV"].mean()


def test_features_no_lookahead_shape():
    """Feature builder returns aligned X/y/meta with expected columns."""
    p = _prices()["sp500"]
    events = directional_changes(p, 0.01)
    X, y, meta = build_features(events)
    assert len(X) == len(y) == len(meta)
    assert {"TMV_l0", "OSV_l0", "regime_tilt", "direction"} <= set(X.columns)
    # label is derived from the NEXT event; last usable row must have exit_idx
    assert (meta["exit_idx"] > meta["conf_idx"]).all()


def test_bootstrap_pvalue_sanity():
    """Clearly positive-mean series -> small p; zero-mean -> large p."""
    rng = np.random.default_rng(1)
    pos = rng.normal(0.5, 1.0, 400)
    zero = rng.normal(0.0, 1.0, 400)
    assert stationary_bootstrap_pvalue(pos, n_boot=500, seed=1) < 0.05
    assert stationary_bootstrap_pvalue(zero, n_boot=500, seed=1) > 0.05


def test_sharpe_annualised_from_daily():
    """curve_metrics annualises from daily returns (the honest way)."""
    rng = np.random.default_rng(2)
    r = rng.normal(0.0004, 0.01, 2520)  # ~10 years of daily returns
    m = curve_metrics(r)
    expected = r.mean() / r.std(ddof=1) * np.sqrt(252)
    assert abs(m["sharpe"] - expected) < 1e-9
    assert -1.0 <= m["max_dd"] <= 0.0
