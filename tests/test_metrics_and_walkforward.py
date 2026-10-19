import numpy as np
import pandas as pd
import pytest

from statarb import metrics
from statarb.data import synthetic_prices
from statarb.walkforward import Config, run_walk_forward


def test_sharpe_known_value():
    r = pd.Series([0.01, -0.01] * 126)
    assert metrics.sharpe(r) == pytest.approx(0.0, abs=1e-9)
    r2 = pd.Series(np.random.default_rng(0).normal(0.001, 0.01, 100_000))
    assert metrics.sharpe(r2) == pytest.approx(0.1 * np.sqrt(252), rel=0.05)


def test_max_drawdown():
    r = pd.Series([0.1, -0.5, 0.2])
    assert metrics.max_drawdown(r) == pytest.approx(-0.5)


def test_walk_forward_is_out_of_sample_only():
    prices = synthetic_prices(n_days=800, n_pairs=2, n_noise=2, seed=5)
    cfg = Config(formation_days=252, trading_days=63, min_corr=0.3, max_pairs=3)
    res = run_walk_forward(prices, cfg)
    assert res.returns.index[0] == prices.index[252]  # first OOS day
    assert res.returns.index.is_unique


def test_future_data_does_not_change_past_results():
    """Corrupting prices after day D must leave all returns before the
    trading window containing D unchanged."""
    prices = synthetic_prices(n_days=900, n_pairs=2, n_noise=2, seed=9)
    cfg = Config(formation_days=252, trading_days=63, min_corr=0.3, max_pairs=3)
    base = run_walk_forward(prices, cfg).returns
    cut = 252 + 63 * 5                                  # start of 6th trading window
    bad = prices.copy()
    bad.iloc[cut:] *= np.exp(np.random.default_rng(0).normal(0, 0.2, bad.iloc[cut:].shape))
    corrupted = run_walk_forward(bad, cfg).returns
    pd.testing.assert_series_equal(base.iloc[: cut - 252], corrupted.iloc[: cut - 252])


def test_no_edge_on_pure_noise():
    """On a universe of independent random walks, net Sharpe should be ~0 or negative."""
    prices = synthetic_prices(n_days=1500, n_pairs=0, n_noise=12, seed=11)
    cfg = Config(min_corr=0.5, correction="bh")
    res = run_walk_forward(prices, cfg)
    assert metrics.sharpe(res.returns) < 0.75
