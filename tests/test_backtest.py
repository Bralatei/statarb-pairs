import numpy as np
import pandas as pd

from statarb.backtest import Costs, backtest_pair, leg_weights
from statarb.kalman import kalman_hedge

NO_COSTS = Costs(0, 0, 0)


def _prices(n=50, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    y = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), idx)
    x = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), idx)
    return y, x


def test_gross_exposure_is_one():
    w = leg_weights(pd.Series([1.0, -1.0]), 1.5)
    assert np.allclose(w.abs().sum(axis=1), 1.0)


def test_positions_are_lagged_one_bar():
    """A signal on day t must earn day t+1's return, never day t's."""
    y, x = _prices()
    pos = pd.Series(0.0, index=y.index)
    pos.iloc[10] = 1.0
    bt = backtest_pair(y, x, pos, 1.0, NO_COSTS)
    assert bt["gross"].iloc[10] == 0.0
    expected = 0.5 * y.pct_change().iloc[11] - 0.5 * x.pct_change().iloc[11]
    assert np.isclose(bt["gross"].iloc[11], expected)


def test_lookahead_canary():
    """An oracle signal that peeks at tomorrow must NOT look profitable
    after the engine's lag... while the same oracle shifted correctly must."""
    y, x = _prices(n=500, seed=1)
    spread_ret = y.pct_change() - x.pct_change()
    cheat = np.sign(spread_ret).fillna(0)            # knows today's move
    honest = np.sign(spread_ret.shift(-1)).fillna(0)  # knows tomorrow's (true oracle)
    r_cheat = backtest_pair(y, x, cheat, 1.0, NO_COSTS)["gross"].sum()
    r_oracle = backtest_pair(y, x, honest, 1.0, NO_COSTS)["gross"].sum()
    assert r_oracle > 1.0                 # oracle with true future info wins big
    assert abs(r_cheat) < r_oracle / 5    # same-day info gives no edge after lag


def test_costs_reduce_returns_by_turnover():
    y, x = _prices()
    pos = pd.Series(0.0, index=y.index)
    pos.iloc[5:15] = 1.0
    c = Costs(commission_bps=5, slippage_bps=5, borrow_bps_annual=0)
    a = backtest_pair(y, x, pos, 1.0, NO_COSTS)
    b = backtest_pair(y, x, pos, 1.0, c)
    # enter (turnover 1) + exit (turnover 1) at 10bps each
    assert np.isclose((a["net"] - b["net"]).sum(), 2 * 10 / 1e4)


def test_kalman_tracks_beta_change():
    rng = np.random.default_rng(4)
    n = 1500
    # log-x must move enough for alpha and beta to be separately identifiable
    lx = pd.Series(np.cumsum(rng.normal(0, 0.03, n)) + 4)
    beta = np.where(np.arange(n) < n // 2, 0.8, 1.3)
    ly = pd.Series(beta * lx.values + rng.normal(0, 0.005, n))
    kf = kalman_hedge(ly, lx, delta=1e-4, obs_var=1e-4)
    assert abs(kf["beta"].iloc[n // 2 - 50] - 0.8) < 0.15
    assert abs(kf["beta"].iloc[-1] - 1.3) < 0.15
