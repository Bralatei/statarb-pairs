import numpy as np
import pandas as pd

from statarb.signals import positions_from_z, rolling_zscore


def test_entry_exit_cycle():
    z = pd.Series([0, 1, 2.1, 1.5, 0.4, 0, -2.2, -1, -0.3])
    pos = positions_from_z(z, entry=2, exit=0.5, stop=4).tolist()
    assert pos == [0, 0, -1, -1, 0, 0, 1, 1, 0]


def test_stop_loss_then_cooldown():
    z = pd.Series([2.1, 3.0, 4.5, 3.0, 2.5, 1.0, 2.2])
    pos = positions_from_z(z, entry=2, exit=0.5, stop=4).tolist()
    # stopped at 4.5, no re-entry at 3.0/2.5 until z drops below entry first
    assert pos == [-1, -1, 0, 0, 0, 0, -1]


def test_no_entry_beyond_stop():
    assert positions_from_z(pd.Series([5.0, 4.5]), 2, 0.5, 4).tolist() == [0, 0]


def test_max_hold():
    z = pd.Series([2.5] + [1.5] * 10)
    pos = positions_from_z(z, 2, 0.5, 4, max_hold=3)
    assert pos.iloc[3] == 0 and pos.iloc[:3].tolist() == [-1, -1, -1]


def test_zscore_is_causal():
    """Changing future values must not change past z-scores."""
    rng = np.random.default_rng(0)
    s = pd.Series(rng.normal(size=300))
    z1 = rolling_zscore(s, 30)
    s2 = s.copy()
    s2.iloc[200:] += 100
    z2 = rolling_zscore(s2, 30)
    pd.testing.assert_series_equal(z1.iloc[:200], z2.iloc[:200])
