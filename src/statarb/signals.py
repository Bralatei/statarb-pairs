"""Spread z-scores and a stateful entry/exit/stop rule.

Convention: position = +1 means LONG the spread (long y, short beta*x);
position = -1 means SHORT the spread.

The signal at bar t uses information up to the close of t. The backtester
applies it from bar t+1, which is what prevents lookahead bias.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def spread_series(log_y: pd.Series, log_x: pd.Series, alpha: float, beta: float) -> pd.Series:
    return log_y - alpha - beta * log_x


def rolling_zscore(spread: pd.Series, window: int) -> pd.Series:
    """Z-score with a trailing window (includes bar t, never later bars)."""
    mu = spread.rolling(window, min_periods=window).mean()
    sd = spread.rolling(window, min_periods=window).std(ddof=1)
    return (spread - mu) / sd.replace(0, np.nan)


def positions_from_z(
    z: pd.Series,
    entry: float = 2.0,
    exit: float = 0.5,
    stop: float = 4.0,
    max_hold: int | None = None,
) -> pd.Series:
    """State machine: enter at |z|>=entry, exit at |z|<=exit, stop at |z|>=stop.

    After a stop-out we wait until |z| falls back below `entry` before
    re-entering, otherwise we would immediately re-enter a broken spread.
    """
    zv = z.values
    pos = np.zeros(len(zv))
    cur, held, cooling = 0, 0, False
    for t, zt in enumerate(zv):
        if np.isnan(zt):
            cur, held = 0, 0
        else:
            if cooling and abs(zt) < entry:
                cooling = False
            if cur == 0:
                if not cooling and entry <= abs(zt) < stop:
                    cur = -1 if zt > 0 else 1   # spread rich -> short it
                    held = 0
            else:
                held += 1
                timed_out = max_hold is not None and held >= max_hold
                if abs(zt) >= stop or timed_out:
                    cur, cooling = 0, True
                elif (cur == -1 and zt <= exit) or (cur == 1 and zt >= -exit):
                    cur = 0
        pos[t] = cur
    return pd.Series(pos, index=z.index, name="position")
