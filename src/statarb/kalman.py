"""Kalman filter for a time-varying hedge ratio.

State: [alpha_t, beta_t] follows a random walk.
Observation: log_y_t = alpha_t + beta_t * log_x_t + e_t.

Each estimate at time t uses only data up to and including t (it is a filter,
not a smoother), so it is safe for backtesting when signals are lagged a bar.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def kalman_hedge(
    log_y: pd.Series,
    log_x: pd.Series,
    delta: float = 1e-5,
    obs_var: float = 1e-3,
    init_beta: float | None = None,
    init_alpha: float = 0.0,
) -> pd.DataFrame:
    """Return DataFrame with columns alpha, beta, spread (one-step prediction error)."""
    y = log_y.values
    x = log_x.values
    n = len(y)
    state = np.array([init_alpha, init_beta if init_beta is not None else 1.0])
    P = np.eye(2)
    Q = delta / (1 - delta) * np.eye(2)   # state noise
    out = np.zeros((n, 3))
    for t in range(n):
        P = P + Q                          # predict
        H = np.array([1.0, x[t]])
        y_hat = H @ state
        err = y[t] - y_hat                 # innovation, uses state from t-1
        S = H @ P @ H + obs_var
        K = P @ H / S
        out[t] = (state[0], state[1], err)  # record prior estimate -> no lookahead
        state = state + K * err            # update
        P = P - np.outer(K, H) @ P
    return pd.DataFrame(out, index=log_y.index, columns=["alpha", "beta", "spread"])
