"""Pair-level backtest with realistic frictions.

Accounting: each pair gets 1 unit of gross capital. A spread position of +1
holds  w_y = +1/(1+|beta|)  in y and  w_x = -beta/(1+|beta|)  in x, so gross
exposure is 1. Daily P&L = w_y * r_y + w_x * r_x on simple returns.

Timing (the most important line in this file):
    weights used on day t = target weights decided at close of day t-1
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Costs:
    commission_bps: float = 1.0   # per side, per unit traded notional
    slippage_bps: float = 2.0     # half-spread + impact proxy
    borrow_bps_annual: float = 50.0  # cost to hold the short leg

    @property
    def trade_bps(self) -> float:
        return self.commission_bps + self.slippage_bps


def leg_weights(position: pd.Series, beta: pd.Series | float) -> pd.DataFrame:
    beta = pd.Series(beta, index=position.index) if np.isscalar(beta) else beta
    denom = 1.0 + beta.abs()
    return pd.DataFrame({"y": position / denom, "x": -position * beta / denom})


def backtest_pair(
    price_y: pd.Series,
    price_x: pd.Series,
    position: pd.Series,
    beta: pd.Series | float,
    costs: Costs | None = None,
) -> pd.DataFrame:
    """Return daily DataFrame: gross, cost, net returns, turnover, weights."""
    costs = costs or Costs()
    target = leg_weights(position, beta)
    held = target.shift(1).fillna(0.0)          # <- no lookahead

    rets = pd.DataFrame({"y": price_y.pct_change(), "x": price_x.pct_change()}).fillna(0.0)
    gross = (held * rets).sum(axis=1)

    # Turnover: change in held weights, accounting for drift is second-order; ignored.
    turnover = held.diff().abs().sum(axis=1).fillna(held.abs().sum(axis=1))
    trade_cost = turnover * costs.trade_bps / 1e4
    short_notional = held.clip(upper=0).abs().sum(axis=1)
    borrow_cost = short_notional * costs.borrow_bps_annual / 1e4 / 252

    net = gross - trade_cost - borrow_cost
    return pd.DataFrame(
        {
            "gross": gross,
            "cost": trade_cost + borrow_cost,
            "net": net,
            "turnover": turnover,
            "w_y": held["y"],
            "w_x": held["x"],
        }
    )


def trade_list(position: pd.Series, net: pd.Series) -> pd.DataFrame:
    """Round-trip trades: entry date, exit date, side, holding days, return."""
    held = position.shift(1).fillna(0)
    trades = []
    start, side = None, 0
    for t, p in held.items():
        if side == 0 and p != 0:
            start, side = t, p
        elif side != 0 and p != side:
            seg = net.loc[start:t].iloc[:-1]
            trades.append((start, t, side, len(seg), float((1 + seg).prod() - 1)))
            start, side = (t, p) if p != 0 else (None, 0)
    if side != 0:
        seg = net.loc[start:]
        trades.append((start, net.index[-1], side, len(seg), float((1 + seg).prod() - 1)))
    return pd.DataFrame(trades, columns=["entry", "exit", "side", "days", "ret"])
