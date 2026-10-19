"""Walk-forward engine: re-select pairs on a rolling formation window, trade them
out-of-sample on the following trading window, repeat.

    |---- formation (select + fit) ----|-- trading (OOS) --|
                         |---- formation ----|-- trading --|
                                      ...

Every number reported comes from trading windows only, which the selection
step never saw.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from .backtest import Costs, backtest_pair, trade_list
from .kalman import kalman_hedge
from .pairs import select_pairs
from .signals import positions_from_z, rolling_zscore, spread_series


@dataclass
class Config:
    formation_days: int = 252
    trading_days: int = 63
    # selection
    min_corr: float = 0.8
    pvalue_alpha: float = 0.05
    correction: str = "bh"
    min_half_life: float = 2.0
    max_half_life: float = 60.0
    max_pairs: int = 10
    # signals
    hedge_method: str = "ols"        # "ols" (static per window) or "kalman"
    z_window: int = 60
    entry_z: float = 2.0
    exit_z: float = 0.5
    stop_z: float = 4.0
    max_hold_mult: float | None = 3.0  # max hold = mult * half-life (None = off)
    kalman_delta: float = 1e-5
    costs: Costs = field(default_factory=Costs)

    @classmethod
    def from_dict(cls, d: dict) -> Config:
        d = dict(d)
        costs = Costs(**d.pop("costs", {}))
        return cls(costs=costs, **d)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class WalkForwardResult:
    returns: pd.Series            # portfolio net daily returns (OOS only)
    gross: pd.Series
    turnover: pd.Series
    n_active: pd.Series           # number of pairs with open positions
    pair_log: pd.DataFrame        # which pairs were selected each period
    trades: pd.DataFrame
    pair_returns: dict = field(default_factory=dict)


def _trade_one_pair(prices, pair, hist_start, trade_start, trade_end, cfg: Config):
    window = prices.iloc[hist_start:trade_end]
    ly, lx = np.log(window[pair.y]), np.log(window[pair.x])

    if cfg.hedge_method == "kalman":
        kf = kalman_hedge(ly, lx, delta=cfg.kalman_delta,
                          init_beta=pair.hedge_ratio, init_alpha=pair.intercept)
        beta = kf["beta"]
        spread = ly - kf["alpha"] - beta * lx
    else:
        beta = pair.hedge_ratio
        spread = spread_series(ly, lx, pair.intercept, beta)

    z = rolling_zscore(spread, cfg.z_window)
    max_hold = (int(np.ceil(cfg.max_hold_mult * pair.half_life))
                if cfg.max_hold_mult else None)
    pos = positions_from_z(z, cfg.entry_z, cfg.exit_z, cfg.stop_z, max_hold)

    # Only trade inside the OOS window; force flat on the final bar's decision
    # so the exit cost is booked inside this window.
    n_hist = trade_start - hist_start
    pos.iloc[:n_hist] = 0.0
    pos.iloc[-1] = 0.0

    bt = backtest_pair(window[pair.y], window[pair.x], pos, beta, cfg.costs)
    bt = bt.iloc[n_hist:]
    trades = trade_list(pos.iloc[n_hist:], bt["net"])
    return bt, pos.iloc[n_hist:], trades


def run_walk_forward(prices: pd.DataFrame, cfg: Config) -> WalkForwardResult:
    n = len(prices)
    F, T = cfg.formation_days, cfg.trading_days
    if n < F + T:
        raise ValueError(f"need at least {F + T} rows, got {n}")

    net_parts, gross_parts, to_parts, active_parts = [], [], [], []
    log_rows, all_trades, pair_rets = [], [], {}

    for start in range(0, n - F - T + 1, T):
        f_end = start + F
        t_end = min(f_end + T, n)
        formation = prices.iloc[start:f_end]
        pairs = select_pairs(
            formation, cfg.min_corr, cfg.pvalue_alpha, cfg.correction,
            cfg.min_half_life, cfg.max_half_life, cfg.max_pairs,
        )
        idx = prices.index[f_end:t_end]
        for p in pairs:
            log_rows.append({"formation_end": prices.index[f_end - 1], **p.__dict__})

        if not pairs:
            zeros = pd.Series(0.0, index=idx)
            net_parts.append(zeros); gross_parts.append(zeros)
            to_parts.append(zeros); active_parts.append(zeros)
            continue

        hist_start = max(0, f_end - cfg.z_window)  # warm-up for the z-score
        nets, grosses, tos, acts = [], [], [], []
        for p in pairs:
            bt, pos, trades = _trade_one_pair(prices, p, hist_start, f_end, t_end, cfg)
            key = f"{p.y}/{p.x}"
            nets.append(bt["net"]); grosses.append(bt["gross"])
            tos.append(bt["turnover"]); acts.append((pos.shift(1).fillna(0) != 0).astype(float))
            pair_rets.setdefault(key, []).append(bt["net"])
            if len(trades):
                trades.insert(0, "pair", key)
                all_trades.append(trades)

        # Capital split equally across the pairs selected this period.
        k = len(pairs)
        net_parts.append(sum(nets) / k)
        gross_parts.append(sum(grosses) / k)
        to_parts.append(sum(tos) / k)
        active_parts.append(sum(acts))

    return WalkForwardResult(
        returns=pd.concat(net_parts).rename("net"),
        gross=pd.concat(gross_parts).rename("gross"),
        turnover=pd.concat(to_parts).rename("turnover"),
        n_active=pd.concat(active_parts).rename("n_active"),
        pair_log=pd.DataFrame(log_rows),
        trades=pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame(),
        pair_returns={k: pd.concat(v) for k, v in pair_rets.items()},
    )
