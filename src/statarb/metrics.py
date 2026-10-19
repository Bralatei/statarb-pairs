"""Performance and robustness statistics."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

TRADING_DAYS = 252


def sharpe(r: pd.Series, rf: float = 0.0) -> float:
    r = r.dropna() - rf / TRADING_DAYS
    sd = r.std(ddof=1)
    return float(np.sqrt(TRADING_DAYS) * r.mean() / sd) if sd > 0 else 0.0


def sortino(r: pd.Series) -> float:
    r = r.dropna()
    downside = np.sqrt((np.minimum(r, 0) ** 2).mean())
    return float(np.sqrt(TRADING_DAYS) * r.mean() / downside) if downside > 0 else 0.0


def equity_curve(r: pd.Series) -> pd.Series:
    return (1 + r.fillna(0)).cumprod()


def max_drawdown(r: pd.Series) -> float:
    eq = equity_curve(r)
    return float((eq / eq.cummax() - 1).min())


def cagr(r: pd.Series) -> float:
    eq = equity_curve(r)
    years = len(r) / TRADING_DAYS
    return float(eq.iloc[-1] ** (1 / years) - 1) if years > 0 else 0.0


def calmar(r: pd.Series) -> float:
    mdd = abs(max_drawdown(r))
    return cagr(r) / mdd if mdd > 0 else 0.0


def bootstrap_sharpe_ci(
    r: pd.Series, n_boot: int = 2000, block: int = 20, ci: float = 0.95, seed: int = 0
) -> tuple[float, float]:
    """Stationary-block bootstrap CI for the Sharpe ratio.

    Blocks preserve short-range autocorrelation that an i.i.d. bootstrap destroys.
    """
    x = r.dropna().values
    n = len(x)
    if n < block * 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    n_blocks = int(np.ceil(n / block))
    for i in range(n_boot):
        starts = rng.integers(0, n - block, n_blocks)
        sample = np.concatenate([x[s : s + block] for s in starts])[:n]
        sd = sample.std(ddof=1)
        out[i] = np.sqrt(TRADING_DAYS) * sample.mean() / sd if sd > 0 else 0.0
    lo, hi = np.quantile(out, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return float(lo), float(hi)


def probabilistic_sharpe(r: pd.Series, benchmark_sr: float = 0.0) -> float:
    """Bailey & Lopez de Prado PSR: P(true SR > benchmark) given skew/kurtosis.

    Works on per-period (daily) Sharpe internally.
    """
    x = r.dropna()
    n = len(x)
    if n < 3 or x.std() == 0:
        return np.nan
    sr = x.mean() / x.std(ddof=1)
    bench = benchmark_sr / np.sqrt(TRADING_DAYS)
    skew = stats.skew(x)
    kurt = stats.kurtosis(x, fisher=False)
    denom = np.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr**2)
    return float(stats.norm.cdf((sr - bench) * np.sqrt(n - 1) / denom))


def summary(r: pd.Series, turnover: pd.Series | None = None) -> dict[str, float]:
    lo, hi = bootstrap_sharpe_ci(r)
    out = {
        "CAGR": cagr(r),
        "AnnVol": float(r.std(ddof=1) * np.sqrt(TRADING_DAYS)),
        "Sharpe": sharpe(r),
        "Sharpe95_lo": lo,
        "Sharpe95_hi": hi,
        "PSR(SR>0)": probabilistic_sharpe(r),
        "Sortino": sortino(r),
        "MaxDD": max_drawdown(r),
        "Calmar": calmar(r),
        "HitRate(days)": float((r[r != 0] > 0).mean()) if (r != 0).any() else np.nan,
    }
    if turnover is not None:
        out["AnnTurnover"] = float(turnover.mean() * TRADING_DAYS)
    return out
