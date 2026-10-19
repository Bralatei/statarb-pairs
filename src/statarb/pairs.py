"""Pair selection: correlation pre-filter -> Engle-Granger cointegration ->
multiple-testing correction -> half-life filter.

Everything here is computed on the *formation* window only.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint


@dataclass(frozen=True)
class Pair:
    y: str               # dependent leg
    x: str               # independent leg
    hedge_ratio: float   # beta in log(y) = alpha + beta*log(x) + eps
    intercept: float
    pvalue: float        # Engle-Granger p-value
    half_life: float     # days


def hedge_ratio(log_y: pd.Series, log_x: pd.Series) -> tuple[float, float]:
    """OLS of log_y on log_x. Returns (alpha, beta)."""
    res = sm.OLS(log_y.values, sm.add_constant(log_x.values)).fit()
    return float(res.params[0]), float(res.params[1])


def half_life(spread: pd.Series) -> float:
    """Half-life of mean reversion from an AR(1)/discrete OU fit.

    d(s_t) = lambda * s_{t-1} + c + e  ->  half-life = -ln(2)/lambda.
    Returns inf if the spread is not mean-reverting (lambda >= 0).
    """
    s = spread.dropna()
    lag = s.shift(1).iloc[1:]
    delta = s.diff().iloc[1:]
    lam = sm.OLS(delta.values, sm.add_constant(lag.values)).fit().params[1]
    return float(-np.log(2) / lam) if lam < 0 else float("inf")


def benjamini_hochberg(pvalues: np.ndarray, alpha: float) -> np.ndarray:
    """Boolean mask of discoveries under BH false-discovery-rate control."""
    p = np.asarray(pvalues)
    n = len(p)
    if n == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(p)
    thresh = alpha * np.arange(1, n + 1) / n
    passed = p[order] <= thresh
    keep = np.zeros(n, dtype=bool)
    if passed.any():
        k = np.max(np.where(passed)[0])
        keep[order[: k + 1]] = True
    return keep


def select_pairs(
    prices: pd.DataFrame,
    min_corr: float = 0.8,
    pvalue_alpha: float = 0.05,
    correction: str = "bh",          # "bh", "bonferroni", or "none"
    min_half_life: float = 2.0,
    max_half_life: float = 60.0,
    max_pairs: int = 10,
) -> list[Pair]:
    """Select tradable pairs from a formation-window price panel."""
    logp = np.log(prices)
    rets_corr = logp.diff().corr()
    level_corr = logp.corr()

    candidates = []
    for a, b in combinations(prices.columns, 2):
        # Cheap pre-filter on log-price correlation; also require positively
        # correlated returns so the pair shares an economic driver.
        if level_corr.loc[a, b] < min_corr or rets_corr.loc[a, b] <= 0:
            continue
        # Engle-Granger is asymmetric; test both orderings, keep the better one.
        best = None
        for y, x in ((a, b), (b, a)):
            _, p, _ = coint(logp[y], logp[x], trend="c")
            if best is None or p < best[2]:
                best = (y, x, p)
        candidates.append(best)

    if not candidates:
        return []

    pvals = np.array([c[2] for c in candidates])
    if correction == "bh":
        keep = benjamini_hochberg(pvals, pvalue_alpha)
    elif correction == "bonferroni":
        keep = pvals <= pvalue_alpha / len(pvals)
    else:
        keep = pvals <= pvalue_alpha

    selected: list[Pair] = []
    used: set[str] = set()
    for (y, x, p), k in sorted(zip(candidates, keep), key=lambda t: t[0][2]):
        if not k or y in used or x in used:   # each ticker in at most one pair
            continue
        alpha_, beta = hedge_ratio(logp[y], logp[x])
        if beta <= 0:
            continue
        hl = half_life(logp[y] - alpha_ - beta * logp[x])
        if not (min_half_life <= hl <= max_half_life):
            continue
        selected.append(Pair(y, x, beta, alpha_, float(p), hl))
        used.update((y, x))
        if len(selected) >= max_pairs:
            break
    return selected
