"""Price data: loading, cleaning, caching, and a synthetic generator.

All downstream code expects a wide DataFrame of *adjusted close* prices:
index = trading dates (DatetimeIndex, ascending), columns = tickers.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load a wide price CSV (first column = date)."""
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return clean_prices(df)


def fetch_yfinance(
    tickers: list[str], start: str, end: str, cache: str | Path | None = None
) -> pd.DataFrame:
    """Download adjusted closes from Yahoo Finance, caching to CSV.

    Requires `pip install yfinance` and internet access. Cached data is
    reused so results are reproducible across runs.
    """
    if cache is not None and Path(cache).exists():
        return load_csv(cache)
    import yfinance as yf  # optional dependency

    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    prices = clean_prices(prices)
    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        prices.to_csv(cache)
    return prices


def clean_prices(df: pd.DataFrame, max_missing_frac: float = 0.05) -> pd.DataFrame:
    """Sort, drop tickers with too much missing data, forward-fill small gaps.

    We forward-fill (never back-fill): back-filling would leak future prices
    into the past.
    """
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df = df.loc[:, df.isna().mean() <= max_missing_frac]
    df = df.ffill(limit=5).dropna(how="any")
    df = df.loc[:, (df > 0).all()]
    return df.astype(float)


def synthetic_prices(
    n_days: int = 2520,
    n_pairs: int = 4,
    n_noise: int = 8,
    seed: int = 42,
    start: str = "2015-01-02",
) -> pd.DataFrame:
    """Generate a universe with known ground truth.

    - `n_pairs` cointegrated pairs: A follows a random walk, B = beta*A + OU spread.
    - `n_noise` independent random walks (should NOT be selected as pairs).

    Ground truth lets the tests check that pair selection finds real pairs
    and rejects spurious ones.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_days)
    cols: dict[str, np.ndarray] = {}
    for i in range(n_pairs):
        log_a = np.log(50) + np.cumsum(rng.normal(0.0002, 0.015, n_days))
        beta = rng.uniform(0.6, 1.4)
        theta = rng.uniform(0.03, 0.12)  # mean-reversion speed -> half-life 6-23 days
        sigma = 0.01
        spread = np.zeros(n_days)
        for t in range(1, n_days):
            spread[t] = spread[t - 1] * (1 - theta) + rng.normal(0, sigma)
        log_b = 0.2 + beta * log_a + spread
        cols[f"P{i}A"] = np.exp(log_a)
        cols[f"P{i}B"] = np.exp(log_b)
    for j in range(n_noise):
        cols[f"N{j}"] = np.exp(np.log(40) + np.cumsum(rng.normal(0.0002, 0.015, n_days)))
    return pd.DataFrame(cols, index=dates)
