"""Command line entry point.

    statarb run --config configs/synthetic.yaml
    statarb run --config configs/etfs.yaml          # needs yfinance + internet
    statarb sensitivity --config configs/synthetic.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import data, metrics
from .report import write_report
from .walkforward import Config, run_walk_forward


def _load(cfg_path: str):
    raw = yaml.safe_load(Path(cfg_path).read_text())
    d = raw["data"]
    if d["source"] == "synthetic":
        prices = data.synthetic_prices(**d.get("params", {}))
    elif d["source"] == "csv":
        prices = data.load_csv(d["path"])
    elif d["source"] == "yfinance":
        prices = data.fetch_yfinance(d["tickers"], d["start"], d["end"], d.get("cache"))
    else:
        raise ValueError(f"unknown data source {d['source']}")
    bench = None
    if d.get("benchmark") and d["benchmark"] in prices:
        bench = prices[d["benchmark"]].pct_change()
        prices = prices.drop(columns=d["benchmark"])
    return prices, bench, Config.from_dict(raw.get("strategy", {})), raw.get("output", "results")


def cmd_run(args):
    prices, bench, cfg, out = _load(args.config)
    print(f"Universe: {prices.shape[1]} tickers, {len(prices)} days "
          f"({prices.index[0].date()} to {prices.index[-1].date()})")
    res = run_walk_forward(prices, cfg)
    stats = write_report(res, cfg, out, bench)
    print(json.dumps(stats["net"], indent=2, default=float))
    print(f"Report written to {out}/")


def cmd_sensitivity(args):
    """Sharpe across a grid of entry/exit thresholds.

    This is a robustness check, NOT an optimizer: we want a broad plateau,
    not the single best cell (picking the max is overfitting).
    """
    prices, _, cfg, out = _load(args.config)
    entries = [1.5, 1.75, 2.0, 2.25, 2.5]
    exits = [0.0, 0.25, 0.5, 0.75, 1.0]
    grid = pd.DataFrame(index=entries, columns=exits, dtype=float)
    for e in entries:
        for x in exits:
            cfg.entry_z, cfg.exit_z = e, x
            grid.loc[e, x] = metrics.sharpe(run_walk_forward(prices, cfg).returns)
    grid.index.name, grid.columns.name = "entry_z", "exit_z"
    Path(out).mkdir(parents=True, exist_ok=True)
    grid.to_csv(Path(out) / "sensitivity.csv")
    print(grid.round(2).to_string())
    print(f"\nmedian {np.nanmedian(grid.values):.2f}, min {np.nanmin(grid.values):.2f}, "
          f"max {np.nanmax(grid.values):.2f}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="statarb")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("run", cmd_run), ("sensitivity", cmd_sensitivity)):
        p = sub.add_parser(name)
        p.add_argument("--config", required=True)
        p.set_defaults(func=fn)
    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
