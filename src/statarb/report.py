"""Write a results folder: metrics, plots, trades, and a Markdown summary."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from . import metrics
from .walkforward import Config, WalkForwardResult


def write_report(res: WalkForwardResult, cfg: Config, out_dir: str | Path,
                 benchmark: pd.Series | None = None) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    net_stats = metrics.summary(res.returns, res.turnover)
    gross_stats = metrics.summary(res.gross)
    stats = {"net": net_stats, "gross": gross_stats}
    if benchmark is not None:
        b = benchmark.reindex(res.returns.index).fillna(0)
        stats["benchmark"] = metrics.summary(b)
        stats["corr_to_benchmark"] = float(res.returns.corr(b))
    if len(res.trades):
        t = res.trades
        stats["trades"] = {
            "count": len(t),
            "win_rate": float((t["ret"] > 0).mean()),
            "avg_ret": float(t["ret"].mean()),
            "avg_days": float(t["days"].mean()),
        }

    (out / "metrics.json").write_text(json.dumps(stats, indent=2, default=float))
    (out / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2, default=str))
    res.trades.to_csv(out / "trades.csv", index=False)
    res.pair_log.to_csv(out / "pairs_selected.csv", index=False)
    pd.DataFrame({"net": res.returns, "gross": res.gross,
                  "turnover": res.turnover}).to_csv(out / "daily_returns.csv")

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1.3, 1]})
    eq_net = metrics.equity_curve(res.returns)
    eq_gross = metrics.equity_curve(res.gross)
    axes[0].plot(eq_gross.index, eq_gross, color="#9aa5b1", lw=1.2, label="Gross")
    axes[0].plot(eq_net.index, eq_net, color="#1f5fa8", lw=1.8, label="Net of costs")
    if benchmark is not None:
        eq_b = metrics.equity_curve(benchmark.reindex(res.returns.index).fillna(0))
        axes[0].plot(eq_b.index, eq_b, color="#c47f2c", lw=1.0, alpha=0.8, label="Benchmark")
    axes[0].set_ylabel("Growth of $1")
    axes[0].set_title(f"Walk-forward pairs strategy  |  net Sharpe {net_stats['Sharpe']:.2f}"
                      f"  (95% CI {net_stats['Sharpe95_lo']:.2f} to {net_stats['Sharpe95_hi']:.2f})")
    axes[0].legend(frameon=False)
    dd = eq_net / eq_net.cummax() - 1
    axes[1].fill_between(dd.index, dd, 0, color="#b5413b", alpha=0.6)
    axes[1].set_ylabel("Drawdown")
    axes[2].plot(res.n_active.index, res.n_active, color="#4a7c59", lw=0.8)
    axes[2].set_ylabel("Open pairs")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "equity.png", dpi=130)
    plt.close(fig)

    lines = ["# Backtest report", "", "| Metric | Net | Gross |", "|---|---|---|"]
    for k in net_stats:
        g = gross_stats.get(k, float("nan"))
        lines.append(f"| {k} | {net_stats[k]:.3f} | {g:.3f} |")
    if "trades" in stats:
        tr = stats["trades"]
        trade_line = (f"Trades: {tr['count']}, win rate {tr['win_rate']:.1%}, "
                      f"avg return {tr['avg_ret']:.2%}, avg hold {tr['avg_days']:.1f} days")
        lines += ["", trade_line]
    lines += ["", "![equity](equity.png)"]
    (out / "REPORT.md").write_text("\n".join(lines))
    return stats
