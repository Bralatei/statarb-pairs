# statarb-pairs

A walk-forward **statistical arbitrage (pairs trading)** research framework in Python.
It selects cointegrated pairs on a rolling formation window, trades their spread
out-of-sample with z-score rules, and reports performance **net of commissions,
slippage and short-borrow costs**, with confidence intervals rather than point estimates.

The emphasis is on research hygiene, not on a flashy backtest:
no lookahead (enforced by tests), multiple-testing control in pair selection,
honest out-of-sample evaluation, and a sensitivity analysis instead of parameter tuning.

![equity curve](results/synthetic/equity.png)

## Results

**Synthetic universe** (4 planted cointegrated pairs + 8 random walks, 10 years, fully reproducible):

| | Net | Gross |
|---|---|---|
| Sharpe | 2.67 (95% CI 2.09 to 3.12) | 2.85 |
| CAGR / Ann. vol | 6.3% / 2.3% | |
| Max drawdown | -1.5% | |
| Trades / win rate / avg hold | 170 / 90% / 8.7 days | |

These numbers validate the *machinery*, not the *strategy*: the data was built
to contain mean-reverting spreads. The useful checks are that
(a) the engine finds the planted pairs, (b) it shows **no edge on pure noise**
(`tests/test_metrics_and_walkforward.py::test_no_edge_on_pure_noise`), and
(c) results degrade gracefully, not cliff-like, across parameters:

```
Net Sharpe by entry/exit threshold (synthetic)
exit_z   0.00  0.25  0.50  0.75  1.00
entry_z
1.50     3.22  3.27  3.39  3.31  3.31
2.00     2.68  2.66  2.67  2.53  2.40
2.50     2.03  2.11  2.09  1.99  1.85
```

**Real data** (28 liquid ETFs, 2012-2026): run `statarb run --config configs/etfs.yaml`.
Expect a far lower Sharpe; published research finds that classic distance and
cointegration pairs returns have decayed sharply since the early 2000s.
<!-- After running, paste your real-data table here and discuss it honestly. -->

## Method

```
prices ──► formation window (252d) ──► pair selection ──► trading window (63d, OOS) ──► roll forward 63d
                                       │
                                       ├─ log-price correlation pre-filter
                                       ├─ Engle-Granger cointegration (both leg orderings)
                                       ├─ Benjamini-Hochberg FDR correction across candidates
                                       ├─ half-life filter (2-60 days, from AR(1) fit)
                                       └─ each ticker in at most one pair
```

1. **Spread**: `s_t = log(y_t) - α - β·log(x_t)`, with β from formation-window OLS
   (static) or a **Kalman filter** (time-varying).
2. **Signal**: rolling 60-day z-score of the spread. Enter at |z| ≥ 2, exit at |z| ≤ 0.5,
   stop-loss at |z| ≥ 4 with a cool-down, time stop at 3× half-life.
3. **Execution**: decisions at close of t, held from t+1. Each pair is dollar-sized
   with gross exposure 1 (`w_y = 1/(1+β)`, `w_x = -β/(1+β)`). Positions are force-closed
   at the end of each trading window so exit costs land inside that window.
4. **Costs**: 1 bp commission + 2 bp slippage per unit turnover, 50 bp/yr borrow on the short leg.
5. **Evaluation**: Sharpe with a stationary block-bootstrap CI, Probabilistic Sharpe
   Ratio, Sortino, max drawdown, Calmar, turnover, trade statistics.

Full reasoning for each choice, its alternatives and its weaknesses is in
[`docs/DESIGN.md`](docs/DESIGN.md).

## Quick start

```bash
git clone https://github.com/<you>/statarb-pairs.git && cd statarb-pairs
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,data]"

python -m pytest -q                                # 21 tests, ~10s
statarb run --config configs/synthetic.yaml        # offline, ~15s
statarb run --config configs/etfs.yaml             # real ETFs via yfinance
statarb sensitivity --config configs/synthetic.yaml  # entry/exit grid, ~5 min
```

Each run writes `metrics.json`, `REPORT.md`, `equity.png`, `trades.csv`,
`pairs_selected.csv` and `daily_returns.csv` to the `output` folder in the config.

## Project layout

```
src/statarb/
  data.py         loading, cleaning (ffill only, never bfill), synthetic ground-truth generator
  pairs.py        cointegration screen, hedge ratio, half-life, BH correction
  kalman.py       time-varying hedge ratio (filter, not smoother: no lookahead)
  signals.py      rolling z-score, entry/exit/stop state machine
  backtest.py     one-bar-lagged execution, costs, trade list
  walkforward.py  rolling formation/trading engine
  metrics.py      Sharpe + bootstrap CI, PSR, drawdown, etc.
  report.py       plots and Markdown report
  cli.py          `statarb run` / `statarb sensitivity`
tests/            21 tests incl. lookahead canaries and a noise-only null test
configs/          synthetic.yaml (offline), etfs.yaml (real data)
docs/DESIGN.md    design decisions, limitations, interview Q&A
```

## Tests worth reading

- `test_positions_are_lagged_one_bar`: a signal on day t earns day t+1's return.
- `test_lookahead_canary`: an oracle using same-day information shows no edge after the lag.
- `test_future_data_does_not_change_past_results`: corrupt prices after day D and
  every earlier return is bit-identical.
- `test_no_edge_on_pure_noise`: on independent random walks, the full pipeline
  produces no meaningful Sharpe.
- `test_half_life_recovers_known_ou`: the estimator recovers a known OU parameter.

## Limitations

- Daily close-to-close fills; no intraday execution or market impact model beyond fixed bps.
- Survivorship: the ETF list is chosen today. ETFs reduce but don't remove this.
- Engle-Granger is a two-asset test with low power on one-year windows; Johansen would
  allow baskets.
- Portfolio is equal-weighted across selected pairs; no volatility targeting or
  correlation-aware sizing, so idle capital lowers raw returns.

## License

MIT
