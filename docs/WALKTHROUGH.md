# Build it yourself: 10 sessions, about 1 hour each

You will rebuild this project from an empty folder in your own repo. The finished
version is your answer key, but **don't copy modules**. The workflow each session:

1. Read the concept notes (10 min).
2. Copy *that session's test file* into your repo. It will fail.
3. Write code until it passes (35-40 min).
4. Answer the "defend it" questions out loud or in your notes, without looking (10 min).
5. Commit with a real message. A clean history of 10+ meaningful commits is itself
   evidence you built it.

When you're stuck, come back to the chat with the session number and your error or
question. I'll explain, but I won't hand you the finished function.

---

## Week 1: foundations

### Session 1: Repo, environment, data
**Concepts:** adjusted vs. raw prices, wide vs. long data, why back-fill leaks the future.
**Build:** `pyproject.toml`, `src/statarb/__init__.py`, `data.py` with `load_csv`,
`clean_prices`, `fetch_yfinance` (with CSV cache).
**Checkpoint:** download 5 ETFs, plot them, save the cache, reload it.
**Defend it:** Why adjusted closes? Why cache? What does `ffill(limit=5)` protect against,
and what would `bfill` break?

### Session 2: Random walks, stationarity, and the synthetic generator
**Concepts:** random walk vs. stationary process, the OU process
`s_t = (1-θ)s_{t-1} + ε`, the ADF test, half-life `ln 2 / θ`.
**Build:** `synthetic_prices` in `data.py`. Plot one planted spread and one noise pair.
Run `adfuller` on each.
**Checkpoint:** ADF rejects a unit root for the planted spread and not for the noise.
**Defend it:** Why build fake data at all? What does ground truth let you test that
real data can't?

### Session 3: Cointegration and hedge ratios
**Concepts:** spurious regression; Engle-Granger two-step; why the test is asymmetric.
**Build:** `pairs.py`: `hedge_ratio` (OLS) and `half_life`.
**Tests:** `test_hedge_ratio_exact`, `test_half_life_recovers_known_ou`,
`test_half_life_random_walk_is_long_or_inf`.
**Defend it:** Correlation vs. cointegration, with an example of two correlated series
that are not cointegrated. Derive the half-life formula from the AR(1) regression.

### Session 4: Pair selection and multiple testing
**Concepts:** familywise error vs. false discovery rate; Bonferroni vs. Benjamini-Hochberg.
Work the BH example in the test by hand on paper first.
**Build:** `benjamini_hochberg` and `select_pairs`.
**Tests:** all of `tests/test_pairs.py`.
**Experiment:** shorten the sample to 400 days and see how many true pairs survive.
Write two sentences on test power.
**Defend it:** You tested 378 pairs. How many false positives do you expect without
correction? Why BH over Bonferroni?

### Session 5: Signals
**Concepts:** z-scores, causality, state machines, why a stop needs a cool-down.
**Build:** `signals.py`: `spread_series`, `rolling_zscore`, `positions_from_z`.
**Tests:** all of `tests/test_signals.py`. Trace `test_stop_loss_then_cooldown`
by hand before coding.
**Defend it:** Why is the exit band not at zero? What breaks without the cool-down?

---

## Week 2: rigor

### Session 6: The backtester and lookahead
**Concepts:** decision time vs. execution time; dollar-neutral vs. beta-neutral sizing;
turnover; commission, slippage and borrow.
**Build:** `backtest.py`: `leg_weights`, `backtest_pair`, `trade_list`.
**Tests:** everything in `test_backtest.py` except the Kalman test.
**Experiment:** delete the `.shift(1)` and rerun `test_lookahead_canary`. Watch it fail.
This is your best interview story.
**Defend it:** Walk through, on a whiteboard, why the one-bar lag prevents lookahead.

### Session 7: Walk-forward engine
**Concepts:** in-sample vs. out-of-sample; rolling formation/trading windows; z-score
warm-up from history; forcing exits at window boundaries.
**Build:** `walkforward.py` with `Config` and `run_walk_forward` (OLS only for now).
**Tests:** `test_walk_forward_is_out_of_sample_only`,
`test_future_data_does_not_change_past_results`.
**Defend it:** Why re-select every quarter? Why must the first OOS day equal the day after
the formation window?

### Session 8: Metrics and the null test
**Concepts:** Sharpe annualization (√252), why a point Sharpe lies, block bootstrap,
skew and the Probabilistic Sharpe Ratio, max drawdown.
**Build:** `metrics.py`.
**Tests:** `test_sharpe_known_value`, `test_max_drawdown`, `test_no_edge_on_pure_noise`.
**Defend it:** A strategy has Sharpe 1.5 over 2 years. Roughly how wide is the CI? Why
use blocks in the bootstrap?

### Session 9: Kalman filter, reporting, CLI
**Concepts:** state-space models; predict and update; filter vs. smoother
(the smoother uses future data).
**Build:** `kalman.py`, the `hedge_method` switch in `walkforward.py`, `report.py`,
`cli.py`, both YAML configs.
**Tests:** `test_kalman_tracks_beta_change`; then `statarb run --config configs/synthetic.yaml`.
**Experiment:** compare OLS and Kalman on synthetic data. Explain why Kalman loses there.
**Defend it:** When would Kalman beat OLS? What does δ control?

### Session 10: Real data, robustness, polish
**Build:**
- Run `configs/etfs.yaml`. Look at which pairs get picked. Do they make economic sense?
- Run `statarb sensitivity`. Is there a plateau or a spike?
- Rerun at 2x and 3x costs. At what cost does the edge vanish?
- Add CI (`.github/workflows/ci.yml`), write your README in your own words, and
  paste your real results, good or bad.

**Defend it:** Go through the full question bank in `docs/DESIGN.md` without notes.
Record yourself giving the 2-minute project pitch.

---

## The 2-minute pitch (fill in your numbers)

> I built a walk-forward pairs trading framework. Each quarter it screens a universe of
> liquid ETFs for cointegrated pairs on the prior year, using Benjamini-Hochberg to
> control false discoveries across hundreds of tests, then trades the spread
> out-of-sample with z-score entry, exit and stop rules, net of commissions, slippage
> and borrow. I enforced no-lookahead with tests, including one that corrupts future
> prices and checks past returns don't change, and I validated the pipeline on synthetic
> data with planted pairs and on pure noise, where it correctly finds no edge. On real
> ETFs from 2012 to 2026 it earned a net Sharpe of ___ with a 95% CI of ___ to ___.
> The biggest lesson was ___.
