# Design decisions and how to defend them

For each component: **what** was chosen, **why**, **what else** was considered, and
**where it breaks**. The last section is a bank of interview questions with answers.

---

## 1. Data

**Choice.** Adjusted daily closes. Forward-fill gaps of at most 5 days, then drop
any rows still missing. Drop tickers missing more than 5% of history.

**Why.** Adjusted prices fold in splits and dividends, so returns are total returns.
Forward-fill uses only past data. **Back-fill would leak the future into the past**:
a ticker's first real price would appear on days before it existed.

**Synthetic generator.** `synthetic_prices` plants pairs with a known OU spread
(`s_t = (1-θ)s_{t-1} + ε`) next to pure random walks. Ground truth lets us test that
selection finds real pairs, rejects noise, and that the whole pipeline earns nothing
when there is nothing to earn.

**Weaknesses.** Yahoo data has occasional bad prints. Universe chosen with hindsight
(survivorship). Adjusted closes are not tradeable prices: the close itself is an auction
you can participate in, but your fill will differ.

## 2. Pair selection

**Pipeline.** Correlation pre-filter, then Engle-Granger on both orderings, then BH FDR
correction, then a half-life band, then a greedy one-pair-per-ticker match.

**Why cointegration, not correlation.** Correlation is about *returns* moving together
day to day; two series can be 0.9 correlated and drift apart forever. Cointegration says a
linear combination of *levels* is stationary, which is the property a mean-reversion
trade actually needs. Correlation is only used as a cheap screen to cut the O(n²) tests.

**Why test both orderings.** Engle-Granger regresses y on x and tests the residual. The
result depends on which is y. We keep the ordering with the lower p-value.

**Why multiple-testing correction.** With 28 tickers there are 378 candidate pairs.
At α = 5% you expect ~19 false positives from noise alone. Benjamini-Hochberg controls
the *false discovery rate* (expected share of selected pairs that are spurious).
Bonferroni controls the probability of *any* false positive, which is too strict and
leaves almost nothing. Measured effect on the synthetic universe: no correction selects
**12** spurious noise-containing pairs over the backtest, BH selects **6**.

**Why a half-life band.** Half-life from `Δs_t = λ s_{t-1} + c + e`, `HL = -ln 2 / λ`.
Under 2 days the "reversion" is likely bid-ask bounce or noise and costs eat it.
Over 60 days the trade will not complete inside a 63-day window.

**Weaknesses.**
- Engle-Granger has low power on 252 days: it misses real pairs that revert slowly
  (`test_selection_finds_true_pairs_and_rejects_noise` documents this).
- AR(1) half-life estimates are biased downward in small samples.
- Cointegration found in-sample can break out-of-sample (mergers, regime change).
  That is why we re-select every quarter and use a stop-loss.

## 3. Hedge ratio: OLS vs Kalman

**OLS.** One β per formation window. Simple, stable, but stale by the end of the window.

**Kalman.** State `[α_t, β_t]` follows a random walk with noise `δ`; observation
`log y_t = α_t + β_t log x_t + e_t`. The filter adapts β daily. We record the
**prior** estimate (before seeing y_t), so the spread at t never uses y_t's own
information to fit β.

**Measured trade-off.** On synthetic data, where true β is constant, Kalman *lowers* net
Sharpe from 2.67 to 1.16 and raises turnover from 12x to 16x a year. It fits noise and
rebalances more. On real data, where β drifts, it may help. This is the right answer to
"which is better?": it depends on whether the relationship is stable, and you
should measure it, not assume.

**Weaknesses.** δ and observation variance are free parameters. The intercept and slope
are poorly identified when log x barely moves (see `test_kalman_tracks_beta_change`).

## 4. Signals

**z-score.** Trailing 60-day mean and standard deviation of the spread. Causal by
construction; `test_zscore_is_causal` checks it.

**Rules.** Enter at |z| ≥ 2, exit at |z| ≤ 0.5, stop at |z| ≥ 4, time stop at 3× half-life.
After a stop, no re-entry until |z| returns below the entry level, otherwise you re-enter
the broken spread the next day.

**Why these numbers.** 2σ is convention, not optimization. The sensitivity grid shows
Sharpe varies smoothly from 1.85 to 3.39 across entries 1.5-2.5 and exits 0-1. A smooth
plateau means the result is not an artifact of one lucky setting. We report the
convention, not the best cell: picking the max of 25 backtests is overfitting.

## 5. Backtest accounting

**Timing.** `held = target.shift(1)`: decide at close t, earn t+1's return. Every
lookahead test hinges on this line.

**Sizing.** Each pair has gross exposure 1: `w_y = 1/(1+|β|)`, `w_x = -β/(1+|β|)`.
Portfolio return = mean of pair returns that period, so capital is split equally across
selected pairs. Idle pairs earn 0 (no cash interest modeled).

**Costs.** Turnover × (commission + slippage) bps, plus annual borrow on short notional.
Entry and exit are both charged (`test_costs_reduce_returns_by_turnover`).

**Why log-price spread, simple-return P&L.** The spread is defined in logs so β is an
elasticity and the spread is scale-free. P&L uses simple returns on actual positions,
because that is what a portfolio actually earns.

**Weaknesses.** Weights drift between rebalances and the engine ignores that (second-order
at daily frequency). No market impact scaling with size. No cash yield on idle capital,
and no margin or funding cost on longs.

## 6. Evaluation

- **Sharpe with block-bootstrap CI.** A point Sharpe hides estimation error. Blocks
  (20 days) keep autocorrelation that an i.i.d. bootstrap would destroy.
- **Probabilistic Sharpe Ratio** (Bailey & López de Prado, 2012): probability the true
  Sharpe exceeds 0, adjusting for skew and fat tails. Mean-reversion strategies often have
  negative skew (many small wins, rare large losses), which inflates naive Sharpe.
- **Noise null test.** The strongest single check: run the whole pipeline on pure random
  walks and confirm the edge disappears.

---

## Interview question bank

**Q: Why should this make money at all?**
Two assets share an economic driver (same sector, same commodity exposure). Temporary
dislocations come from liquidity shocks, flows into one ETF, or slow information diffusion.
You are paid to provide liquidity and wait. The edge erodes as more capital does the same
trade, which is why published pairs returns have fallen over time.

**Q: How do you know there's no lookahead?**
Three tests: a one-bar lag test, an oracle canary that shows same-day information has no
value after the lag, and a corruption test where changing prices after day D leaves every
earlier return identical. Selection only sees the formation window, and the Kalman filter
records prior estimates.

**Q: Your synthetic Sharpe is 2.7. Isn't that suspicious?**
Yes, if it were real data. The synthetic data was built with mean-reverting spreads, so a
high Sharpe there only proves the engine can find a planted signal. The meaningful checks
are the noise test (no edge where none exists) and the real-data run, which is much weaker.

**Q: Difference between correlation and cointegration?**
Correlation: co-movement of returns, short horizon. Cointegration: a linear combination of
non-stationary levels is stationary, so the gap reverts. Two random walks can be highly
correlated over a sample and still not be cointegrated.

**Q: Why not Johansen?**
Johansen handles more than two series and is symmetric. For pairs, Engle-Granger is
simpler and easier to explain; testing both orderings fixes its asymmetry. Johansen is
the natural extension to baskets.

**Q: How many pairs did you test, and what about false positives?**
Up to n(n-1)/2 per window, every window. BH controls the false discovery rate; the
measured effect is half the spurious pairs. Further protection: re-selection each quarter
and stop-losses for pairs that break.

**Q: How did you pick the 2.0 / 0.5 thresholds?**
Convention, fixed before looking at results. The sensitivity grid shows a broad plateau.
I did not pick the best cell, because that would be fitting to the backtest.

**Q: What happens when a pair breaks?**
The z-score blows out. The 4σ stop exits, the cool-down prevents re-entry, the time stop
caps holding, and quarterly re-selection drops the pair.

**Q: What's the capacity?**
For liquid ETFs, millions per pair at daily frequency before impact dominates. The binding
constraint is the number of genuinely cointegrated pairs, not liquidity.

**Q: What would you do next?**
Johansen baskets; vol-targeted sizing; a regime filter (e.g., pause in high-VIX
periods); intraday data for execution; cost sensitivity at 2x and 3x bps; and a
Deflated Sharpe Ratio to account for every configuration tried.

## References

- Gatev, Goetzmann & Rouwenhorst (2006), *Pairs Trading: Performance of a Relative-Value
  Arbitrage Rule*, Review of Financial Studies.
- Do & Faff (2010), *Does Simple Pairs Trading Still Work?*, Financial Analysts Journal.
- Engle & Granger (1987), *Co-integration and Error Correction*, Econometrica.
- Benjamini & Hochberg (1995), *Controlling the False Discovery Rate*, JRSS-B.
- Bailey & López de Prado (2012), *The Sharpe Ratio Efficient Frontier*, Journal of Risk.
- Chan (2013), *Algorithmic Trading: Winning Strategies and Their Rationale*, Wiley
  (Kalman filter hedge ratios).
