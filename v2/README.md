# v2: fixed and refactored strategy code

The original notebooks in `training/` and `Projects/` are left untouched. `v2/` holds the fixed, function-based version,
so each pipeline component (preselection, beta forecast, market-drop scenario, return / risk estimate, optimizer,
post-allocation) can be swapped and compared against the baseline.

**Two tracks (2026-10, see [Development](#development-two-objectives)):** *defensive* (maximise crash protection while fully invested) holds the 30 / 20 lowest-blend stocks; *return* (risk-adjusted return, protection as a side condition) holds the lowest third. Both use inverse vol, a 25% sector cap and quarterly rebalancing: `components.recommended(date, universe, track=...)`.

**Defensive setup:** each quarter, predict every stock's next-swan beta as ½ × the mean of its last 5 swan betas + ½ × its 1-year market beta. Hold the 30 lowest (S&P500, point-in-time members) or 20 lowest (S&P100), with at most 25% of the stocks and 25% of the weight in one sector, at inverse-volatility weights: `components.recommended(date, universe)`. See [Component experiments](#component-experiments) and [Robustness](#robustness) for why.

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python v2/mvo2.py      # Swan9 baseline -> v2/output/, asserts reference PR 0.2225
.venv/bin/python v2/compare.py   # original notebook vs v2 on the same frozen data
.venv/bin/python v2/swan_beta.py # recompute every Swan Beta from prices, check it against All_SwanBeta100.xlsx
.venv/bin/python v2/walkforward.py  # predict Swan5..Swan11 one by one from the 5 swans before each
.venv/bin/python v2/backtest.py SP100 hand   # monthly-rebalance backtest 2004 -> today (SP100|SP500, hand|rule swans)
.venv/bin/python v2/experiments.py SP500      # component grid on point-in-time S&P500 members (~45 min, cached after)
.venv/bin/python v2/components.py SP500 2026-10-01   # recommended holdings on a date (default: last price date)
.venv/bin/python v2/robustness.py SP500       # number of stocks, sector cap, rebalance frequency, cost (~45 min, cached after)
.venv/bin/python v2/risk_model.py             # factor risk model smoke test
.venv/bin/python v2/factor_experiment.py SP500   # factor risk model in the weighting step vs the recommended setup
.venv/bin/python v2/explore.py SP500             # every preselection (K-means, DBI, DTW, Holt, ...) x weighting (HRP, HERC, ...)
.venv/bin/python v2/develop.py SP500             # defensive and return tracks, pre-registered variants, overfitting checks
.venv/bin/python v2/defense.py                   # does it make money, where the drawdowns come from, sizing / multi-asset layers
.venv/bin/python v2/defense_selection.py         # momentum and stock-trend filters at the selection level
.venv/bin/python v2/equity_defense.py            # drawdown control inside a fully invested stock book
.venv/bin/python v2/allocation.py                # long-term stock + ETF allocations against a 15% max drawdown target
.venv/bin/python v2/rp_robustness.py             # the chosen risk parity under other start months, costs, lookbacks, assets, sleeves
```

| File | What |
|---|---|
| `mvo2.py` | MVO 2.0 as one function per step; `run(params)` returns stats, weights, curves and PR |
| `compare.py` | runs the original `training/MVO` notebook offline next to v2 and prints weights / PR side by side |
| `swan_beta.py` | Swan Beta matrix and swan market stats (DD, T1, T2) from `data/swans.csv` and the frozen prices |
| `walkforward.py` | walk-forward test over Swan5–Swan11 on the unified data, with and without analyst targets |
| `backtest.py` | monthly rebalance from 2004, using only swans confirmed and prices known at each date; compares MVO with ablations and simple baselines |
| `components.py` | swappable parts: point-in-time universe, beta forecasters (swan mean, market beta, blend, ridge, random forest), selection, weighting (equal, inverse vol, min vol, MVO); `recommended()` |
| `experiments.py` | forecaster test on Swan5–Swan11 and the full forecaster × weighting backtest grid |
| `robustness.py` | the recommended setup across stock counts, sector caps, rebalance frequencies and costs; its sector weights |
| `risk_model.py` | Barra-style factor risk model from prices: market beta, sectors, size, momentum, volatility; ERC weights |
| `factor_experiment.py` | risk-forecast accuracy (factor vs sample covariance) and factor-model weightings vs the recommended setup |
| `preselect.py` | the team's preselection methods (K-means, MVO DB's DBI voting, DTW clustering, Holt) plus a Barra-feature ridge forecaster |
| `explore.py` | which component matters: preselection test at each swan, preselection × weighting grid, random controls |
| `develop.py` | the two objectives: fixed-in-advance variants, β-matched index + T-bill benchmark, leave-one-swan-out and sub-period checks |
| `defense.py` | original vs modified vs buy-and-hold and low-vol ETFs; drawdown and return sources; trend, vol-target, bond and risk-parity layers |
| `defense_selection.py` | momentum filter and stock-level trend filter added to the selection |
| `equity_defense.py` | crash-day covariance min-var, sector risk parity, stress-beta cap, regime switch; all fully invested |
| `allocation.py` | stock sleeve + Treasury / gold / TIPS / commodity ETFs: fixed mix, risk parity, vol target, drawdown control |
| `rp_robustness.py` | robustness of the chosen allocation (A1 risk parity R1 + IEF + GLD) |
| `live.py` | live use of the chosen allocation: target holdings, orders from your holdings, risk / market report |

## Baseline: Swan9 from Swan4–Swan8, S&P 100, K-means N=7

| PR | original | v2 without nominal fix | v2 |
|---|---|---|---|
| MVO weights | 0.2327 | 0.2327 | **0.2225** |
| post-allocation: MVO weight | 0.0563 | 0.2327 | 0.2225 |
| post-allocation: equal | 0.2117 | 0.2117 | 0.2220 |
| post-allocation: adjusted proportion | 0.1839 | 0.2195 | 0.2081 |
| post-allocation: direct | 0.1663 | 0.2031 | 0.2081 |

## Unified data (used by `walkforward.py`)
- Swan Beta and MarketDD (DD, T1, T2) are computed by `swan_beta.py` from `data/swans.csv` and the frozen prices, replacing the two beta Excel files and `MarketDD.xlsx`. In `mvo2.run`, `beta_file=None` and `marketdd_file=None` select them; `mvo2.SWAN9` still uses the original files for comparison.
- Universe: when a swan has optdata, the tickers in it (roughly the index on that date); otherwise every ticker with price history. This is survivorship-biased for the early swans, because only companies that still exist today are in the price file.
- Without analyst targets (`optdata_file=None`), the expected return is the past-year trend `R_60`, floored at 0.
- **Predicted-swan length.** With a consistent T2 (days for the index to regain its prior peak), T1 + T2 is longer than the ~282-day look-back for every target. In the original data it already was for Swan5–7, which is why only Swan8 onward were ever run. The recovery is now compressed to end on the window's last day; the depth is unchanged. When it fits (e.g. `SWAN9`), nothing changes.
- When max-Sharpe has no solution (e.g. in 2008 only OXY beats the 1% rf), the optimizer falls back to min volatility. The rf is compared with `E_2/100` = return-per-drawdown ÷ 100, a unit mismatch inherited from the original.

## Walk-forward: each swan predicted from the 5 before it (S&P 100, K-means N=7, w1=0.17)

| target | variant | PR | PR equal-wt preselection | portfolio actual β | full-cycle excess* |
|---|---|---|---|---|---|
| Swan5 2008 | no P_fwd | 0.285 | 0.352 | 0.715 | +6.0% |
| Swan6 2011 | no P_fwd | 0.906 | 0.760 | 0.094 | +8.0% |
| Swan7 2015 | no P_fwd | 0.197 | 0.345 | 0.803 | +10.5% |
| Swan8 2018 | no P_fwd | 0.503 | 0.592 | 0.497 | +3.5% |
| Swan9 2020 | no P_fwd | 0.263 | 0.191 | 0.739 | −2.5% |
| | P_fwd (look-ahead) | 0.225 | 0.191 | 0.777 | −3.8% |
| | original input files | 0.222 | 0.182 | 0.777 | −3.8% |
| Swan10D 2022 | no P_fwd | 0.787 | 0.616 | 0.213 | +20.2% |
| | P_fwd | 0.629 | 0.616 | 0.371 | +4.8% |
| Swan11 2025 (out of sample) | no P_fwd | 0.678 | 0.773 | 0.295 | −4.3% |
| | P_fwd | 0.679 | 0.773 | 0.295 | −2.0% |

\* Portfolio minus index return, from the portfolio date until the index is back at its pre-crash peak.

- The predicted portfolio β is about 0.41–0.46 every time; the actual β ranges from 0.09 to 0.80.
- Without P_fwd: mean PR 0.517. Equal-weighting the preselected stocks instead gives 0.518, and MVO beats it in only 3 of 7 swans.
- Analyst targets never helped: worse on Swan9 and Swan10D, a tie on Swan11.
- Seven events is a small sample. Treat these numbers as a baseline to compare upgrades against, not as proof.

## Backtest: monthly rebalance, 2004-01 → 2026-10, 10 bp cost per unit of turnover

`backtest.py`: each month the original pipeline is rebuilt from data known that day (the last 5 *confirmed* swans, prices up to the day before, trend returns), then held to the next rebalance. S&P100, hand swans:

| | MVO | no injection | Min-vol | Preselect EW | Universe EW | ^OEX | Switch |
|---|---|---|---|---|---|---|---|
| CAGR | 10.9% | 10.9% | 10.6% | 11.9% | 14.6% | 8.9% | 9.4% |
| Sharpe (rf=0) | 0.74 | 0.74 | 0.76 | 0.80 | 0.83 | 0.55 | 0.62 |
| max drawdown | 35% | 39% | 36% | 36% | 50% | 56% | 37% |
| beta | 0.59 | 0.59 | 0.57 | 0.65 | 0.96 | 1.00 | 0.83 |

- **The crash injection adds little**: same return, a few points less drawdown. The protection comes from the swan-beta preselection.
- **Detecting a swan and switching then is too late.** By the time the index is 10% down, much of the fall has happened. *Switch* (hold the index, move to MVO when the index is ≥10% below its 1-year high) earns about the index's return.
- ^OEX / ^GSPC are price indices with no dividends, while stock prices include them, so they understate the market by ~2%/yr. Use OEF / SPY (total return) when comparing returns; `experiments.py` and `robustness.py` report both.
- Results before 2026-10-21 skipped each rebalance day's return (about 12 trading days a year, for every portfolio and the index) and understated every CAGR by ~2%/yr. All numbers here are after the fix; the backtest's ^OEX now matches the actual index exactly.

## Component experiments

`experiments.py`, S&P500 universe = **point-in-time members** (`data/universe`), so stocks enter only once they joined the index and leave when they were dropped. S&P100 = today's list (no point-in-time source, survivorship-biased). Monthly rebalance 2004 → 2026-10, 10 bp per unit of turnover. Selection: lowest 30 (S&P500) / 20 (S&P100) predicted betas, among stocks with at least 2 of the last 5 swan betas.

**1. Which forecaster predicts the next swan's beta?** Each of Swan5–Swan11 predicted the day before its peak. Mean over the 7 swans:

| | S&P500 rank corr | S&P500 actual β of lowest 30 | S&P100 rank corr | S&P100 actual β of lowest 20 |
|---|---|---|---|---|
| all candidates | | 0.95 | | 0.88 |
| K-means cluster (original) | | 0.43 | | 0.46 |
| swan mean (last 5) | 0.41 | 0.48 | 0.40 | 0.53 |
| 1-year market beta | 0.53 | 0.41 | 0.54 | 0.43 |
| **blend (½ swan mean + ½ market beta)** | 0.53 | **0.38** | 0.53 | **0.43** |
| ridge (6 features, trained on earlier swans) | 0.54 | 0.38 | 0.53 | 0.45 |
| random forest (same) | 0.52 | 0.41 | 0.51 | 0.45 |

The swan betas alone are a weaker predictor than plain market beta. Combining the two is best. Machine learning on ~10 past swans does not beat the simple average.

**2. Backtest of forecaster × weighting** (selected rows; all 24 in `v2/output/experiments_<universe>_grid.csv`):

| S&P500 point-in-time | CAGR | vol | Sharpe | max DD | beta | turnover/yr | PR mean | PR worst |
|---|---|---|---|---|---|---|---|---|
| K-means + MVO (original) | 10.8% | 16.0% | 0.72 | 34% | 0.57 | 7.1 | 0.65 | 0.03 |
| K-means + equal | 10.1% | 15.9% | 0.69 | 37% | 0.61 | 3.9 | 0.58 | 0.11 |
| **blend + inverse vol** | 9.9% | 14.9% | 0.71 | 39% | 0.54 | 2.1 | 0.64 | 0.11 |
| swan mean + inverse vol | 10.4% | 15.9% | 0.71 | 42% | 0.65 | 0.9 | 0.57 | 0.11 |
| market beta + inverse vol | 8.9% | 14.3% | 0.67 | 37% | 0.49 | 2.9 | 0.63 | 0.07 |
| ridge + inverse vol | 8.3% | 14.4% | 0.63 | 38% | 0.50 | 2.5 | 0.64 | 0.10 |
| universe equal weight | 11.6% | 20.1% | 0.65 | 58% | 1.04 | 0.7 | 0.03 | −0.17 |
| SPY (incl. dividends) | 10.8% | 18.6% | 0.65 | 55% | 0.99 | 0 | 0.02 | 0.01 |

| S&P100 (today's list) | CAGR | vol | Sharpe | max DD | beta | turnover/yr | PR mean | PR worst |
|---|---|---|---|---|---|---|---|---|
| K-means + MVO (original) | 10.9% | 15.7% | 0.74 | 35% | 0.59 | 4.1 | 0.56 | 0.20 |
| **blend + inverse vol** | 10.5% | 14.3% | 0.77 | 37% | 0.58 | 1.4 | 0.57 | 0.09 |
| swan mean + inverse vol | 12.6% | 15.1% | 0.86 | 39% | 0.66 | 0.8 | 0.49 | 0.10 |
| universe equal weight | 14.6% | 18.6% | 0.83 | 50% | 0.96 | 0.6 | 0.09 | −0.12 |
| OEF (incl. dividends) | 10.8% | 18.4% | 0.65 | 54% | 0.99 | 0 | 0.02 | 0.00 |

**3. Conclusions**
- **Every combination does the job.** Return close to the market (SPY 10.8%), beta 0.5–0.65, ~20% less volatility, and a max drawdown of 34–42% instead of 55%. The rows shown lost less than the index in every one of the 7 swans since 2004 (PR worst > 0). COVID is the weak spot for all.
- **No combination is significantly better than another on Sharpe.** In a monthly-block bootstrap, the 90% interval of the Sharpe difference vs the original includes 0 for every variant (e.g. blend + inverse vol: S&P500 [−0.17, +0.14], S&P100 [−0.10, +0.17]). The only significant difference is that random forest + min-vol is worse. With 22 years and 7 crashes, differences of ±0.1 Sharpe are noise.
- Blend is the best beta predictor in both universes and matches the original's crash protection at a third of its turnover, with no K-means seed, crash injection, analyst targets or tuned thresholds. MVO and min-vol never help consistently; equal and inverse-vol weights are as good.
- Survivorship bias remains. Point-in-time membership is in place, but Yahoo has no prices for about half the 2004 members (coverage 52% in 2004, 60% in 2008, 67% in 2012, 86% in 2020, 97% in 2026). Removing it needs paid data (Norgate, CRSP).

## Robustness

`robustness.py`, blend + inverse vol, 2004 → 2026-10.

**Number of stocks** (S&P500, monthly, 10 bp): smooth trade-off, no cliff. Fewer stocks gives more protection and less return.

| n | 15 | 20 | **30** | 50 | 75 |
|---|---|---|---|---|---|
| CAGR | 9.5% | 9.8% | 9.9% | 9.9% | 9.9% |
| Sharpe | 0.68 | 0.70 | 0.71 | 0.72 | 0.72 |
| beta | 0.50 | 0.51 | 0.54 | 0.57 | 0.61 |
| PR mean | 0.67 | 0.66 | 0.64 | 0.60 | 0.54 |

S&P100 (n = 10–30): Sharpe 0.77–0.82, PR mean 0.51–0.62.

**Sector concentration.** Without a cap, the S&P500 version holds on average 47% utilities (up to 83%) and 32% consumer staples. The S&P100 version holds 35% staples (up to 47%) and 20% each of health care and utilities. The original K-means + MVO is worse: 6.8 stocks on average, top sector 55% on average and at times 100%. A sector cap fixes this at almost no cost:

| S&P500, n=30, monthly, 10 bp | CAGR | vol | Sharpe | max DD | beta | PR mean | PR worst |
|---|---|---|---|---|---|---|---|
| no cap | 9.9% | 14.9% | 0.71 | 39% | 0.54 | 0.64 | 0.11 |
| 35% | 9.7% | 14.3% | 0.72 | 39% | 0.55 | 0.62 | 0.18 |
| **25%** | 9.9% | 14.2% | 0.74 | 41% | 0.57 | 0.60 | 0.14 |
| 20% | 10.1% | 14.5% | 0.74 | 42% | 0.60 | 0.57 | 0.12 |

**Rebalance frequency and cost.** Quarterly is as good as monthly at ~60% of the turnover. At 0.3% cost (closer to real), S&P500 n=30: monthly Sharpe 0.68, quarterly 0.71, semi-annual 0.70. The original K-means + MVO turns over 7×/yr. At 0.5% cost its monthly Sharpe falls to 0.54 (blend: 0.65).

**Timing luck.** Quarterly rebalancing, 0.3% cost, starting in month 1, 2 or 3 of each quarter:

| | S&P500 Sharpe | S&P500 PR worst | S&P100 Sharpe | S&P100 PR worst |
|---|---|---|---|---|
| original (K-means + MVO) | 0.76 / 0.81 / 0.76 | −0.09 / −0.10 / 0.07 | 0.76 / 0.72 / 0.70 | 0.22 / 0.20 / 0.24 |
| blend, n=30/20, no cap | 0.70 / 0.67 / 0.67 | 0.04 / 0.08 / 0.10 | 0.74 / 0.75 / 0.75 | 0.06 / 0.07 / 0.09 |
| **blend, 25% sector cap** | 0.70 / 0.70 / 0.70 | 0.10 / 0.11 / 0.13 | 0.78 / 0.77 / 0.77 | 0.01 / 0.01 / 0.07 |

The original's S&P500 return is higher (11.6–12.9% vs 9.4% quarterly), but it swings with the start month and in 2 of 3 cases it lost *more* than the index in one swan. That is what 7 concentrated stocks look like. The capped blend gives the same result whichever month it starts.

**Recommendation:** blend beta → lowest 30 (S&P500) / 20 (S&P100) → inverse vol → 25% sector cap → quarterly rebalance. Expect roughly the market's return minus 1%/yr on S&P500 (SPY 10.8%, this 9.4–9.9%), beta ~0.57, max drawdown ~40% instead of 55%, and less loss than the index in every swan.

## Factor risk model

`risk_model.py` builds Σ = X F Xᵀ + D on each date from prices only, using data before that date:

| factor | exposure |
|---|---|
| market | 1-year daily beta to the benchmark |
| sectors | one 0/1 column per sector (`data/universe/sectors.csv`) |
| size | log of 3-month average dollar volume. Yahoo has no shares-outstanding history, and dollar volume tracks market cap. |
| momentum | 12-month return skipping the last month |
| volatility | 1-year daily volatility |

Style exposures are z-scored and clipped at ±3. Factor returns come from a daily cross-sectional regression over the past year. F and the specific variances D are exponentially weighted (half-life 90 days), and D has a floor.

**Does it forecast risk better?** At each month-end it predicts the volatility of several portfolios (the strategy's, equal weight, 10 random 20/30-stock ones). That prediction is compared with the realised volatility over the next 3 months:

| | S&P500 factor | S&P500 sample cov | S&P100 factor | S&P100 sample cov |
|---|---|---|---|---|
| mean \|log(realised / predicted)\| (lower = better) | **0.31** | 0.34 | **0.32** | 0.35 |
| rank correlation, predicted vs realised | **0.59** | 0.55 | **0.52** | 0.47 |
| realised / predicted for the portfolio optimised on the sample covariance | **0.94** | 1.04 | 0.94 | 0.99 |

The factor model forecasts better. In particular, it does not understate the risk of an optimised portfolio the way the sample covariance does (ratio > 1 = more risk than predicted).

**Does it improve the portfolio?** `factor_experiment.py`: the same selection weighted differently, plus whole-universe optimisers. Quarterly rebalance, 0.3% cost; the ranges are over the 3 possible start months.

| S&P500 point-in-time | Sharpe | CAGR | vol | max DD | beta | PR mean | PR worst | turnover/yr | holdings |
|---|---|---|---|---|---|---|---|---|---|
| **recommended (inverse vol)** | 0.70–0.71 | 9.5–9.6% | 14.4–14.6% | 40% | 0.58 | 0.59–0.60 | 0.10–0.13 | 1.5 | 30 |
| same stocks, factor ERC | 0.69–0.71 | 9.4–9.6% | 14.4–14.5% | 40–42% | 0.59 | 0.56–0.58 | 0.10–0.13 | 1.5 | 30 |
| same stocks, factor min-var | 0.69–0.71 | 8.9–9.2% | 13.6–13.7% | 41–42% | 0.54–0.55 | 0.59–0.61 | 0.13–0.20 | 2.6 | 23 |
| same stocks, sample min-var | 0.67–0.69 | 8.7–8.8% | 13.6–13.7% | 41% | 0.54 | 0.58–0.61 | 0.11–0.21 | 2.7 | 18 |
| all candidates, factor min-var | 0.64–0.68 | 8.2–8.5% | 13.4–13.6% | 41–43% | 0.60–0.62 | 0.38–0.41 | −0.02–0.13 | 2.5 | 62 |
| all candidates, factor min-var, predicted swan β ≤ 0.5 | 0.66–0.70 | 8.4–9.0% | 13.5–13.7% | 41–42% | 0.60 | 0.45–0.47 | −0.02–0.14 | 2.3 | 52 |
| SPY (incl. dividends) | 0.64–0.65 | 10.7–10.8% | 18.6% | 55% | 0.99 | 0.02 | 0.01 | – | – |

| S&P100 | Sharpe | CAGR | max DD | beta | PR mean | PR worst | turnover/yr |
|---|---|---|---|---|---|---|---|
| **recommended (inverse vol)** | 0.77–0.79 | 10.7–11.0% | 37–41% | 0.60 | 0.53–0.55 | 0.01–0.07 | 1.0 |
| same stocks, factor ERC | 0.77–0.80 | 10.7–11.1% | 38–42% | 0.61 | 0.52–0.54 | 0.01–0.07 | 1.0 |
| same stocks, factor min-var | 0.69–0.72 | 9.2–9.5% | 40–43% | 0.58 | 0.50–0.52 | −0.04–0.05 | 1.9 |
| same stocks, sample min-var | 0.70–0.74 | 9.4–9.8% | 40–42% | 0.58 | 0.50–0.53 | −0.07–0.00 | 1.9 |
| all candidates, factor min-var | 0.75–0.76 | 10.0–10.1% | 39–40% | 0.65 | 0.38–0.40 | −0.12–0.01 | 1.7 |

**Conclusions**
- **Keep inverse-vol weighting.** No factor-model weighting is better in both universes. ERC gives the same result as inverse vol, since 30 defensive stocks have similar correlations. Factor min-variance lowers volatility (13.6% vs 14.4%) and beta on S&P500 at the same Sharpe, but with 70% more turnover and fewer names, and it is worse on S&P100.
- **The swan-beta selection adds something a generic low-risk portfolio does not.** Minimum variance over all candidates under the factor model, the textbook low-volatility portfolio, has about the same volatility but clearly less crash protection: PR mean 0.38–0.47 vs 0.59, and it lost more than the index in a swan. Adding the predicted swan-beta constraint recovers part of the gap. Picking stocks by how they fell in past crashes is not the same as picking the least volatile ones.
- **Use the risk model to measure, not to weight.** It forecasts portfolio risk better than the sample covariance and does not understate an optimised portfolio's risk. That makes it the right tool for risk reports, for checking the portfolio's factor exposures, and for any future optimiser.

## Component exploration: what matters?

`explore.py`. All preselection methods rank the same pool (stocks with all 5 window betas, point-in-time S&P500 members) and hold the n safest (30 / 20). Weightings: equal, inverse vol, HRP (pypfopt), HERC (`components.herc`, one-level version), factor-model min-variance, and the original MVO steps (injection + trend/drawdown + max Sharpe). Backtests use quarterly rebalancing at 0.3% cost, averaged over the 3 start months, 2004 → 2026-10.

**A. Pick quality: average actual beta in the next swan of the n stocks picked** (Swan5–Swan11 mean; lower is better)

| | random n | K-means (MVO2) | K-means + DBI (MVO DB) | DTW (team code) | DTW (corrected) | Holt | swan mean | market β | blend | Barra ridge |
|---|---|---|---|---|---|---|---|---|---|---|
| S&P500 | 0.93 | 0.47 | 0.49 | 0.44 | 0.45 | 0.71 | 0.45 | 0.39 | **0.38** | **0.38** |
| S&P100 | 0.87 | 0.50 | 0.51 | 0.52 | 0.57 | 0.71 | 0.54 | **0.45** | **0.45** | 0.47 |

Blend beats every swan-only method in 5–6 of 7 swans. The exception is always **Swan9 (COVID)**, where low-market-beta stocks (utilities, REITs) fell hard and the swan-history methods did better.

**B. Main effects on the backtest.** Each row/column is averaged over the other dimension (S&P500 / S&P100):

| | PR mean | Sharpe | max DD |
|---|---|---|---|
| range across preselection methods | **0.26 / 0.22** | 0.08 / 0.10 | 0.06 / 0.06 |
| range across weightings | 0.04 / 0.03 | 0.13 / 0.08 | 0.12 / 0.07 |
| random 30 / 20 stocks (mean of 5) | 0.09 / 0.14 | 0.51 / 0.65 | 0.57 / 0.51 |
| best preselection rows | blend, market β, Barra ridge: 0.60–0.63 / 0.53–0.56 | | |
| swan-only rows (K-means, DBI, DTW, swan mean) | 0.51–0.57 / 0.46–0.52 | | |
| Holt | 0.29–0.47 / 0.29–0.39 | | |
| original MVO weighting | | worst everywhere: 0.49–0.58 / 0.65–0.78 | 0.44–0.49 / 0.37–0.47 |

**Conclusions**
1. **The preselection is the strategy.** It decides crash protection (PR mean moves by 0.22–0.26 between methods, by only 0.03–0.04 between weightings) and is worth +0.4–0.5 PR over a random pick. The weighting mostly affects return volatility and drawdown.
2. **Clustering adds nothing over sorting by average swan beta.** K-means, K-means + DBI and DTW give about the same picks as simply sorting by the mean of the last 5 swan betas; on S&P500 corrected DTW gives exactly the same ranking. DBI picks the K-means seeds with the best-separated clusters, which says nothing about future betas. DTW warps the order of the swans, but a stock's betas in Swan5 and Swan6 are separate events, not a time series that can be shifted. The team's DTW code also passes the square distance matrix to `linkage` as if its rows were features. Fixing that does not change the result.
3. **Holt is harmful.** Extrapolating a trend through 5 unrelated crashes gives the worst picks: actual beta 0.71, barely better than random on some swans.
4. **Plain market beta is the strongest single signal, but swan history adds something.** Blend and Barra ridge are best overall. In a pooled regression of the next swan beta on swan mean, market beta and volatility, the swan mean adds a small amount once market beta is known (coefficient 0.11–0.14, p ≈ 0.05–0.06, negative in Swan4 and Swan10D). But it is what saves the portfolio in an unusual crash like COVID. Averaging the last 3, 5 or all swans makes no difference (rank corr 0.39 / 0.40 / 0.38).
5. **Weighting: avoid the original MVO; the rest are similar.** The original MVO steps hold 6–9 stocks, turn over 3–4×/yr and have the worst Sharpe and drawdown in both universes. Equal, inverse vol, HRP and factor min-var are within noise; factor min-var gives the lowest drawdown on S&P500. HERC has more turnover and is not better.
6. **The Barra model in preselection** (style exposures and sectors as extra features of the ridge forecaster) gives about the same picks as the simple blend. It is used in weighting (factor min-var) and for risk measurement.
7. **Protection vs return.** Swan-only selections hold higher-beta stocks (~0.6 vs ~0.5) and so earn a little more (Sharpe 0.69–0.73 vs 0.66–0.67 on S&P500), at the cost of less crash protection. Which to prefer depends on the strategy's objective, which should be decided explicitly.

## Development: two objectives

`develop.py`. The variants were fixed *before* looking at results, and the decision rules are stated up front, so nothing is tuned to one crash, COVID in particular.

- **Defensive track:** fully invested, maximise crash protection. Judged by PR mean and median, PR mean without COVID (Swan9), the worst leave-one-swan-out PR mean, down capture, and the cost vs the index.
- **Return track:** risk-adjusted return, with crash protection as a side condition. Judged by Sharpe in each sub-period (2004–2014, 2015–2026) and excess return over the **β-matched index + T-bill mix**. That benchmark holds the index at the portfolio's beta and cash for the rest (`data/prices/tbill_13w.csv`); beating it means the stock selection is worth more than simply holding less stock.
- **Adoption rule:** a variant replaces the baseline only if it is better in both universes, without COVID, in both sub-periods, and in at least 5 of 7 swans.

All variants are quarterly with 0.3% cost, inverse vol, a 25% sector cap, averaged over the 3 start months, 2004 → 2026-10.

**Defensive track** (S&P500 point-in-time / S&P100):

| variant (score, lower = safer) | PR mean | PR mean w/o COVID | worst leave-one-out | PR worst | swans beating D0 | max DD | CAGR − index |
|---|---|---|---|---|---|---|---|
| **D0 blend**: ½ swan mean + ½ market β | **0.58 / 0.53** | **0.66 / 0.61** | **0.51 / 0.46** | 0.11 / 0.03 | – | 40% / 39% | −1.2% / +0.1% |
| D1 stress: rank avg of swan mean and β on the worst 10% market days | 0.42 / 0.46 | 0.47 / 0.52 | 0.34 / 0.39 | 0.07 / 0.10 | 1 / 1 | 47% / 39% | −2.9% / −0.4% |
| D2 both safe: max of the two ranks | 0.56 / 0.50 | 0.63 / 0.58 | 0.49 / 0.43 | 0.10 / 0.06 | 2 / 2 | 43% / 43% | −1.8% / −0.7% |
| D3 worst case: rank avg of worst swan β and market β | 0.52 / 0.51 | 0.58 / 0.58 | 0.46 / 0.44 | 0.16 / 0.09 | 1 / 3 | 43% / 39% | −2.2% / −0.2% |
| D4 three: worst swan β, stress β, market β | 0.49 / 0.50 | 0.55 / 0.56 | 0.42 / 0.42 | 0.15 / 0.10 | 1 / 1 | 43% / 38% | −2.6% / −0.6% |

**Return track:**

| | Sharpe | Sharpe 2004–14 | Sharpe 2015–26 | CAGR | excess vs β-matched 2004–14 | excess vs β-matched 2015–26 | max DD | β | PR mean |
|---|---|---|---|---|---|---|---|---|---|
| S&P500 D0 (30 stocks) | 0.70 | 0.79 | 0.63 | 9.5% | +5.4% | −0.9% | 40% | 0.58 | 0.58 |
| S&P500 **R1 lowest third** | **0.74** | 0.77 | **0.70** | 10.3% | +5.1% | −0.8% | 41% | 0.68 | 0.44 |
| S&P500 R2 lowest half | 0.72 | 0.73 | 0.72 | 10.5% | +4.6% | −1.1% | 44% | 0.75 | 0.34 |
| SPY | 0.64 | 0.48 | 0.82 | 10.8% | – | – | 55% | 1.00 | 0 |
| S&P100 D0 (20 stocks) | 0.78 | 0.81 | 0.76 | 10.9% | +6.4% | +0.4% | 39% | 0.61 | 0.53 |
| S&P100 **R1 lowest third** | **0.82** | 0.82 | **0.82** | 11.3% | +6.4% | +0.5% | 39% | 0.64 | 0.45 |
| S&P100 R2 lowest half | 0.82 | 0.82 | 0.83 | 11.8% | +6.6% | +0.2% | 40% | 0.71 | 0.38 |
| OEF | 0.65 | 0.44 | 0.85 | 10.8% | – | – | 54% | 1.00 | 0 |

**Conclusions**
1. **Defensive track: keep D0.** None of the new scores passes the adoption rule. The worst-case and three-signal scores improve the worst swan (COVID) by ~0.05 PR, but lose 0.04–0.10 of mean PR and are worse in 5–6 of the 7 swans. That is exactly the "fit to COVID" trade the rule is meant to reject. The stress beta alone is clearly worse.
2. **Return track: hold the lowest third (R1).** It has a higher Sharpe than D0 and the index in both universes, the most even Sharpe across the two sub-periods, the lowest turnover (~0.85×/yr), and still roughly halves the index's losses in the average swan (PR mean ~0.45). Lowest half (R2) is about the same with less protection.
3. **The strategy's edge over "just hold less stock" was earned before 2015.** Against the β-matched index + T-bill mix, both tracks earned +4.6–6.6%/yr in 2004–2014 and about −1% to +0.5%/yr in 2015–2026. Since 2015 the index (mega-cap growth) has had the higher Sharpe (0.82–0.85). What the strategy still delivered then is crash protection: PR 0.53–1.0 in Swan8, Swan10D and Swan11. Whether the pre-2015 edge (the low-beta anomaly) comes back is the main open question.

## Drawdowns and returns: diagnosis and layers

`defense.py`, S&P500 point-in-time, quarterly rebalance, 0.3% cost, 2004 → 2026-10.

**Does it make money?**

| | CAGR | Sharpe | max DD (when) | GFC 07-10→09-03 | COVID | 2022 | β | alpha |
|---|---|---|---|---|---|---|---|---|
| original K-means + MVO, quarterly | 11.7% | 0.77 | 37% (COVID) | −20% | −37% | −2% | 0.58 | +4.7% |
| original, monthly as the team ran it | 10.8% | 0.72 | 34% (GFC) | −30% | −33% | −5% | 0.57 | +3.9% |
| defensive D0 | 9.6% | 0.71 | 40% (GFC) | −39% | −31% | −10% | 0.58 | +2.6% |
| return R1 | 10.5% | 0.75 | 40% (GFC) | −40% | −33% | −12% | 0.68 | +2.6% |
| SPY | 10.8% | 0.65 | 55% (GFC) | −55% | −34% | −24% | 1.00 | 0 |

Alpha is CAGR − T-bill − β × (SPY − T-bill). Since 2011-11 (when the low-vol ETFs start): SPY 14.9% / Sharpe 0.92, USMV 11.8% / 0.90, R1 11.7% / 0.88, D0 10.0% / 0.77, original 10.7% / 0.72.

**Why the drawdowns are large and the returns do not beat buy-and-hold**
- **It is a long-only stock portfolio with β ≈ 0.6.** In a crash where everything falls, it loses about β × the market's fall plus its own stock risk: 0.58 × 55% ≈ 32% from the market alone in 2008. No stock selection removes that systematic part. In COVID even the defensive stocks fell as much as the market.
- **The return is β × the market plus a small alpha.** The alpha is +2.6% to +4.7%/yr, almost all of it before 2015. From 2015 to 2026 the market returned 13.7%/yr, and holding 0.6 of it cost about 5%/yr.
- The original has a little more alpha but concentrates in ~7 stocks. Its result depends on timing, and in COVID it lost more than the index.

**Additions tested** (fixed in advance; each also applied to SPY to see whether the layer, rather than the stock picking, does the work):

| | CAGR | Sharpe | Sharpe 2004–14 / 2015–26 | max DD | worst year |
|---|---|---|---|---|---|
| R1 alone | 10.5% | 0.75 | 0.79 / 0.71 | 40% | −23% |
| R1 + momentum filter (selection) | 9.9% | 0.71 | 0.76 / 0.65 | 40% | −24% |
| R1 + stock trend filter (selection) | 6.5% | 0.71 | 0.98 / **0.49** | 27% | −11% |
| R1 + index trend filter (10-month average) | 8.4% | 0.89 | 1.20 / **0.62** | 16% | −10% |
| R1 + 10% vol target | 9.3% | 0.88 | 1.07 / 0.71 | 29% | −13% |
| R1 60% / Treasuries 40% | 7.8% | 0.90 | 1.10 / 0.73 | 21% | −8% |
| **R1 + Treasuries + gold, risk parity** | 7.2% | **1.10** | 1.42 / **0.82** | **16%** | −8% |
| SPY + Treasuries + gold, risk parity | 7.0% | 1.05 | 1.18 / 0.94 | 17% | −12% |
| **R1 60% / Treasuries 20% / gold 20%** (no leverage) | **9.2%** | **0.96** | 1.04 / **0.88** | **23%** | −10% |
| SPY 60% / Treasuries 20% / gold 20% | 9.6% | 0.85 | 0.71 / 0.99 | 33% | −20% |
| **R1 risk parity, levered to 10% vol** (avg 1.65×, T-bill + 0.5% financing) | **10.4%** | 1.01 | | **21%** | −12% |
| SPY risk parity, levered to 10% vol | 9.9% | 0.97 | | 22% | −17% |

**Conclusions**
1. **Stock-level additions do not fix it.** The momentum filter changes nothing. The stock trend filter cuts the drawdown but whipsaws after 2015, so it fails the sub-period rule.
2. **The index trend filter** has the best drawdown control, but its gain is mostly 2008 and it lags after 2015. That is the same one-event dependence the COVID rule guards against.
3. **What lowers the drawdown robustly is owning other assets.** Risk parity across the equity sleeve, 7–10y Treasuries and gold cuts the max drawdown to ~16% with a Sharpe above 1, and it is better than its parts in both sub-periods. Levered to 10% volatility, it earns about the market's return (10.4% vs 10.8%) with a max drawdown of ~21% instead of 55%.
4. **The stock selection adds only a little on top.** SPY in the same risk-parity mix is close behind (Sharpe 1.05 vs 1.10, max DD 17% vs 16%). The defense comes mainly from the asset mix; swan-beta selection is a modest improvement to the equity sleeve.
5. Caveats: Treasuries failed in 2022 (IEF −15%) and only gold helped; gold's strong 2024–26 run flatters the mix. Leverage needs futures or margin and has its own risks.

## Lowering drawdowns inside the stock book

`equity_defense.py`: still fully invested and long-only, fixed in advance. S&P500 point-in-time, quarterly, 0.3% cost, mean of 3 start months.

| | CAGR | Sharpe | Sharpe 2004–14 / 2015–26 | max DD | GFC | COVID | 2022 | PR mean |
|---|---|---|---|---|---|---|---|---|
| D0 | 9.5% | 0.70 | 0.79 / 0.63 | 40% | −39% | −30% | −10% | 0.58 |
| R1 | 10.3% | 0.74 | 0.77 / 0.70 | 41% | −40% | −33% | −12% | 0.44 |
| E1 min variance on crash-day covariance (60 lowest blend) | 9.0% | 0.67 | 0.68 / 0.66 | 43% | −42% | −31% | −10% | 0.52 |
| E2 equal risk per sector (D0 stocks) | 9.4% | 0.69 | 0.71 / 0.68 | 41% | −41% | −29% | −12% | 0.56 |
| E3 min variance with stress beta ≤ 0.4 | 7.6% | 0.61 | 0.60 / 0.61 | 45% | −43% | −32% | −18% | 0.36 |
| E4 R1 above the 10-month average, D0 below (always invested) | 10.0% | 0.73 | 0.76 / 0.70 | 40% | −39% | −33% | −12% | 0.46 |
| original K-means + MVO | 12.1% | 0.78 | 0.97 / 0.61 | 36% (32–38%) | −26% | −35% | −4% | 0.63 |

**None of the four lowers the drawdown.** A fully invested stock book bottoms out around 35–40% in 2008, when nearly every stock fell.

The original fell less in 2008 (−26%) because MVO put 100% in utilities and consumer staples, with no health care, energy or industrials. D0's 25% sector cap spreads into those sectors, and they fell more in 2008. The same concentration hurt in COVID: 69% utilities in January 2020, which fell 35%, more than the index. So inside the stock book, betting on the most defensive sectors helps in some crashes and hurts in others; no variant is protective in all of them.

**With ETFs allowed, the drawdown comes down.** The only additions that lowered the drawdown robustly were Treasury and gold ETFs (previous section). The simplest is a fixed **60% R1 / 20% IEF / 20% GLD**: 9.2% CAGR, max drawdown 23% (SPY 55%), worst year −10% (SPY −37%), Sharpe 0.96. That beats both R1 alone and SPY in both sub-periods. Here the stock selection clearly matters: the same mix with SPY as the stock part draws down 33%.

## Long-term allocation with stocks and ETFs (personal use, no leverage or options)

Goal set on 2026-10-23: max drawdown ≤ 15% first, then as much return as possible. `allocation.py`, variants fixed in advance, monthly rebalance, 0.1% cost, 2004 → 2026-10:

| stock sleeve R1 (swan-beta return track) + | CAGR | Sharpe | max DD | worst year | CAGR 2004–14 / 2015–26 | GFC | COVID | 2022 |
|---|---|---|---|---|---|---|---|---|
| nothing (100% stocks) | 10.6% | 0.76 | 41% | −23% | – | −40% | −33% | −12% |
| A0 60% / 20% IEF / 20% GLD | 9.4% | 0.97 | 22.7% | −8.7% | 10.3% / 8.5% | −18% | −21% | −12% |
| A5 A0 + 8% vol target (rest SHY) | 8.3% | 1.02 | 20.2% | −9.4% | 9.5% / 7.2% | −11% | −20% | −10% |
| A6 A0 + drawdown control | 8.6% | 0.98 | 20.6% | −9.8% | 9.8% / 7.4% | −13% | −19% | −11% |
| **A1 risk parity: R1, IEF, GLD** | **7.1%** | **1.09** | **15.5%** | −8.2% | 8.7% / 5.6% | +6% | −8% | −13% |
| A2 risk parity + TIP + DBC | 5.4% | 0.94 | 15.8% | −7.2% | 6.3% / 4.6% | +2% | −7% | −10% |
| A3 / A4 A2 + vol target / drawdown control | 5.0–5.2% | 0.94 | 15.2–15.8% | ~−7% | | | | |

With D0 as the sleeve the numbers are similar (A4 is the only variant at ≤ 15%: 14.7% max DD, 4.8% CAGR). SPY as the sleeve: A0 9.9% / 31.6% max DD, A1 7.0% / 17.5%.

**Conclusions**
1. **The frontier is clear: about 7%/yr at a 15–16% max drawdown, 8–9% at 20–22%, 10–11% at 40%, and SPY's 10.8% at 55%.** Nothing tested, and nothing realistic without leverage, gets 20%/yr with a ≤ 15% drawdown over 2004–2026.
2. **Inflation assets do not help.** Adding TIP and commodities (DBC) leaves the drawdown at ~16% and lowers the return by ~1.7%/yr. The vol target and drawdown control add little on top of risk parity; on the 60/20/20 mix they cut ~2 points of drawdown for ~1% of return.
3. **Where the swan-beta selection matters:** in an equity-heavy mix it cuts the drawdown clearly (A0: 22.7% with R1 vs 31.6% with SPY). In risk parity, where stocks are only ~25–30% of the risk budget, it matters little (15.5% vs 17.5%).
4. The 15–16% drawdown of the risk-parity mixes comes from 2022, when stocks and bonds fell together. Since 2015 their returns are ~5–6%/yr, because bonds have been weak.

## Chosen allocation: robustness and live use

Chosen on 2026-10-23: **A1 risk parity of the R1 sleeve, IEF and GLD** (inverse 1-year vol, split reset monthly, sleeve rebuilt quarterly). `rp_robustness.py`, 2004 → 2026-10:

| change from the base | CAGR | Sharpe | max DD | worst year |
|---|---|---|---|---|
| **base** | **7.1%** | **1.09** | **15.5%** | −8.1% |
| sleeve rebuilt in month 2 / 3 of the quarter | 7.0% / 7.1% | 1.08 / 1.09 | 15.6% | −8.2% |
| costs 0.5% stocks, 0.2% ETFs | 7.0% | 1.08 | 15.7% | −8.2% |
| vol lookback 6 / 24 months | 7.2% / 7.2% | 1.11 / 1.10 | 14.7% / 16.4% | −8.0% / −8.8% |
| split reset quarterly | 7.2% | 1.09 | 15.7% | −8.0% |
| TLT instead of IEF | 9.0% | 1.03 | 19.0% | −10.8% |
| SHY instead of IEF | 3.3% | 1.39 | 6.9% | −3.3% |
| no gold | 5.6% | 0.95 | 15.5% | −10.6% |
| ends 2023-12 (before gold's run) | 6.6% | 1.03 | 15.5% | −8.1% |
| sleeve of 30 / 50 / 75 stocks | 7.0% | 1.04–1.06 | 16.9% / 16.4% / 16.0% | ~−7% |
| sleeve D0 (30 stocks, sector cap) | 6.9% | 1.05 | 16.2% | −7.2% |
| sleeve SPY | 7.0% | 1.07 | 17.5% | −11.7% |

The result does not hinge on timing, costs or the lookback: ~7% a year with a 15–17% max drawdown. Gold adds ~1.5%/yr (7.1% vs 5.6%); stopping at 2023-12, before gold's 2024–26 run, the mix still earns 6.6%. Longer bonds (TLT) buy return with drawdown; in 2022 they fell with stocks. A 30-stock sleeve, for small accounts, costs ~1 point of max drawdown.

**Live use** (`v2/live.py`; everything it writes goes to `live/`, which is git-ignored because it holds your positions):

```
.venv/bin/python data/update_data.py                                     # 1. data up to date (~4 min)
.venv/bin/python v2/live.py target --capital 20000 [--stocks 30 --whole]   # 2. target holdings
.venv/bin/python v2/live.py orders --holdings live/holdings.csv            # 3. trades, sells first
.venv/bin/python v2/live.py report --holdings live/holdings.csv            #    market, split, risk, drift
```

- Schedule: the split is reset on the first trading day of each month; the sleeve is rebuilt in January, April, July and October (`target` reuses the quarter's saved sleeve of the same size otherwise). Between dates, rebalance only if `report` flags a drift above 5%.
- `holdings.csv` has columns `ticker,shares`, plus a `CASH` row in dollars. `orders` refuses to run if your holdings are worth more than 2% away from the capital the target was built for; re-run `target` with the actual value.
- Small accounts: the default sleeve is ~160 stocks at ~$30–50 each per $20k, which needs fractional shares. Without them use `--stocks 30 --whole`: whole shares are rounded to the nearest share within the cash, leaving ~$50 and ≤ 1% off target per name. Stocks are only ~25–30% of the money; the rest is IEF and GLD.
- The split moves with volatility: on 2026-10-23 it was stocks 28%, IEF 62%, GLD 11% (gold has been volatile). That is the rule as backtested, not a view.

## Fixes compared with the original MVO2 notebook
1. **P_fwd price basis.** optdata `P_fwd` is a dollar target quoted on the portfolio date. The original compared it with Yahoo prices adjusted back from *today*, so splits made R_fwd several times too large (NEE and AAPL ×4, WMT ×3, AMZN and GOOGL ×20) and the optimizer piled into them. v2 compares it with the nominal price of that day (`data/prices/nominal_close_SP100.csv`).
2. **Type E outputs used the wrong prices.** `output()` multiplied weights sorted by PR_SR against prices sorted alphabetically (yfinance sorts them). The MVO-weight curve gave PR 0.056 instead of 0.233. v2 always aligns by ticker.
3. **K-means crashed whenever a second cluster was needed.** `concat(ignore_index=True)` dropped the tickers, and `Series[0]` is an error on pandas 3. `n_init=10` is now pinned, because sklearn ≥1.4 runs only one initialisation by default and so gives different clusters.
4. `data_cleaning` filled the wrong rows (it used the loop counter as the row). The 1-year and half-year returns were identical, since both used the first 30 days. Weekly std was scaled by √(number of weeks) instead of √52.
5. **Dead code removed.** The GARCH block was never used (12 monthly points < 100). The 90-day max-drawdown loop just recomputed the 1-year MDD. Also removed: unused report-only columns (WEDataBank), quandl and pandas_datareader.

Known and kept as in the original: renormalising after the 3% filter can push a weight to 0.175, above the 0.17 cap.

## Data notes
- **Swan Beta is computed in-house, not taken from Bloomberg.** Swan β = a stock's peak→trough return ÷ the benchmark's peak→trough return, on Yahoo adjusted close. The benchmark is ^OEX from Swan1 onward and ^GSPC for Swan-4 to Swan-1, before ^OEX existed. The 14 windows are in `data/swans.csv`. They were found by searching for the windows that reproduce `All_SwanBeta100.xlsx`, and they reproduce every column to within 2e-4. `All_SwanBeta500.xlsx` uses SPY as the benchmark instead.
- **Hand list vs rule.** The hand list cannot be reproduced exactly by any threshold: Swan1 starts on 1987-08-17, not the 08-25 peak, and Swan5 is only the 2008-09-24 → 2009-03-09 leg. `swan_beta.detect_swans()` is a fixed rule instead: an episode where the index is more than 5% below its trailing 1-year high and at some point at least 15% below it; trough = the episode's lowest close, peak = the highest close in the year before. From 1987 it finds 11 swans: the 10 modern hand swans (Swan1–6, 8–11) plus 2010-07 (−16.3%). It misses only Swan7 (2015, −13.3%). A 20% threshold would miss 2018 (−19.96%) and 2025 (−19.9%) by a hair, and 10% gives 38 events, mostly minor.
- **Confirmed date.** A swan's trough is only final once the index is back within 5% of its 1-year high. The backtest uses a swan only after that date (e.g. Swan9 from 2020-06-05, Swan11 from 2025-05-13). MarketDD's T2 is also recomputed with prices up to each date, so nothing from the future leaks in.
- **The only ^OEX decline of 12% or more after Swan10D is 2025-02-19 → 2025-04-08 (−19.9%)**, up to 2026-10-16. It is the natural Swan11 / out-of-sample test. `optdata_SP100_20241202.csv` is point-in-time data for a portfolio built just before it.
- **MarketDD.xlsx does not match the beta windows for several swans.** DD is close everywhere, but T1 and/or T2 differ for Swan1, 3, 4, 6, 7 and 10D. Examples: Swan7 has T1 85 vs 27 days, and Swan10D has T2 415 vs 297, since ^OEX recovered on 2023-12-18. These feed the predicted market-drop curve. `swan_beta.py` prints the full comparison.
- **`optdata_MVO2.xlsx` (Swan9's analyst targets) is look-ahead.** It has no date, and P_fwd only lines up with prices around May–Aug 2020, i.e. after the crash: OXY and BA targets are 0.4–0.5× their Feb-2020 price, and AAPL is still pre-split. The other S&P 100 optdata files match their filename dates (median P_fwd / price 1.03–1.11).
- One set of inputs is enough. v2 reads only `data/SwanBeta`, `data/MVO_Components`, `data/optdata`, `data/prices` and `data/swans.csv`. The copies are redundant: `data/MVO Components/` (a subset of `data/MVO_Components/`), `dtw_result/dtw_data/`, and `Projects/Gradient Matrix Test/Swan10d/Code/`. The latter two have older ticker lists.
- `All_SwanBeta100.xlsx` (102 tickers, up to Swan10D, exact) and `All_SwanBeta100_MVO2.xlsx` (95 tickers, closer to the 2020 index, Swan4/Swan8 off by up to 0.02, Swan10a/10b instead of 10D) are two versions of the same table. The baseline keeps `_MVO2` so it stays comparable with the original.
- WBA and AVG are no longer on Yahoo. BK is downloaded as BNY, and the 2026-05 BNY membership entry is read as BK.
- MCD, OKE, PSA, WRB, VTRS and FERG had empty S&P500 price columns: Yahoo dropped them when the prices were frozen. `update_data.py` filled their full history on 2026-10-23, and `develop.py`, `allocation.py`, `defense.py` and `rp_robustness.py` were re-run (changes ≤ 0.5 points). The other caches (`experiments`, `explore`, `factor`, `robustness`, `equity_defense`, `defense_selection`) still come from the data before the fill.
- S&P500 prices cover the Swan Beta file's tickers plus every member since 1996 that Yahoo still has (1,213 columns). A reused ticker can map an old member to a newer company with the same symbol; this has not been checked one by one.
- ^OEX / ^GSPC are price-only; OEF (from 2000) and SPY (from 1993) are the total-return benchmarks in the price files.
