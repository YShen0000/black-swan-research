# v2: fixed and refactored strategy code

The original notebooks in `training/` and `Projects/` are left untouched. `v2/` holds the fixed, function-based version,
so each pipeline component (preselection, beta forecast, market-drop scenario, return / risk estimate, optimizer,
post-allocation) can be swapped and compared against the baseline.

**Recommended setup (2026-10):** each month, predict every stock's next-swan beta as ½ × the mean of its last 5 swan betas + ½ × its 1-year market beta. Hold the 30 lowest (S&P500, point-in-time members) or 20 lowest (S&P100) at inverse-volatility weights: `components.recommended(date, universe)`. See [Component experiments](#component-experiments) for why.

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python v2/mvo2.py      # Swan9 baseline -> v2/output/, asserts reference PR 0.2225
.venv/bin/python v2/compare.py   # original notebook vs v2 on the same frozen data
.venv/bin/python v2/swan_beta.py # recompute every Swan Beta from prices, check it against All_SwanBeta100.xlsx
.venv/bin/python v2/walkforward.py  # predict Swan5..Swan11 one by one from the 5 swans before each
.venv/bin/python v2/backtest.py SP100 hand   # monthly-rebalance backtest 2004 -> today (SP100|SP500, hand|rule swans)
.venv/bin/python v2/experiments.py SP500      # component grid on point-in-time S&P500 members (~45 min, cached after)
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

`backtest.py`: each month the original pipeline is rebuilt from data known that day (the last 5 *confirmed* swans, prices up to the day before, trend returns), then held for the month. S&P100, hand swans:

| | MVO | no injection | Min-vol | Preselect EW | Universe EW | ^OEX | Switch |
|---|---|---|---|---|---|---|---|
| CAGR | 9.6% | 9.5% | 9.8% | 10.6% | 12.9% | 6.9% | 7.2% |
| Sharpe (rf=0) | 0.68 | 0.67 | 0.72 | 0.74 | 0.76 | 0.46 | 0.51 |
| max drawdown | 32% | 36% | 34% | 36% | 47% | 52% | 33% |
| beta | 0.59 | 0.60 | 0.57 | 0.65 | 0.96 | 1.00 | 0.83 |

- **The crash injection adds little**: same return, a few points less drawdown. The protection comes from the swan-beta preselection.
- **Detecting a swan and switching then is too late.** By the time the index is 10% down, much of the fall has happened. *Switch* (hold the index, move to MVO when the index is ≥10% below its 1-year high) earns about the index's return.
- ^OEX / ^GSPC are price indices with no dividends, while stock prices include them, so they understate the market by ~2%/yr. Use OEF / SPY (total return) when comparing returns; `experiments.py` reports both.

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
| K-means + MVO (original) | 10.3% | 15.5% | 0.71 | 37% | 0.57 | 7.1 | 0.64 | 0.02 |
| K-means + equal | 9.3% | 15.4% | 0.66 | 34% | 0.61 | 3.9 | 0.58 | 0.09 |
| **blend + inverse vol** | 9.0% | 14.5% | 0.67 | 37% | 0.54 | 2.1 | 0.62 | 0.08 |
| swan mean + inverse vol | 9.5% | 15.4% | 0.67 | 37% | 0.65 | 0.9 | 0.56 | 0.09 |
| market beta + inverse vol | 8.0% | 13.9% | 0.63 | 37% | 0.50 | 2.9 | 0.61 | 0.04 |
| ridge + inverse vol | 7.6% | 14.0% | 0.59 | 39% | 0.51 | 2.5 | 0.61 | 0.07 |
| universe equal weight | 10.3% | 19.6% | 0.60 | 58% | 1.04 | 0.7 | 0.03 | −0.14 |
| SPY (incl. dividends) | 9.0% | 18.1% | 0.57 | 52% | 0.99 | 0 | 0.02 | 0.01 |

| S&P100 (today's list) | CAGR | vol | Sharpe | max DD | beta | turnover/yr | PR mean | PR worst |
|---|---|---|---|---|---|---|---|---|
| K-means + MVO (original) | 9.6% | 15.2% | 0.68 | 32% | 0.59 | 4.1 | 0.54 | 0.19 |
| **blend + inverse vol** | 9.4% | 13.9% | 0.72 | 34% | 0.58 | 1.4 | 0.57 | 0.07 |
| swan mean + inverse vol | 11.1% | 14.6% | 0.80 | 36% | 0.66 | 0.8 | 0.49 | 0.08 |
| universe equal weight | 12.9% | 18.1% | 0.76 | 47% | 0.96 | 0.6 | 0.10 | −0.09 |
| OEF (incl. dividends) | 8.9% | 17.9% | 0.56 | 51% | 0.99 | 0 | 0.01 | −0.03 |

**3. Conclusions**
- **Every combination does the job.** About the market's return (SPY 9.0%) with beta 0.5–0.65, ~20% less volatility and a max drawdown of 34–40% instead of 52%. The rows shown lost less than the index in every one of the 7 swans since 2004 (PR worst > 0, PR mean 0.5–0.7). COVID is the weak spot for all (PR 0.02–0.24).
- **No combination is significantly better than another on Sharpe.** In a monthly-block bootstrap, the 90% interval of the Sharpe difference vs the original includes 0 for every variant (e.g. blend + inverse vol: S&P500 [−0.19, +0.11], S&P100 [−0.09, +0.19]). With 22 years and 7 crashes, differences of ±0.1 Sharpe are noise.
- **Why blend + inverse vol is recommended anyway:** it is the best beta predictor in both universes, and its crash protection (PR mean) matches the original. It needs a third to a half of the original's turnover, which matters once real costs are above 10 bp. It also drops the tuned parts: no K-means seed, crash injection, analyst targets or 0.5/1.3/0.7 thresholds, so there is less to overfit. The cost is slightly lower return than the original on S&P500 (9.0% vs 10.3%, not significant).
- If return matters more than protection: swan mean + inverse vol has the lowest turnover and a higher return, but more beta (0.65) and less protection.
- MVO (max Sharpe on a return estimate) and min-vol never help consistently; equal and inverse-vol weights are as good.
- Survivorship bias remains. Point-in-time membership is in place, but Yahoo has no prices for about half the 2004 members (mostly later delisted or acquired; coverage 52% in 2004, 60% in 2008, 67% in 2012, 86% in 2020, 97% in 2026). The point-in-time universe equal weight beating SPY by 1.3%/yr is the remaining bias plus the equal-weight effect. Removing it needs paid data (Norgate, CRSP).

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
- WBA and AVG are no longer on Yahoo. BK is downloaded as BNY.
- S&P500 prices cover the Swan Beta file's tickers plus every member since 1996 that Yahoo still has (1,213 columns). A reused ticker can map an old member to a newer company with the same symbol; this has not been checked one by one.
- ^OEX / ^GSPC are price-only; OEF (from 2000) and SPY (from 1993) are the total-return benchmarks in the price files.
