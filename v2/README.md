# v2: fixed and refactored strategy code

The original notebooks in `training/` and `Projects/` are left untouched. `v2/` holds the fixed, function-based version,
so each pipeline component (preselection, beta forecast, market-drop scenario, return / risk estimate, optimizer,
post-allocation) can be swapped and compared against the baseline.

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python v2/mvo2.py      # Swan9 baseline -> v2/output/, asserts reference PR 0.2225
.venv/bin/python v2/compare.py   # original notebook vs v2 on the same frozen data
.venv/bin/python v2/swan_beta.py # recompute every Swan Beta from prices, check it against All_SwanBeta100.xlsx
.venv/bin/python v2/walkforward.py  # predict Swan5..Swan11 one by one from the 5 swans before each
.venv/bin/python v2/backtest.py SP100 hand   # monthly-rebalance backtest 2004 -> today (SP100|SP500, hand|rule swans)
```

| File | What |
|---|---|
| `mvo2.py` | MVO 2.0 as one function per step; `run(params)` returns stats, weights, curves and PR |
| `compare.py` | runs the original `training/MVO` notebook offline next to v2 and prints weights / PR side by side |
| `swan_beta.py` | Swan Beta matrix and swan market stats (DD, T1, T2) from `data/swans.csv` and the frozen prices |
| `walkforward.py` | walk-forward test over Swan5–Swan11 on the unified data, with and without analyst targets |
| `backtest.py` | monthly rebalance from 2004, using only swans confirmed and prices known at each date; compares MVO with ablations and simple baselines |

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
| Swan8 2018 | no P_fwd | 0.577 | 0.817 | 0.423 | +7.8% |
| Swan9 2020 | no P_fwd | 0.263 | 0.191 | 0.739 | −2.5% |
| | P_fwd (look-ahead) | 0.225 | 0.191 | 0.777 | −3.8% |
| | original input files | 0.222 | 0.182 | 0.777 | −3.8% |
| Swan10D 2022 | no P_fwd | 0.787 | 0.616 | 0.213 | +20.2% |
| | P_fwd | 0.629 | 0.616 | 0.371 | +4.8% |
| Swan11 2025 (out of sample) | no P_fwd | 0.678 | 0.773 | 0.295 | −4.3% |
| | P_fwd | 0.679 | 0.773 | 0.295 | −2.0% |

\* Portfolio minus index return, from the portfolio date until the index is back at its pre-crash peak.

- The predicted portfolio β is about 0.41–0.46 every time; the actual β ranges from 0.09 to 0.80.
- Without P_fwd: mean PR 0.527. Equal-weighting the preselected stocks instead gives 0.550, and MVO beats it in only 3 of 7 swans.
- Analyst targets never helped: worse on Swan9 and Swan10D, a tie on Swan11.
- Seven events is a small sample. Treat these numbers as a baseline to compare upgrades against, not as proof.

## Backtest: monthly rebalance, 2004-01 → 2026-10, 10 bp cost per unit of turnover

Each month the portfolio is rebuilt from data known that day: the last 5 *confirmed* swans, prices up to the day before, and trend returns (no analyst targets). It is then held for the month.

| hand swans | S&P100 MVO | no injection | Min-vol | Preselect EW | Universe EW | ^OEX | Switch |
|---|---|---|---|---|---|---|---|
| CAGR | 10.1% | 10.1% | 10.2% | 11.0% | 12.9% | 6.9% | 7.5% |
| vol | 15.2% | 15.2% | 14.3% | 15.1% | 18.1% | 18.0% | 16.3% |
| Sharpe (rf=0) | 0.71 | 0.71 | 0.75 | 0.77 | 0.76 | 0.46 | 0.52 |
| max drawdown | 32% | 36% | 34% | 36% | 47% | 52% | 33% |
| beta | 0.59 | 0.59 | 0.57 | 0.64 | 0.96 | 1.00 | 0.83 |

| hand swans | S&P500 MVO | no injection | Min-vol | Preselect EW | Universe EW | ^GSPC | Switch |
|---|---|---|---|---|---|---|---|
| CAGR | 10.5% | 10.3% | 8.3% | 9.6% | 13.6% | 7.2% | 7.2% |
| vol | 15.0% | 15.1% | 14.2% | 14.8% | 19.2% | 18.2% | 15.9% |
| Sharpe (rf=0) | 0.74 | 0.73 | 0.63 | 0.70 | 0.76 | 0.48 | 0.52 |
| max drawdown | 35% | 36% | 36% | 37% | 52% | 53% | 36% |
| beta | 0.55 | 0.56 | 0.54 | 0.59 | 1.02 | 1.00 | 0.80 |

Per-swan PR and the rule-based swan runs are printed by `backtest.py` and saved under `v2/output/`. The rule list gives the same picture: S&P100 MVO CAGR 11.6%, max drawdown 44%; S&P500 9.2%, 37%.

- **The strategy does what it is meant to.** Beta is about 0.55–0.6, the max drawdown is 32–35% vs 52–53% for the index, and it lost less than the index in every swan since 2004. COVID was the weakest case (PR 0.08–0.24).
- **Survivorship bias is large.** Equal-weighting today's constituents beats the index by about 6%/yr, because companies that later failed or were dropped are missing. So the strategy's CAGR edge over the index is not trustworthy. The fair comparison is with Universe EW, which has the same bias: the strategy earns ~2–3%/yr less but with ~40% lower beta and drawdown, at a similar Sharpe.
- **The crash injection adds little.** Same CAGR and Sharpe, 1–4 points less drawdown. The protection comes from the swan-beta preselection.
- **MVO adds little over equal-weighting the preselection**, at 5–6× the turnover. Min-vol is no better.
- **Detecting a swan and switching then is too late.** By the time the index is 10% down, much of the fall has happened; Switch earns about the index's return with only a little less drawdown.

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
