# v2: fixed and refactored strategy code

The original notebooks in `training/` and `Projects/` are left untouched. `v2/` holds the fixed, function-based version,
so each pipeline component (preselection, beta forecast, market-drop scenario, return / risk estimate, optimizer,
post-allocation) can be swapped and compared against the baseline.

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python v2/mvo2.py      # Swan9 baseline -> v2/output/, asserts reference PR 0.2225
.venv/bin/python v2/compare.py   # original notebook vs v2 on the same frozen data
```

| File | What |
|---|---|
| `mvo2.py` | MVO 2.0 as one function per step; `run(params)` returns stats, weights, curves and PR |
| `compare.py` | runs the original `training/MVO` notebook offline next to v2 and prints weights / PR side by side |

## Baseline: Swan9 from Swan4–Swan8, S&P 100, K-means N=7

| PR | original | v2 without nominal fix | v2 |
|---|---|---|---|
| MVO weights | 0.2327 | 0.2327 | **0.2225** |
| post-allocation: MVO weight | 0.0563 | 0.2327 | 0.2225 |
| post-allocation: equal | 0.2117 | 0.2117 | 0.2220 |
| post-allocation: adjusted proportion | 0.1839 | 0.2195 | 0.2081 |
| post-allocation: direct | 0.1663 | 0.2031 | 0.2081 |

## Fixes compared with the original MVO2 notebook
1. **P_fwd price basis.** optdata `P_fwd` is a dollar target quoted on the portfolio date. The original compared it with Yahoo prices adjusted back from *today*, so splits made R_fwd several times too large (NEE and AAPL ×4, WMT ×3, AMZN and GOOGL ×20) and the optimizer piled into them. v2 compares it with the nominal price of that day (`data/prices/nominal_close_SP100.csv`).
2. **Type E outputs used the wrong prices.** `output()` multiplied weights sorted by PR_SR against prices sorted alphabetically (yfinance sorts them). The MVO-weight curve gave PR 0.056 instead of 0.233. v2 always aligns by ticker.
3. **K-means crashed whenever a second cluster was needed.** `concat(ignore_index=True)` dropped the tickers, and `Series[0]` is an error on pandas 3. `n_init=10` is now pinned, because sklearn ≥1.4 runs only one initialisation by default and so gives different clusters.
4. `data_cleaning` filled the wrong rows (it used the loop counter as the row). The 1-year and half-year returns were identical, since both used the first 30 days. Weekly std was scaled by √(number of weeks) instead of √52.
5. **Dead code removed.** The GARCH block was never used (12 monthly points < 100). The 90-day max-drawdown loop just recomputed the 1-year MDD. Also removed: unused report-only columns (WEDataBank), quandl and pandas_datareader.

Known and kept as in the original: renormalising after the 3% filter can push a weight to 0.175, above the 0.17 cap.

## Data notes
- **Swan Beta is computed in-house, not taken from Bloomberg.** Swan β = stock return ÷ ^OEX return from the swan's peak to its trough, on Yahoo adjusted close. This reproduces the stored values exactly (error < 1e-4) for Swan8 (2018-10-03 → 2018-12-24), Swan9 (2020-02-19 → 2020-03-23) and Swan10D (2022-01-03 → 2022-10-12). `All_SwanBeta500.xlsx` uses SPY as the benchmark instead. Swan1–Swan7 were not checked.
- `MarketDD.xlsx` matches ^OEX: Swan9 drop 31.53%, Swan10D 27.2%.
- **`optdata_MVO2.xlsx` (Swan9's analyst targets) is look-ahead.** It has no date, and P_fwd only lines up with prices around May–Aug 2020, i.e. after the crash: OXY and BA targets are 0.4–0.5× their Feb-2020 price, and AAPL is still pre-split. The other S&P 100 optdata files match their filename dates (median P_fwd / price 1.03–1.11).
- One set of inputs is enough. v2 reads only `data/SwanBeta`, `data/MVO_Components`, `data/optdata` and `data/prices`. The copies are redundant: `data/MVO Components/` (a subset of `data/MVO_Components/`), `dtw_result/dtw_data/`, and `Projects/Gradient Matrix Test/Swan10d/Code/`. The latter two have older ticker lists.
- `All_SwanBeta100.xlsx` (102 tickers, up to Swan10D, matches recomputed betas exactly) and `All_SwanBeta100_MVO2.xlsx` (95 tickers, closer to the 2020 index, Swan4/Swan8 off by up to 0.02, Swan10a/10b instead of 10D) are two versions of the same table. The baseline keeps `_MVO2` so it stays comparable with the original.
- WBA and AVG are no longer on Yahoo. BK is downloaded as BNY.
