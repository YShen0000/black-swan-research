"""Swan Beta and swan market statistics, computed from frozen prices and the swan windows in data/swans.csv.

    Swan beta of a stock = its peak->trough return / the benchmark's peak->trough return  (adjusted close)
    DD = benchmark drop, T1 = trading days peak..trough (inclusive), T2 = trading days until the benchmark regains the peak

The windows in data/swans.csv were reverse-engineered: they reproduce every column of
data/SwanBeta/All_SwanBeta100.xlsx exactly. Running this file re-checks that, and compares with MarketDD.xlsx.

    python v2/swan_beta.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"
SWANS = pd.read_csv(DATA / "swans.csv", index_col="swan", parse_dates=["peak", "trough"])
CLOSE = pd.read_csv(DATA / "prices/close_SP100.csv.gz", index_col=0, parse_dates=True)


def swan_betas(prices=CLOSE, swans=SWANS):
    """Stocks x swans matrix of Swan Beta (NaN where the stock has no price at the peak or trough)."""
    stocks = prices.drop(columns=swans.benchmark.unique(), errors="ignore")
    out = {}
    for s, w in swans.iterrows():
        ret = prices.loc[w.trough] / prices.loc[w.peak] - 1
        out[s] = ret[stocks.columns] / ret[w.benchmark]
    return pd.DataFrame(out)


def market_stats(prices=CLOSE, swans=SWANS):
    """Per swan: benchmark drop DD, fall length T1, recovery length T2 (NaN if not yet recovered)."""
    rows = {}
    for s, w in swans.iterrows():
        b = prices[w.benchmark].dropna()
        after = b[w.trough:]
        recovered = after[after >= b[w.peak]].index
        rows[s] = {"DD": 1 - b[w.trough] / b[w.peak], "T1": len(b[w.peak:w.trough]),
                   "T2": len(b[w.trough:recovered[0]]) - 1 if len(recovered) else np.nan}
    return pd.DataFrame(rows).T


if __name__ == "__main__":
    stored = pd.read_excel(DATA / "SwanBeta/All_SwanBeta100.xlsx", index_col=0)
    ours = swan_betas()
    err = (ours.reindex(stored.index) - stored[ours.columns]).abs()
    print("max |recomputed - stored| per swan:", err.max().round(5).to_dict())
    assert (err.max() < 1e-3).all(), "Swan Beta no longer reproduces All_SwanBeta100.xlsx"

    m = pd.read_excel(DATA / "MVO_Components/MarketDD.xlsx", index_col=0)
    cmp = market_stats().join(m[["Actual DD", "T1", "T2"]].add_prefix("MarketDD "))
    print("\nswan windows vs MarketDD.xlsx")
    print(cmp.round(4).to_string())
