"""Swan Beta and swan market statistics, computed from frozen prices and the swan windows in data/swans.csv.

    Swan beta of a stock = its peak->trough return / the benchmark's peak->trough return  (adjusted close)
    DD = benchmark drop, T1 = trading days peak..trough (inclusive), T2 = trading days until the benchmark regains the peak

Two swan lists:
- hand: data/swans.csv, the team's list. The windows were reverse-engineered and reproduce every column of
  data/SwanBeta/All_SwanBeta100.xlsx exactly.
- rule: detect_swans(), a fixed rule (index >= 15% below its trailing 1-year high), so new swans are found the same way.
Both get a `confirmed` date: the first day the index is back within 5% of its 1-year high, when the trough is final.
Running this file re-checks the stored betas and compares the two lists and MarketDD.xlsx.

    python v2/swan_beta.py
"""
from pathlib import Path

from functools import lru_cache

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"
LOOK, CALM = 252, 0.05   # 1-year high; "calm" = within 5% of it


@lru_cache
def load_prices(universe="SP100", kind="close"):
    """Frozen prices from data/prices: kind "close" (adjusted) or "nominal"."""
    prefix = "" if kind == "close" else "nominal_"
    return pd.read_csv(DATA / f"prices/{prefix}close_{universe}.csv.gz", index_col=0, parse_dates=True)


CLOSE = load_prices()


def drawdown(bench):
    """Drop below the trailing 1-year high."""
    return 1 - bench / bench.rolling(LOOK, min_periods=1).max()


def confirm_dates(swans, prices=CLOSE):
    """First day after each trough when the index is back within CALM of its 1-year high: the trough is then final."""
    out = {}
    for s, w in swans.iterrows():
        dd = drawdown(prices[w.benchmark].dropna())
        calm = dd[w.trough:][dd[w.trough:] < CALM].index
        out[s] = calm[0] if len(calm) else pd.NaT
    return pd.Series(out)


def detect_swans(prices=CLOSE, threshold=0.15, oex_from="1987-01-01"):
    """Rule-based swans: each episode where the index sits > CALM below its 1-year high and at some point >= threshold.
    trough = lowest close of the episode, peak = highest close in the year before it.
    Benchmark ^GSPC before oex_from (when the S&P100 list starts), ^OEX after, as in the hand list."""
    rows = []
    for bench, a, z in [("^GSPC", None, oex_from), ("^OEX", oex_from, None)]:
        b = prices[bench].dropna()
        dd = drawdown(b).loc[a:z]
        episode = (dd > CALM).astype(int)
        ids = (episode.diff() != 0).cumsum()[episode == 1]
        for _, days in dd[episode == 1].groupby(ids):
            if days.max() < threshold:
                continue
            trough = b[days.index].idxmin()
            peak = b[:trough].iloc[-LOOK:].idxmax()
            rows.append({"swan": f"R{trough:%Y-%m}", "benchmark": bench, "peak": peak, "trough": trough})
    return pd.DataFrame(rows).set_index("swan")


SWANS = pd.read_csv(DATA / "swans.csv", index_col="swan", parse_dates=["peak", "trough"])


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
    both = ours.columns.intersection(stored.columns)   # swans added after the original files have nothing to check against
    err = (ours.reindex(stored.index)[both] - stored[both]).abs()
    print("max |recomputed - stored| per swan:", err.max().round(5).to_dict())
    assert (err.max() < 1e-3).all(), "Swan Beta no longer reproduces All_SwanBeta100.xlsx"

    rule = detect_swans().join(market_stats(swans=detect_swans()).DD)
    near = lambda t: ", ".join(SWANS.index[(SWANS.trough - t).abs() <= pd.Timedelta(days=45)])
    rule["hand list"] = rule.trough.map(near)
    rule["confirmed"] = confirm_dates(rule)
    print("rule-based swans (15% below 1-year high):")
    print(rule.round(3).to_string())

    m = pd.read_excel(DATA / "MVO_Components/MarketDD.xlsx", index_col=0)
    cmp = market_stats().join(m[["Actual DD", "T1", "T2"]].add_prefix("MarketDD "))
    print("\nswan windows vs MarketDD.xlsx")
    print(cmp.round(4).to_string())
