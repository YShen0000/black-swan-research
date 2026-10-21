"""Compare strategy component combinations (see components.py), using only data available at each date.

    python v2/experiments.py SP500        # universe SP500 (point-in-time members) or SP100 (today's list)

1. Forecaster test: for each swan Swan5..Swan11, predict every stock's beta the day before the peak and compare with
   what happened (rank correlation; average actual beta of the n stocks predicted safest).
2. Backtest grid: monthly rebalance 2004 -> today for every forecaster x weighting, plus the original
   (K-means + MVO), the point-in-time universe equal-weighted, and the index.
Writes v2/output/experiments_<universe>_*.csv.
"""
import datetime
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import backtest
import components as C
import swan_beta

FORECASTERS = ["swan_mean", "market_beta", "blend", "ridge", "rf"]
WEIGHTINGS = ["equal", "inv_vol", "min_vol", "mvo"]
N_SELECT = {"SP100": 20, "SP500": 30}
N_KMEANS = {"SP100": 7, "SP500": 17}


def setup(universe, which="hand"):
    swans = backtest.swan_list(universe, which)
    C._SWANS[universe] = swans
    C._BETAS[universe] = swan_beta.swan_betas(swan_beta.load_prices(universe), swans)
    return swans


def context(t, universe, swans):
    """Everything known the day before t: confirmed swans, last-5 window, betas, candidates' features."""
    known = swans[swans.confirmed < t]
    window = list(known.index[-5:])
    history = list(known.index[known.trough >= "1987-01-01"])
    betas = C._BETAS[universe][known.index]
    x = C.features(t, C.eligible(t, universe), betas, window, universe).dropna()
    marketdd = swan_beta.market_stats(swan_beta.load_prices(universe).loc[:t - datetime.timedelta(days=1)],
                                      known.loc[history])
    return dict(known=list(known.index), window=window, history=history, betas=betas, x=x, marketdd=marketdd,
                start_swan=window[0], end_swan=window[-1], N=N_KMEANS[universe])


def forecaster_test(universe, swans):
    rows = []
    n = N_SELECT[universe]
    for target in ["Swan5", "Swan6", "Swan7", "Swan8", "Swan9", "Swan10D", "Swan11"]:
        t = swans.peak[target]
        ctx = context(t, universe, swans)
        actual = C._BETAS[universe][target].reindex(ctx["x"].index)
        row = {"target": target, "candidates": len(ctx["x"]), "all": actual.mean()}
        km = C.select_kmeans(ctx["betas"].loc[ctx["x"].index], ctx["window"], ctx["N"])
        row["kmeans: actual beta of selection"] = actual[km.index].mean()
        for f in FORECASTERS:
            pred = C.forecast(f, ctx["x"], universe, ctx["known"])
            row[f"{f}: rank corr"] = pred.corr(actual, method="spearman")
            row[f"{f}: actual beta of lowest {n}"] = actual[pred.nsmallest(n).index].mean()
        rows.append(row)
    t = pd.DataFrame(rows).set_index("target")
    t.loc["mean"] = t.mean()
    return t


def grid_weights(universe, swans, start="2004-01-01"):
    close = swan_beta.load_prices(universe)
    dates = close.loc[start:].index.to_series().groupby(close.loc[start:].index.to_period("M")).first()
    out = {}
    for t in dates:
        ctx = context(t, universe, swans)
        x = ctx["x"]
        w = {}
        km = C.select_kmeans(ctx["betas"].loc[x.index], ctx["window"], ctx["N"])
        w["kmeans | mvo (original)"] = C.weigh("mvo", km, t, universe, ctx)
        w["kmeans | equal"] = C.weigh("equal", km, t, universe, ctx)
        for f in FORECASTERS:
            sel = C.select_lowest(C.forecast(f, x, universe, ctx["known"]), N_SELECT[universe])
            for k in WEIGHTINGS:
                w[f"{f} | {k}"] = C.weigh(k, sel, t, universe, ctx)
        alive = C.eligible(t, universe)
        w["universe EW"] = pd.Series(1 / len(alive), index=alive)
        w["index"] = pd.Series({C.BENCH[universe]: 1.0})
        out[t] = w
        print(t.date(), ctx["window"][0], "..", ctx["window"][-1], "| candidates", len(x), "| eligible", len(alive),
              flush=True)
    return out


if __name__ == "__main__":
    universe = sys.argv[1] if len(sys.argv) > 1 else "SP500"
    swans = setup(universe)
    out = Path(__file__).parent / "output"
    out.mkdir(exist_ok=True)
    pd.set_option("display.width", 250, "display.max_columns", 50)

    ft = forecaster_test(universe, swans)
    ft.to_csv(out / f"experiments_{universe}_forecasters.csv")
    print(ft.T.round(3).to_string())

    cache = out / f"experiments_{universe}_weights.pkl"   # the slow part; delete to recompute
    weights = pd.read_pickle(cache) if cache.exists() else grid_weights(universe, swans)
    pd.to_pickle(weights, cache)
    tr = f"{C.TOTAL_RETURN[universe]} (index incl. dividends)"
    for w in weights.values():
        w[tr] = pd.Series({C.TOTAL_RETURN[universe]: 1.0})
    r, turnover = backtest.daily_returns(weights, swan_beta.load_prices(universe))
    bench = C.BENCH[universe]
    r = r.rename(columns={"index": bench})
    table = backtest.summary(r, bench, swans, turnover.rename(columns={"index": bench}))
    pr = table[table.index.str.startswith("PR ")]
    table.loc["PR mean"] = pr.mean()
    table.loc["PR worst"] = pr.min()
    r.to_csv(out / f"experiments_{universe}_daily.csv")
    table.to_csv(out / f"experiments_{universe}_grid.csv")
    print(table.T.sort_values("Sharpe(rf=0)", ascending=False).round(3).to_string())
