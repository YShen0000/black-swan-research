"""Robustness of the recommended setup (blend beta -> lowest n -> inverse vol): number of stocks, sector cap,
rebalance frequency and trading cost; plus its sector concentration.

    python v2/robustness.py SP500     # or SP100; ~45 min for SP500 the first time, cached after

Writes v2/output/robustness_<universe>.csv (every variant x frequency x cost) and
v2/output/robustness_<universe>_sectors.csv (monthly sector weights of the uncapped default).
"""
import sys
from pathlib import Path

import pandas as pd

import backtest
import components as C
import experiments as E
import swan_beta

N_GRID = {"SP500": [15, 20, 30, 50, 75], "SP100": [10, 15, 20, 30]}
CAPS = [0.35, 0.25, 0.20]
FREQS = {"monthly": 1, "quarterly": 3, "semi-annual": 6}
COSTS = [0.001, 0.003, 0.005]


def variants(universe, swans, start="2004-01-01"):
    close = swan_beta.load_prices(universe)
    dates = close.loc[start:].index.to_series().groupby(close.loc[start:].index.to_period("M")).first()
    n0 = E.N_SELECT[universe]
    out = {}
    for t in dates:
        ctx = E.context(t, universe, swans)
        pred = C.forecast("blend", ctx["x"], universe, ctx["known"])
        w = {}
        for n in N_GRID[universe]:
            w[f"n={n}"] = C.weigh("inv_vol", C.select_lowest(pred, n), t, universe, ctx)
        for cap in CAPS:
            sel = C.select_lowest(pred, n0, cap)
            w[f"n={n0} sector cap {cap:.0%}"] = C.cap_sectors(C.weigh("inv_vol", sel, t, universe, ctx), cap)
        w["index"] = pd.Series({C.BENCH[universe]: 1.0})
        w[C.TOTAL_RETURN[universe]] = pd.Series({C.TOTAL_RETURN[universe]: 1.0})
        out[t] = w
        print(t.date(), flush=True)
    return out


if __name__ == "__main__":
    universe = sys.argv[1] if len(sys.argv) > 1 else "SP500"
    swans = E.setup(universe)
    close = swan_beta.load_prices(universe)
    out = Path(__file__).parent / "output"
    out.mkdir(exist_ok=True)
    cache = out / f"robustness_{universe}_weights.pkl"
    weights = pd.read_pickle(cache) if cache.exists() else variants(universe, swans)
    pd.to_pickle(weights, cache)

    # original setup for reference, from the experiments grid cache if it exists
    grid = out / f"experiments_{universe}_weights.pkl"
    if grid.exists():
        g = pd.read_pickle(grid)
        for t in weights:
            weights[t]["original (K-means + MVO)"] = g[t]["kmeans | mvo (original)"]

    bench = C.BENCH[universe]
    rows = {}
    dates = sorted(weights)
    for fname, k in FREQS.items():
        sub = {t: weights[t] for t in dates[::k]}
        for cost in COSTS:
            r, turnover = backtest.daily_returns(sub, close, cost)
            r = r.rename(columns={"index": bench})
            t = backtest.summary(r, bench, swans, turnover.rename(columns={"index": bench}))
            pr = t[t.index.str.startswith("PR ")]
            t.loc["PR mean"], t.loc["PR worst"] = pr.mean(), pr.min()
            for v in t.columns:
                rows[(v, fname, f"{cost:.1%}")] = t[v]
    table = pd.DataFrame(rows).T
    table.index.names = ["variant", "rebalance", "cost"]
    table.to_csv(out / f"robustness_{universe}.csv")

    key = f"n={E.N_SELECT[universe]}"
    sec = pd.DataFrame({t: w[key].groupby(C.sectors()(w[key].index)).sum() for t, w in weights.items()}).T.fillna(0)
    sec.to_csv(out / f"robustness_{universe}_sectors.csv")

    cols = ["CAGR", "vol", "Sharpe(rf=0)", "max drawdown", "beta", "turnover/yr", "PR mean", "PR worst"]
    pd.set_option("display.width", 250)
    print(table.xs(("monthly", "0.1%"), level=["rebalance", "cost"])[cols].round(3).to_string())
    print()
    print(table.xs(f"n={E.N_SELECT[universe]}", level="variant")[cols].round(3).to_string())
    print()
    print("sector weights of the default (mean / max over months):")
    print(pd.DataFrame({"mean": sec.mean(), "max": sec.max()}).sort_values("mean", ascending=False).round(3).to_string())
