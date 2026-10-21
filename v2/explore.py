"""Which component of the strategy matters? Preselection x weighting, with random and generic controls.

    python v2/explore.py SP500        # or SP100; first SP500 run ~1 h, cached after

A. Preselection test: on the day before each swan peak (Swan5..Swan11), every method picks n stocks from the same
   pool; we record the average *actual* swan beta of the pick (lower = better protection). Random picks are the
   floor a method has to beat.
B. Backtest grid 2004 -> today: every preselection x every weighting, rebuilt monthly from data known then.
   Evaluated with quarterly rebalancing at 0.3% cost, averaged over the 3 possible start months, and monthly at 0.1%.
C. Main effects: how much the results move when only the preselection changes vs only the weighting changes.
"""
import datetime
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import backtest
import components as C
import experiments as E
import mvo2
import preselect as P
import risk_model as RM
import swan_beta

METHODS = ["kmeans", "kmeans_dbi", "dtw_team", "dtw", "holt", "swan_mean", "market_beta", "blend", "barra_ridge"]
WEIGHTINGS = ["equal", "inv_vol", "hrp", "herc", "factor_minvar", "mvo"]
N_RANDOM = 5


def pool(t, ctx, universe):
    return sorted(set(ctx["x"].index) & set(P.window_betas(universe, tuple(ctx["window"])).index))


def preselection_test(universe, swans):
    n = E.N_SELECT[universe]
    rows = []
    for target in ["Swan5", "Swan6", "Swan7", "Swan8", "Swan9", "Swan10D", "Swan11"]:
        t = swans.peak[target]
        ctx = E.context(t, universe, swans)
        actual = C._BETAS[universe][target]
        cand = pool(t, ctx, universe)
        rng = np.random.default_rng(0)
        row = {"target": target, "pool": len(cand), "pool average": actual[cand].mean(),
               "random n (mean of 500)": np.mean([actual[rng.choice(cand, n, replace=False)].mean() for _ in range(500)])}
        for m in METHODS:
            row[m] = actual[P.ranking(m, t, ctx, universe)[:n]].mean()
        rows.append(row)
    d = pd.DataFrame(rows).set_index("target")
    d.loc["mean"] = d.mean()
    return d


def weigh(kind, pick, t, universe, ctx, fcov):
    sel = pd.Series(ctx["x"].reindex(pick).mkt_beta.values, index=pick)   # beta only used by "mvo" (injection)
    if kind == "factor_minvar":
        from factor_experiment import minvar
        return minvar(fcov.loc[pick, pick], 0.10, sector_cap=1.0)
    if kind == "mvo":   # original steps 2-4: inject, trend return / drawdown, max Sharpe; injection uses swan-mean beta
        sel = ctx["x"].reindex(pick).swan_mean.fillna(1.0)
    return C.weigh(kind, sel, t, universe, ctx)


def monthly(universe, swans, start="2004-01-01"):
    close = swan_beta.load_prices(universe)
    dates = close.loc[start:].index.to_series().groupby(close.loc[start:].index.to_period("M")).first()
    n = E.N_SELECT[universe]
    out = {}
    for t in dates:
        ctx = E.context(t, universe, swans)
        cand = pool(t, ctx, universe)
        fcov = RM.factor_model(t, cand, universe)[0]
        w = {}
        for m in METHODS:
            pick = P.ranking(m, t, ctx, universe)[:n]
            for k in WEIGHTINGS:
                w[(m, k)] = weigh(k, pick, t, universe, ctx, fcov)
        rng = np.random.default_rng(t.toordinal())
        for i in range(N_RANDOM):
            pick = list(rng.choice(cand, n, replace=False))
            for k in ["equal", "inv_vol"]:
                w[(f"random {i}", k)] = weigh(k, pick, t, universe, ctx, fcov)
        w[("pool", "equal")] = pd.Series(1 / len(cand), index=cand)
        w[("index", "")] = pd.Series({C.BENCH[universe]: 1.0})
        w[(C.TOTAL_RETURN[universe], "")] = pd.Series({C.TOTAL_RETURN[universe]: 1.0})
        out[t] = w
        print(t.date(), ctx["window"][0], "..", ctx["window"][-1], "| pool", len(cand), flush=True)
    return out


def evaluate(weights, universe, swans, close):
    bench = C.BENCH[universe]
    dates = sorted(weights)
    runs = [("monthly 0.1%", dates, 0.001)] + [(f"q{p + 1}", dates[p::3], 0.003) for p in range(3)]
    res = {}
    for label, sub, cost in runs:
        flat = {t: {f"{m}|{k}": v for (m, k), v in weights[t].items()} for t in sub}
        r, tv = backtest.daily_returns(flat, close, cost)
        r = r.rename(columns={"index|": bench}).loc[sub[0] + datetime.timedelta(days=1):]
        s = backtest.summary(r, bench, swans, tv.rename(columns={"index|": bench}))
        pr = s[s.index.str.startswith("PR ")]
        s.loc["PR mean"], s.loc["PR worst"] = pr.mean(), pr.min()
        res[label] = s.T
    quarterly = sum(res[f"q{p + 1}"] for p in range(3)) / 3
    return res["monthly 0.1%"], quarterly


if __name__ == "__main__":
    universe = sys.argv[1] if len(sys.argv) > 1 else "SP500"
    swans = E.setup(universe)
    close = swan_beta.load_prices(universe)
    out = Path(__file__).parent / "output"
    out.mkdir(exist_ok=True)
    pd.set_option("display.width", 250, "display.max_rows", 200)

    a = preselection_test(universe, swans)
    a.to_csv(out / f"explore_{universe}_preselection.csv")
    print("A. average actual swan beta of the n stocks each method picks (lower = better)")
    print(a.T.round(3).to_string())

    cache = out / f"explore_{universe}_weights.pkl"
    weights = pd.read_pickle(cache) if cache.exists() else monthly(universe, swans)
    pd.to_pickle(weights, cache)
    monthly_t, quarterly_t = evaluate(weights, universe, swans, close)
    monthly_t.to_csv(out / f"explore_{universe}_monthly.csv")
    quarterly_t.to_csv(out / f"explore_{universe}_quarterly.csv")

    cols = ["CAGR", "vol", "Sharpe(rf=0)", "max drawdown", "beta", "turnover/yr", "PR mean", "PR worst"]
    print("\nB. quarterly, 0.3% cost, mean of 3 start months")
    print(quarterly_t[cols].sort_values("PR mean", ascending=False).round(3).to_string())

    grid = quarterly_t.loc[[f"{m}|{k}" for m in METHODS for k in WEIGHTINGS]]
    grid.index = pd.MultiIndex.from_tuples([tuple(i.split("|")) for i in grid.index])
    rnd = quarterly_t.loc[[f"random {i}|{k}" for i in range(N_RANDOM) for k in ["equal", "inv_vol"]]]
    print("\nC. main effects (quarterly): average over the other dimension; random = mean of 5 random picks")
    for metric in ["PR mean", "Sharpe(rf=0)", "max drawdown", "CAGR"]:
        tab = grid[metric].unstack()
        print(f"\n{metric}")
        print(tab.round(3).to_string())
        print(f"  preselection effect (range of row means): {np.ptp(tab.mean(axis=1)):.3f}"
              f" | weighting effect (range of column means): {np.ptp(tab.mean(axis=0)):.3f}"
              f" | random pick: {rnd[metric].mean():.3f}")
