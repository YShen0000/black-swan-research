"""Develop both objectives with variants fixed *before* looking at results (no tuning on outcomes).

    python v2/develop.py SP500        # or SP100; ~40 min for SP500 the first time, cached after

Defensive track: fully invested, maximise crash protection. Hold n = 30 (S&P500) / 20 (S&P100), inverse vol,
25% sector cap. Scores, lower = safer (rank = percentile rank among the candidates):
  D0 blend        ½ swan mean + ½ 1-year market beta (the current recommendation)
  D1 stress       mean rank of swan mean and stress beta (beta on the 10% worst market days of the past year)
  D2 both safe    max of the ranks of swan mean and market beta: a stock must be safe on both
  D3 worst case   mean rank of the stock's worst (highest) swan beta in the window and market beta
  D4 three        mean rank of worst swan beta, stress beta and market beta
Return track: risk-adjusted return, crash protection as a side condition. Inverse vol, 25% sector cap.
  R1 tercile      lowest third of the candidates by the D0 score
  R2 half         lowest half
Benchmarks: SPY / OEF (total return), candidates equal weight, and for each portfolio the beta-matched
index + T-bill mix (same full-period beta), which shows whether stock selection beats simply holding less stock.

Overfitting checks: every swan left out in turn (PR without Swan9 = COVID shown explicitly), two sub-periods
(2004-2014, 2015-2026), both universes, and the share of swans in which a variant beats D0.
"""
import datetime
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import backtest
import components as C
import experiments as E
import swan_beta

CAP = 0.25
SPLIT = "2015-01-01"


def stress_beta(t, tickers, universe, q=0.10):
    close = swan_beta.load_prices(universe).loc[:t - datetime.timedelta(days=1)]
    r = close[list(tickers) + [C.BENCH[universe]]].iloc[-253:].pct_change().iloc[1:]
    b = r.pop(C.BENCH[universe])
    bad = b <= b.quantile(q)
    return r[bad].apply(lambda s: s.cov(b[bad])) / b[bad].var()


def scores(t, ctx, universe):
    x = ctx["x"]
    w = ctx["betas"].loc[x.index, ctx["window"]]
    rank = lambda s: s.rank(pct=True)
    s = pd.DataFrame({"swan_mean": x.swan_mean, "mkt": x.mkt_beta, "worst_swan": w.max(axis=1),
                      "stress": stress_beta(t, x.index, universe)})
    return pd.DataFrame({
        "D0 blend": (s.swan_mean + s.mkt) / 2,
        "D1 stress": (rank(s.swan_mean) + rank(s.stress)) / 2,
        "D2 both safe": np.maximum(rank(s.swan_mean), rank(s.mkt)),
        "D3 worst case": (rank(s.worst_swan) + rank(s.mkt)) / 2,
        "D4 three": (rank(s.worst_swan) + rank(s.stress) + rank(s.mkt)) / 3,
    })


def hold(score, t, universe, ctx, n):
    pick = C.select_lowest(score.dropna(), n, CAP)
    return C.cap_sectors(C.weigh("inv_vol", pick, t, universe, ctx), CAP)


def monthly(universe, swans, start="2004-01-01"):
    close = swan_beta.load_prices(universe)
    dates = close.loc[start:].index.to_series().groupby(close.loc[start:].index.to_period("M")).first()
    n = E.N_SELECT[universe]
    out = {}
    for t in dates:
        ctx = E.context(t, universe, swans)
        sc = scores(t, ctx, universe)
        w = {k: hold(sc[k], t, universe, ctx, n) for k in sc}
        m = len(sc)
        w["R1 tercile"] = hold(sc["D0 blend"], t, universe, ctx, m // 3)
        w["R2 half"] = hold(sc["D0 blend"], t, universe, ctx, m // 2)
        w["candidates EW"] = pd.Series(1 / m, index=sc.index)
        w["index"] = pd.Series({C.TOTAL_RETURN[universe]: 1.0})
        out[t] = w
        print(t.date(), "candidates", m, flush=True)
    return out


def cash_daily(index):
    y = pd.read_csv(swan_beta.DATA / "prices/tbill_13w.csv", index_col=0, parse_dates=True).yield_pct
    return (y.reindex(index).ffill() / 100 / 252).fillna(0)


def metrics(r, b, cash, swans):
    """Objective scorecard of daily returns r vs total-return index b."""
    beta = r.cov(b) / b.var()
    mix = beta * b + (1 - beta) * cash
    w = (1 + r).cumprod()
    yrs = len(r) / 252
    cagr = lambda s: (1 + s).prod() ** (252 / len(s)) - 1
    m, bm = (1 + r).resample("ME").prod() - 1, (1 + b).resample("ME").prod() - 1
    up, dn = bm > 0, bm < 0
    out = {"CAGR": w.iloc[-1] ** (1 / yrs) - 1, "vol": r.std() * np.sqrt(252),
           "Sharpe": r.mean() / r.std() * np.sqrt(252), "max DD": (1 - w / w.cummax()).max(), "beta": beta,
           "up capture": m[up].mean() / bm[up].mean(), "down capture": m[dn].mean() / bm[dn].mean(),
           "excess vs β-matched": cagr(r) - cagr(mix), "CAGR - index": cagr(r) - cagr(b)}
    for name, part in [("2004-14", slice(None, SPLIT)), ("2015-26", slice(SPLIT, None))]:
        rp, bp, mp = r.loc[part], b.loc[part], mix.loc[part]
        out[f"Sharpe {name}"] = rp.mean() / rp.std() * np.sqrt(252)
        out[f"excess vs β-matched {name}"] = cagr(rp) - cagr(mp)
    wealth, bw = (1 + r).cumprod(), (1 + b).cumprod()
    for s, sw in swans[swans.peak >= r.index[0]].iterrows():
        pd_, bd = 1 - wealth[sw.trough] / wealth[sw.peak], 1 - bw[sw.trough] / bw[sw.peak]
        out[f"PR {s}"] = 1 - pd_ / bd
    return pd.Series(out)


def evaluate(weights, universe, swans):
    close = swan_beta.load_prices(universe)
    dates = sorted(weights)
    tr = C.TOTAL_RETURN[universe]
    tables = []
    for p in range(3):   # quarterly rebalance, each start month, 0.3% cost
        sub = {t: weights[t] for t in dates[p::3]}
        r, tv = backtest.daily_returns(sub, close, 0.003)
        r = r.loc[dates[p] + datetime.timedelta(days=1):]
        cash = cash_daily(r.index)
        t = pd.DataFrame({k: metrics(r[k], r["index"], cash, swans) for k in r.columns}).T
        t["turnover/yr"] = tv.sum() / (len(r) / 252)
        tables.append(t)
    t = sum(tables) / 3
    pr = t[[c for c in t.columns if c.startswith("PR ")]]
    t["PR mean"], t["PR worst"], t["PR median"] = pr.mean(axis=1), pr.min(axis=1), pr.median(axis=1)
    t["PR mean w/o Swan9"] = pr.drop(columns="PR Swan9").mean(axis=1)
    t["PR mean, worst leave-one-out"] = pd.concat([pr.drop(columns=c).mean(axis=1) for c in pr.columns], axis=1).min(axis=1)
    t["swans beating D0"] = (pr.sub(pr.loc["D0 blend"], axis=1) > 0).sum(axis=1)
    return t.rename(index={"index": tr})


if __name__ == "__main__":
    universe = sys.argv[1] if len(sys.argv) > 1 else "SP500"
    swans = E.setup(universe)
    out = Path(__file__).parent / "output"
    out.mkdir(exist_ok=True)
    cache = out / f"develop_{universe}_weights.pkl"
    weights = pd.read_pickle(cache) if cache.exists() else monthly(universe, swans)
    pd.to_pickle(weights, cache)
    t = evaluate(weights, universe, swans)
    t.to_csv(out / f"develop_{universe}.csv")
    pd.set_option("display.width", 250)
    print(f"{universe}, quarterly rebalance, 0.3% cost, mean of 3 start months, 2004 -> today")
    print("\ndefensive scorecard")
    print(t[["PR mean", "PR median", "PR worst", "PR mean w/o Swan9", "PR mean, worst leave-one-out", "swans beating D0",
             "down capture", "max DD", "CAGR - index"]].round(3).to_string())
    print("\nreturn scorecard")
    print(t[["CAGR", "vol", "Sharpe", "Sharpe 2004-14", "Sharpe 2015-26", "excess vs β-matched",
             "excess vs β-matched 2004-14", "excess vs β-matched 2015-26", "up capture", "beta", "turnover/yr"]].round(3).to_string())
    print("\nPR by swan")
    print(t[[c for c in t.columns if c.startswith("PR Swan")]].round(2).to_string())
