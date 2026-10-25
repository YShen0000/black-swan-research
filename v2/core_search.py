"""D1's core, retested: how the stock sleeve is weighted and how much of the core is in stocks. Fixed in advance.

Sleeves (S&P500 point-in-time, rebuilt on the first trading day of each month; blend = ½ mean of the last 5 swan
betas + ½ 1-year market beta, lower = safer; <= 25% per sector everywhere):
  K0  R1: lowest third by blend, inverse vol                                    (current, ~160 stocks)
  K1  lowest 20 by blend, inverse vol
  K2  lowest 30 by blend, inverse vol
  K3  Barra min-var: minimum variance under the factor model (risk_model.py) over the lowest third by blend,
      <= 5% per stock (so >= 20 names); if more than 30 names, the 30 largest weights, renormalised
  K4  swan-beta cap + min-var: minimum variance under the factor model over all candidates with the portfolio's
      blend <= 0.5 (if out of reach: mean of the 20 lowest + 0.02), <= 5% per stock, 30 largest weights
Core mixes: stocks / IEF / GLD at 50/25/25 (current), 70/15/15 and 100/0/0. D1 = half core + half rotation C3, in
4 tranches (strategy_robustness.py). 15 combinations, plus each sleeve alone and each core alone.
A change to the live setup is adopted only if it beats the current D1 (K0, 50/25/25) on Sharpe in 2005-14, in
2015-26 and without COVID, with max drawdown <= 25%. More stock is a choice of return against drawdown: the table
shows it, the rule above does not apply to it.

    python v2/core_search.py build     # sleeves, ~20 min, cached in output/core_search_weights.pkl
    python v2/core_search.py           # the tables
"""
import sys
from pathlib import Path

import pandas as pd

import backtest
import develop as D
import experiments as E
import risk_model as RM
import strategy_robustness as SR
import strategy_search as S
import swan_beta
from factor_experiment import minvar

OUT = Path(__file__).parent / "output"
U = "SP500"
SLEEVES = ["K0", "K1", "K2", "K3", "K4"]
MIXES = {"50/25/25": 0.5, "70/15/15": 0.7, "100/0/0": 1.0}
EQUITY_ETFS = ["SPY", "EFA", "EEM", "VNQ"]


def top30(w):
    w = w.nlargest(30)
    return w / w.sum()


def build():
    swans = E.setup(U)
    out = {}
    for t in sorted(pd.read_pickle(OUT / "develop_SP500_weights.pkl")):
        ctx = E.context(t, U, swans)
        x = ctx["x"]
        blend = (x.swan_mean + x.mkt_beta) / 2
        third = blend.nsmallest(len(blend) // 3).index
        w = {"K0": D.hold(blend, t, U, ctx, len(blend) // 3), "K1": D.hold(blend, t, U, ctx, 20),
             "K2": D.hold(blend, t, U, ctx, 30), "K3": top30(minvar(RM.factor_model(t, third, U)[0], 0.05))}
        cov = RM.factor_model(t, blend.index, U)[0]
        try:
            k4 = minvar(cov, 0.05, beta=blend, beta_max=0.5)
        except RuntimeError:
            k4 = minvar(cov, 0.05, beta=blend, beta_max=blend.nsmallest(20).mean() + 0.02)
        w["K4"] = top30(k4)
        out[t] = w
        print(t.date(), {k: len(v) for k, v in w.items()}, flush=True)
    return out


def tranched(r, fns):
    stock = SLEEVES
    copies = [S.run_all(r, fns, stock=stock)] + [S.run_all(r, fns, stock=stock, every=10 ** 6, dates=SR.mid_month(r.index, o))
                                                  for o in (5, 10, 15)]
    return {k: pd.concat([c[k] for c in copies], axis=1).mean(axis=1) for k in fns}


if __name__ == "__main__":
    cache = OUT / "core_search_weights.pkl"
    if sys.argv[1:] == ["build"]:
        pd.to_pickle(build(), cache)
        sys.exit()
    weights = pd.read_pickle(cache)
    close = swan_beta.load_prices(U)
    sl, _ = backtest.daily_returns(weights, close, 0.003)
    r = S.load()
    for k in SLEEVES:
        r[k] = sl[k].reindex(r.index)
    rot = S.rotation(SR.ROT)
    core = lambda k, x: S.fixed({k: x, "IEF": (1 - x) / 2, "GLD": (1 - x) / 2} if x < 1 else {k: 1.0})
    fns = {(k, m, "D1"): (lambda c: lambda d, h: c(d, h).mul(0.5).add(rot(d, h).mul(0.5), fill_value=0))(core(k, x))
           for k in SLEEVES for m, x in MIXES.items()}
    runs = tranched(r, fns)
    runs.update(S.run_all(r, {(k, m, "core only"): core(k, x) for k in SLEEVES for m, x in MIXES.items()}, stock=SLEEVES))
    spy, cash = r.SPY.loc[S.START:], r.cash.loc[S.START:]
    rows = {key: S.evaluate(s.loc[S.START:], spy, cash) for key, s in runs.items()}
    for k in SLEEVES:
        rows[(k, "stocks only", "")] = S.evaluate(r[k].loc[S.START:], spy, cash)
    t = pd.DataFrame(rows).T
    base = t.loc[("K0", "50/25/25", "D1")]
    better = lambda c: t[c] > base[c]
    t["beats current D1 (Sharpe 05-14, 15-26, ex-COVID; DD<=25%)"] = (
        better("Sharpe 2005-14") & better("Sharpe 2015-26") & better("Sharpe ex-COVID") & (t["max DD"] <= 0.25))
    t.to_csv(OUT / "core_search.csv")
    pd.set_option("display.width", 260, "display.max_rows", 100)
    cols = ["CAGR", "vol", "Sharpe", "max DD", "worst year", "CAGR 2005-14", "CAGR 2015-26", "Sharpe 2005-14",
            "Sharpe 2015-26", "Sharpe ex-COVID"]
    for part in ["D1", "core only", "stocks only"]:
        sub = t[t.index.get_level_values(2 if part != "stocks only" else 1) == part]
        print(f"\n{part}")
        print(sub[cols].astype(float).round(3).to_string())
        if part == "D1":
            print(sub[list(S.DF.CRISES) + [t.columns[-1]]].to_string(float_format=lambda x: f"{x:.3f}"))
    n = pd.DataFrame({t_: {k: len(v) for k, v in w.items()} for t_, w in weights.items()}).T
    print("\nholdings per sleeve (min / median / max):")
    print(n.describe().loc[["min", "50%", "max"]].round(0).to_string())
    ends = [d for d in S.A.month_ends(r.index) if d >= pd.Timestamp(S.START)]
    eq = pd.Series({d: rot(d, r.loc[:d]).reindex(EQUITY_ETFS).fillna(0).sum() for d in ends})
    print(f"\nrotation's share in equity ETFs (SPY/EFA/EEM/VNQ): mean {eq.mean():.0%}, min {eq.min():.0%}, max {eq.max():.0%}")
    for m, x in MIXES.items():
        tot = 0.5 * x + 0.5 * eq
        print(f"  D1 with core {m}: total equity share mean {tot.mean():.0%} (range {tot.min():.0%}-{tot.max():.0%})")
