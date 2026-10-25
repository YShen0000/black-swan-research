"""Looking for more return with the drawdown under control (long only, stocks and ETFs, no leverage). Fixed before
any result was seen; every rule-based variant runs in 4 tranches (month end, 5 / 10 / 15 trading days in).

factor_lab.py found no stock factor with a stable premium in 2005-26 S&P500 (swan beta flips sign with the market),
and mean_reversion.py found RSI(2) too weak. What is left is which markets to own and when:
  A1  rotation top 3 of SPY, QQQ, IWM, EFA, EEM, VNQ, DBC, IEF, TLT, GLD (C3 plus Nasdaq-100 and small caps)
  A2  the same, top 2;     A4  the same, top 1
  A3  rotation top 3 of the 9 sector SPDRs + QQQ + IEF, TLT, GLD
  T1  QQQ while above its 10-month average, else IEF 50 / GLD 50
  T2  SPY, same rule (control: is it the timing or the choice of QQQ?)
  T3  swan factor timed: highest third by blend (high crash beta) while SPY is above its 10-month average,
      else R1 50 / IEF 25 / GLD 25
  M1  half D1 core (R1 70 / IEF 15 / GLD 15) + half A1;   M2  half D1 core + half T1
References: C3 (current rotation), D1 70/15/15 (live), SPY, QQQ.
Hindsight warning: QQQ is in the list because it is the index that did best; T1 vs T2 and A1 vs C3 show how much
of any gain is that choice.
Scored as in strategy_search.py, plus the CAGR split 2005-15 / 2016-26.

    python v2/return_search.py build    # high-blend sleeve, ~5 min
    python v2/return_search.py
"""
import sys
from pathlib import Path

import pandas as pd

import backtest
import develop as D
import experiments as E
import strategy_robustness as SR
import strategy_search as S
import swan_beta

OUT = Path(__file__).parent / "output"
U = "SP500"
BIG = ["SPY", "QQQ", "IWM", "EFA", "EEM", "VNQ", "DBC", "IEF", "TLT", "GLD"]
SECT = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY", "QQQ", "IEF", "TLT", "GLD"]
STOCK = ["R1", "H1"]


def build():
    swans = E.setup(U)
    out = {}
    for t in sorted(pd.read_pickle(OUT / "develop_SP500_weights.pkl")):
        ctx = E.context(t, U, swans)
        x = ctx["x"]
        blend = (x.swan_mean + x.mkt_beta) / 2
        out[t] = {"R1": D.hold(blend, t, U, ctx, len(blend) // 3), "H1": D.hold(-blend, t, U, ctx, len(blend) // 3)}
        print(t.date(), flush=True)
    return out


def timed(signal, on, off):
    def f(d, hist):
        R = S.Rules(hist, d)
        w = pd.Series(on if R.up(signal) else off, dtype=float)
        miss = [a for a in w.index if not R.ready(a)]
        w["cash"] = w.get("cash", 0) + w[miss].sum()
        w[miss] = 0
        return w
    return f


def half(a, b):
    return lambda d, h: a(d, h).mul(0.5).add(b(d, h).mul(0.5), fill_value=0)


def tranched(r, fns):
    copies = [S.run_all(r, fns, stock=STOCK)] + [S.run_all(r, fns, stock=STOCK, every=10 ** 6,
                                                            dates=SR.mid_month(r.index, o)) for o in (5, 10, 15)]
    return {k: pd.concat([c[k] for c in copies], axis=1).mean(axis=1) for k in fns}


def setup():
    w = pd.read_pickle(OUT / "return_search_weights.pkl")
    sl, _ = backtest.daily_returns(w, swan_beta.load_prices(U), 0.003)
    r = S.load()
    for k in STOCK:
        r[k] = sl[k].reindex(r.index)
    return r


def variants():
    core = S.fixed({"R1": .7, "IEF": .15, "GLD": .15})
    off = {"IEF": .5, "GLD": .5}
    return {
        "SPY": S.fixed({"SPY": 1.0}), "QQQ": S.fixed({"QQQ": 1.0}),
        "C3 rotation (current)": S.rotation(SR.ROT), "D1 70/15/15 (live)": half(core, S.rotation(SR.ROT)),
        "A1 rotation top 3 of 10": S.rotation(BIG), "A2 top 2 of 10": S.rotation(BIG, 2),
        "A4 top 1 of 10": S.rotation(BIG, 1), "A3 sectors+QQQ+bonds/gold top 3": S.rotation(SECT),
        "T1 QQQ timed": timed("QQQ", {"QQQ": 1.0}, off), "T2 SPY timed": timed("SPY", {"SPY": 1.0}, off),
        "T3 swan factor timed": timed("SPY", {"H1": 1.0}, {"R1": .5, "IEF": .25, "GLD": .25}),
        "M1 half core + half A1": half(core, S.rotation(BIG)), "M2 half core + half T1": half(core, timed("QQQ", {"QQQ": 1.0}, off)),
    }


def score(runs, r):
    spy, cash = r.SPY.loc[S.START:], r.cash.loc[S.START:]
    rows = {}
    for k, s in runs.items():
        s = s.loc[S.START:]
        st = S.evaluate(s, spy, cash)
        for nm, part in [("2005-15", slice("2005", "2015")), ("2016-26", slice("2016", "2026"))]:
            p = S.DF.stats(s.loc[part], spy.loc[part], cash)
            st[f"CAGR {nm}"], st[f"max DD {nm}"] = p["CAGR"], p["max DD"]
        rows[k] = st
    return pd.DataFrame(rows).T


if __name__ == "__main__":
    if sys.argv[1:] == ["build"]:
        pd.to_pickle(build(), OUT / "return_search_weights.pkl")
        sys.exit()
    r = setup()
    t = score(tranched(r, variants()), r)
    t.to_csv(OUT / "return_search.csv")
    pd.set_option("display.width", 260)
    cols = ["CAGR", "vol", "Sharpe", "max DD", "worst year", "CAGR 2005-15", "CAGR 2016-26", "max DD 2005-15",
            "max DD 2016-26", "Sharpe ex-COVID"]
    print(t[cols].astype(float).round(3).to_string())
    print(); print(t[list(S.DF.CRISES)].astype(float).round(3).to_string())
