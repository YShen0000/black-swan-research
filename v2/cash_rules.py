"""Allow cash instead of forcing stock positions. Fixed in advance; inside D1 70/15/15 (half core, half rotation in
4 tranches), long only, no leverage.

  E1  regime: the core's stocks (R1) go to T-bills while SPY is below its 10-month average (month-end decision)
  E2  bad regime only: the same, but only while SPY is below its 10-month average AND more than 10% below its
      1-year high (both at the decision day); IEF / GLD stay as they are
  E3  per stock: a selected stock below its 200-day average at the monthly rebuild is not bought; its slot is in
      T-bills until the next rebuild (no re-weighting of the others)
  E4  per stock: the same with a negative 6-1 month return as the downtrend test
Regime decisions are tranched like the rotation (month end, 5 / 10 / 15 trading days in). Adopted only if D1 with it
has a higher Sharpe in 2005-14, 2015-26 and without COVID, and a max drawdown no higher than now.

    python v2/cash_rules.py
"""
import datetime
from pathlib import Path

import pandas as pd

import backtest
import defense as DF
import strategy_robustness as SR
import strategy_search as S
import swan_beta

OUT = Path(__file__).parent / "output"
U = "SP500"
CORE = {"IEF": .15, "GLD": .15}


def regime(bad_only):
    def f(d, hist):
        R = S.Rules(hist, d)
        p = R.px["SPY"].dropna()
        down = not R.up("SPY")
        if bad_only:
            down = down and p.iloc[-1] < 0.9 * p.iloc[-252:].max()
        w = pd.Series({"cash" if down else "R1": .7, **CORE})
        miss = [a for a in w.index if a != "cash" and not R.ready(a)]
        w["cash"] = w.get("cash", 0) + w[miss].sum()
        w[miss] = 0
        return w
    return f


def per_stock(weights, close, cash, test):
    """Monthly sleeve where names failing `test` at the rebuild hold T-bills instead."""
    c2 = close.copy()
    c2["CASHX"] = (1 + cash.reindex(close.index).fillna(0)).cumprod()
    out, share = {}, {}
    for t, w in weights.items():
        p = close.loc[:t - datetime.timedelta(days=1), w.index].iloc[-253:]
        ok = test(p)
        out[t] = {"s": pd.concat([w[ok], pd.Series({"CASHX": w[~ok].sum()})])}
        share[t] = w[~ok].sum()
    r, _ = backtest.daily_returns(out, c2, 0.003)
    return r["s"], pd.Series(share)


if __name__ == "__main__":
    r = S.load()
    close = swan_beta.load_prices(U)
    w = {t: v["R1"] for t, v in pd.read_pickle(OUT / "search_sleeves_weights.pkl").items()}
    shares = {}
    r["E3"], shares["E3"] = per_stock(w, close, r.cash, lambda p: p.iloc[-1] > p.iloc[-200:].mean())
    r["E4"], shares["E4"] = per_stock(w, close, r.cash, lambda p: p.iloc[-22] / p.iloc[-127] > 1)
    for k in ["E3", "E4"]:
        r[k] = r[k].reindex(r.index)
    rot = S.rotation(SR.ROT)
    half = lambda c: (lambda d, h: c(d, h).mul(0.5).add(rot(d, h).mul(0.5), fill_value=0))
    fns = {"D1 (live)": half(S.fixed({"R1": .7, **CORE})), "E1 regime": half(regime(False)),
           "E2 bad regime": half(regime(True)), "E3 skip stocks below 200d": half(S.fixed({"E3": .7, **CORE})),
           "E4 skip stocks with 6-1 < 0": half(S.fixed({"E4": .7, **CORE}))}
    stock = ["R1", "E3", "E4"]
    copies = [S.run_all(r, fns, stock=stock)] + [S.run_all(r, fns, stock=stock, every=10 ** 6, dates=SR.mid_month(r.index, o))
                                                 for o in (5, 10, 15)]
    d1 = {k: pd.concat([c[k] for c in copies], axis=1).mean(axis=1).loc[S.START:] for k in fns}
    spy, cash = r.SPY.loc[S.START:], r.cash.loc[S.START:]
    rows = {k: S.evaluate(s, spy, cash) for k, s in d1.items()}
    for k in ["R1", "E3", "E4"]:
        rows[f"sleeve {k} alone"] = S.evaluate(r[k].loc[S.START:], spy, cash)
    t = pd.DataFrame(rows).T
    t.to_csv(OUT / "cash_rules.csv")
    pd.set_option("display.width", 250)
    cols = ["CAGR", "Sharpe", "max DD", "worst year", "CAGR 2005-14", "CAGR 2015-26", "Sharpe 2005-14", "Sharpe 2015-26",
            "Sharpe ex-COVID"] + list(DF.CRISES)
    print(t[cols].astype(float).round(3).to_string())
    for k, s in shares.items():
        print(f"{k}: sleeve share in cash mean {s.mean():.0%}, 2008-09..2009-03 {s.loc['2008-09':'2009-03'].mean():.0%},"
              f" 2015-26 {s.loc['2015':].mean():.0%}")
    ends = [d for d in S.A.month_ends(r.index) if d >= pd.Timestamp(S.START)]
    for k, bad in [("E1", False), ("E2", True)]:
        f = regime(bad)
        off = pd.Series({d: f(d, r.loc[:d]).get("cash", 0) > 0 for d in ends})
        print(f"{k}: core stocks in cash {off.mean():.0%} of month ends, switches {int(off.astype(int).diff().abs().sum())}")
    b = t.loc["D1 (live)"]
    print("adopt:", {k: bool(all(t.loc[k, c] > b[c] for c in ["Sharpe 2005-14", "Sharpe 2015-26", "Sharpe ex-COVID"])
                             and t.loc[k, "max DD"] <= b["max DD"]) for k in fns if k != "D1 (live)"})
