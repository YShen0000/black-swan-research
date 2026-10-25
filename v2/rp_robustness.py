"""Robustness of the chosen allocation: risk parity of the R1 stock sleeve, 7-10y Treasuries (IEF) and gold (GLD).

    python v2/rp_robustness.py      # needs the develop / robustness weight caches

Each row changes one thing from the base (R1, quarterly sleeve rebalance starting in January, 1-year vol lookback,
monthly allocation rebalance, IEF, 0.1% ETF / 0.3% stock cost). 2004 -> today unless stated.
"""
from pathlib import Path

import numpy as np
import pandas as pd

import allocation as A
import backtest
import defense as DF
import swan_beta

OUT = Path(__file__).parent / "output"


def sleeve_returns(weights_by_date, key, phase=0, cost=0.003):
    close = swan_beta.load_prices("SP500")
    dates = sorted(weights_by_date)
    r, _ = backtest.daily_returns({t: {key: weights_by_date[t][key]} for t in dates[phase::3]}, close, cost)
    return r[key]


if __name__ == "__main__":
    dev = pd.read_pickle(OUT / "develop_SP500_weights.pkl")
    rob = pd.read_pickle(OUT / "robustness_SP500_weights.pkl")
    base_sleeve = sleeve_returns(dev, "R1 tercile")
    idx = base_sleeve.index[1:]
    px = DF.etf().reindex(idx)
    spy = swan_beta.load_prices("SP500")["SPY"].reindex(idx).pct_change()
    cash = DF.tbill_daily(idx)
    etf = {k.lower(): px[k].pct_change() for k in ["IEF", "TLT", "SHY", "GLD", "USMV"]}
    etf["shy_bond"] = etf["shy"]   # SHY as the bond asset (the plain "shy" column is the cash leg in A.run)

    def go(sleeve, names=("s", "ief", "gld"), lookback=252, every=1, cost=0.001, end=None):
        a = pd.DataFrame({"s": sleeve.reindex(idx), **etf}).loc[:end]
        s = A.run(a[list(names) + ["shy"]], A.risk_parity(list(names), lookback), None, cost, every)
        st = DF.stats(s, spy.loc[s.index], cash)
        for nm, part in [("2004-14", slice(None, "2014-12-31")), ("2015-26", slice("2015-01-01", None))]:
            p = DF.stats(s.loc[part], spy.loc[part], cash)
            st[f"CAGR {nm}"], st[f"max DD {nm}"] = p["CAGR"], p["max DD"]
        return st

    rows = {"base: R1 + IEF + GLD": go(base_sleeve)}
    for p in [1, 2]:
        rows[f"sleeve rebalanced from month {p + 1} of the quarter"] = go(sleeve_returns(dev, "R1 tercile", p))
    rows["stock cost 0.5%, ETF cost 0.2%"] = go(sleeve_returns(dev, "R1 tercile", 0, 0.005), cost=0.002)
    for lb in [126, 504]:
        rows[f"vol lookback {lb // 21} months"] = go(base_sleeve, lookback=lb)
    rows["allocation rebalanced quarterly"] = go(base_sleeve, every=3)
    rows["bonds: TLT (20y+) instead of IEF"] = go(base_sleeve, names=("s", "tlt", "gld"))
    rows["bonds: SHY (1-3y) instead of IEF"] = go(base_sleeve, names=("s", "shy_bond", "gld"))
    rows["no gold: R1 + IEF"] = go(base_sleeve, names=("s", "ief"))
    rows["ends 2023-12 (before gold's run)"] = go(base_sleeve, end="2023-12-31")
    for n in [30, 50, 75]:
        key = f"n={n}"
        rows[f"stock sleeve: {n} stocks (no sector cap)"] = go(sleeve_returns(rob, key))
    rows["stock sleeve: D0, 30 stocks with sector cap"] = go(sleeve_returns(dev, "D0 blend"))
    rows["stock sleeve: SPY"] = go(spy)
    usmv = etf["usmv"]
    rows["stock sleeve: USMV (from 2011-11)"] = go(usmv.where(usmv.index >= "2011-11-01"))
    rows["R1 sleeve, from 2011-11 (to compare with USMV)"] = go(base_sleeve.where(base_sleeve.index >= "2011-11-01"))
    t = pd.DataFrame(rows).T
    t.to_csv(OUT / "rp_robustness.csv")
    pd.set_option("display.width", 250)
    print(t[["CAGR", "vol", "Sharpe", "max DD", "worst year", "CAGR 2004-14", "CAGR 2015-26", "max DD 2004-14",
             "max DD 2015-26", "GFC 2007-10→2009-03", "COVID 2020-02→03", "2022 bear"]].to_string(
        float_format=lambda x: f"{x:.3f}"))
