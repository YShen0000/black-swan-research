"""How often should the E2 brake be checked? Fixed in advance, on D1 70/15/15 without the brake (4 tranches), with the
same overlay for every variant: while braked, the core's stocks (35% of the money) earn T-bills instead of R1, and
each switch costs 0.3% on that 35%.

Condition, as live.py: SPY (total return) below the average of its last 10 month-end prices (the current month
counted at the day's price) AND more than 10% below its 1-year high.
  B4   4 tranches, each checked only on its decision day (month end, 5 / 10 / 15 trading days in)     (live)
  BD   checked every close; all 35% switch together; traded at that close
  BD2  same, traded at the next close (one day late)
  BDS  checked every close, sticky: braked once the condition holds, released only when SPY is back above its
       10-month average
Intraday stops cannot be tested: only daily closes are available.

    python v2/brake_timing.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

import strategy_robustness as SR
import strategy_search as S

OUT = Path(__file__).parent / "output"
SHARE, COST, DD = 0.35, 0.003, 0.10


def signals(spy_ret):
    """Daily: (below the 10-month average, more than DD below the 1-year high)."""
    p = (1 + spy_ret.fillna(0)).cumprod()
    month = p.index.to_period("M")
    ends = p.groupby(month).last()
    prior9 = ends.rolling(9).sum().shift(1)                    # 9 completed month ends before each month
    avg = (prior9.reindex(month).values + p.values) / 10
    below = pd.Series(p.values < avg, index=p.index)
    deep = p < (1 - DD) * p.rolling(252, min_periods=200).max()
    return below, deep


def tranche_state(cond, index):
    days = {0: list(S.A.month_ends(index))} | {o: SR.mid_month(index, o) for o in (5, 10, 15)}
    out = []
    for o, ds in days.items():
        out.append(cond.reindex(ds).astype(float).reindex(index).ffill().fillna(0))
    return pd.concat(out, axis=1).mean(axis=1)


def overlay(d1, r1, cash, state, lag=1):
    """state decided at a close; effective from `lag` days later."""
    s = state.shift(lag).fillna(0)
    return d1 - SHARE * s * (r1 - cash) - SHARE * s.diff().abs().fillna(0) * COST


if __name__ == "__main__":
    r = S.load()
    rot = S.rotation(SR.ROT)
    fns = {"D1": lambda d, h: S.fixed({"R1": .7, "IEF": .15, "GLD": .15})(d, h).mul(0.5).add(rot(d, h).mul(0.5), fill_value=0)}
    copies = [S.run_all(r, fns)] + [S.run_all(r, fns, every=10 ** 6, dates=SR.mid_month(r.index, o)) for o in (5, 10, 15)]
    d1 = pd.concat([c["D1"] for c in copies], axis=1).mean(axis=1)
    below, deep = signals(r.SPY)
    cond = below & deep
    sticky, on = [], False
    for b, c in zip(below, cond):
        on = c or (on and b)
        sticky.append(float(on))
    states = {"B4 4 tranches (live)": tranche_state(cond, r.index), "BD daily": cond.astype(float),
              "BDS daily, sticky": pd.Series(sticky, index=r.index)}
    r1, cash = r.R1.fillna(0), r.cash
    runs = {"D1 no brake": d1}
    for k, st in states.items():
        runs[k] = overlay(d1, r1, cash, st)
    runs["BD2 daily, next close"] = overlay(d1, r1, cash, states["BD daily"], lag=2)
    spy, c = r.SPY.loc[S.START:], cash.loc[S.START:]
    rows = {}
    for k, s in runs.items():
        st = S.evaluate(s.loc[S.START:], spy, c)
        rows[k] = st
    t = pd.DataFrame(rows).T
    sw = {k: v.loc[S.START:].diff().abs().sum() / 22 for k, v in states.items()}
    t["switches/yr"] = pd.Series(sw).reindex(t.index)
    t.loc["BD2 daily, next close", "switches/yr"] = sw["BD daily"]
    t["days braked"] = pd.Series({k: v.loc[S.START:].mean() for k, v in states.items()}).reindex(t.index)
    t.to_csv(OUT / "brake_timing.csv")
    pd.set_option("display.width", 250)
    cols = ["CAGR", "Sharpe", "max DD", "worst year", "CAGR 2005-14", "CAGR 2015-26", "Sharpe 2005-14", "Sharpe 2015-26",
            "Sharpe ex-COVID", "switches/yr", "days braked"] + list(S.DF.CRISES)
    print(t[cols].astype(float).round(3).to_string())
