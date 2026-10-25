"""Overfitting checks for the two best variants of strategy_search.py: C3 (rotation, top 3 of 8 asset ETFs by mean
3/6/12-month return, each above its 10-month average) and D1 (half R1 50/25/25, half C3). Each row changes one
thing: trend length, lookbacks, number held, decision day, costs, each asset left out, the half/half ratio, and
gold left out everywhere (its 2024-26 run). Plus 3 sub-periods and rolling 5-year windows against the references.
The decision day turned out to matter (COVID -0% at month end, -22% ten days later), so the last table splits each
strategy into 4 tranches decided on different days; those are the numbers to trust.

    python v2/strategy_robustness.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

import allocation as A
import defense as DF
import strategy_search as S

OUT = Path(__file__).parent / "output"
ROT = ["SPY", "EFA", "EEM", "VNQ", "DBC", "IEF", "TLT", "GLD"]
CORE = {"R1": 0.5, "IEF": 0.25, "GLD": 0.25}


def mid_month(index, offset=10):
    """The decision day `offset` trading days into each month (instead of the month end)."""
    first = index.to_series().groupby(index.to_period("M")).head(offset + 1)
    return list(first.groupby(first.index.to_period("M")).tail(1))


def tranched(r, fns, offsets=(5, 10, 15)):
    """Average of copies decided at the month end and `offsets` trading days into each month: four sub-portfolios,
    so the result does not hang on one decision day (rebalance timing luck)."""
    copies = [S.run_all(r, fns)] + [S.run_all(r, fns, every=10 ** 6, dates=mid_month(r.index, o)) for o in offsets]
    return {k: pd.concat([c[k] for c in copies], axis=1).mean(axis=1) for k in fns}, copies


def pair(rot=ROT, top=3, core=CORE, share=0.5):
    c3 = S.rotation(rot, top)
    return {"C3": c3, "D1": lambda d, h: S.fixed(core)(d, h).mul(1 - share).add(c3(d, h).mul(share), fill_value=0)}


def go(r, label, fns, setting=None, **kw):
    old = S.SMA_MONTHS, S.LOOKS
    if setting:
        S.SMA_MONTHS, S.LOOKS = setting
    runs = S.run_all(r, fns, **kw)
    S.SMA_MONTHS, S.LOOKS = old
    out = {}
    for k, s in runs.items():
        s = s.loc[S.START:]
        st = S.evaluate(s, r.SPY.loc[S.START:], r.cash.loc[S.START:])
        out[(label, k)] = st[["CAGR", "Sharpe", "max DD", "worst year", "CAGR 2005-14", "CAGR 2015-26", "Sharpe ex-COVID",
                               "COVID 2020-02→03", "2022 bear"]]
    return out, runs


if __name__ == "__main__":
    r = S.load()
    rows = {}
    add = lambda res: rows.update(res[0])
    base = go(r, "base", pair())
    add(base)
    for m in [8, 12]:
        add(go(r, f"trend average {m} months", pair(), (m, S.LOOKS)))
    for looks in [(12,), (6, 12), (1, 3, 6, 12)]:
        add(go(r, f"lookbacks {looks}", pair(), (S.SMA_MONTHS, looks)))
    for top in [2, 4]:
        add(go(r, f"hold top {top}", pair(top=top)))
    add(go(r, "decided 10 trading days into the month", pair(), every=10 ** 6, dates=mid_month(r.index)))
    for a in ROT:
        add(go(r, f"without {a}", pair(rot=[x for x in ROT if x != a])))
    add(go(r, "no gold anywhere (core R1 50 / IEF 50)", pair(rot=[x for x in ROT if x != "GLD"],
                                                            core={"R1": .5, "IEF": .5})))
    for share in [0.3, 0.7]:
        add(go(r, f"D1 with {share:.0%} in C3", {"D1": pair(share=share)["D1"]}))
    add(go(r, "SPY for R1 in the core", {"D1": pair(core={"SPY": .5, "IEF": .25, "GLD": .25})["D1"]}))
    t = pd.DataFrame(rows).T
    t.to_csv(OUT / "strategy_robustness.csv")
    pd.set_option("display.width", 250)
    print(t.astype(float).round(3).to_string())

    # costs x3, sub-periods and rolling 5-year windows on the base runs
    refs = S.run_all(r, {"R1 50/25/25": S.fixed(CORE), "SPY": S.fixed({"SPY": 1.0}),
                         "SPY 60 / IEF 40": S.fixed({"SPY": .6, "IEF": .4})})
    allr = {**base[1], **refs}
    allr = {k: s.loc[S.START:] for k, s in allr.items()}
    print("\nthree sub-periods, CAGR / max DD")
    parts = [("2005-11", slice("2005", "2011")), ("2012-18", slice("2012", "2018")), ("2019-26", slice("2019", None))]
    sub = {}
    for k, s in allr.items():
        for nm, p in parts:
            x = s.loc[p]
            w = (1 + x).cumprod()
            sub[(k, nm)] = f"{w.iloc[-1] ** (252 / len(x)) - 1:.1%} / {(1 - w / w.cummax()).max():.1%}"
    print(pd.Series(sub).unstack().to_string())
    print("\nrolling 5-year CAGR: share of windows (monthly steps) where the row beats the column")
    m = {k: (1 + s).resample("ME").prod() for k, s in allr.items()}
    roll = pd.DataFrame({k: v.rolling(60).apply(np.prod, raw=True) ** (1 / 5) - 1 for k, v in m.items()}).dropna()
    print(pd.DataFrame({c: {k: (roll[k] > roll[c]).mean() for k in ["C3", "D1"]} for c in roll}).round(2).to_string())
    print("worst rolling 5-year CAGR:", roll.min().round(3).to_dict())
    cost3 = {}
    assets = r.copy()
    assets["shy"] = 0.0
    c = pd.Series(0.003, index=assets.columns)
    c[["R1", "M1", "M2"]] = 0.009
    c[["cash", "shy"]] = 0.0
    for k, f in pair().items():
        s = A.run(assets, f, None, c, drift=True).loc[S.START:]
        cost3[k] = S.evaluate(s, r.SPY.loc[S.START:], r.cash.loc[S.START:])[["CAGR", "Sharpe", "max DD"]]
    print("\ncosts x3:"); print(pd.DataFrame(cost3).T.astype(float).round(3).to_string())
    turn = {}
    for k, f in pair().items():
        w_hist = []
        rr = r.copy(); rr["shy"] = 0.0
        ends = A.month_ends(rr.index)
        for d in ends[ends >= pd.Timestamp(S.START)]:
            w_hist.append(f(d, rr.loc[:d].drop(columns="shy")).reindex(rr.columns).fillna(0))
        w_hist = pd.DataFrame(w_hist)
        turn[k] = w_hist.diff().abs().sum(axis=1).mean() * 12 / 2
    print("\none-way turnover per year (target changes only):", {k: round(v, 2) for k, v in turn.items()})

    tr, copies = tranched(r, pair())
    rows = {}
    for k in tr:
        for o, c in zip(["month end", "day 5", "day 10", "day 15"], copies):
            rows[(k, f"decided {o}")] = S.evaluate(c[k].loc[S.START:], r.SPY.loc[S.START:], r.cash.loc[S.START:])
        rows[(k, "4 tranches")] = S.evaluate(tr[k].loc[S.START:], r.SPY.loc[S.START:], r.cash.loc[S.START:])
    cols = ["CAGR", "Sharpe", "max DD", "worst year", "CAGR 2005-14", "CAGR 2015-26", "Sharpe 2005-14", "Sharpe 2015-26",
            "Sharpe ex-COVID", "GFC 2007-10→2009-03", "COVID 2020-02→03", "2022 bear", "2025 tariffs"]
    print("\ndecision day and 4 tranches:")
    print(pd.DataFrame(rows).T[cols].astype(float).round(3).to_string())
    pd.to_pickle(tr, OUT / "strategy_tranched_returns.pkl")
