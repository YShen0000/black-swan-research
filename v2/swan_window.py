"""Which past swans should the score use? Fixed in advance. At each date only swans confirmed before it are used
(all from 1987, as the strategy does); the score is ½ (swan part) + ½ 1-year market beta, then R1's rule (lowest
third, inverse vol, <= 25% per sector).

  W5   mean of the last 5 swans                                         (current)
  W3   mean of the last 3
  WA   mean of all swans since 1987
  WS   mean of the severe swans only (^GSPC peak-to-trough drop >= 25%)
  WR   recency-weighted mean of all, half-life 2 swans
  WM   median of the last 5 (robust to one odd swan)
  W0   no swan part: 1-year market beta alone (control: what do swans add at all?)

Checks: (1) pick quality: at each target swan's peak (Swan5 ... Swan11) the next-swan beta of the lowest third,
mean and per swan; (2) the sleeve alone, monthly, 2005-26; (3) D1 70/15/15 with it, 4 tranches.
A window replaces W5 only if it improves pick quality in at least 5 of 7 swans and D1's Sharpe in 2005-14, 2015-26
and without COVID.

    python v2/swan_window.py build      # ~15 min, cached
    python v2/swan_window.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import backtest
import components as C
import develop as D
import experiments as E
import strategy_robustness as SR
import strategy_search as S
import swan_beta

OUT = Path(__file__).parent / "output"
U = "SP500"
KEYS = ["W5", "W3", "WA", "WS", "WR", "WM", "W0"]


def swan_parts(swans, betas, t, tickers):
    known = swans[(swans.confirmed < t) & (swans.trough >= "1987-01-01")]
    gspc = swan_beta.load_prices(U)["^GSPC"]
    drop = 1 - gspc.reindex(known.trough).values / gspc.reindex(known.peak).values
    b = betas.reindex(tickers)[known.index]
    names = list(known.index)
    age = np.arange(len(names))[::-1]                       # 0 = most recent
    rw = 0.5 ** (age / 2)
    severe = [n for n, d in zip(names, drop) if d >= 0.25]
    return {"W5": b[names[-5:]].mean(axis=1), "W3": b[names[-3:]].mean(axis=1), "WA": b.mean(axis=1),
            "WS": b[severe].mean(axis=1),
            "WR": (b * rw).sum(axis=1, min_count=1) / (b.notna() * rw).sum(axis=1),
            "WM": b[names[-5:]].median(axis=1)}


def scores(swans, t, ctx):
    x = ctx["x"]
    parts = swan_parts(swans, C._BETAS[U], t, x.index)
    out = {k: ((v + x.mkt_beta) / 2).dropna() for k, v in parts.items()}
    out["W0"] = x.mkt_beta
    return out


def build():
    swans = E.setup(U)
    out = {}
    for t in sorted(pd.read_pickle(OUT / "develop_SP500_weights.pkl")):
        ctx = E.context(t, U, swans)
        out[t] = {k: D.hold(s, t, U, ctx, len(s) // 3) for k, s in scores(swans, t, ctx).items()}
        print(t.date(), flush=True)
    return out


def pick_quality():
    swans = E.setup(U)
    rows = {}
    for target in ["Swan5", "Swan6", "Swan7", "Swan8", "Swan9", "Swan10D", "Swan11"]:
        t = swans.peak[target]
        ctx = E.context(t, U, swans)
        actual = C._BETAS[U][target]
        rows[target] = {k: actual.reindex(s.nsmallest(len(s) // 3).index).mean() for k, s in scores(swans, t, ctx).items()}
        rows[target]["all candidates"] = actual.reindex(ctx["x"].index).mean()
    return pd.DataFrame(rows).T


if __name__ == "__main__":
    cache = OUT / "swan_window_weights.pkl"
    if sys.argv[1:] == ["build"]:
        pd.to_pickle(build(), cache)
        sys.exit()
    pd.set_option("display.width", 250)
    pq = pick_quality()
    print("actual beta in the target swan of the lowest third picked (lower = better)")
    print(pq.round(3).to_string())
    print("mean:", pq.mean().round(3).to_dict())
    better = {k: int((pq[k] < pq["W5"]).sum()) for k in KEYS if k != "W5"}
    print("swans where the window beats W5:", better)

    w = pd.read_pickle(cache)
    sl, _ = backtest.daily_returns(w, swan_beta.load_prices(U), 0.003)
    r = S.load()
    for k in KEYS:
        r[k] = sl[k].reindex(r.index)
    rot = S.rotation(SR.ROT)
    fns = {k: (lambda c: lambda d, h: c(d, h).mul(0.5).add(rot(d, h).mul(0.5), fill_value=0))(
        S.fixed({k: .7, "IEF": .15, "GLD": .15})) for k in KEYS}
    copies = [S.run_all(r, fns, stock=KEYS)] + [S.run_all(r, fns, stock=KEYS, every=10 ** 6, dates=SR.mid_month(r.index, o))
                                                for o in (5, 10, 15)]
    d1 = {k: pd.concat([c[k] for c in copies], axis=1).mean(axis=1) for k in KEYS}
    spy, cash = r.SPY.loc[S.START:], r.cash.loc[S.START:]
    rows = {}
    for k in KEYS:
        rows[(k, "sleeve")] = S.evaluate(r[k].loc[S.START:], spy, cash)
        rows[(k, "D1 70/15/15")] = S.evaluate(d1[k].loc[S.START:], spy, cash)
    t = pd.DataFrame(rows).T
    t.to_csv(OUT / "swan_window.csv")
    cols = ["CAGR", "Sharpe", "max DD", "worst year", "CAGR 2005-14", "CAGR 2015-26", "Sharpe 2005-14", "Sharpe 2015-26",
            "Sharpe ex-COVID"] + list(S.DF.CRISES)
    print(); print(t[cols].astype(float).round(3).to_string())
    b = t.loc[("W5", "D1 70/15/15")]
    adopt = {k: bool(better[k] >= 5 and all(t.loc[(k, "D1 70/15/15"), c] > b[c]
                                            for c in ["Sharpe 2005-14", "Sharpe 2015-26", "Sharpe ex-COVID"]))
             for k in KEYS if k != "W5"}
    print("adopt (rule above):", adopt)
