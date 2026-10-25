"""Higher-return mixes for a 20-25% max drawdown target: R1 sleeve at 50 / 60 / 70% of the money, the rest in IEF and
GLD, split either equally or by inverse 1-year vol. SPY at the same weights and the current risk parity as
references. Monthly sleeve and reset, weights drift in between, 0.3% stock / 0.1% ETF cost, 2004 -> today.

    python v2/frontier.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

import allocation as A
import backtest
import defense as DF
import experiments as E
import swan_beta

OUT = Path(__file__).parent / "output"


def inv_vol_rest(x):
    def f(d, hist):
        inv = 1 / hist[["ief", "gld"]].iloc[-252:].std()
        return pd.concat([pd.Series({"s": x}), (1 - x) * inv / inv.sum()])
    return f


def recovery(s):
    """Months from the max-drawdown peak until the old high is regained (None if not yet)."""
    w = (1 + s).cumprod()
    trough = (1 - w / w.cummax()).idxmax()
    peak = w[:trough].idxmax()
    back = w[trough:][w[trough:] >= w[peak]]
    return None if back.empty else round((back.index[0] - peak).days / 30.4)


if __name__ == "__main__":
    swans = E.setup("SP500")
    close = swan_beta.load_prices("SP500")
    dev = pd.read_pickle(OUT / "develop_SP500_weights.pkl")
    r, _ = backtest.daily_returns({t: {"R1": w["R1 tercile"], "SPY": pd.Series({"SPY": 1.0})} for t, w in dev.items()},
                                  close, 0.003)
    idx = r.index[1:]
    r = r.loc[idx]
    px = DF.etf().reindex(idx)
    etf = {k.lower(): px[k].pct_change() for k in ["IEF", "GLD", "SHY"]}
    spy, cash = r.SPY, DF.tbill_daily(idx)
    sw = swans[swans.peak >= idx[0]]

    def go(sleeve, fn):
        a = pd.DataFrame({"s": r[sleeve], **etf})[["s", "ief", "gld", "shy"]]
        return A.run(a, fn, None, 0.001, drift=True)

    runs = {}
    for x in [0.5, 0.6, 0.7]:
        runs[f"R1 {x:.0%}, IEF/GLD equal"] = go("R1", A.fixed({"s": x, "ief": (1 - x) / 2, "gld": (1 - x) / 2}))
        runs[f"R1 {x:.0%}, IEF/GLD inverse vol"] = go("R1", inv_vol_rest(x))
    for x in [0.5, 0.6, 0.7]:
        runs[f"SPY {x:.0%}, IEF/GLD equal"] = go("SPY", A.fixed({"s": x, "ief": (1 - x) / 2, "gld": (1 - x) / 2}))
    runs["current: risk parity R1/IEF/GLD"] = go("R1", A.risk_parity(["s", "ief", "gld"]))
    runs["SPY 100%"] = r.SPY

    rows = {}
    for k, s in runs.items():
        st = DF.stats(s, spy, cash)
        for nm, part in [("2004-14", slice(None, "2014-12-31")), ("2015-26", slice("2015-01-01", None))]:
            p = DF.stats(s.loc[part], spy.loc[part], cash)
            st[f"CAGR {nm}"], st[f"Sharpe {nm}"], st[f"max DD {nm}"] = p["CAGR"], p["Sharpe"], p["max DD"]
        st["to 2023-12: CAGR"] = DF.stats(s.loc[:"2023-12-31"], spy, cash)["CAGR"]
        st["recovery months"] = recovery(s)
        w = (1 + s).cumprod()
        for n, v in sw.iterrows():
            st[n] = w.loc[v.trough] / w.loc[v.peak] - 1
        yearly = (1 + s).groupby(s.index.year).prod() - 1
        st["losing years"] = int((yearly < 0).sum())
        rows[k] = st
    t = pd.DataFrame(rows).T
    t.to_csv(OUT / "frontier.csv")
    pd.set_option("display.width", 250)
    f = lambda c: t[c].astype(float).round(3).to_string()
    print(f(["CAGR", "vol", "Sharpe", "max DD", "worst year", "losing years", "recovery months", "to 2023-12: CAGR"]))
    print(); print(t[["MDD peak", "MDD trough"]].to_string())
    print(); print(f(["CAGR 2004-14", "Sharpe 2004-14", "max DD 2004-14", "CAGR 2015-26", "Sharpe 2015-26", "max DD 2015-26"]))
    print(); print(f(list(DF.CRISES)))
    print(); print(f(list(sw.index)))
    yearly = pd.DataFrame({k: (1 + s).groupby(s.index.year).prod() - 1 for k, s in runs.items()
                           if k.startswith("R1") and "equal" in k or k.startswith("current") or k == "SPY 100%"})
    print(); print(yearly.round(3).to_string())
