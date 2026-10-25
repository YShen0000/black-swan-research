"""Beyond swan beta: momentum, trend and rotation, alone and combined with the swan sleeve. Stocks and ETFs only, no
leverage, no shorting. Variants and the bar are fixed here before any result was seen.

Bar for a "qualified" long-term strategy (all must hold):
  Q1 CAGR >= 10%        Q2 max drawdown <= 25%        Q3 CAGR >= 8% in both 2005-14 and 2015-26
  Q4 Sharpe >= SPY's in both sub-periods                Q5 Sharpe without COVID (2020-02-19 -> 06-30) >= R1 50/25/25's

Stock sleeves (search_sleeves.py, monthly): R1 swan, M1 12-1 momentum, M2 swan + momentum ranks; SPY.
Rules (month-end decision, traded next day, weights drift in between):
  trend   an asset is held only while its price is above its 10-month average of month-end prices, else T-bills
  score   relative momentum = mean of its 3-, 6- and 12-month returns
  An asset needs 12 months of history to be used; until then its slot is in T-bills. ETF and SPY history starts
  2002, the stock sleeves 2004; everything is scored from 2005-01 so that each starts with a full year of signals.
Variants:
  benchmarks  SPY; SPY 60 / IEF 40; R1 50 / IEF 25 / GLD 25 ("R1 50/25/25"); risk parity R1/IEF/GLD
  A sleeves   M1, M2 alone; M1 and M2 in 50/25/25
  B trend     B1 GTAA-5 (Faber): 20% each SPY, EFA, IEF, VNQ, DBC, each with the trend rule
              B2 B1 with R1 for SPY;  B3 R1 50/25/25 with the trend rule on each part;  B4 same at 70/15/15
  C rotation  C1 dual momentum (GEM): if SPY's 12-month return beats T-bills, the better of SPY and EFA on
              12-month return, else IEF;  C2 C1 with R1 for SPY
              C3 top 3 of SPY, EFA, EEM, VNQ, DBC, IEF, TLT, GLD by score, equal weight, each with the trend rule
              C4 C3 with R1 for SPY;  C5 top 3 of the nine sector SPDRs by score, with the trend rule
  D combos    D1 half R1 50/25/25, half C3;  D2 half R1 50/25/25, half B2
Costs per unit of weight change: 0.3% stock sleeves (on top of their own 0.3% inside), 0.1% ETFs, 0 T-bills.

    python v2/strategy_search.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

import allocation as A
import backtest
import defense as DF
import swan_beta

OUT = Path(__file__).parent / "output"
SECTORS = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]
SMA_MONTHS, LOOKS = 10, (3, 6, 12)
COVID = ("2020-02-19", "2020-06-30")


def load():
    close = swan_beta.load_prices("SP500")
    sl = pd.read_pickle(OUT / "search_sleeves_weights.pkl")
    w = {t: {**v, "SPY": pd.Series({"SPY": 1.0})} for t, v in sl.items()}
    r, _ = backtest.daily_returns(w, close, 0.003)
    r = r.loc[r.index[1:]].drop(columns="SPY")
    etf = DF.etf().loc["2002-01-01":]
    days = etf.index.union(r.index)
    out = pd.concat([etf.reindex(days).pct_change(fill_method=None).drop(columns=["SPLV", "USMV"]), r.reindex(days)], axis=1)
    out["SPY"] = close["SPY"].reindex(days).pct_change(fill_method=None)
    out["cash"] = DF.tbill_daily(days)
    return out.iloc[1:]


class Rules:
    """Signals from the return history up to the decision day (month ends only)."""

    def __init__(self, hist, d):
        self.d = d
        self.px = (1 + hist.fillna(0)).cumprod().where(hist.notna().cumsum() > 0)
        self.me = self.px.groupby(self.px.index.to_period("M")).tail(1)

    def ready(self, a):
        return self.px[a].notna().sum() >= 252

    def up(self, a):
        m = self.me[a].dropna().iloc[-SMA_MONTHS:]
        return self.ready(a) and len(m) == SMA_MONTHS and self.px[a].iloc[-1] > m.mean()

    def ret(self, a, months):
        m = self.me[a].dropna()
        return m.iloc[-1] / m.iloc[-1 - months] - 1 if len(m) > months else np.nan

    def score(self, a):
        return np.mean([self.ret(a, k) for k in LOOKS]) if self.ready(a) else np.nan


def fixed(weights):
    def f(d, hist):
        R = Rules(hist, d)
        w = pd.Series(weights, dtype=float)
        miss = [a for a in w.index if not R.ready(a)]
        w["cash"] = w.get("cash", 0) + w[miss].sum()
        w[miss] = 0
        return w
    return f


def trend(weights):
    def f(d, hist):
        R = Rules(hist, d)
        w = pd.Series(weights, dtype=float)
        off = [a for a in w.index if not R.up(a)]
        w["cash"] = w.get("cash", 0) + w[off].sum()
        w[off] = 0
        return w
    return f


def gem(us):
    def f(d, hist):
        R = Rules(hist, d)
        cash12 = R.ret("cash", 12)
        if R.ret(us, 12) > cash12:
            return pd.Series({us if R.ret(us, 12) >= R.ret("EFA", 12) else "EFA": 1.0})
        return pd.Series({"IEF": 1.0})
    return f


def rotation(universe, top=3):
    def f(d, hist):
        R = Rules(hist, d)
        s = pd.Series({a: R.score(a) for a in universe}).dropna().nlargest(top)
        w = pd.Series(0.0, index=list(universe) + ["cash"])
        for a in s.index:
            w["cash" if not R.up(a) else a] += 1 / top
        w["cash"] += 1 - w.sum()
        return w
    return f


def half_half(f1, f2):
    def f(d, hist):
        return f1(d, hist).mul(0.5).add(f2(d, hist).mul(0.5), fill_value=0)
    return f


def variants():
    core = {"R1": 0.5, "IEF": 0.25, "GLD": 0.25}
    g5 = {"SPY": .2, "EFA": .2, "IEF": .2, "VNQ": .2, "DBC": .2}
    rot = ["SPY", "EFA", "EEM", "VNQ", "DBC", "IEF", "TLT", "GLD"]
    return {
        "SPY": fixed({"SPY": 1.0}),
        "SPY 60 / IEF 40": fixed({"SPY": .6, "IEF": .4}),
        "R1 50/25/25": fixed(core),
        "risk parity R1/IEF/GLD": A.risk_parity(["R1", "IEF", "GLD"]),
        "A M1 momentum alone": fixed({"M1": 1.0}),
        "A M2 swan+momentum alone": fixed({"M2": 1.0}),
        "A M1 50/25/25": fixed({"M1": .5, "IEF": .25, "GLD": .25}),
        "A M2 50/25/25": fixed({"M2": .5, "IEF": .25, "GLD": .25}),
        "B1 GTAA-5 trend": trend(g5),
        "B2 GTAA-5 trend, R1 for SPY": trend({**{k: v for k, v in g5.items() if k != "SPY"}, "R1": .2}),
        "B3 R1 50/25/25 + trend": trend(core),
        "B4 R1 70/15/15 + trend": trend({"R1": .7, "IEF": .15, "GLD": .15}),
        "C1 dual momentum SPY/EFA/IEF": gem("SPY"),
        "C2 dual momentum R1/EFA/IEF": gem("R1"),
        "C3 rotation top 3 of 8": rotation(rot),
        "C4 rotation top 3 of 8, R1 for SPY": rotation(["R1"] + rot[1:]),
        "C5 sector rotation top 3 of 9": rotation(SECTORS),
        "D1 half R1 50/25/25, half C3": half_half(fixed(core), rotation(rot)),
        "D2 half R1 50/25/25, half B2": half_half(fixed(core), trend({**{k: v for k, v in g5.items() if k != "SPY"}, "R1": .2})),
    }


def evaluate(s, spy, cash):
    st = DF.stats(s, spy, cash)
    for nm, part in [("2005-14", slice(None, "2014-12-31")), ("2015-26", slice("2015-01-01", None))]:
        p = DF.stats(s.loc[part], spy.loc[part], cash)
        st[f"CAGR {nm}"], st[f"Sharpe {nm}"], st[f"max DD {nm}"] = p["CAGR"], p["Sharpe"], p["max DD"]
    ex = s.drop(s.loc[COVID[0]:COVID[1]].index)
    st["Sharpe ex-COVID"] = ex.mean() / ex.std() * np.sqrt(252)
    return st


def run_all(r, fns, stock=("R1", "M1", "M2"), **kw):
    """stock: the stock-sleeve columns, which cost 0.3% per unit of weight change (ETFs 0.1%)."""
    assets = r.copy()
    assets["shy"] = 0.0   # A.run's idle leg, unused: T-bills are the explicit "cash" asset
    cost = pd.Series(0.001, index=assets.columns)
    cost[list(stock)] = 0.003
    cost[["cash", "shy"]] = 0.0
    return {k: A.run(assets, f, None, cost, drift=True, **kw) for k, f in fns.items()}


START = "2005-01-01"


def table(runs, r):
    runs = {k: s.loc[START:] for k, s in runs.items()}
    spy, cash = r.SPY.loc[START:], r.cash.loc[START:]
    t = pd.DataFrame({k: evaluate(s, spy, cash) for k, s in runs.items()}).T
    ref = t.loc["R1 50/25/25", "Sharpe ex-COVID"]
    t["Q1 CAGR>=10%"] = t.CAGR >= 0.10
    t["Q2 maxDD<=25%"] = t["max DD"] <= 0.25
    t["Q3 both halves>=8%"] = (t["CAGR 2005-14"] >= 0.08) & (t["CAGR 2015-26"] >= 0.08)
    t["Q4 Sharpe>=SPY both"] = (t["Sharpe 2005-14"] >= t.loc["SPY", "Sharpe 2005-14"]) & \
                               (t["Sharpe 2015-26"] >= t.loc["SPY", "Sharpe 2015-26"])
    t["Q5 ex-COVID>=R1 core"] = t["Sharpe ex-COVID"] >= ref
    t["passes"] = t.filter(like="Q").sum(axis=1)
    return t


if __name__ == "__main__":
    r = load()
    runs = run_all(r, variants())
    t = table(runs, r)
    t.to_csv(OUT / "strategy_search.csv")
    pd.set_option("display.width", 300, "display.max_columns", 40)
    num = ["CAGR", "vol", "Sharpe", "max DD", "worst year", "CAGR 2005-14", "CAGR 2015-26", "Sharpe 2005-14",
           "Sharpe 2015-26", "max DD 2005-14", "max DD 2015-26", "Sharpe ex-COVID"]
    print(t[num].astype(float).round(3).to_string())
    print()
    print(t[list(DF.CRISES) + [c for c in t if c.startswith("Q")] + ["passes"]].to_string(
        float_format=lambda x: f"{x:.3f}"))
