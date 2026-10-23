"""Long-term allocation with stocks + ETFs only (no leverage, no options); target: max drawdown <= 15%.
Variants fixed in advance:

  A0  60% stock sleeve / 20% IEF / 20% GLD                               (from defense.py)
  A1  risk parity: sleeve, IEF, GLD                                       (inverse 1-year vol, monthly)
  A2  risk parity: sleeve, IEF, GLD, TIP, DBC     adds inflation assets for stock-and-bond selloffs like 2022
  A3  A2 + volatility target 8%: scale the mix to 8% expected vol (3-month realised), rest in SHY, never above 100%
  A4  A2 + drawdown control: risky share 100% while the portfolio is < 5% below its high, falling linearly to 50% at
      12% below, rest in SHY; decided at each month end
  A5  A0 + volatility target 8%
  A6  A0 + drawdown control
Stock sleeves: return track R1 and defensive D0 (swan-beta selection), and SPY as the control. An asset enters the
risk parity once it has 6 months of history (GLD 2004-11, TIP 2003-12, DBC 2006-02). Monthly rebalance; 0.1% cost
per unit of weight change. 2004 -> today.

    python v2/allocation.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

import defense as DF

OUT = Path(__file__).parent / "output"
DD_START, DD_FLOOR, DD_MIN_SHARE = 0.05, 0.12, 0.5
VOL_TARGET = 0.08


def month_ends(index):
    return index.to_series().groupby(index.to_period("M")).tail(1).index


def run(assets, target_fn, control=None, cost=0.001):
    """assets: daily returns (incl. 'shy'); target_fn(date, history) -> weights of the risky mix at a month end.
    control: None, 'vol' or 'dd'. Weights decided at month end are held (daily rebalanced) for the next month."""
    ends = set(month_ends(assets.index))
    risky = [c for c in assets.columns if c != "shy"]
    w = pd.Series(0.0, index=assets.columns)
    w["shy"] = 1.0
    nav, peak, out, mix_hist = 1.0, 1.0, [], []
    for i, (d, row) in enumerate(assets.iterrows()):
        ret = float((w * row.fillna(0)).sum())
        out.append(ret)
        nav *= 1 + ret
        peak = max(peak, nav)
        if d in ends and i >= 126:
            target = target_fn(d, assets.iloc[: i + 1][risky]).reindex(risky).fillna(0)
            mix_ret = assets.iloc[max(0, i - 62): i + 1][risky].fillna(0) @ target
            share = 1.0
            if control == "vol":
                share = min(1.0, VOL_TARGET / (mix_ret.std() * np.sqrt(252)))
            elif control == "dd":
                dd = 1 - nav / peak
                share = 1.0 if dd <= DD_START else max(DD_MIN_SHARE, 1 - (dd - DD_START) / (DD_FLOOR - DD_START) * (1 - DD_MIN_SHARE))
            new = (target * share).reindex(assets.columns).fillna(0)
            new["shy"] += 1 - share
            out[-1] -= (new - w).abs().sum() * cost
            w = new
    return pd.Series(out, index=assets.index)


def fixed(weights):
    def f(d, hist):
        w = pd.Series(weights, dtype=float)
        if hist["gld"].iloc[-126:].isna().any():   # GLD not yet trading: its share goes to IEF
            w["ief"] += w.pop("gld")
        return w / w.sum()
    return f


def risk_parity(names):
    def f(d, hist):
        h = hist[names].iloc[-252:]
        ok = [c for c in names if h[c].iloc[-126:].notna().all()]
        inv = 1 / h[ok].std()
        return inv / inv.sum()
    return f


if __name__ == "__main__":
    r = DF.sleeves()
    px = DF.etf().reindex(r.index)
    cash = DF.tbill_daily(r.index)
    etf = {k.lower(): px[k].pct_change() for k in ["IEF", "GLD", "TIP", "DBC", "SHY"]}
    variants = {
        "A0 60/20/20": (fixed({"s": .6, "ief": .2, "gld": .2}), None),
        "A1 risk parity 3": (risk_parity(["s", "ief", "gld"]), None),
        "A2 risk parity 5": (risk_parity(["s", "ief", "gld", "tip", "dbc"]), None),
        "A3 A2 + vol target 8%": (risk_parity(["s", "ief", "gld", "tip", "dbc"]), "vol"),
        "A4 A2 + drawdown control": (risk_parity(["s", "ief", "gld", "tip", "dbc"]), "dd"),
        "A5 A0 + vol target 8%": (fixed({"s": .6, "ief": .2, "gld": .2}), "vol"),
        "A6 A0 + drawdown control": (fixed({"s": .6, "ief": .2, "gld": .2}), "dd"),
    }
    rows = {}
    for sleeve in ["return R1", "defensive D0", "SPY"]:
        assets = pd.DataFrame({"s": r[sleeve], **etf})
        for name, (fn, control) in variants.items():
            s = run(assets, fn, control)
            st = DF.stats(s, r.SPY, cash)
            for part_name, part in [("2004-14", slice(None, "2014-12-31")), ("2015-26", slice("2015-01-01", None))]:
                p = DF.stats(s.loc[part], r.SPY.loc[part], cash)
                st[f"CAGR {part_name}"], st[f"Sharpe {part_name}"], st[f"max DD {part_name}"] = p["CAGR"], p["Sharpe"], p["max DD"]
            st["meets max DD <= 15%"] = st["max DD"] <= 0.15
            rows[(sleeve, name)] = st
        rows[(sleeve, "stock sleeve alone")] = DF.stats(r[sleeve], r.SPY, cash)
    t = pd.DataFrame(rows).T
    t.to_csv(OUT / "allocation.csv")
    pd.set_option("display.width", 280)
    cols = ["CAGR", "vol", "Sharpe", "max DD", "worst year", "CAGR 2004-14", "CAGR 2015-26", "Sharpe 2004-14",
            "Sharpe 2015-26", "max DD 2004-14", "max DD 2015-26", "meets max DD <= 15%"]
    print(t[cols].to_string(float_format=lambda x: f"{x:.3f}"))
    print()
    print(t[list(DF.CRISES)].to_string(float_format=lambda x: f"{x:.3f}"))
