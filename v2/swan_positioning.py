"""Two ways to put the swan signal where it matters. Rules fixed before looking at results; S&P500 point-in-time,
monthly sleeves (R1 / D0 from develop.py), weights drift between resets, 0.3% stock / 0.1% ETF cost, 2004 -> today.

1. Crash-loss sizing: hold as much stock as a predicted swan loss allows.
   At each month end: predicted swan beta of the sleeve b = ½ weighted mean of its stocks' last-5 swan betas
   + ½ its 1-year market beta (the blend forecast at portfolio level; SPY: b = 1). Predicted index drop D = mean
   (C1) or worst (C2) ^GSPC peak->trough drop of the swans confirmed so far since 1987. Stock weight
   w = min(1, 15% / (b * D)), the 15% loss budget being the max-drawdown target. Bonds and gold are assumed to lose
   nothing (conservative); the rest 1 - w is split between IEF and GLD by inverse 1-year vol.
   Controls: the same rule on SPY; a fixed stock weight equal to the variant's average w, same IEF / GLD split.
2. Switch: hold SPY normally, the defensive sleeve D0 during a selloff.
   Selloff starts on the first close with ^OEX W1 10% / W2 15% below its 1-year high (15% = the swan rule), ends
   when it is back within 5% (the swan confirmation rule). Switched the next day at the latest month's D0 weights,
   0.3% cost per unit of turnover. Shown alone, in 60% / 20% IEF / 20% GLD, and in the risk parity.
   Controls: SPY, D0 and R1 held all the time.
Adoption (as before): better Sharpe than its control in 2004-14 and in 2015-26 and without COVID, and a smaller
loss in at least 5 of the 7 swans since 2004.

    python v2/swan_positioning.py
"""
from bisect import bisect_right
from pathlib import Path

import numpy as np
import pandas as pd

import allocation as A
import backtest
import components as C
import defense as DF
import experiments as E
import swan_beta

OUT = Path(__file__).parent / "output"
U = "SP500"
BUDGET = 0.15
ENTRY = {"W1": 0.10, "W2": 0.15}
EXIT = 0.05


def selloff_state(oex, entry):
    dd = 1 - oex / oex.rolling(252, min_periods=1).max()
    on, out = False, []
    for x in dd:
        on = x >= entry if not on else x >= EXIT
        out.append(on)
    return pd.Series(out, index=oex.index, dtype=float)


if __name__ == "__main__":
    swans = E.setup(U)
    close = swan_beta.load_prices(U)
    gspc = close["^GSPC"]
    betas = C._BETAS[U]
    dev = pd.read_pickle(OUT / "develop_SP500_weights.pkl")
    months = sorted(dev)
    r, _ = backtest.daily_returns({t: {"R1": dev[t]["R1 tercile"], "D0": dev[t]["D0 blend"],
                                       "SPY": pd.Series({"SPY": 1.0})} for t in months}, close, 0.003)
    idx = r.index[1:]
    r = r.loc[idx]
    px = DF.etf().reindex(idx)
    etf = {k.lower(): px[k].pct_change() for k in ["IEF", "GLD", "SHY"]}
    spy, cash = r.SPY, DF.tbill_daily(idx)
    known_drop = (1 - gspc.reindex(swans.trough).values / gspc.reindex(swans.peak).values)
    swans = swans.assign(index_drop=known_drop)

    def sleeve_beta(d, key):
        if key == "SPY":
            return 1.0
        w = dev[months[bisect_right(months, d) - 1]][{"R1": "R1 tercile", "D0": "D0 blend"}[key]]
        window = list(swans.index[swans.confirmed < d][-5:])
        b = betas.reindex(w.index)[window].mean(axis=1)
        ok = b.notna()
        swan = (w[ok] * b[ok]).sum() / w[ok].sum()
        p = close.loc[:d, list(w.index) + ["^GSPC"]].iloc[-253:].pct_change().iloc[1:]
        m = p.pop("^GSPC")
        port = p.fillna(0) @ w
        return (swan + port.cov(m) / m.var()) / 2

    def hedge_split(hist):
        inv = 1 / hist[["ief", "gld"]].iloc[-252:].std()
        return inv / inv.sum()

    log = {}

    def sizing(key, how):
        def f(d, hist):
            known = swans[(swans.confirmed < d) & (swans.trough >= "1987-01-01")]
            D = known.index_drop.mean() if how == "mean" else known.index_drop.max()
            w = min(1.0, BUDGET / (sleeve_beta(d, key) * D))
            log.setdefault((key, how), {})[d] = w
            return pd.concat([pd.Series({"s": w}), (1 - w) * hedge_split(hist)])
        return f

    def fixed_share(x):
        return lambda d, hist: pd.concat([pd.Series({"s": x}), (1 - x) * hedge_split(hist)])

    def mix(stock, fn):
        a = pd.DataFrame({"s": stock, **etf})[["s", "ief", "gld", "shy"]]
        return A.run(a, fn, None, 0.001, drift=True)

    sw = swans[(swans.peak >= idx[0])]

    def stats(s, ctrl=None):
        st = DF.stats(s, spy, cash)
        for nm, part in [("2004-14", slice(None, "2014-12-31")), ("2015-26", slice("2015-01-01", None))]:
            st[f"Sharpe {nm}"] = DF.stats(s.loc[part], spy.loc[part], cash)["Sharpe"]
        ex = s.drop(s.loc["2020-02-19":"2020-06-30"].index)
        st["Sharpe ex-COVID"] = ex.mean() / ex.std() * np.sqrt(252)
        wealth = (1 + s).cumprod()
        st["swan losses"] = pd.Series({n: 1 - wealth.loc[w.trough] / wealth.loc[w.peak] for n, w in sw.iterrows()})
        return st

    # 1. crash-loss sizing
    rows = {}
    for key in ["R1", "SPY"]:
        for v, how in [("C1 mean drop", "mean"), ("C2 worst drop", "max")]:
            rows[f"{v}: {key}"] = stats(mix(r[key], sizing(key, how)))
            avg = pd.Series(log[(key, how)]).mean()
            rows[f"{v}: {key}"]["avg stock weight"] = avg
            rows[f"   control: {key} fixed at {avg:.0%}"] = stats(mix(r[key], fixed_share(round(avg, 2))))
    rows["risk parity R1 + IEF + GLD (current)"] = stats(mix(r.R1, A.risk_parity(["s", "ief", "gld"])))
    sizing_t = pd.DataFrame(rows).T

    # 2. switch
    oex = close["^OEX"].reindex(idx).ffill()
    rows, legs = {}, {"SPY": r.SPY, "D0": r.D0, "R1": r.R1}
    for v, e in ENTRY.items():
        on = selloff_state(oex, e).shift(1).fillna(0)
        legs[f"{v} switch at {e:.0%}"] = on * r.D0 + (1 - on) * r.SPY - on.diff().abs().fillna(0) * 2 * 0.003
        print(f"{v}: {int(on.diff().clip(lower=0).sum())} selloffs, {on.mean():.0%} of days in D0")
    for name, s in legs.items():
        rows[("stocks only", name)] = stats(s)
        rows[("60/20/20", name)] = stats(mix(s, A.fixed({"s": .6, "ief": .2, "gld": .2})))
        rows[("risk parity", name)] = stats(mix(s, A.risk_parity(["s", "ief", "gld"])))
    switch_t = pd.DataFrame(rows).T

    pd.set_option("display.width", 250)
    cols = ["CAGR", "vol", "Sharpe", "max DD", "worst year", "Sharpe 2004-14", "Sharpe 2015-26", "Sharpe ex-COVID"]
    for name, t in [("1. crash-loss sizing", sizing_t), ("2. switch SPY -> D0 in selloffs", switch_t)]:
        print(f"\n{name}")
        print(t[cols + (["avg stock weight"] if "avg stock weight" in t else [])].astype(float).round(3).to_string())
        losses = pd.DataFrame({k: v for k, v in t["swan losses"].items()}).T
        print(losses.round(3).to_string())
        t.drop(columns="swan losses").to_csv(OUT / f"swan_positioning_{name[0]}.csv")
    print("\nstock weight of C2 R1 over time (year ends):")
    w = pd.Series(log[("R1", "max")])
    print(w.groupby(w.index.year).last().round(2).to_string())
