"""Stop-loss rules on top of D1 70/15/15 (live). Fixed in advance; long only, no leverage.

  L15 / L25  trailing stop per stock: a stock in the R1 sleeve that closes 15% / 25% below its highest close since
             the month's rebuild is sold at the next close (0.3% cost) and the money sits in T-bills until the next
             monthly rebuild, when the stock is bought back if it is still selected
  B10 / B15  portfolio brake: when D1 closes 10% / 15% below its own high, the core's stocks (35% of the money) go to
             T-bills from the next day; they come back when D1 is within 5% / 7.5% of its high again
             (approximation: the brake swaps the 35% R1 share for T-bills inside the tranched D1 return)
Compared with D1 as is: CAGR, max DD, sub-periods, without COVID, crises, how often the stops fire.
A rule is adopted only if it lowers D1's max drawdown and does not lower its Sharpe in 2005-14, 2015-26 or without
COVID.

    python v2/stop_loss.py
"""
import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import strategy_robustness as SR
import strategy_search as S
import swan_beta

OUT = Path(__file__).parent / "output"
STOCK_SHARE = 0.35


def stopped_sleeve(weights, close, cash, stop, cost=0.003):
    """Daily returns of monthly-rebuilt sleeves where each stock has a trailing stop inside the month."""
    dates = sorted(weights)
    rets, fired, drifted = [], 0, pd.Series(dtype=float)
    for t, nxt in zip(dates, dates[1:] + [close.index[-1] + datetime.timedelta(days=1)]):
        w = weights[t]
        px = close.loc[t:nxt, w.index].ffill()
        rel = (px / px.iloc[0]).values
        hit = rel / np.maximum.accumulate(rel, axis=0) <= 1 - stop
        c = cash.reindex(px.index).fillna(0).values
        val = rel.copy()
        for j in range(rel.shape[1]):
            k = np.argmax(hit[:, j])
            if hit[k, j] and k + 1 < len(rel):
                s = k + 1                                   # sold at the next close
                fired += 1
                growth = np.cumprod(np.r_[1.0, 1 + c[s + 1:]])
                val[s:, j] = rel[s, j] * (1 - cost) * growth
        v = pd.Series(val @ w.values, index=px.index)
        r = v.pct_change().dropna()
        tv = w.sub(drifted, fill_value=0).abs().sum()
        r.iloc[0] = (1 + r.iloc[0]) * (1 - cost * tv) - 1
        rets.append(r)
        last = pd.Series(val[-1] * w.values, index=w.index)
        held = last[~hit.any(axis=0)] if hit.any() else last
        drifted = held / last.sum()                         # stopped-out names are cash at the month end
    return pd.concat(rets), fired


def brake(d1, r1, cash, start, back):
    """D1 with the core's stocks swapped for T-bills while D1 is `start` below its high, until within `back`."""
    nav, peak, on, out, switches = 1.0, 1.0, False, [], 0
    for d, x in d1.items():
        y = x - STOCK_SHARE * (r1[d] - cash[d]) if on else x
        out.append(y)
        nav *= 1 + y
        peak = max(peak, nav)
        dd = 1 - nav / peak
        new = dd >= start if not on else dd > back
        if new != on:
            out[-1] -= STOCK_SHARE * 0.003
            switches += 1
        on = new
    return pd.Series(out, index=d1.index), switches


if __name__ == "__main__":
    r = S.load()
    close = swan_beta.load_prices("SP500")
    w = {t: v["R1"] for t, v in pd.read_pickle(OUT / "search_sleeves_weights.pkl").items()}
    fired = {}
    for s in (0.15, 0.25):
        k = f"L{int(s * 100)}"
        x, fired[k] = stopped_sleeve(w, close, r.cash, s)
        r[k] = x.reindex(r.index)
    keys = ["R1", "L15", "L25"]
    rot = S.rotation(SR.ROT)
    fns = {k: (lambda c: lambda d, h: c(d, h).mul(0.5).add(rot(d, h).mul(0.5), fill_value=0))(
        S.fixed({k: .7, "IEF": .15, "GLD": .15})) for k in keys}
    copies = [S.run_all(r, fns, stock=keys)] + [S.run_all(r, fns, stock=keys, every=10 ** 6, dates=SR.mid_month(r.index, o))
                                                for o in (5, 10, 15)]
    d1 = {k: pd.concat([c[k] for c in copies], axis=1).mean(axis=1).loc[S.START:] for k in keys}
    spy, cash = r.SPY.loc[S.START:], r.cash.loc[S.START:]
    runs = {"D1 (live)": d1["R1"], "D1, stock stop 15%": d1["L15"], "D1, stock stop 25%": d1["L25"]}
    sw = {}
    for start, back in [(0.10, 0.05), (0.15, 0.075)]:
        k = f"D1, brake at {start:.0%}"
        runs[k], sw[k] = brake(d1["R1"], r.R1.loc[S.START:].fillna(0), cash, start, back)
    rows = {k: S.evaluate(s, spy, cash) for k, s in runs.items()}
    for k in ["R1", "L15", "L25"]:
        rows[f"sleeve {k} alone"] = S.evaluate(r[k].loc[S.START:], spy, cash)
    t = pd.DataFrame(rows).T
    t.to_csv(OUT / "stop_loss.csv")
    pd.set_option("display.width", 250)
    cols = ["CAGR", "Sharpe", "max DD", "worst year", "CAGR 2005-14", "CAGR 2015-26", "Sharpe 2005-14", "Sharpe 2015-26",
            "Sharpe ex-COVID"] + list(S.DF.CRISES)
    print(t[cols].astype(float).round(3).to_string())
    yrs = len(spy) / 252
    print("\nstock stops fired per year:", {k: round(v / 22, 1) for k, v in fired.items()},
          "| brake switches per year:", {k: round(v / yrs, 1) for k, v in sw.items()})
    b = t.loc["D1 (live)"]
    print("adopt:", {k: bool(t.loc[k, "max DD"] < b["max DD"] and all(t.loc[k, c] >= b[c] for c in
                     ["Sharpe 2005-14", "Sharpe 2015-26", "Sharpe ex-COVID"])) for k in runs if k != "D1 (live)"})
