"""Select only stocks in an up-trend, so a stopped-out stock is not simply bought back. Fixed in advance; the sleeve
stays fully invested in stocks (no cash switch), S&P500 point-in-time, rebuilt monthly, inverse vol, <= 25% per sector.

  R1  lowest third of all candidates by blend                                             (current)
  T1  lowest third by blend among candidates whose last close is above their 200-day average
  T2  T1 + 15% trailing stop per stock (cash until the next rebuild; a stopped stock comes back only if the T1 filter
      lets it)
  T3  R1 + 15% trailing stop, and a stock stopped out this month is left out of next month's sleeve (the rest of the
      sleeve is scaled up)
  T4  lowest third by blend among candidates with a positive 6-1 month return (another trend definition)
Each alone and inside D1 70/15/15 (rotation in 4 tranches). Adopted only if D1 with it has a higher Sharpe in 2005-14,
2015-26 and without COVID, and a max drawdown no higher than now.

    python v2/trend_selection.py build    # ~5 min, cached
    python v2/trend_selection.py
"""
import datetime
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import develop as D
import experiments as E
import stop_loss as SL
import strategy_robustness as SR
import strategy_search as S
import swan_beta

OUT = Path(__file__).parent / "output"
U = "SP500"
STOP = 0.15


def build():
    swans = E.setup(U)
    close = swan_beta.load_prices(U)
    out = {}
    for t in sorted(pd.read_pickle(OUT / "develop_SP500_weights.pkl")):
        ctx = E.context(t, U, swans)
        x = ctx["x"]
        blend = (x.swan_mean + x.mkt_beta) / 2
        p = close.loc[:t - datetime.timedelta(days=1), x.index].iloc[-253:]
        up200 = blend[p.iloc[-1] > p.iloc[-200:].mean()]
        up_mom = blend[(p.iloc[-22] / p.iloc[-127] - 1) > 0]
        out[t] = {"R1": D.hold(blend, t, U, ctx, len(blend) // 3),
                  "T1": D.hold(up200, t, U, ctx, max(1, len(up200) // 3)),
                  "T4": D.hold(up_mom, t, U, ctx, max(1, len(up_mom) // 3))}
        print(t.date(), len(up200), len(up_mom), flush=True)
    return out


def no_rebuy(weights, close, cash, stop=STOP, cost=0.003):
    """R1 with a trailing stop; names stopped in a month are dropped from the next month's weights."""
    dates = sorted(weights)
    rets, fired, drifted, banned = [], 0, pd.Series(dtype=float), set()
    for t, nxt in zip(dates, dates[1:] + [close.index[-1] + datetime.timedelta(days=1)]):
        w = weights[t].drop(list(banned), errors="ignore")
        w = w / w.sum()
        px = close.loc[t:nxt, w.index].ffill()
        rel = (px / px.iloc[0]).values
        hit = rel / np.maximum.accumulate(rel, axis=0) <= 1 - stop
        c = cash.reindex(px.index).fillna(0).values
        val, banned = rel.copy(), set()
        for j in range(rel.shape[1]):
            k = np.argmax(hit[:, j])
            if hit[k, j] and k + 1 < len(rel):
                s = k + 1
                fired += 1
                banned.add(w.index[j])
                val[s:, j] = rel[s, j] * (1 - cost) * np.cumprod(np.r_[1.0, 1 + c[s + 1:]])
        v = pd.Series(val @ w.values, index=px.index)
        r = v.pct_change().dropna()
        tv = w.sub(drifted, fill_value=0).abs().sum()
        r.iloc[0] = (1 + r.iloc[0]) * (1 - cost * tv) - 1
        rets.append(r)
        last = pd.Series(val[-1] * w.values, index=w.index)
        drifted = last.drop(list(banned)) / last.sum()
    return pd.concat(rets), fired


if __name__ == "__main__":
    cache = OUT / "trend_selection_weights.pkl"
    if sys.argv[1:] == ["build"]:
        pd.to_pickle(build(), cache)
        sys.exit()
    w = pd.read_pickle(cache)
    close = swan_beta.load_prices(U)
    r = S.load()
    from backtest import daily_returns
    sl, _ = daily_returns(w, close, 0.003)
    for k in ["R1", "T1", "T4"]:
        r[k] = sl[k].reindex(r.index)
    fired = {}
    x, fired["T2"] = SL.stopped_sleeve({t: v["T1"] for t, v in w.items()}, close, r.cash, STOP)
    r["T2"] = x.reindex(r.index)
    x, fired["T3"] = no_rebuy({t: v["R1"] for t, v in w.items()}, close, r.cash)
    r["T3"] = x.reindex(r.index)
    keys = ["R1", "T1", "T2", "T3", "T4"]
    rot = S.rotation(SR.ROT)
    fns = {k: (lambda c: lambda d, h: c(d, h).mul(0.5).add(rot(d, h).mul(0.5), fill_value=0))(
        S.fixed({k: .7, "IEF": .15, "GLD": .15})) for k in keys}
    copies = [S.run_all(r, fns, stock=keys)] + [S.run_all(r, fns, stock=keys, every=10 ** 6, dates=SR.mid_month(r.index, o))
                                                for o in (5, 10, 15)]
    d1 = {k: pd.concat([c[k] for c in copies], axis=1).mean(axis=1) for k in keys}
    spy, cash = r.SPY.loc[S.START:], r.cash.loc[S.START:]
    rows = {}
    for k in keys:
        rows[(k, "sleeve")] = S.evaluate(r[k].loc[S.START:], spy, cash)
        rows[(k, "D1 70/15/15")] = S.evaluate(d1[k].loc[S.START:], spy, cash)
    t = pd.DataFrame(rows).T
    t.to_csv(OUT / "trend_selection.csv")
    pd.set_option("display.width", 250)
    cols = ["CAGR", "Sharpe", "max DD", "worst year", "CAGR 2005-14", "CAGR 2015-26", "Sharpe 2005-14", "Sharpe 2015-26",
            "Sharpe ex-COVID"] + list(S.DF.CRISES)
    print(t[cols].astype(float).round(3).to_string())
    print("\nstops fired per year:", {k: round(v / 22, 1) for k, v in fired.items()})
    n = pd.DataFrame({d: {k: len(v[k]) for k in ["R1", "T1", "T4"]} for d, v in w.items()}).T
    print("holdings min / median:", n.min().to_dict(), n.median().to_dict())
    b = t.loc[("R1", "D1 70/15/15")]
    print("adopt:", {k: bool(all(t.loc[(k, "D1 70/15/15"), c] > b[c] for c in ["Sharpe 2005-14", "Sharpe 2015-26",
                     "Sharpe ex-COVID"]) and t.loc[(k, "D1 70/15/15"), "max DD"] <= b["max DD"]) for k in keys if k != "R1"})
