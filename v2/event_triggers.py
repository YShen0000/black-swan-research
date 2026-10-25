"""Event triggers vs the calendar for the chosen allocation (risk parity of the R1 sleeve, IEF, GLD). Fixed in advance:

  base  sleeve rebuilt quarterly, split reset monthly (live.py until 2026-10)
  S1    sleeve rebuilt monthly
  S2    base + sleeve rebuilt the day after each new swan is confirmed (data/swans.csv)
  S3    S1 + the same swan trigger
  P1    base + split reset as soon as any weight is 5 points off its target
  P2    base + split reset on the first close with ^OEX 10%, 15% and 20% below its 1-year high
        (once per level per episode; an episode ends when ^OEX is back within 5% of its high)
Weights drift with returns between resets (as in a real account), 0.3% stock / 0.1% ETF cost, 2004 -> today.

    python v2/event_triggers.py     # needs the develop cache; event-date weights are cached on the first run
"""
from pathlib import Path

import pandas as pd

import allocation as A
import backtest
import defense as DF
import develop as D
import experiments as E
import swan_beta

OUT = Path(__file__).parent / "output"
U = "SP500"
LEVELS, EPISODE_END = (0.10, 0.15, 0.20), 0.05


def swan_dates(swans, close):
    """First trading day after each swan's confirmation (context() only uses swans confirmed before t)."""
    days = close.index
    return [days[days > c][0] for c in swans.confirmed if pd.Timestamp("2004-01-01") < c < days[-1]]


def event_weights(dates, swans):
    cache = OUT / "event_triggers_weights.pkl"
    out = pd.read_pickle(cache) if cache.exists() else {}
    for t in dates:
        if t not in out:
            ctx = E.context(t, U, swans)
            sc = D.scores(t, ctx, U)["D0 blend"]
            out[t] = D.hold(sc, t, U, ctx, len(sc) // 3)
    pd.to_pickle(out, cache)
    return out


def drawdown_dates(oex):
    dd = 1 - oex / oex.rolling(252, min_periods=1).max()
    out, hit = [], set()
    for d, x in dd.items():
        if x < EPISODE_END:
            hit = set()
        for lv in LEVELS:
            if x >= lv and lv not in hit:
                hit.add(lv)
                out.append(d)
    return out


if __name__ == "__main__":
    swans = E.setup(U)
    close = swan_beta.load_prices(U)
    dev = {t: w["R1 tercile"] for t, w in pd.read_pickle(OUT / "develop_SP500_weights.pkl").items()}
    months = sorted(dev)
    sd = swan_dates(swans, close)
    ev = event_weights(sd, swans)

    def sleeve(ws):
        r, tv = backtest.daily_returns({t: {"s": w} for t, w in sorted(ws.items())}, close, 0.003)
        return r["s"], tv.sum() / (len(r) / 252)

    quarterly = {t: dev[t] for t in months[::3]}
    sleeves = {"quarterly": sleeve(quarterly), "monthly": sleeve(dev),
               "quarterly + swan": sleeve({**quarterly, **ev}), "monthly + swan": sleeve({**dev, **ev})}
    idx = sleeves["quarterly"][0].index[1:]
    px = DF.etf().reindex(idx)
    etf = {k.lower(): px[k].pct_change() for k in ["IEF", "GLD", "SHY"]}
    spy = close["SPY"].reindex(idx).pct_change()
    cash = DF.tbill_daily(idx)
    dd_days = [d for d in drawdown_dates(close["^OEX"].dropna()) if d >= idx[0]]
    rp = A.risk_parity(["s", "ief", "gld"])

    def go(sl, **kw):
        s, tv = sleeves[sl]
        a = pd.DataFrame({"s": s.reindex(idx), **etf})[["s", "ief", "gld", "shy"]]
        r = A.run(a, rp, None, 0.001, drift=True, **kw)
        st = DF.stats(r, spy, cash)
        for nm, part in [("2004-14", slice(None, "2014-12-31")), ("2015-26", slice("2015-01-01", None))]:
            p = DF.stats(r.loc[part], spy.loc[part], cash)
            st[f"CAGR {nm}"], st[f"Sharpe {nm}"], st[f"max DD {nm}"] = p["CAGR"], p["Sharpe"], p["max DD"]
        ex = r.drop(r.loc["2020-02-19":"2020-06-30"].index)
        st["Sharpe ex-COVID"] = ex.mean() / ex.std() * 252 ** 0.5
        st["sleeve turnover/yr"] = tv
        return st

    rows = {"base": go("quarterly"), "S1 sleeve monthly": go("monthly"), "S2 base + swan trigger": go("quarterly + swan"),
            "S3 S1 + swan trigger": go("monthly + swan"), "P1 base + 5-point drift band": go("quarterly", band=0.05),
            "P2 base + drawdown trigger": go("quarterly", dates=dd_days)}
    t = pd.DataFrame(rows).T
    t.to_csv(OUT / "event_triggers.csv")
    pd.set_option("display.width", 250)
    print("swan trigger dates:", [d.date() for d in sd])
    print("drawdown trigger dates:", [d.date() for d in dd_days])
    cols = ["CAGR", "Sharpe", "max DD", "worst year", "CAGR 2004-14", "CAGR 2015-26", "Sharpe 2004-14", "Sharpe 2015-26",
            "max DD 2004-14", "max DD 2015-26", "Sharpe ex-COVID", "sleeve turnover/yr"]
    print(t[cols].astype(float).round(3).to_string())
    print()
    print(t[list(DF.CRISES)].astype(float).round(3).to_string())
