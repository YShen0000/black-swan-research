"""Selection-level additions to the defensive (D0) and return (R1) tracks, fixed in advance:

  M  momentum filter: drop the candidates in the bottom third of 12-month return (skipping the last month)
     before picking the lowest blend scores (avoids buying falling knives)
  S  stock trend filter: of the picked stocks, those below their own 200-day average are held as T-bills instead
     (decided at each rebalance)

    python v2/defense_selection.py      # S&P500 point-in-time, ~35 min, cached after
"""
import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import backtest
import components as C
import defense as DF
import develop as D
import experiments as E
import swan_beta

OUT = Path(__file__).parent / "output"
CASH = "__CASH__"


def momentum(t, tickers):
    px = swan_beta.load_prices("SP500").loc[:t - datetime.timedelta(days=1), list(tickers)].iloc[-253:]
    return px.iloc[-22] / px.iloc[0] - 1


def below_trend(t, tickers):
    px = swan_beta.load_prices("SP500").loc[:t - datetime.timedelta(days=1), list(tickers)].iloc[-200:]
    return px.iloc[-1] < px.mean()


def monthly(swans, start="2004-01-01"):
    close = swan_beta.load_prices("SP500")
    dates = close.loc[start:].index.to_series().groupby(close.loc[start:].index.to_period("M")).first()
    out = {}
    for t in dates:
        ctx = E.context(t, "SP500", swans)
        score = D.scores(t, ctx, "SP500")["D0 blend"]
        mom = momentum(t, score.index)
        keep = mom >= mom.quantile(1 / 3)
        m = len(score)
        w = {}
        for name, s, n in [("D0", score, 30), ("R1", score, m // 3)]:
            w[f"{name} + M"] = D.hold(s[keep], t, "SP500", ctx, n if name == "D0" else keep.sum() // 3)
            base = D.hold(s, t, "SP500", ctx, n)
            out_ = below_trend(t, base.index)
            w[f"{name} + S"] = pd.concat([base[~out_], pd.Series({CASH: base[out_].sum()})]) if out_.any() else base
        out[t] = w
        print(t.date(), flush=True)
    return out


if __name__ == "__main__":
    swans = E.setup("SP500")
    cache = OUT / "defense_selection_weights.pkl"
    weights = pd.read_pickle(cache) if cache.exists() else monthly(swans)
    pd.to_pickle(weights, cache)
    close = swan_beta.load_prices("SP500").copy()
    cash = DF.tbill_daily(close.index)
    close[CASH] = (1 + cash).cumprod()               # T-bills as a price series
    dates = sorted(weights)
    r, _ = backtest.daily_returns({t: weights[t] for t in dates[::3]}, close, 0.003)
    r = r.loc[r.index[1]:]
    base = DF.sleeves()
    r = r.join(base[["defensive D0", "return R1", "SPY"]], how="inner")
    c = DF.tbill_daily(r.index)
    t = pd.DataFrame({k: DF.stats(r[k], r.SPY, c) for k in r.columns}).T
    for name, part in [("2004-14", slice(None, "2014-12-31")), ("2015-26", slice("2015-01-01", None))]:
        t[f"Sharpe {name}"] = pd.Series({k: DF.stats(r[k].loc[part], r.SPY.loc[part], c)["Sharpe"] for k in r.columns})
    t.to_csv(OUT / "defense_selection.csv")
    pd.set_option("display.width", 250)
    print(t[["CAGR", "vol", "Sharpe", "Sharpe 2004-14", "Sharpe 2015-26", "max DD", "worst year", "beta"]
            + list(DF.CRISES)].round(3).to_string())
