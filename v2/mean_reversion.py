"""Short-term mean reversion on daily closes, the closest thing to intraday "做T" without intraday data.
Textbook rule (Connors RSI(2)), fixed in advance, long only:
  enter  close above its 200-day average and 2-day RSI below 10
  exit   close above its 5-day average
Signals at the close of day d; traded at the close of d+1 (conservative: no same-close fill). 0.1% cost per side.
Universes: SPY alone; the nine sector SPDRs, equal slots (1/9 each), each slot in T-bills when flat.
Time in the market, number of trades, average trade and the usual stats; plus as an overlay on idle cash.

    python v2/mean_reversion.py
"""
import numpy as np
import pandas as pd

import defense as DF
import strategy_search as S
import swan_beta

SECTORS = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]
COST = 0.001


def rsi(p, n=2):
    d = p.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def position(p, lag=1):
    """1 while long, 0 while flat, decided at each close; shifted by `lag` days for the fill."""
    sma200, sma5, r = p.rolling(200).mean(), p.rolling(5).mean(), rsi(p)
    pos, on = [], False
    for c, a, b, x in zip(p, sma200, sma5, r):
        if not on and c > a and x < 10:
            on = True
        elif on and c > b:
            on = False
        pos.append(float(on))
    return pd.Series(pos, index=p.index).shift(lag + 1).fillna(0)   # +1: a close decision earns from the next day


def strategy(px, cash, lag=1):
    """Daily returns: the asset's return while in position (entered at the close lag days after the signal)."""
    pos = position(px, lag)
    r = px.pct_change().fillna(0)
    out = pos * r + (1 - pos) * cash - pos.diff().abs().fillna(0) * COST
    trades = int((pos.diff() > 0).sum())
    rets = []
    entry = None
    for d, v in pos.diff().fillna(0).items():
        if v > 0:
            entry = px.shift(1)[d]   # bought at the previous close (the fill day's close before this return)
        elif v < 0 and entry:
            rets.append(px.shift(1)[d] / entry - 1)
    return out, pos, trades, (np.mean(rets) if rets else np.nan)


if __name__ == "__main__":
    etf = DF.etf()
    close = swan_beta.load_prices("SP500")
    spy = close["SPY"].reindex(etf.index)
    cash = DF.tbill_daily(etf.index)
    start = S.START
    rows = {}
    for lag, nm in [(1, "next close"), (0, "same close (optimistic)")]:
        s, pos, n, avg = strategy(spy, cash, lag)
        st = S.evaluate(s.loc[start:], spy.pct_change().loc[start:], cash.loc[start:])
        st["time in market"], st["trades/yr"], st["avg trade"] = pos.loc[start:].mean(), n / 22, avg
        rows[f"SPY RSI2, fill {nm}"] = st
        legs = {k: strategy(etf[k], cash, lag) for k in SECTORS}
        s = pd.concat([v[0] for v in legs.values()], axis=1).mean(axis=1)
        st = S.evaluate(s.loc[start:], spy.pct_change().loc[start:], cash.loc[start:])
        st["time in market"] = pd.concat([v[1] for v in legs.values()], axis=1).mean(axis=1).loc[start:].mean()
        st["trades/yr"] = sum(v[2] for v in legs.values()) / 22
        st["avg trade"] = np.nanmean([v[3] for v in legs.values()])
        rows[f"9 sectors RSI2, fill {nm}"] = st
    rows["SPY buy and hold"] = S.evaluate(spy.pct_change().loc[start:], spy.pct_change().loc[start:], cash.loc[start:])
    t = pd.DataFrame(rows).T
    pd.set_option("display.width", 250)
    cols = ["CAGR", "vol", "Sharpe", "max DD", "worst year", "CAGR 2005-14", "CAGR 2015-26", "Sharpe 2005-14",
            "Sharpe 2015-26", "time in market", "trades/yr", "avg trade"]
    print(t[cols].astype(float).round(4).to_string())
