"""Does the strategy make money, where do the drawdowns come from, and what lowers them?

    python v2/defense.py            # S&P500 point-in-time; needs the develop / experiments weight caches

1. Original (K-means + MVO), defensive track (D0) and return track (R1) vs buy-and-hold SPY and the low-vol ETFs,
   same data and rules: drawdown depth and dates, worst year, crisis windows, return split into beta and alpha.
2. Layers added on top, all fixed in advance (standard textbook settings, nothing tuned):
   sizing      T  trend filter: hold the sleeve while SPY is above its 10-month average at month end, else T-bills
               V  volatility target: exposure = min(1, 10% / sleeve's 3-month realised vol), rest in T-bills, monthly
   allocation  B  60% sleeve / 40% 7-10y Treasuries (IEF), monthly
               P  risk parity: inverse 1-year vol across sleeve, IEF and gold (GLD, from 2004-11), monthly
               PT risk parity with the trend filter on the sleeve
   Each layer is also applied to SPY itself: if SPY + layer does as well, the layer, not the stock selection, is
   doing the work. 0.1% cost per unit of exposure change.
Selection-level additions are in defense_selection.py.
"""
import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import backtest
import experiments as E
import swan_beta

OUT = Path(__file__).parent / "output"
CRISES = {"GFC 2007-10→2009-03": ("2007-10-09", "2009-03-09"), "COVID 2020-02→03": ("2020-02-19", "2020-03-23"),
          "2022 bear": ("2022-01-03", "2022-10-12"), "2025 tariffs": ("2025-02-19", "2025-04-08")}


def etf():
    return pd.read_csv(swan_beta.DATA / "prices/etf_close.csv.gz", index_col=0, parse_dates=True)


def tbill_daily(index):
    y = pd.read_csv(swan_beta.DATA / "prices/tbill_13w.csv", index_col=0, parse_dates=True).yield_pct
    return (y.reindex(index).ffill() / 100 / 252).fillna(0)


def sleeves():
    """Daily returns 2004 -> today: strategies (quarterly, 0.3% cost, January start) and benchmarks."""
    close = swan_beta.load_prices("SP500")
    dev = pd.read_pickle(OUT / "develop_SP500_weights.pkl")
    exp = pd.read_pickle(OUT / "experiments_SP500_weights.pkl")
    dates = sorted(dev)
    w = {t: {"original (K-means+MVO)": exp[t]["kmeans | mvo (original)"], "defensive D0": dev[t]["D0 blend"],
             "return R1": dev[t]["R1 tercile"]} for t in dates[::3]}
    r, _ = backtest.daily_returns(w, close, 0.003)
    m, _ = backtest.daily_returns({t: {"original monthly": exp[t]["kmeans | mvo (original)"]} for t in dates}, close, 0.001)
    r["original monthly (as the team ran it)"] = m.iloc[:, 0]
    px = etf().reindex(r.index)
    r["SPY"] = close["SPY"].reindex(r.index).pct_change()
    for k in ["IEF", "GLD", "USMV", "SPLV"]:
        r[k] = px[k].pct_change()
    return r.loc[r.index[1]:]


def stats(s, spy, cash):
    s = s.dropna()
    w = (1 + s).cumprod()
    dd = 1 - w / w.cummax()
    trough = dd.idxmax()
    peak = w[:trough].idxmax()
    yrs = len(s) / 252
    cagr = w.iloc[-1] ** (1 / yrs) - 1
    b = s.cov(spy.loc[s.index]) / spy.loc[s.index].var()
    rf = (1 + cash.loc[s.index]).prod() ** (1 / yrs) - 1
    spy_cagr = (1 + spy.loc[s.index]).prod() ** (1 / yrs) - 1
    yearly = (1 + s).groupby(s.index.year).prod() - 1
    out = {"CAGR": cagr, "vol": s.std() * np.sqrt(252), "Sharpe": s.mean() / s.std() * np.sqrt(252),
           "max DD": dd.max(), "MDD peak": peak.date(), "MDD trough": trough.date(), "worst year": yearly.min(),
           "beta": b, "alpha (CAGR − rf − β·(SPY−rf))": cagr - rf - b * (spy_cagr - rf)}
    for name, (a, z) in CRISES.items():
        if s.index[0] <= pd.Timestamp(a):
            out[name] = w.loc[:z].iloc[-1] / w.loc[:a].iloc[-1] - 1
    return pd.Series(out)


def layer(kind, s, spy_tr, cash, px):
    """Apply a sizing / allocation layer to daily sleeve returns s. Decisions at month end, applied next month."""
    month_end = s.groupby(s.index.to_period("M")).tail(1).index
    if kind == "T":
        sma = spy_tr.rolling(210).mean()
        on = (spy_tr > sma).reindex(month_end).astype(float)
        expo = on.reindex(s.index).shift(1).ffill().fillna(1)
        return expo * s + (1 - expo) * cash - expo.diff().abs().fillna(0) * 0.001
    if kind == "V":
        vol = s.rolling(63).std() * np.sqrt(252)
        expo = (0.10 / vol).clip(upper=1).reindex(month_end).reindex(s.index).shift(1).ffill().fillna(1)
        return expo * s + (1 - expo) * cash - expo.diff().abs().fillna(0) * 0.001
    ief, gld = px["IEF"].pct_change().reindex(s.index), px["GLD"].pct_change().reindex(s.index)
    if kind == "B":
        assets = pd.DataFrame({"s": s, "ief": ief})
        w = pd.DataFrame({"s": 0.6, "ief": 0.4}, index=s.index)
    else:
        sleeve = layer("T", s, spy_tr, cash, px) if kind == "PT" else s
        assets = pd.DataFrame({"s": sleeve, "ief": ief, "gld": gld})
        inv = 1 / (assets.rolling(252, min_periods=126).std())
        target = inv.div(inv.sum(axis=1), axis=0).reindex(month_end)
        w = target.reindex(s.index).shift(1).ffill().fillna({"s": 0.6, "ief": 0.4, "gld": 0.0}).fillna(0)
        w = w.div(w.sum(axis=1), axis=0)
    out = (w * assets.fillna(0)).sum(axis=1)
    return out - w.diff().abs().sum(axis=1).fillna(0) * 0.001


if __name__ == "__main__":
    pd.set_option("display.width", 260, "display.max_columns", 30)
    swans = E.setup("SP500")
    r = sleeves()
    close = swan_beta.load_prices("SP500")
    spy_tr = close["SPY"].reindex(r.index)
    cash = tbill_daily(r.index)
    px = etf().reindex(r.index)

    base = pd.DataFrame({k: stats(r[k], r.SPY, cash) for k in r.columns}).T
    base.to_csv(OUT / "defense_baseline.csv")
    print("1. same data and rules, 2004 -> today (USMV / SPLV from 2011: compare with SPY over the same years below)")
    print(base.round(3).to_string())
    since = r.loc["2011-11-01":]
    print("\n   since 2011-11 (all with data)")
    print(pd.DataFrame({k: stats(since[k], since.SPY, cash) for k in ["original (K-means+MVO)", "defensive D0",
                                                                        "return R1", "SPY", "USMV", "SPLV"]}).T
          [["CAGR", "vol", "Sharpe", "max DD", "beta", "COVID 2020-02→03", "2022 bear"]].round(3).to_string())

    rows = {}
    for sleeve in ["SPY", "original (K-means+MVO)", "defensive D0", "return R1"]:
        rows[(sleeve, "none")] = stats(r[sleeve], r.SPY, cash)
        for kind in ["T", "V", "B", "P", "PT"]:
            rows[(sleeve, kind)] = stats(layer(kind, r[sleeve], spy_tr, cash, px), r.SPY, cash)
    lay = pd.DataFrame(rows).T
    lay.to_csv(OUT / "defense_layers.csv")
    cols = ["CAGR", "vol", "Sharpe", "max DD", "worst year", "beta"] + list(CRISES)
    print("\n2. layers (T trend, V vol target 10%, B 60/40 with IEF, P risk parity with IEF+GLD, PT P + trend)")
    print(lay[cols].round(3).to_string())
    for name, part in [("2004-14", slice(None, "2014-12-31")), ("2015-26", slice("2015-01-01", None))]:
        sub = {k: v for k, v in rows.items()}
        sp = pd.DataFrame({k: stats((r[k[0]] if k[1] == "none" else layer(k[1], r[k[0]], spy_tr, cash, px)).loc[part],
                                    r.SPY.loc[part], cash) for k in sub}).T
        print(f"\n   {name}")
        print(sp[["CAGR", "Sharpe", "max DD"]].unstack().round(3).to_string())
