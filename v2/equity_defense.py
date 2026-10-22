"""Lower drawdowns while staying a fully invested, long-only stock portfolio. Variants fixed in advance:

  E1 stress min-var     minimum variance over the 60 lowest blend scores, with the covariance estimated only on
                        the 20% worst market days of the past 3 years (Ledoit-Wolf shrunk); <= 10% per stock,
                        <= 25% per sector. Diversify for the correlations that hold in a crash.
  E2 sector risk parity the D0 stocks; each sector gets equal risk (1 / sector volatility), inverse vol inside it.
                        Crashes hit different sectors, so spread the risk across them.
  E3 stress-beta cap    minimum variance (factor model) over all candidates with the portfolio's stress beta
                        (beta on the 10% worst market days of the past year) <= 0.4; <= 5% per stock, <= 25% per sector.
  E4 regime switch      hold R1 (lowest third) while SPY is above its 10-month average, D0 (lowest 30) below it.
                        Always 100% in stocks.
Baselines: D0, R1, SPY, and the original K-means + MVO.

    python v2/equity_defense.py      # S&P500 point-in-time, ~30 min, cached after
"""
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

import backtest
import components as C
import defense as DF
import develop as D
import experiments as E
import risk_model as RM
import swan_beta
from factor_experiment import minvar

OUT = Path(__file__).parent / "output"
U = "SP500"


def stress_cov(t, tickers, years=3, q=0.20):
    close = swan_beta.load_prices(U).loc[:t - datetime.timedelta(days=1)]
    r = close[list(tickers) + ["^GSPC"]].iloc[-252 * years - 1:].pct_change().iloc[1:]
    b = r.pop("^GSPC")
    bad = r[b <= b.quantile(q)].dropna(axis=1, thresh=20).fillna(0)
    cov = LedoitWolf().fit(bad.values).covariance_ * 252
    return pd.DataFrame(cov, index=bad.columns, columns=bad.columns)


def sector_risk_parity(w_stocks, t, ctx):
    tickers = w_stocks.index
    cov = C.weekly_cov(t, tickers, U)
    sec = C.sectors()(tickers)
    out = pd.Series(0.0, index=tickers)
    for s in sec.unique():
        idx = tickers[sec == s]
        inner = 1 / np.sqrt(np.diag(cov.loc[idx, idx]))
        inner = inner / inner.sum()
        out[idx] = inner / np.sqrt(inner @ cov.loc[idx, idx].values @ inner)
    return out / out.sum()


def monthly(swans, start="2004-01-01"):
    close = swan_beta.load_prices(U)
    dates = close.loc[start:].index.to_series().groupby(close.loc[start:].index.to_period("M")).first()
    spy = close["SPY"]
    out = {}
    for t in dates:
        ctx = E.context(t, U, swans)
        score = D.scores(t, ctx, U)["D0 blend"].dropna()
        d0 = D.hold(score, t, U, ctx, 30)
        r1 = D.hold(score, t, U, ctx, len(score) // 3)
        w = {"D0": d0, "R1": r1}
        top = score.nsmallest(60).index
        scov = stress_cov(t, top)
        w["E1 stress min-var"] = minvar(scov, 0.10)
        w["E2 sector risk parity"] = sector_risk_parity(d0, t, ctx)
        fcov = RM.factor_model(t, score.index, U)[0]
        sb = D.stress_beta(t, score.index, U)
        try:
            w["E3 stress-beta cap"] = minvar(fcov, 0.05, beta=sb, beta_max=0.4)
        except RuntimeError:   # 0.4 out of reach: lowest a 5% cap allows (mean of the 20 lowest) + 0.02
            w["E3 stress-beta cap"] = minvar(fcov, 0.05, beta=sb, beta_max=sb.nsmallest(20).mean() + 0.02)
        p = spy.loc[:t - datetime.timedelta(days=1)]
        w["E4 regime switch"] = r1 if p.iloc[-1] > p.iloc[-210:].mean() else d0
        w["SPY"] = pd.Series({"SPY": 1.0})
        out[t] = w
        print(t.date(), flush=True)
    return out


if __name__ == "__main__":
    swans = E.setup(U)
    cache = OUT / "equity_defense_weights.pkl"
    weights = pd.read_pickle(cache) if cache.exists() else monthly(swans)
    pd.to_pickle(weights, cache)
    exp = pd.read_pickle(OUT / "experiments_SP500_weights.pkl")
    for t in weights:
        weights[t]["original (K-means+MVO)"] = exp[t]["kmeans | mvo (original)"]

    close = swan_beta.load_prices(U)
    dates = sorted(weights)
    phases = []
    for p in range(3):   # quarterly, 0.3% cost, each start month
        r, tv = backtest.daily_returns({t: weights[t] for t in dates[p::3]}, close, 0.003)
        r = r.loc[dates[p] + datetime.timedelta(days=1):]
        cash = DF.tbill_daily(r.index)
        s = pd.DataFrame({k: DF.stats(r[k], r.SPY, cash) for k in r.columns}).T
        num = s.drop(columns=["MDD peak", "MDD trough"]).astype(float)
        for name, part in [("2004-14", slice(None, "2014-12-31")), ("2015-26", slice("2015-01-01", None))]:
            num[f"Sharpe {name}"] = [DF.stats(r[k].loc[part], r.SPY.loc[part], cash)["Sharpe"] for k in r.columns]
            num[f"max DD {name}"] = [DF.stats(r[k].loc[part], r.SPY.loc[part], cash)["max DD"] for k in r.columns]
        num["turnover/yr"] = tv.sum() / (len(r) / 252)
        bw, sw = (1 + r).cumprod(), None
        prs = {}
        for sn, w in swans[swans.peak >= r.index[0]].iterrows():
            drop = 1 - bw.loc[w.trough] / bw.loc[w.peak]
            prs[sn] = 1 - drop / drop["SPY"]
        prs = pd.DataFrame(prs)
        num["PR mean"], num["PR worst"] = prs.mean(axis=1), prs.min(axis=1)
        phases.append(num)
    t = sum(phases) / 3
    t["max DD range over start months"] = [f"{min(p.loc[k, 'max DD'] for p in phases):.0%}–{max(p.loc[k, 'max DD'] for p in phases):.0%}" for k in t.index]
    t.to_csv(OUT / "equity_defense.csv")
    pd.set_option("display.width", 260)
    cols = ["CAGR", "vol", "Sharpe", "Sharpe 2004-14", "Sharpe 2015-26", "max DD", "max DD range over start months",
            "max DD 2004-14", "max DD 2015-26", "worst year", "beta", "PR mean", "PR worst", "turnover/yr"]
    print("S&P500 point-in-time, quarterly, 0.3% cost, mean of 3 start months")
    print(t[cols].round(3).to_string())
    print()
    print(t[list(DF.CRISES)].round(3).to_string())
