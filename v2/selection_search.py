"""Stock selection for return as well as protection. Fixed in advance (S&P500 point-in-time, rebuilt on the first
trading day of each month; blend = ½ mean of the last 5 swan betas + ½ 1-year market beta):

  R1  lowest third by blend, inverse vol, <= 25% per sector                          (current)
  N1  sector-neutral swan: in each sector the lowest third by blend (>= 1 stock), inverse vol inside the sector;
      sector weights = the sector's share of the candidates' 3-month average dollar volume (the cap proxy: there
      is no shares-outstanding history), so sectors are held roughly as in the index
  N0  control for N1: all candidates, the same sector weights, inverse vol inside -- sector-neutral without selection
  N2  asymmetric beta: lowest third by (swan mean - 1-year market beta), i.e. stocks that fell less in past swans
      than their everyday beta implies; inverse vol, <= 25% per sector
Compared stocks-only and inside D1 (core stocks 70 / IEF 15 / GLD 15, half rotation in 4 tranches), with SPY.
A sleeve replaces R1 in D1 only if D1 with it has a higher Sharpe in 2005-14, in 2015-26 and without COVID, a max
drawdown <= 25%, and the sleeve alone lost less than SPY in at least 5 of the 7 swans since 2005.

    python v2/selection_search.py build     # ~5 min, cached in output/selection_search_weights.pkl
    python v2/selection_search.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import backtest
import components as C
import develop as D
import experiments as E
import risk_model as RM
import strategy_robustness as SR
import strategy_search as S
import swan_beta

OUT = Path(__file__).parent / "output"
U = "SP500"
SLEEVES = ["R1", "N0", "N1", "N2"]
CORE = 0.7


def sector_neutral(pick, t, ctx, sector_weight):
    """Inverse vol inside each sector, sectors at `sector_weight`."""
    sec = C.sectors()(pick)
    out = []
    for s, names in pd.Series(pick, index=pick).groupby(sec.values):
        w = C.weigh("inv_vol", pd.Series(0.0, index=names.index), t, U, ctx)
        out.append(w * sector_weight[s])
    w = pd.concat(out)
    return w / w.sum()


def build():
    swans = E.setup(U)
    dv = RM.dollar_volume(U)
    out = {}
    for t in sorted(pd.read_pickle(OUT / "develop_SP500_weights.pkl")):
        ctx = E.context(t, U, swans)
        x = ctx["x"]
        blend = (x.swan_mean + x.mkt_beta) / 2
        sec = C.sectors()(blend.index)
        size = dv.loc[t - pd.Timedelta(days=91):t - pd.Timedelta(days=1)].reindex(columns=blend.index).mean().fillna(0)
        sw = size.groupby(sec.values).sum()
        sw = sw / sw.sum()
        third = blend.groupby(sec.values).apply(lambda b: b.nsmallest(max(1, len(b) // 3))).index.get_level_values(-1)
        out[t] = {"R1": D.hold(blend, t, U, ctx, len(blend) // 3),
                  "N0": sector_neutral(blend.index, t, ctx, sw),
                  "N1": sector_neutral(third, t, ctx, sw),
                  "N2": D.hold(x.swan_mean - x.mkt_beta, t, U, ctx, len(blend) // 3)}
        print(t.date(), {k: len(v) for k, v in out[t].items()}, flush=True)
    return out


def tranched(r, fns):
    copies = [S.run_all(r, fns, stock=SLEEVES)] + [
        S.run_all(r, fns, stock=SLEEVES, every=10 ** 6, dates=SR.mid_month(r.index, o)) for o in (5, 10, 15)]
    return {k: pd.concat([c[k] for c in copies], axis=1).mean(axis=1) for k in fns}


def alpha_rows(x, spy, cash):
    out = {}
    for nm, (a, b) in {"2005-14": ("2005", "2014"), "2015-26": ("2015", "2026")}.items():
        s, m, c = x.loc[a:b], spy.loc[a:b], cash.loc[a:b]
        yrs = len(s) / 252
        beta = s.cov(m) / m.var()
        cagr = lambda v: (1 + v).prod() ** (1 / yrs) - 1
        out[f"beta {nm}"] = beta
        out[f"excess over beta-matched {nm}"] = cagr(s) - cagr(beta * m + (1 - beta) * c)
    return out


if __name__ == "__main__":
    cache = OUT / "selection_search_weights.pkl"
    if sys.argv[1:] == ["build"]:
        pd.to_pickle(build(), cache)
        sys.exit()
    weights = pd.read_pickle(cache)
    close = swan_beta.load_prices(U)
    sl, tv = backtest.daily_returns(weights, close, 0.003)
    r = S.load()
    for k in SLEEVES:
        r[k] = sl[k].reindex(r.index)
    swans = E.setup(U)
    sw = swans[swans.peak >= S.START]
    spy, cash = r.SPY.loc[S.START:], r.cash.loc[S.START:]

    rows, losses = {}, {}
    for k in SLEEVES + ["SPY"]:
        x = r[k].loc[S.START:]
        rows[(k, "stocks only")] = {**S.evaluate(x, spy, cash), **alpha_rows(x, spy, cash)}
        w = (1 + x).cumprod()
        losses[k] = pd.Series({n: 1 - w.loc[v.trough] / w.loc[v.peak] for n, v in sw.iterrows()})
    losses = pd.DataFrame(losses).T
    beats = (losses.drop("SPY") < losses.loc["SPY"]).sum(axis=1)

    rot = S.rotation(SR.ROT)
    fns = {(k, "D1 70/15/15"): (lambda c: lambda d, h: c(d, h).mul(0.5).add(rot(d, h).mul(0.5), fill_value=0))(
        S.fixed({k: CORE, "IEF": (1 - CORE) / 2, "GLD": (1 - CORE) / 2})) for k in SLEEVES}
    for key, s in tranched(r, fns).items():
        rows[key] = S.evaluate(s.loc[S.START:], spy, cash)
    t = pd.DataFrame(rows).T
    base = t.loc[("R1", "D1 70/15/15")]
    d1 = t.xs("D1 70/15/15", level=1)
    adopt = {k: bool((d1.loc[k, ["Sharpe 2005-14", "Sharpe 2015-26", "Sharpe ex-COVID"]] >
                      base[["Sharpe 2005-14", "Sharpe 2015-26", "Sharpe ex-COVID"]]).all()
                     and d1.loc[k, "max DD"] <= 0.25 and beats.get(k, 0) >= 5) for k in SLEEVES if k != "R1"}
    t.to_csv(OUT / "selection_search.csv")
    pd.set_option("display.width", 260)
    cols = ["CAGR", "vol", "Sharpe", "max DD", "worst year", "CAGR 2005-14", "CAGR 2015-26", "Sharpe 2005-14",
            "Sharpe 2015-26", "Sharpe ex-COVID"]
    print(t[cols].astype(float).round(3).to_string())
    print("\nstocks only: beta and excess return over SPY + T-bills at the same beta")
    print(t.xs("stocks only", level=1)[[c for c in t if c.startswith(("beta", "excess"))]].astype(float).round(3).to_string())
    print("\ncrises:"); print(t[list(S.DF.CRISES)].astype(float).round(3).to_string())
    print("\nloss in each swan (stocks only):"); print(losses.round(3).to_string())
    print("swans with a smaller loss than SPY:", beats.to_dict())
    secs = {k: pd.DataFrame({d: weights[d][k].groupby(C.sectors()(weights[d][k].index)).sum()
                             for d in weights if d >= pd.Timestamp("2015-01-01")}).T.mean() for k in SLEEVES}
    print("\naverage sector weights 2015-26:"); print(pd.DataFrame(secs).round(3).to_string())
    print("\nholdings (median):", {k: int(np.median([len(weights[d][k]) for d in weights])) for k in SLEEVES},
          "| turnover/yr:", (tv.sum() / (len(sl) / 252)).round(2).to_dict())
    print("adopt in D1 (rule above):", adopt)
