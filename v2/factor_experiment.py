"""Factor risk model (risk_model.py) in the weighting step, compared with the recommended setup.

    python v2/factor_experiment.py SP500     # or SP100; ~1 h for SP500 the first time, cached after

Portfolios, rebuilt every month from data known then (candidates = stocks with swan betas, as in experiments.py):
  recommended         blend beta -> lowest n, 25% sector cap -> inverse vol            (components.recommended)
  rec + factor ERC    same stocks, equal risk contribution under the factor model, 25% sector cap
  rec + factor minvar same stocks, minimum variance under the factor model, <= 10% per stock, <= 25% per sector
  rec + sample minvar same, with the plain sample covariance instead (isolates the risk model's value)
  universe minvar     all candidates, minimum variance under the factor model, <= 5% per stock, <= 25% per sector
  universe minvar β   same, plus predicted (blend) swan beta of the portfolio <= 0.5 (or the lowest reachable)
Also checks the risk forecasts: predicted vs realised volatility over the next 3 months, factor model vs sample cov.
"""
import datetime
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier

import backtest
import components as C
import experiments as E
import risk_model as RM
import swan_beta

CAP25 = 0.25


def minvar(cov, cap, sector_cap=CAP25, beta=None, beta_max=None):
    sec = C.sectors()(cov.index)
    for constrained in (True, False):   # drop the sector cap if it makes the problem infeasible
        try:
            ef = EfficientFrontier(None, cov, weight_bounds=(0, max(cap, 1.0 / len(cov) + 1e-6)))
            if constrained:
                ef.add_sector_constraints(sec.to_dict(), {}, {s: sector_cap for s in sec.unique()})
            if beta is not None:
                ef.add_constraint(lambda w, b=beta[cov.index].values: b @ w <= beta_max)
            ef.min_volatility()
            w = pd.Series(ef.clean_weights())
            return w[w > 0] / w[w > 0].sum()
        except Exception:
            continue
    raise RuntimeError("min variance infeasible")


def monthly(universe, swans, start="2004-01-01"):
    close = swan_beta.load_prices(universe)
    dates = close.loc[start:].index.to_series().groupby(close.loc[start:].index.to_period("M")).first()
    n = E.N_SELECT[universe]
    weights, risk = {}, {}
    rng = np.random.default_rng(0)
    for t in dates:
        ctx = E.context(t, universe, swans)
        pred = C.forecast("blend", ctx["x"], universe, ctx["known"])
        cand = ctx["x"].index
        fcov = RM.factor_model(t, cand, universe)[0]
        scov = RM.sample_cov(t, cand, universe)
        sel = C.select_lowest(pred, n, CAP25).index
        w = {"recommended": C.cap_sectors(C.weigh("inv_vol", pred[sel], t, universe, ctx), CAP25),
             "rec + factor ERC": C.cap_sectors(RM.erc(fcov.loc[sel, sel]), CAP25),
             "rec + factor minvar": minvar(fcov.loc[sel, sel], 0.10),
             "rec + sample minvar": minvar(scov.loc[sel, sel], 0.10),
             "universe minvar": minvar(fcov, 0.05),
             # 0.5, or the lowest a 5% cap allows (mean of the 20 lowest) + 0.02 when 0.5 is out of reach
             "universe minvar β<=0.5": minvar(fcov, 0.05, beta=pred, beta_max=max(0.5, pred.nsmallest(20).mean() + 0.02)),
             "index": pd.Series({C.BENCH[universe]: 1.0}),
             C.TOTAL_RETURN[universe]: pd.Series({C.TOTAL_RETURN[universe]: 1.0})}
        weights[t] = w
        # risk forecasts for the portfolios above plus 10 random equal-weight portfolios of n candidates
        tests = {k: v for k, v in w.items() if k in ("recommended", "universe minvar", "rec + sample minvar")}
        tests["equal weight all"] = pd.Series(1 / len(cand), index=cand)
        for i in range(10):
            pick = rng.choice(cand, n, replace=False)
            tests[f"random {i}"] = pd.Series(1 / n, index=pick)
        risk[t] = {k: (np.sqrt(v @ fcov.loc[v.index, v.index] @ v), np.sqrt(v @ scov.loc[v.index, v.index] @ v), v)
                   for k, v in tests.items()}
        print(t.date(), "candidates", len(cand), "| universe minvar holds", len(w["universe minvar"]), flush=True)
    return weights, risk


def forecast_check(risk, close, horizon=63):
    rows = []
    for t, tests in risk.items():
        px = close.loc[t:].iloc[:horizon + 1]
        if len(px) < horizon + 1:
            continue
        for k, (f, s, w) in tests.items():
            r = px[w.index].pct_change().iloc[1:].fillna(0) @ w
            rows.append({"date": t, "portfolio": "random" if k.startswith("random") else k,
                         "factor": f, "sample": s, "realised": r.std() * np.sqrt(252)})
    d = pd.DataFrame(rows)
    out = {}
    for m in ["factor", "sample"]:
        ratio = d.realised / d[m]
        out[m] = {"realised / predicted, median": ratio.median(), "mean |log ratio|": np.log(ratio).abs().mean(),
                  "rank corr (all)": d[m].corr(d.realised, method="spearman")}
        for p, g in d.groupby("portfolio"):
            out[m][f"median ratio: {p}"] = (g.realised / g[m]).median()
    return pd.DataFrame(out)


def evaluate(weights, universe, swans, close):
    bench = C.BENCH[universe]
    dates = sorted(weights)
    rows = {}
    for label, sub, cost in [("monthly 0.1%", dates, 0.001)] + [
            (f"quarterly 0.3% start {p + 1}", dates[p::3], 0.003) for p in range(3)]:
        r, tv = backtest.daily_returns({t: weights[t] for t in sub}, close, cost)
        r = r.rename(columns={"index": bench}).loc[sub[0] + datetime.timedelta(days=1):]
        s = backtest.summary(r, bench, swans, tv.rename(columns={"index": bench}))
        pr = s[s.index.str.startswith("PR ")]
        s.loc["PR mean"], s.loc["PR worst"] = pr.mean(), pr.min()
        hold = {k: np.mean([len(weights[t][k]) for t in sub]) for k in s.columns if k in weights[sub[0]]}
        top = {k: np.mean([weights[t][k].groupby(C.sectors()(weights[t][k].index)).sum().max() for t in sub])
               for k in hold}
        s.loc["avg holdings"], s.loc["avg top sector"] = pd.Series(hold), pd.Series(top)
        for k in s.columns:
            rows[(k, label)] = s[k]
    return pd.DataFrame(rows).T


if __name__ == "__main__":
    universe = sys.argv[1] if len(sys.argv) > 1 else "SP500"
    swans = E.setup(universe)
    close = swan_beta.load_prices(universe)
    out = Path(__file__).parent / "output"
    out.mkdir(exist_ok=True)
    cache = out / f"factor_{universe}_weights.pkl"
    weights, risk = pd.read_pickle(cache) if cache.exists() else monthly(universe, swans)
    pd.to_pickle((weights, risk), cache)

    pd.set_option("display.width", 250)
    fc = forecast_check(risk, close)
    fc.to_csv(out / f"factor_{universe}_forecast_check.csv")
    print("risk forecast vs realised volatility over the next 3 months:")
    print(fc.round(3).to_string())

    table = evaluate(weights, universe, swans, close)
    table.to_csv(out / f"factor_{universe}.csv")
    cols = ["CAGR", "vol", "Sharpe(rf=0)", "max drawdown", "beta", "turnover/yr", "PR mean", "PR worst",
            "avg holdings", "avg top sector"]
    for label in table.index.get_level_values(1).unique():
        print(f"\n{label}")
        print(table.xs(label, level=1)[cols].round(3).to_string())
