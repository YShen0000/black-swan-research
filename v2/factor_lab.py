"""Swan beta as a factor, next to the usual price factors. S&P500 point-in-time, month starts 2004 -> today.

For every eligible member on the first trading day t of each month, using data before t only:
  swan      mean of the stock's betas in the last 5 confirmed swans
  blend     ½ swan + ½ 1-year market beta (the strategy's score)
  mkt_beta  1-year daily beta;  down_beta  beta on index-down days;  vol  1-year daily volatility
  mom12_1   return from 12 months to 1 month before t;  mom6_1  6 to 1 months;  rev1  the last month's return
  high52    last price / 52-week high;  trend  last price / 200-day average;  size  log 3-month dollar volume
Forward returns: t to the next month start (1m) and to 3 months later (3m).
Each factor: monthly rank IC (Spearman with the forward return), its mean and t-stat in 2005-15 and 2016-26, and
the long-only quintile spread (equal weight, top minus bottom quintile, 1m, annualised).

    python v2/factor_lab.py build      # panel, ~5 min, cached in output/factor_panel.pkl
    python v2/factor_lab.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import experiments as E
import risk_model as RM
import swan_beta

OUT = Path(__file__).parent / "output"
U = "SP500"
FACTORS = ["swan", "blend", "mkt_beta", "down_beta", "vol", "mom12_1", "mom6_1", "rev1", "high52", "trend", "size"]


def build():
    swans = E.setup(U)
    close = swan_beta.load_prices(U)
    dv = RM.dollar_volume(U)
    dates = sorted(pd.read_pickle(OUT / "develop_SP500_weights.pkl"))
    rows = []
    for i, t in enumerate(dates):
        x = E.context(t, U, swans)["x"]
        p = close.loc[:t - pd.Timedelta(days=1), x.index].iloc[-253:]
        f = pd.DataFrame({"swan": x.swan_mean, "blend": (x.swan_mean + x.mkt_beta) / 2, "mkt_beta": x.mkt_beta,
                          "down_beta": x.down_beta, "vol": x.vol,
                          "mom12_1": p.iloc[-22] / p.iloc[0] - 1, "mom6_1": p.iloc[-22] / p.iloc[-127] - 1,
                          "rev1": p.iloc[-1] / p.iloc[-22] - 1, "high52": p.iloc[-1] / p.max(),
                          "trend": p.iloc[-1] / p.iloc[-200:].mean(),
                          "size": np.log(dv.loc[t - pd.Timedelta(days=91):t - pd.Timedelta(days=1)]
                                         .reindex(columns=x.index).mean())})
        for k, h in [("fwd1", 1), ("fwd3", 3)]:
            end = dates[i + h] if i + h < len(dates) else None
            if end is not None:
                q = close.loc[t:end, x.index]
                f[k] = q.ffill().iloc[-1] / q.iloc[0] - 1   # bought at t's close, as the backtests do
        f["date"] = t
        rows.append(f.reset_index(names="ticker"))
        print(t.date(), len(f), flush=True)
    return pd.concat(rows, ignore_index=True)


def ic_table(panel, fwd="fwd1"):
    out = {}
    for f in FACTORS:
        ic = panel.dropna(subset=[f, fwd]).groupby("date").apply(lambda g: g[f].corr(g[fwd], method="spearman"))
        row = {}
        for nm, part in [("2005-15", slice("2005", "2015")), ("2016-26", slice("2016", "2026"))]:
            s = ic.loc[part]
            row[f"IC {nm}"] = s.mean()
            row[f"t {nm}"] = s.mean() / s.std() * np.sqrt(len(s))
        out[f] = row
    return pd.DataFrame(out).T


def quintiles(panel, fwd="fwd1"):
    out = {}
    for f in FACTORS:
        d = panel.dropna(subset=[f, fwd]).copy()
        d["q"] = d.groupby("date")[f].transform(lambda s: pd.qcut(s.rank(method="first"), 5, labels=False))
        m = d.groupby(["date", "q"])[fwd].mean().unstack()
        row = {}
        for nm, part in [("2005-15", slice("2005", "2015")), ("2016-26", slice("2016", "2026"))]:
            mm = m.loc[part]
            ann = (1 + mm).prod() ** (12 / len(mm)) - 1
            row[f"Q1 (low) {nm}"], row[f"Q5 (high) {nm}"] = ann[0], ann[4]
            row[f"Q5-Q1 {nm}"] = ann[4] - ann[0]
        out[f] = row
    return pd.DataFrame(out).T


if __name__ == "__main__":
    cache = OUT / "factor_panel.pkl"
    if sys.argv[1:] == ["build"]:
        pd.to_pickle(build(), cache)
        sys.exit()
    panel = pd.read_pickle(cache)
    pd.set_option("display.width", 250)
    print("rank IC with next-month return (positive = high values earn more)")
    print(ic_table(panel).round(3).to_string())
    print("\nrank IC with next-3-month return")
    print(ic_table(panel, "fwd3").round(3).to_string())
    print("\nequal-weight quintiles, next-month return, annualised")
    print(quintiles(panel).round(3).to_string())
    print("\nfactor correlations (mean monthly rank corr)")
    print(panel.groupby("date")[FACTORS].corr(method="spearman").groupby(level=1).mean().loc[FACTORS, FACTORS].round(2).to_string())
