"""Monthly-rebalance backtest: on the first trading day of each month, rebuild the portfolio from data known then
(swans whose trough is confirmed, prices up to the day before), hold it for the month, repeat.

    python v2/backtest.py                    # S&P100, hand swan list
    python v2/backtest.py SP500 rule         # universe SP100|SP500, swan list hand|rule

Portfolios compared:
    MVO              the strategy: K-means preselection, crash injection, trend return / drawdown, max Sharpe
    MVO no injection same without the crash injection
    Min-vol          same preselection, minimum-volatility weights (no return estimate)
    Preselect EW     equal weight on the preselected stocks
    Universe EW      equal weight on every stock with prices (same survivor universe, no swan logic)
    Index            ^OEX for S&P100, ^GSPC for S&P500
    Switch           Index normally; MVO while the index is >= 10% below its 1-year high at the rebalance

Caveats: the universe is today's constituent list (survivorship bias); the cost is 10 bp per unit of turnover.
"""
import datetime
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import mvo2
import swan_beta

LOOKBACK = datetime.timedelta(days=410)
COST = 0.001
VARIANTS = {"MVO": {}, "MVO no injection": {"inject": False}, "Min-vol": {"returns": "minvol"}}


def swan_list(universe, which):
    close = swan_beta.load_prices(universe)
    swans = (swan_beta.SWANS if which == "hand" else swan_beta.detect_swans(close)).copy()
    if universe != "SP100":
        swans["benchmark"] = "^GSPC"
    swans["confirmed"] = swan_beta.confirm_dates(swans, close)
    return swans


def monthly_weights(universe="SP100", which="hand", start="2004-01-01", N=None):
    """Weights per rebalance date for every portfolio."""
    close = swan_beta.load_prices(universe)
    bench = "^OEX" if universe == "SP100" else "^GSPC"
    swans = swan_list(universe, which)
    betas = swan_beta.swan_betas(close, swans)
    stocks = close.columns.difference(swan_beta.INDICES)
    dd = swan_beta.drawdown(close[bench].dropna())
    dates = close.loc[start:].index.to_series().groupby(close.loc[start:].index.to_period("M")).first()
    out = {}
    for t in dates:
        known = swans[swans.confirmed < t]
        history = list(known.index[known.trough >= "1987-01-01"])   # modern swans, as Swan1.. in the original
        p = dict(prices=universe, betas=betas[known.index], history=history,
                 marketdd=swan_beta.market_stats(close.loc[:t - datetime.timedelta(days=1)], known.loc[history]),
                 start_swan=known.index[-5], end_swan=known.index[-1], start=t - LOOKBACK, end=t,
                 N=N or (7 if universe == "SP100" else 17), w1=0.17, min_weight=0.03, MA_days=30, optdata_file=None)
        w = {}
        for name, extra in VARIANTS.items():
            built = mvo2.build_portfolio(dict(p, **extra))
            w[name] = built["weights"]
        pre = built["beta"].index
        w["Preselect EW"] = pd.Series(1 / len(pre), index=pre)
        alive = mvo2.get_close(stocks, t - LOOKBACK, t, close).dropna(axis=1, thresh=200).columns
        w["Universe EW"] = pd.Series(1 / len(alive), index=alive)
        w["Index"] = pd.Series({bench: 1.0})
        w["Switch"] = w["MVO"] if dd[:t - datetime.timedelta(days=1)].iloc[-1] >= 0.10 else w["Index"]
        out[t] = w
        print(t.date(), "swans", known.index[-5], "..", known.index[-1], "| MVO", len(w["MVO"]), "stocks", flush=True)
    return out


def daily_returns(weights, close):
    """Buy at the rebalance close, let weights drift, pay COST per unit of turnover at each rebalance."""
    dates = sorted(weights)
    names = weights[dates[0]].keys()
    rets, turnover = {n: [] for n in names}, {n: [] for n in names}
    drifted = {n: pd.Series(dtype=float) for n in names}
    for t, nxt in zip(dates, dates[1:] + [close.index[-1] + datetime.timedelta(days=1)]):
        for n in names:
            w = weights[t][n]
            px = close.loc[t:nxt - datetime.timedelta(days=1), w.index].ffill()
            value = (px / px.iloc[0]) @ w
            r = value.pct_change().dropna()
            tv = w.sub(drifted[n], fill_value=0).abs().sum()
            r = pd.concat([pd.Series({t: -COST * tv}), r])
            rets[n].append(r)
            turnover[n].append(tv)
            last = w * px.iloc[-1] / px.iloc[0]
            drifted[n] = last / last.sum()
    return pd.DataFrame({n: pd.concat(v) for n, v in rets.items()}), pd.DataFrame(turnover, index=dates)


def summary(r, bench, swans, turnover):
    wealth = (1 + r).cumprod()
    years = (r.index[-1] - r.index[0]).days / 365.25
    stats = pd.DataFrame({
        "CAGR": wealth.iloc[-1] ** (1 / years) - 1,
        "vol": r.std() * np.sqrt(252),
        "Sharpe(rf=0)": r.mean() / r.std() * np.sqrt(252),
        "max drawdown": (1 - wealth / wealth.cummax()).max(),
        "beta": r.cov()[bench] / r[bench].var(),
        "turnover/yr": turnover.sum() / years,
    })
    for s, w in swans[swans.peak >= r.index[0]].iterrows():
        drop = 1 - wealth.loc[w.trough] / wealth.loc[w.peak]
        stats[f"PR {s}"] = 1 - drop / drop[bench]
    return stats.T


if __name__ == "__main__":
    universe = sys.argv[1] if len(sys.argv) > 1 else "SP100"
    which = sys.argv[2] if len(sys.argv) > 2 else "hand"
    close = swan_beta.load_prices(universe)
    weights = monthly_weights(universe, which)
    r, turnover = daily_returns(weights, close)
    bench = "^OEX" if universe == "SP100" else "^GSPC"
    r = r.rename(columns={"Index": bench})
    turnover = turnover.rename(columns={"Index": bench})
    table = summary(r, bench, swan_list(universe, which), turnover)
    out = Path(__file__).parent / "output"
    out.mkdir(exist_ok=True)
    r.to_csv(out / f"backtest_{universe}_{which}_daily.csv")
    table.to_csv(out / f"backtest_{universe}_{which}.csv")
    pd.set_option("display.width", 250)
    print(f"\n{universe}, {which} swans, {r.index[0].date()} -> {r.index[-1].date()}, monthly rebalance, {COST:.1%} cost")
    print(table.round(3).to_string())
