"""MVO 2.0 baseline as plain functions, one per pipeline step, so each step can be swapped and upgraded.

    preselect -> inject predicted swan -> estimate return / risk -> optimize -> post-allocate -> evaluate

Same logic as training/MVO/MVO2_0_TRAINING(run this code).ipynb, with its bugs fixed (see v2/README.md).

    python v2/mvo2.py            # Swan9 baseline, writes v2/output/, checks the reference PR
"""
import datetime
import math
from pathlib import Path

import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier
from sklearn.cluster import KMeans

DATA = Path(__file__).resolve().parent.parent / "data"

SWAN9 = dict(   # predict Swan9 (COVID) from Swan4..Swan8, S&P100
    beta_file="SwanBeta/All_SwanBeta100_MVO2.xlsx", optdata_file="optdata/optdata_MVO2.xlsx", optdata_sheet=1,
    start_swan="Swan4", end_swan="Swan8",
    start=datetime.datetime(2019, 1, 1), end=datetime.datetime(2020, 2, 15),                       # look-back window
    start_swan_date=datetime.datetime(2020, 2, 15), end_swan_date=datetime.datetime(2021, 3, 23),  # evaluation window
    crash_peak="2020-02-19", crash_trough="2020-03-23",
    benchmark="^OEX", N=7, w1=0.17, min_weight=0.03, MA_days=30,
)


# ---------- data ----------

CLOSE = pd.read_csv(DATA / "prices/close_SP100.csv", index_col=0, parse_dates=True)            # split+dividend adjusted
NOMINAL = pd.read_csv(DATA / "prices/nominal_close_SP100.csv", index_col=0, parse_dates=True)  # price quoted that day


def get_close(tickers, start, end, prices=CLOSE):
    """Frozen daily prices, columns in the order asked, end exclusive (like yfinance)."""
    return prices.loc[start:end - datetime.timedelta(days=1)].reindex(columns=list(tickers))


def load_betas(beta_file, start_swan, end_swan):
    betas = pd.read_excel(DATA / beta_file, index_col=0)
    window = betas.loc[:, start_swan:end_swan]
    return betas[window.isna().sum(axis=1) < window.shape[1] - 1]   # keep stocks with at least 2 betas in the window


def swan_weights(betas, end_swan):
    """Weight of each past swan = its mean correlation with the later swans (upper triangle of the beta correlation)."""
    corr = betas.corr(numeric_only=True).loc["Swan1":end_swan, "Swan1":end_swan]
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool)))
    avg = upper.mean(axis=1)
    return avg / avg.sum()


# ---------- step 1: preselection ----------

def preselect_kmeans(betas, start_swan, end_swan, N, min_size=10, seed=1):
    """K-means on the swan betas; keep the cluster(s) with the lowest centre. Returns predicted beta per stock."""
    x = betas.loc[:, start_swan:end_swan].dropna()
    km = KMeans(n_clusters=N, random_state=seed, n_init=10).fit(x)   # n_init pinned: sklearn>=1.4 defaults to 1
    centre = km.cluster_centers_.mean(axis=1).clip(min=0)
    pred = pd.DataFrame({"group": km.labels_, "beta": centre[km.labels_]}, index=x.index).sort_values(["beta", "group"])
    first = pred[pred.group == pred.group.iloc[0]]
    if len(first) < min_size:   # too few: add the next-safest cluster
        first = pd.concat([first, pred[pred.group == pred.group.iloc[len(first)]]])
    return first.beta


# ---------- step 2: inject the predicted swan into the look-back prices ----------

def predicted_market_drop(total_days, end_swan, swan_w):
    """Linear fall over T1 days, log recovery over T2 days, then 0 (T1 correlation-weighted, T2 / DD plain means)."""
    m = pd.read_excel(DATA / "MVO_Components/MarketDD.xlsx", index_col=0).loc[:end_swan]
    t1 = int((m["T1"].values * swan_w.values).sum())
    t2 = int(round(m["T2"].mean(), 0))
    dd = m["Actual DD"].mean()
    t0 = total_days - t1 - t2
    assert t0 >= 0, f"look-back window ({total_days} days) shorter than predicted swan ({t1}+{t2} days)"
    base = (t2 + 1) ** (1 / (dd * 100))
    md = [round(dd / (t1 - 1) * i, 4) for i in range(t1)]
    md += [round(dd - math.log(i + 2, base) / 100, 4) for i in range(t2)]
    return np.array(md + [0] * t0)


def inject(prices, beta, market_drop):
    """Each stock falls beta * market drop; missing prices filled log-linearly."""
    shocked = prices * (1 - np.outer(market_drop, beta[prices.columns].values))
    return np.exp(np.log(shocked).interpolate(limit_direction="both"))


# ---------- step 3: expected return and risk ----------

def max_drawdown(returns):
    wealth = returns.add(1).cumprod()
    return -(wealth / wealth.cummax() - 1).min()


def estimate(prices, end, optdata, MA_days):
    """Per-stock forward return estimate, 1y max drawdown, weekly vol, and the annualised weekly covariance."""
    one_year_ago = prices.index[prices.index <= end - datetime.timedelta(days=365)][-1]
    half_year_ago = prices.index[prices.index <= end - datetime.timedelta(days=182)][-1]
    ma = prices.rolling(MA_days).mean().dropna().loc[one_year_ago:]
    ma_now, ma_1y, ma_half = ma.iloc[-1], ma.iloc[0], ma.loc[half_year_ago]

    # optdata P_fwd is a dollar target quoted at `end`; prices are adjusted back from today -> compare nominal prices
    to_nominal = NOMINAL.loc[:end].iloc[-1] / CLOSE.loc[:end].iloc[-1]
    r_fwd = optdata.P_fwd.loc[ma_now.index] / (ma_now * to_nominal[ma_now.index]) - 1
    r_60 = ma_now / ma_1y - 1
    r_60_half = ma_now / ma_half - 1
    # trust the analyst target only when it is roughly in line with the past year's trend
    r = pd.Series(np.select([r_fwd < 0.5 * r_60, r_fwd > 1.3 * r_60],
                            [r_60_half, np.maximum(1.3 * r_60, 0.7 * r_fwd)], r_fwd), index=r_fwd.index).clip(lower=0)

    week = pd.to_datetime(prices.index).strftime("%W").values
    week_end = np.where(week != np.roll(week, 1))[0][1:] - 1
    weekly = prices.iloc[week_end].pct_change().dropna()
    weekly = weekly[weekly.index >= one_year_ago]
    return pd.DataFrame({
        "Rfwd_adjusted": r,
        "Fall_MDD": prices.pct_change().loc[one_year_ago:].apply(max_drawdown),
        "std_f": np.log1p(weekly).std(ddof=0) * np.sqrt(52),
    }), weekly.cov() * 52


# ---------- step 4: optimisation ----------

def optimize(expected, cov, w1, min_weight, rf=0.01):
    ef = EfficientFrontier(expected, cov, weight_bounds=(0, w1))
    ef.max_sharpe(rf)
    w = pd.Series(ef.clean_weights())
    w = w[w >= min_weight]
    return w / w.sum()   # ponytail: renormalising can push a weight a bit over w1 (0.17 -> 0.175), as in the original


# ---------- step 5: post-allocation ----------

def post_allocation(weights, pr_sr):
    """Alternative weightings of the MVO picks, ranked by PR*sqrt(SR), using the allocation weight table."""
    alloc = pd.read_excel(DATA / "MVO_Components/allocation weight.xlsx", index_col="stock selected")
    order = pr_sr[weights.index].sort_values(ascending=False, kind="stable")
    n = len(order)
    weight1, number1, weight2, number2 = alloc.loc[n].values
    proportion = order / order.sum()
    adjusted, residual = [], 0.0
    for i, p in enumerate(proportion):
        a = min(weight1, p + residual / (n - i))
        adjusted.append(a)
        residual += p - a
    direct = [weight1] * int(number1) + [weight2] * int(number2)
    return pd.DataFrame({"weight": weights[order.index], "equal": 1 / n, "proportion": proportion,
                         "adjusted proportion": adjusted, "direct": direct + [0.0] * (n - len(direct))}, index=order.index)


# ---------- step 6: evaluation on the real swan ----------

def evaluate(weights, start_swan_date, end_swan_date, benchmark, crash_peak, crash_trough):
    """Value of 10,000 invested in `weights` vs the benchmark; PR = 1 - portfolio drop / benchmark drop."""
    px = get_close(weights.index, start_swan_date, end_swan_date)
    bench = get_close([benchmark], start_swan_date, end_swan_date)[benchmark]
    curve = pd.DataFrame({"portfolio_value": (px * (10000 * weights / px.iloc[0])).sum(axis=1),
                          "oex": 10000 * bench / bench.iloc[0]}, index=px.index)
    t = curve.loc[[crash_peak, crash_trough]]
    drop = (1 - t.iloc[1] / t.iloc[0]).abs()
    return curve, 1 - drop.portfolio_value / drop.oex


# ---------- pipeline ----------

def run(p=SWAN9, out=None):
    betas = load_betas(p["beta_file"], p["start_swan"], p["end_swan"])
    beta = preselect_kmeans(betas, p["start_swan"], p["end_swan"], p["N"])

    prices = get_close(beta.index, p["start"], p["end"]).dropna(how="all", axis=1)
    dropped = sorted(set(beta.index) - set(prices.columns))   # no price data (e.g. WBA, delisted from Yahoo)
    beta = beta[prices.columns]
    md = predicted_market_drop(len(prices), p["end_swan"], swan_weights(betas, p["end_swan"]))
    shocked = inject(prices, beta, md)

    optdata = pd.read_excel(DATA / p["optdata_file"], index_col=0, sheet_name=p["optdata_sheet"])
    stats, cov = estimate(shocked, p["end"], optdata, p["MA_days"])
    stats["beta"] = beta
    stats["PR"] = 1 - beta
    stats["SR"] = stats.Rfwd_adjusted / stats.std_f
    stats["PR_SR"] = stats.PR * stats.SR ** 0.5

    weights = optimize(stats.Rfwd_adjusted / stats.Fall_MDD / 100, cov, p["w1"], p["min_weight"])
    alloc = post_allocation(weights, stats.PR_SR)
    window = (p["start_swan_date"], p["end_swan_date"], p["benchmark"], p["crash_peak"], p["crash_trough"])
    curves = {c: evaluate(alloc[c], *window) for c in alloc.columns}
    res = dict(stats=stats, weights=weights, market_drop=md, allocation=alloc, dropped=dropped,
               curve=curves["weight"][0], PR=curves["weight"][1], PR_by_allocation={c: v[1] for c, v in curves.items()})
    if out:
        out = Path(out)
        out.mkdir(exist_ok=True)
        stats.to_csv(out / "preselected_Stocks_info.csv")
        weights.rename("weight").to_csv(out / "weights.csv")
        pd.DataFrame(md).to_csv(out / "market_drop.csv")
        alloc.to_csv(out / "post_allocation.csv")
        res["curve"].to_csv(out / "Injected_result.csv")
        get_close(weights.index, p["start_swan_date"], p["end_swan_date"]).to_csv(out / "tradeinfo.csv")
        for c, (curve, _) in curves.items():
            curve.to_csv(out / f"{c.replace(' ', '_')}_result.csv")
    return res


if __name__ == "__main__":
    r = run(out=Path(__file__).parent / "output")
    print(r["weights"].round(4).to_string())
    print("PR:", round(r["PR"], 4), {k: round(v, 4) for k, v in r["PR_by_allocation"].items()})
    assert abs(r["PR"] - 0.2225) < 1e-3, "Swan9 reference PR changed"
    assert abs(r["PR_by_allocation"]["weight"] - r["PR"]) < 1e-12
