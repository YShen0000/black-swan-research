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
from pypfopt.exceptions import OptimizationError
from sklearn.cluster import KMeans

import swan_beta

DATA = Path(__file__).resolve().parent.parent / "data"

SWAN9 = dict(   # predict Swan9 (COVID) from Swan4..Swan8, S&P100, on the original input files (reference PR 0.2225)
    beta_file="SwanBeta/All_SwanBeta100_MVO2.xlsx", marketdd_file="MVO_Components/MarketDD.xlsx",
    optdata_file="optdata/optdata_MVO2.xlsx", optdata_sheet=1,
    start_swan="Swan4", end_swan="Swan8",
    start=datetime.datetime(2019, 1, 1), end=datetime.datetime(2020, 2, 15),                       # look-back window
    start_swan_date=datetime.datetime(2020, 2, 15), end_swan_date=datetime.datetime(2021, 3, 23),  # evaluation window
    crash_peak="2020-02-19", crash_trough="2020-03-23",
    benchmark="^OEX", N=7, w1=0.17, min_weight=0.03, MA_days=30,
)


# ---------- data ----------

CLOSE = swan_beta.load_prices("SP100")   # split+dividend adjusted; load_prices(universe, "nominal") = price quoted that day


def get_close(tickers, start, end, prices=CLOSE):
    """Frozen daily prices, columns in the order asked, end exclusive (like yfinance)."""
    return prices.loc[start:end - datetime.timedelta(days=1)].reindex(columns=list(tickers))


def load_betas(beta_file, start_swan, end_swan, betas=None):
    """Swan Beta table: given, an original Excel file, or (None) computed from data/swans.csv + prices."""
    if betas is None:
        betas = swan_beta.swan_betas() if beta_file is None else pd.read_excel(DATA / beta_file, index_col=0)
    window = betas.loc[:, start_swan:end_swan]
    return betas[window.isna().sum(axis=1) < window.shape[1] - 1]   # keep stocks with at least 2 betas in the window


def load_marketdd(marketdd_file):
    """Per-swan DD / T1 / T2: the original MarketDD.xlsx, or (None) computed from data/swans.csv + prices."""
    if marketdd_file is None:
        return swan_beta.market_stats()
    return pd.read_excel(DATA / marketdd_file, index_col=0).rename(columns={"Actual DD": "DD"})


def load_optdata(optdata_file, sheet=0):
    """Analyst targets (P_fwd) per ticker, or None to run without them."""
    if optdata_file is None:
        return None
    if optdata_file.endswith(".csv"):
        return pd.read_csv(DATA / optdata_file, index_col=0)
    return pd.read_excel(DATA / optdata_file, index_col=0, sheet_name=sheet)


def swan_weights(betas, history):
    """Weight of each past swan in `history` = its mean correlation with the later ones (upper triangle of the beta correlation)."""
    corr = betas[history].corr()
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

def predicted_market_drop(total_days, marketdd, swan_w):
    """Linear fall over T1 days, log recovery over T2 days, then 0 (T1 correlation-weighted, T2 / DD plain means).

    `marketdd` holds the past swans (Swan1..end_swan). If T1 + T2 is longer than the look-back window, the recovery
    is compressed to end on the window's last day: same depth, and the latest prices (used for returns) stay clean.
    """
    m = marketdd.loc[swan_w.index]
    t1 = int((m["T1"] * swan_w).sum())
    t2 = min(int(round(m["T2"].mean(), 0)), total_days - t1)
    dd = m["DD"].mean()
    t0 = total_days - t1 - t2
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


def estimate(prices, end, optdata, MA_days, universe="SP100"):
    """Per-stock forward return estimate, 1y max drawdown, weekly vol, and the annualised weekly covariance.

    With optdata: analyst target bounded by the past year's trend. Without (optdata None): the trend alone.
    """
    one_year_ago = prices.index[prices.index <= end - datetime.timedelta(days=365)][-1]
    half_year_ago = prices.index[prices.index <= end - datetime.timedelta(days=182)][-1]
    ma = prices.rolling(MA_days).mean().dropna().loc[one_year_ago:]
    ma_now, ma_1y, ma_half = ma.iloc[-1], ma.iloc[0], ma.loc[half_year_ago]

    r_60 = ma_now / ma_1y - 1
    r_60_half = ma_now / ma_half - 1
    if optdata is None:
        r = r_60.clip(lower=0)
    else:
        # P_fwd is a dollar target quoted at `end`; prices are adjusted back from today -> compare nominal prices
        to_nominal = (swan_beta.load_prices(universe, "nominal").loc[:end].iloc[-1]
                      / swan_beta.load_prices(universe).loc[:end].iloc[-1])
        r_fwd = pd.to_numeric(optdata.P_fwd, errors="coerce").loc[ma_now.index] / (ma_now * to_nominal[ma_now.index]) - 1
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

def optimize(expected, cov, w1, min_weight, rf=0.01, method="max_sharpe"):
    """Max Sharpe; if it has no solution (too few stocks beat rf under the w1 cap, e.g. before 2008), min volatility.
    method="min_volatility" skips the return estimate altogether.
    The w1 cap is raised to 1/n when there are too few stocks for it to be feasible (e.g. 5 stocks at 17%)."""
    w1 = max(w1, 1 / len(expected) + 1e-6)
    try:
        if method != "max_sharpe":
            raise ValueError
        ef = EfficientFrontier(expected, cov, weight_bounds=(0, w1))
        ef.max_sharpe(rf)
    except (ValueError, OptimizationError):
        ef = EfficientFrontier(expected, cov, weight_bounds=(0, w1))
        ef.min_volatility()
        method = "min_volatility"
    w = pd.Series(ef.clean_weights())
    w = w[w >= min_weight]
    return w / w.sum(), method   # ponytail: renormalising can push a weight a bit over w1 (0.17 -> 0.175), as in the original


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

def evaluate(weights, start_swan_date, end_swan_date, benchmark, crash_peak, crash_trough, prices=CLOSE):
    """Value of 10,000 invested in `weights` vs the benchmark; PR = 1 - portfolio drop / benchmark drop."""
    px = get_close(weights.index, start_swan_date, end_swan_date, prices)
    bench = get_close([benchmark], start_swan_date, end_swan_date, prices)[benchmark]
    curve = pd.DataFrame({"portfolio_value": (px * (10000 * weights / px.iloc[0])).sum(axis=1),
                          "oex": 10000 * bench / bench.iloc[0]}, index=px.index)
    t = curve.loc[[crash_peak, crash_trough]]
    drop = (1 - t.iloc[1] / t.iloc[0]).abs()
    return curve, 1 - drop.portfolio_value / drop.oex


# ---------- pipeline ----------

def build_portfolio(p):
    """Steps 1-4: preselect, inject, estimate, optimize, using only data up to p["end"].

    p: dict like SWAN9, plus optional keys
      prices   universe of frozen prices, "SP100" (default) or "SP500"
      betas / marketdd   precomputed tables (else from beta_file / marketdd_file; None -> computed from data/swans.csv)
      history  swans used for the correlation weights and the predicted drop (default Swan1..end_swan)
      tickers  stocks allowed into preselection
      returns  "pfwd" (analyst targets, needs optdata_file), "trend" (past-year trend) or "minvol" (no return estimate)
      inject   False skips the crash injection
    """
    universe = p.get("prices", "SP100")
    close = swan_beta.load_prices(universe)
    betas = load_betas(p.get("beta_file"), p["start_swan"], p["end_swan"], p.get("betas"))
    if p.get("tickers") is not None:
        betas = betas[betas.index.isin(p["tickers"])]
    beta = preselect_kmeans(betas, p["start_swan"], p["end_swan"], p["N"])

    prices = get_close(beta.index, p["start"], p["end"], close)
    prices = prices.dropna(axis=1, thresh=int(0.9 * len(prices)))   # need ~a full look-back (no IPOs / delisted)
    dropped = sorted(set(beta.index) - set(prices.columns))
    beta = beta[prices.columns]
    history = p.get("history") or list(betas.loc[:, "Swan1":p["end_swan"]].columns)
    marketdd = p["marketdd"] if p.get("marketdd") is not None else load_marketdd(p.get("marketdd_file"))
    md = predicted_market_drop(len(prices), marketdd, swan_weights(betas, history))
    shocked = inject(prices, beta, md if p.get("inject", True) else 0 * md)

    returns = p.get("returns") or ("pfwd" if p.get("optdata_file") else "trend")
    optdata = load_optdata(p["optdata_file"], p.get("optdata_sheet", 0)) if returns == "pfwd" else None
    stats, cov = estimate(shocked, p["end"], optdata, p["MA_days"], universe)
    stats["beta"] = beta
    stats["PR"] = 1 - beta
    stats["SR"] = stats.Rfwd_adjusted / stats.std_f
    stats["PR_SR"] = stats.PR * stats.SR ** 0.5

    weights, method = optimize(stats.Rfwd_adjusted / stats.Fall_MDD / 100, cov, p["w1"], p["min_weight"],
                               method="min_volatility" if returns == "minvol" else "max_sharpe")
    return dict(stats=stats, weights=weights, optimizer=method, market_drop=md, dropped=dropped, beta=beta)


def run(p=SWAN9, out=None):
    """build_portfolio, then post-allocation and evaluation on the swan in p (crash_peak -> crash_trough)."""
    built = build_portfolio(p)
    stats, weights, method, md, dropped, beta = (built[k] for k in ["stats", "weights", "optimizer", "market_drop",
                                                                     "dropped", "beta"])
    alloc = post_allocation(weights, stats.PR_SR)
    window = (p["start_swan_date"], p["end_swan_date"], p["benchmark"], p["crash_peak"], p["crash_trough"],
              swan_beta.load_prices(p.get("prices", "SP100")))
    curves = {c: evaluate(alloc[c], *window) for c in alloc.columns}
    res = dict(stats=stats, weights=weights, optimizer=method, market_drop=md, allocation=alloc, dropped=dropped,
               curve=curves["weight"][0], PR=curves["weight"][1], PR_by_allocation={c: v[1] for c, v in curves.items()},
               PR_preselect_equal=evaluate(pd.Series(1 / len(beta), index=beta.index), *window)[1])
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
