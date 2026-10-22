"""Swappable strategy components, so combinations can be compared in experiments.py.

    python v2/components.py SP500 2026-10-01 [defensive|return]   # recommended holdings -> v2/output/holdings_*.csv

    universe    which stocks can be held on date t (point-in-time S&P500 membership, or today's S&P100 list)
    forecaster  predicted swan beta per stock, using only data before t
    selection   which stocks to hold, from the predicted betas
    weighting   how much of each

Original MVO2 = forecaster "kmeans" + selection "kmeans" + weighting "mvo".
"""
import datetime
from functools import lru_cache

import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import mvo2
import swan_beta

YEAR = datetime.timedelta(days=365)
LOOKBACK = datetime.timedelta(days=410)
BENCH = {"SP100": "^OEX", "SP500": "^GSPC"}
INDICES = swan_beta.INDICES
TOTAL_RETURN = {"SP100": "OEF", "SP500": "SPY"}   # ETFs incl. dividends; ^OEX / ^GSPC are price-only


# ---------- universe ----------

@lru_cache
def _sp500_periods():
    p = pd.read_csv(swan_beta.DATA / "universe/sp500_ticker_start_end.csv", parse_dates=["start_date", "end_date"])
    p["ticker"] = p.ticker.str.replace(".", "-")   # Yahoo spelling
    return p


def members(t, universe):
    """S&P500: members on t (fja05680/sp500, from 1996). S&P100, or S&P500 before 1996: every stock in the price file
    (today's list -> survivorship-biased)."""
    if universe == "SP100" or t < pd.Timestamp("1996-01-02"):
        return swan_beta.load_prices(universe).columns.difference(INDICES)
    p = _sp500_periods()
    return pd.Index(p.ticker[(p.start_date <= t) & (p.end_date.isna() | (p.end_date >= t))].unique())


def eligible(t, universe):
    """Members on t that have prices for ~all of the look-back window."""
    close = swan_beta.load_prices(universe)
    px = close.loc[t - LOOKBACK:t - datetime.timedelta(days=1), close.columns.intersection(members(t, universe))]
    return px.columns[px.notna().mean() >= 0.9]


# ---------- features ----------

def market_features(t, tickers, universe):
    """1-year daily market beta, downside beta (index-down days only) and annualised vol, before t."""
    close = swan_beta.load_prices(universe)
    r = close.loc[t - YEAR:t - datetime.timedelta(days=1), list(tickers) + [BENCH[universe]]].pct_change().iloc[1:]
    b = r.pop(BENCH[universe])
    down = b < 0
    return pd.DataFrame({
        "mkt_beta": r.apply(lambda s: s.cov(b)) / b.var(),
        "down_beta": r[down].apply(lambda s: s.cov(b[down])) / b[down].var(),
        "vol": r.std() * np.sqrt(252),
    })


def swan_features(betas, window):
    w = betas[window]
    return pd.DataFrame({"swan_mean": w.mean(axis=1), "swan_last": w.ffill(axis=1).iloc[:, -1],
                         "swan_std": w.std(axis=1)}).where(w.notna().sum(axis=1) >= 2)


FEATURES = ["swan_mean", "swan_last", "swan_std", "mkt_beta", "down_beta", "vol"]


def features(t, tickers, betas, window, universe):
    return swan_features(betas.loc[betas.index.intersection(tickers)], window).join(
        market_features(t, tickers, universe), how="inner")


# ---------- forecasters: predicted beta in the next swan ----------

@lru_cache(maxsize=64)
def _training_set(universe, swans_key):
    """One row per (past swan k, stock): features as of the day before k's peak -> actual beta in k."""
    swans = _SWANS[universe].loc[list(swans_key)]
    betas = _BETAS[universe]
    rows = []
    for k in range(5, len(swans)):
        target = swans.index[k]
        t = swans.peak.iloc[k]
        x = features(t, eligible(t, universe), betas, list(swans.index[k - 5:k]), universe).dropna()
        x["y"] = betas[target].reindex(x.index).clip(-1, 3)
        rows.append(x.dropna())
    return pd.concat(rows)


@lru_cache(maxsize=64)
def _model(universe, swans_key, kind):
    data = _training_set(universe, swans_key)
    model = (make_pipeline(StandardScaler(), Ridge(alpha=10.0)) if kind == "ridge"
             else RandomForestRegressor(n_estimators=300, min_samples_leaf=25, random_state=0, n_jobs=-1))
    return model.fit(data[FEATURES], data.y)


def forecast(kind, x, universe, known):
    """x: features of the candidates (index = tickers). known: swans confirmed before t, chronological."""
    if kind == "swan_mean":       # original's per-stock estimate (MVO DB "unequal beta")
        return x.swan_mean
    if kind == "market_beta":     # plain 1-year beta
        return x.mkt_beta
    if kind == "blend":
        return (x.swan_mean + x.mkt_beta) / 2
    if kind in ("ridge", "rf"):   # trained on all earlier swans, refit when a new swan is confirmed
        return pd.Series(_model(universe, tuple(known), kind).predict(x[FEATURES]), index=x.index)
    raise ValueError(kind)


# ---------- selection ----------

@lru_cache
def sectors():
    """Ticker -> sector (data/universe/sectors.csv, today's classification). Missing tickers get their own name."""
    s = pd.read_csv(swan_beta.DATA / "universe/sectors.csv", index_col="ticker").sector
    return lambda tickers: pd.Series([s.get(t, t) for t in tickers], index=tickers)


def select_lowest(pred, n, sector_cap=None):
    """The n lowest predicted betas; with sector_cap, at most floor(cap * n) (>= 1) stocks per sector."""
    if sector_cap is None:
        return pred.nsmallest(n)
    most = max(1, int(sector_cap * n))
    pred = pred.sort_values()
    rank = pred.groupby(sectors()(pred.index)).cumcount()
    return pred[rank < most].head(n)


def cap_sectors(w, cap):
    """Scale sectors above `cap` down to it and spread the excess over the others, pro rata, until none is above."""
    sec = sectors()(w.index)
    for _ in range(50):
        total = w.groupby(sec).sum()
        over = total[total > cap + 1e-9]
        if over.empty:
            break
        scale = sec.map(lambda s: cap / total[s] if s in over.index else 1.0)
        excess = (w * (1 - scale)).sum()
        w = w * scale
        free = ~sec.isin(over.index)
        w[free] += excess * w[free] / w[free].sum()
    return w


def select_kmeans(betas, window, N):
    """Original: K-means on the window betas, keep the cluster(s) with the lowest centre."""
    return mvo2.preselect_kmeans(betas[window], window[0], window[-1], N)


# ---------- weighting ----------

def weekly_cov(t, tickers, universe):
    px = mvo2.get_close(tickers, t - YEAR, t, swan_beta.load_prices(universe)).ffill().bfill()
    week = px.resample("W-FRI").last().pct_change().dropna()
    return week.cov() * 52


def weigh(kind, beta, t, universe, ctx, cap=0.17):
    tickers = beta.index
    if kind == "equal":
        return pd.Series(1 / len(tickers), index=tickers)
    if kind == "inv_vol":       # ponytail: inverse-vol as the risk-parity stand-in; full ERC if correlations matter
        vol = np.sqrt(np.diag(weekly_cov(t, tickers, universe)))
        w = pd.Series(1 / vol, index=tickers)
        return w / w.sum()
    if kind == "hrp":           # Hierarchical Risk Parity (López de Prado), pypfopt, on 1 year of daily returns
        from pypfopt import HRPOpt
        r = mvo2.get_close(tickers, t - YEAR, t, swan_beta.load_prices(universe)).ffill().bfill().pct_change().iloc[1:]
        w = pd.Series(HRPOpt(r).optimize())
        return w / w.sum()
    if kind == "herc":
        return herc(weekly_cov(t, tickers, universe))
    if kind == "min_vol":
        ef = EfficientFrontier(None, weekly_cov(t, tickers, universe), weight_bounds=(0, max(cap, 1.5 / len(tickers))))
        ef.min_volatility()
        w = pd.Series(ef.clean_weights())
        return w[w > 0] / w[w > 0].sum()
    if kind == "mvo":           # original steps 2-4: inject the predicted swan, trend return / drawdown, max Sharpe
        p = dict(ctx, betas=ctx["betas"].loc[tickers], start=t - LOOKBACK, end=t, prices=universe,
                 w1=cap, min_weight=0.03, MA_days=30, optdata_file=None)
        prices = mvo2.get_close(tickers, p["start"], t, swan_beta.load_prices(universe))
        md = mvo2.predicted_market_drop(len(prices), ctx["marketdd"], mvo2.swan_weights(ctx["betas"], ctx["history"]))
        stats, cov = mvo2.estimate(mvo2.inject(prices, beta.clip(lower=0), md), t, None, 30, universe)
        return mvo2.optimize(stats.Rfwd_adjusted / stats.Fall_MDD / 100, cov, cap, 0.03)[0]
    raise ValueError(kind)


def herc(cov, max_k=8):
    """Hierarchical Equal Risk Contribution (Raffinot), flat version: cluster the stocks on correlation distance
    (Ward), pick the cluster count by silhouette, weight clusters by 1 / cluster volatility and stocks inside a
    cluster by inverse variance. ponytail: one level instead of the paper's recursive split; add it if HERC wins."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    from sklearn.metrics import silhouette_score
    sd = np.sqrt(np.diag(cov))
    dist = np.sqrt(np.clip(0.5 * (1 - cov.values / np.outer(sd, sd)), 0, None))
    np.fill_diagonal(dist, 0)
    Z = linkage(squareform(dist, checks=False), "ward")
    ks = range(2, min(max_k, len(cov) - 1) + 1)
    labels = max((fcluster(Z, k, "maxclust") for k in ks),
                 key=lambda lab: silhouette_score(dist, lab, metric="precomputed") if len(set(lab)) > 1 else -1)
    w = pd.Series(0.0, index=cov.index)
    for c in set(labels):
        idx = cov.index[labels == c]
        inner = 1 / np.diag(cov.loc[idx, idx])
        inner /= inner.sum()
        w[idx] = inner / np.sqrt(inner @ cov.loc[idx, idx].values @ inner)
    return w / w.sum()


# set by experiments.setup(): swan list and beta matrix per universe
_SWANS, _BETAS = {}, {}


# ---------- recommended setup (see v2/README.md, "Component experiments") ----------

def recommended(t, universe="SP500", n=None, which="hand", sector_cap=0.25, track="defensive"):
    """Blend beta forecast (half mean of the last 5 swan betas, half 1-year market beta) -> lowest n, at most 25% of
    them per sector -> inverse vol, sectors capped at 25%. Rebalance quarterly (see v2/README.md, "Development").
    track="defensive": n = 30 (S&P500) / 20 (S&P100). track="return": the lowest third of the candidates."""
    import experiments   # late import: experiments imports this module
    swans = experiments.setup(universe, which)
    ctx = experiments.context(pd.Timestamp(t), universe, swans)
    if n is None:
        n = experiments.N_SELECT[universe] if track == "defensive" else len(ctx["x"]) // 3
    pick = select_lowest(forecast("blend", ctx["x"], universe, ctx["known"]), n, sector_cap)
    w = weigh("inv_vol", pick, pd.Timestamp(t), universe, ctx)
    return (w if sector_cap is None else cap_sectors(w, sector_cap)).sort_values(ascending=False)


if __name__ == "__main__":
    import sys
    from pathlib import Path
    universe = sys.argv[1] if len(sys.argv) > 1 else "SP500"
    t = pd.Timestamp(sys.argv[2] if len(sys.argv) > 2 else swan_beta.load_prices(universe).index[-1])
    track = sys.argv[3] if len(sys.argv) > 3 else "defensive"
    w = recommended(t, universe, track=track)
    assert abs(w.sum() - 1) < 1e-9 and (w > 0).all()
    out = Path(__file__).parent / "output"
    out.mkdir(exist_ok=True)
    w.rename("weight").to_csv(out / f"holdings_{universe}_{track}_{t:%Y-%m-%d}.csv")
    print(f"{universe} {track} {t.date()}: {len(w)} stocks, top sector {w.groupby(sectors()(w.index)).sum().max():.0%}")
    print(w.round(4).to_string())
