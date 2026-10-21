"""Preselection methods from the team's notebooks, plus Barra-feature and random controls, for explore.py.

Each method looks only at the swan-beta window (the last 5 confirmed swans) and returns a ranking of stocks, safest
first. The clustering is done once per window on every stock with all 5 betas; on a date, the ranking is filtered
to that date's candidates and the first n are held.

  kmeans        MVO2: K-means (seed 1), clusters ordered by centre; inside a cluster by own average beta
  kmeans_dbi    MVO DB: K-means for seeds 1..1000, keep the 5 seeds with the best Davies-Bouldin index, rank stocks
                by how often they land in the safest cluster(s), then by average beta
  dtw_team      dtw_result notebook: DTW distance between beta sequences, Ward linkage and DBI-chosen cluster count,
                computed the way the notebook does (the square distance matrix passed as if it were features)
  dtw           same idea done as intended: condensed distance matrix, average linkage, silhouette-chosen count
  holt          Holt linear-trend forecast of each stock's next beta (Projects/Holt Estimation)
  barra_ridge   ridge regression on the 6 forecaster features plus factor-model exposures (size, momentum,
                volatility, sector), trained on earlier swans
"""
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing

import components as C
import risk_model as RM


@lru_cache(maxsize=None)
def window_betas(universe, window):
    return C._BETAS[universe][list(window)].dropna()


def _kmeans_order(x, N, seed):
    km = KMeans(n_clusters=N, random_state=seed, n_init=10).fit(x)
    centre = pd.Series(km.cluster_centers_.mean(axis=1).clip(min=0))
    d = pd.DataFrame({"cluster": km.labels_, "centre": centre[km.labels_].values, "avg": x.mean(axis=1)}, index=x.index)
    return d.sort_values(["centre", "cluster", "avg"]), km.labels_


@lru_cache(maxsize=None)
def kmeans(universe, window, N):
    return list(_kmeans_order(window_betas(universe, window), N, 1)[0].index)


@lru_cache(maxsize=None)
def kmeans_dbi(universe, window, N, seeds=1000, top=5, min_size=10):
    x = window_betas(universe, window)
    dbi = {}
    for s in range(1, seeds + 1):
        dbi[s] = davies_bouldin_score(x, KMeans(n_clusters=N, random_state=s, n_init=10).fit(x).labels_)
    count = pd.Series(0, index=x.index)
    for s in sorted(dbi, key=dbi.get)[:top]:
        order = _kmeans_order(x, N, s)[0]
        first = order.cluster.iloc[0]
        pick = order.index[order.cluster == first]
        if len(pick) < min_size:   # as the notebook: add the next cluster
            nxt = order.cluster.iloc[len(pick)]
            pick = pick.append(order.index[order.cluster == nxt])
        count[pick] += 1
    return list(pd.DataFrame({"count": count, "avg": x.mean(axis=1)}).sort_values(["count", "avg"],
                                                                                  ascending=[False, True]).index)


def _dtw_matrix(x):
    """All-pairs DTW distance (absolute difference cost) between the rows of x, vectorised over pairs."""
    a = x.values
    n, m = a.shape
    cost = np.abs(a[:, None, :, None] - a[None, :, None, :])          # n x n x m x m
    D = np.full((n, n, m + 1, m + 1), np.inf)
    D[:, :, 0, 0] = 0
    for i in range(1, m + 1):
        for j in range(1, m + 1):
            D[:, :, i, j] = cost[:, :, i - 1, j - 1] + np.minimum(np.minimum(D[:, :, i - 1, j], D[:, :, i, j - 1]),
                                                                  D[:, :, i - 1, j - 1])
    return D[:, :, m, m]


@lru_cache(maxsize=None)
def dtw(universe, window, team=True):
    x = window_betas(universe, window)
    dist = _dtw_matrix(x)
    if team:   # as in the notebook: rows of the square matrix used as feature vectors
        Z = linkage(dist, method="ward", optimal_ordering=True)
        score = lambda lab: davies_bouldin_score(dist, lab)
    else:
        Z = linkage(squareform(dist, checks=False), method="average")
        score = lambda lab: -silhouette_score(dist, lab, metric="precomputed")
    best = min((fcluster(Z, t=k, criterion="maxclust") for k in range(2, 15)),
               key=lambda lab: score(lab) if len(set(lab)) > 1 else np.inf)
    avg = x.mean(axis=1)
    cluster_avg = avg.groupby(best).mean()
    d = pd.DataFrame({"c": pd.Series(best, index=x.index).map(cluster_avg), "avg": avg})
    return list(d.sort_values(["c", "avg"]).index)


@lru_cache(maxsize=None)
def holt(universe, window):
    x = window_betas(universe, window)
    out = {}
    for t, row in x.iterrows():
        try:
            out[t] = ExponentialSmoothing(row.values.astype(float), trend="add").fit(optimized=True).forecast(1)[0]
        except Exception:
            out[t] = row.mean()
    return list(pd.Series(out).sort_values().index)


# ---------- forecaster with Barra exposures ----------

def barra_features(t, x, universe):
    """x: components.features for the candidates on t; adds factor-model style exposures and sector dummies."""
    e = RM.exposures(t, x.index, universe).drop(columns="market")
    return x.join(e, how="inner").fillna(0)


@lru_cache(maxsize=16)
def _barra_model(universe, swans_key):
    swans = C._SWANS[universe].loc[list(swans_key)]
    rows = []
    for k in range(5, len(swans)):
        t = swans.peak.iloc[k]
        if t < pd.Timestamp("1990-01-01"):   # dollar volume (size) is thin before 1990
            continue
        x = C.features(t, C.eligible(t, universe), C._BETAS[universe], list(swans.index[k - 5:k]), universe).dropna()
        f = barra_features(t, x, universe)
        f["y"] = C._BETAS[universe][swans.index[k]].reindex(f.index).clip(-1, 3)
        rows.append(f.dropna(subset=["y"]))
    data = pd.concat(rows).fillna(0)
    cols = [c for c in data.columns if c != "y"]
    return make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(data[cols], data.y), cols


def barra_ridge(t, x, universe, known):
    model, cols = _barra_model(universe, tuple(known))
    f = barra_features(t, x, universe).reindex(columns=cols, fill_value=0)
    return pd.Series(model.predict(f), index=f.index)


# ---------- one entry point ----------

def ranking(method, t, ctx, universe):
    """Safest-first list of candidate tickers for `method` on date t. Candidates: stocks with all 5 window betas,
    so every method ranks the same pool (the clustering methods need complete windows)."""
    cand = set(ctx["x"].index) & set(window_betas(universe, tuple(ctx["window"])).index)
    win = tuple(ctx["window"])
    if method == "kmeans":
        order = kmeans(universe, win, ctx["N"])
    elif method == "kmeans_dbi":
        order = kmeans_dbi(universe, win, ctx["N"])
    elif method == "dtw_team":
        order = dtw(universe, win, True)
    elif method == "dtw":
        order = dtw(universe, win, False)
    elif method == "holt":
        order = holt(universe, win)
    elif method == "barra_ridge":
        order = list(barra_ridge(t, ctx["x"], universe, ctx["known"]).sort_values().index)
    elif method in ("swan_mean", "market_beta", "blend"):
        order = list(C.forecast(method, ctx["x"], universe, ctx["known"]).sort_values().index)
    else:
        raise ValueError(method)
    return [s for s in order if s in cand]
