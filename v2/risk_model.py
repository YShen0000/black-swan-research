"""Barra-style factor risk model built from prices only:  Σ = X F Xᵀ + D

Exposures X on date t (data before t only):
  market      1-year daily beta to the benchmark
  sectors     one 0/1 column per sector (data/universe/sectors.csv; sectors with < 3 stocks pooled as "Other")
  size        log of 3-month average dollar volume (no shares-outstanding history on Yahoo; tracks log market cap)
  momentum    12-month return skipping the last month
  volatility  1-year daily volatility
  Style exposures (size, momentum, volatility) are z-scored across the stocks and clipped at ±3.
Factor returns: each day of the past year, a cross-sectional least-squares fit of stock returns on X (X held fixed).
F = exponentially weighted covariance of the daily factor returns (half-life 90 days), annualised.
D = each stock's exponentially weighted residual variance, annualised.
"""
import datetime
from functools import lru_cache

import numpy as np
import pandas as pd

import components as C
import swan_beta

YEAR = 252
HALF_LIFE = 90
STYLES = ["size", "momentum", "volatility"]


@lru_cache
def dollar_volume(universe):
    return pd.read_csv(swan_beta.DATA / f"prices/dollar_volume_{universe}.csv.gz", index_col=0, parse_dates=True)


def _returns(t, tickers, universe, days=YEAR):
    close = swan_beta.load_prices(universe).loc[:t - datetime.timedelta(days=1)]
    return close[list(tickers) + [C.BENCH[universe]]].iloc[-days - 1:].pct_change().iloc[1:]


def exposures(t, tickers, universe):
    r = _returns(t, tickers, universe)
    bench = r.pop(C.BENCH[universe])
    px = swan_beta.load_prices(universe).loc[:t - datetime.timedelta(days=1), list(tickers)].iloc[-YEAR - 1:]
    dv = dollar_volume(universe).loc[t - datetime.timedelta(days=91):t - datetime.timedelta(days=1)].reindex(columns=tickers)
    style = pd.DataFrame({
        "size": np.log(dv.mean().where(lambda s: s > 0)),
        "momentum": px.iloc[-22] / px.iloc[0] - 1,
        "volatility": r.std() * np.sqrt(YEAR),
    })
    style = ((style - style.mean()) / style.std()).clip(-3, 3).fillna(0)
    sec = C.sectors()(pd.Index(tickers))
    sec = sec.where(sec.map(sec.value_counts()) >= 3, "Other")
    return pd.concat([(r.apply(lambda s: s.cov(bench)) / bench.var()).rename("market"), style,
                      pd.get_dummies(sec).astype(float)], axis=1)


def _ewm_weights(n):
    w = 0.5 ** (np.arange(n)[::-1] / HALF_LIFE)
    return w / w.sum()


def factor_model(t, tickers, universe):
    """Returns (Σ, X, F, D) for `tickers` on date t."""
    X = exposures(t, tickers, universe)
    R = _returns(t, tickers, universe).drop(columns=C.BENCH[universe]).fillna(0)
    f, *_ = np.linalg.lstsq(X.values, R.values.T, rcond=None)          # K x T factor returns
    resid = R.values.T - X.values @ f                                   # N x T
    w = _ewm_weights(R.shape[0])
    fc = f - (f * w).sum(axis=1, keepdims=True)
    F = (fc * w) @ fc.T * YEAR
    D = (resid ** 2 * w).sum(axis=1) * YEAR
    D = np.maximum(D, np.median(D) * 0.1)                              # floor: no stock with ~zero specific risk
    cov = X.values @ F @ X.values.T + np.diag(D)
    return (pd.DataFrame(cov, index=X.index, columns=X.index), X,
            pd.DataFrame(F, index=X.columns, columns=X.columns), pd.Series(D, index=X.index))


def sample_cov(t, tickers, universe):
    """Plain covariance of the past year's daily returns, annualised (what the risk model is compared with)."""
    return _returns(t, tickers, universe).drop(columns=C.BENCH[universe]).cov() * YEAR


def erc(cov, sweeps=100):
    """Equal risk contribution weights (each stock contributes the same variance), long only.
    Cyclical coordinate descent on min ½wᵀΣw - Σ log(w_i)/n, which keeps every weight positive."""
    s = cov.values
    n = len(s)
    w = 1 / np.sqrt(np.diag(s))
    for _ in range(sweeps):
        for i in range(n):
            b = s[i] @ w - s[i, i] * w[i]
            w[i] = (-b + np.sqrt(b * b + 4 * s[i, i] / n)) / (2 * s[i, i])
    return pd.Series(w / w.sum(), index=cov.index)


if __name__ == "__main__":
    # smoke test: the model reproduces the variance of an equal-weight portfolio about as well as the sample cov
    t = pd.Timestamp("2020-01-02")
    tickers = C.eligible(t, "SP500")
    cov, X, F, D = factor_model(t, tickers, "SP500")
    ew = pd.Series(1 / len(tickers), index=tickers)
    print("stocks", len(tickers), "factors", X.shape[1], "| EW vol: factor %.3f sample %.3f" % (
        np.sqrt(ew @ cov @ ew), np.sqrt(ew @ sample_cov(t, tickers, "SP500") @ ew)))
    print("specific share of variance, median stock: %.2f" % (D / np.diag(cov)).median())
    assert np.all(np.linalg.eigvalsh(cov.values) > 0)
    w = erc(cov)
    rc = w * (cov @ w)
    print("ERC risk contributions max/min: %.3f" % (rc.max() / rc.min()))
    assert rc.max() / rc.min() < 1.01
