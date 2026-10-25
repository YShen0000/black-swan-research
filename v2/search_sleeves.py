"""Monthly stock sleeves for strategy_search.py (S&P500 point-in-time, first trading day of each month, 2004 -> today).
Each holds a third of the candidates, inverse-vol weighted, <= 25% per sector (as R1):

  R1   lowest blend score (½ mean of the last 5 swan betas + ½ 1-year market beta)   -- existing return track
  M1   highest 12-1 month momentum (return from 12 months to 1 month before t)
  M2   lowest mean of rank(blend score) and rank(-momentum): low swan beta and strong trend together

    python v2/search_sleeves.py      # ~10 min, cached in output/search_sleeves_weights.pkl
"""
from pathlib import Path

import pandas as pd

import develop as D
import experiments as E
import swan_beta

OUT = Path(__file__).parent / "output"
U = "SP500"


def momentum(t, tickers):
    p = swan_beta.load_prices(U).loc[:t - pd.Timedelta(days=1), list(tickers)].iloc[-253:]
    return p.iloc[-22] / p.iloc[0] - 1


def build():
    swans = E.setup(U)
    dates = sorted(pd.read_pickle(OUT / "develop_SP500_weights.pkl"))
    out = {}
    for t in dates:
        ctx = E.context(t, U, swans)
        x = ctx["x"]
        blend = (x.swan_mean + x.mkt_beta) / 2
        mom = momentum(t, x.index).reindex(x.index)
        n = len(x) // 3
        out[t] = {"R1": D.hold(blend, t, U, ctx, n), "M1": D.hold(-mom, t, U, ctx, n),
                  "M2": D.hold(blend.rank() + (-mom).rank(), t, U, ctx, n)}
        print(t.date(), len(x), flush=True)
    return out


if __name__ == "__main__":
    cache = OUT / "search_sleeves_weights.pkl"
    pd.to_pickle(build(), cache)
