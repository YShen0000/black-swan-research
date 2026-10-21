"""Daily dollar volume (split-adjusted close x volume) for every stock that has prices -> dollar_volume_<universe>.csv.gz.
Used as the size factor's proxy: Yahoo has no history of shares outstanding, and dollar volume tracks market cap closely.

    python data/prices/freeze_volume.py [SP100|SP500]
"""
import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

HERE = Path(__file__).parent
universe = sys.argv[1] if len(sys.argv) > 1 else "SP500"
close = pd.read_csv(HERE / f"close_{universe}.csv.gz", index_col=0)
tickers = [t for t in close.columns if close[t].notna().any() and not t.startswith("^") and t not in ("SPY", "OEF")]
RENAMED = {"BK": "BNY"}

end = str((pd.Timestamp(close.index[-1]) + pd.Timedelta(days=1)).date())


def download(ts):
    px = yf.download([RENAMED.get(t, t) for t in ts], start=close.index[0], end=end, auto_adjust=False, progress=False)
    px = px.rename(columns={v: k for k, v in RENAMED.items()}, level=1)
    return (px["Close"] * px["Volume"]).reindex(columns=ts)


dv = download(tickers)
for _ in range(3):   # Yahoo drops random tickers under load; retry the empty ones
    empty = [t for t in tickers if dv[t].isna().all()]
    if empty:
        dv[empty] = download(empty)
dv.index = pd.to_datetime(dv.index).date
dv.index.name = "Date"
dv.to_csv(HERE / f"dollar_volume_{universe}.csv.gz", float_format="%.6g")
print(dv.shape, "no volume:", [t for t in tickers if dv[t].isna().all()])
