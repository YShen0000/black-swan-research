"""Download Yahoo daily prices once and freeze them to CSV, so every run sees the same data.

    python data/prices/freeze_prices.py          # S&P100 universe
    python data/prices/freeze_prices.py SP500    # S&P500 universe

close_<universe>.csv.gz          split + dividend adjusted close (what yfinance auto_adjust returns). Use for returns / risk
                            and Swan Beta.
nominal_close_<universe>.csv.gz  the price actually quoted that day. Use when comparing with a dollar figure
                            from that day, e.g. optdata P_fwd (analyst target price).
Universe: every ticker in that universe's Swan Beta files, plus ^OEX, ^GSPC, SPY. History from 1962 (or listing) to the run date.
"""
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import yfinance as yf

HERE = Path(__file__).parent
DATA = HERE.parent
START, END = "1962-01-01", str(date.today())
UNIVERSES = {"SP100": ["SwanBeta/All_SwanBeta100.xlsx", "SwanBeta/All_SwanBeta100_MVO2.xlsx"],
             "SP500": ["SwanBeta/All_SwanBeta500.xlsx"]}
universe = sys.argv[1] if len(sys.argv) > 1 else "SP100"

tickers = set()
for f in UNIVERSES[universe]:
    tickers |= set(pd.read_excel(DATA / f, index_col=0).index.dropna().astype(str))
tickers = sorted(tickers) + ["^OEX", "^GSPC", "SPY"]
RENAMED = {"BK": "BNY"}  # Yahoo serves the old history under the new ticker


def download(ts):
    # Yahoo back-adjusts "Close" for every split up to today; END = today, so all of them are undone below
    px = yf.download([RENAMED.get(t, t) for t in ts], start=START, end=END,
                     auto_adjust=False, actions=True, progress=False)
    return px.rename(columns={v: k for k, v in RENAMED.items()}, level=1)


px = download(tickers)
for _ in range(3):  # Yahoo drops random tickers under load; retry the empty ones
    empty = [t for t in tickers if t not in px["Close"] or px["Close"][t].isna().all()]
    if empty:
        px = px.drop(columns=empty, level=1, errors="ignore").join(download(empty))

splits = px["Stock Splits"].reindex(columns=tickers).fillna(0).replace(0, 1)
later_splits = splits[::-1].cumprod()[::-1].shift(-1).fillna(1)   # product of the splits after each day
out = {f"close_{universe}.csv.gz": px["Adj Close"], f"nominal_close_{universe}.csv.gz": px["Close"].reindex(columns=tickers) * later_splits}
for name, df in out.items():
    df = df.reindex(columns=tickers).dropna(how="all")
    df.index = pd.to_datetime(df.index).date
    df.index.name = "Date"
    df.to_csv(HERE / name, float_format="%.8g")   # significant digits: 1960s adjusted prices are ~0.01

close = out[f"close_{universe}.csv.gz"].reindex(columns=tickers)
print(close.shape, close.index[0], close.index[-1], "missing:", [t for t in tickers if close[t].isna().all()])  # delisted / renamed since the beta file
