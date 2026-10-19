"""Download Yahoo daily prices once and freeze them to CSV, so every run sees the same data.

    python data/prices/freeze_prices.py

close_SP100.csv          split + dividend adjusted close (what yfinance auto_adjust returns). Use for returns / risk.
nominal_close_SP100.csv  the price actually quoted that day. Use when comparing with a dollar figure
                         from that day, e.g. optdata P_fwd (analyst target price).
"""
from datetime import date
from pathlib import Path

import pandas as pd
import yfinance as yf

HERE = Path(__file__).parent
DATA = HERE.parent
START, END = "2019-01-01", "2025-01-01"   # ponytail: covers Swan9 / Swan10D / Swan11 on S&P100; add S&P500 when researched

tickers = set()
for f in ["SwanBeta/All_SwanBeta100.xlsx", "SwanBeta/All_SwanBeta100_MVO2.xlsx"]:
    tickers |= set(pd.read_excel(DATA / f, index_col=0).index.dropna().astype(str))
tickers = sorted(tickers) + ["^OEX", "SPY"]
RENAMED = {"BK": "BNY"}  # Yahoo serves the old history under the new ticker


def download(ts):
    # up to today: Yahoo back-adjusts "Close" for every split up to today, so we need all of them to undo it
    px = yf.download([RENAMED.get(t, t) for t in ts], start=START, end=str(date.today()),
                     auto_adjust=False, actions=True, progress=False)
    return px.rename(columns={v: k for k, v in RENAMED.items()}, level=1)


px = download(tickers)
for _ in range(3):  # Yahoo drops random tickers under load; retry the empty ones
    empty = [t for t in tickers if t not in px["Close"] or px["Close"][t].isna().all()]
    if empty:
        px = px.drop(columns=empty, level=1, errors="ignore").join(download(empty))

splits = px["Stock Splits"].reindex(columns=tickers).fillna(0).replace(0, 1)
later_splits = splits[::-1].cumprod()[::-1].shift(-1).fillna(1)   # product of the splits after each day
out = {"close_SP100.csv": px["Adj Close"], "nominal_close_SP100.csv": px["Close"].reindex(columns=tickers) * later_splits}
for name, df in out.items():
    df = df.reindex(columns=tickers).loc[:pd.Timestamp(END) - pd.Timedelta(days=1)]
    df.index = pd.to_datetime(df.index).date
    df.index.name = "Date"
    df.round(6).to_csv(HERE / name)

close = out["close_SP100.csv"].reindex(columns=tickers)
print(close.loc[:END].shape, "missing:", [t for t in tickers if close[t].isna().all()])  # WBA, AVG: delisted
