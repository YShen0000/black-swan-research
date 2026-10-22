"""Other asset classes and comparison ETFs (adjusted close, incl. distributions) -> etf_close.csv.gz.

    python data/prices/freeze_etfs.py

IEF 7-10y Treasuries, TLT 20y+ Treasuries, SHY 1-3y Treasuries, GLD gold (from 2004-11),
USMV / SPLV low-volatility S&P ETFs (from 2011), the off-the-shelf version of a defensive equity book.
"""
from pathlib import Path

import yfinance as yf

TICKERS = ["IEF", "TLT", "SHY", "GLD", "USMV", "SPLV"]
px = yf.download(TICKERS, start="2002-01-01", auto_adjust=True, progress=False)["Close"]
px.index = px.index.date
px.index.name = "Date"
px.to_csv(Path(__file__).parent / "etf_close.csv.gz", float_format="%.6g")
print(px.shape, {t: str(px[t].first_valid_index()) for t in TICKERS})
