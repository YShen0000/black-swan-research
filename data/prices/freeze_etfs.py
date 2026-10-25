"""Other asset classes and comparison ETFs (adjusted close, incl. distributions) -> etf_close.csv.gz.

    python data/prices/freeze_etfs.py

IEF 7-10y Treasuries, TLT 20y+ Treasuries, SHY 1-3y Treasuries, GLD gold (from 2004-11),
TIP inflation-protected Treasuries (from 2003-12), DBC broad commodities (from 2006-02),
USMV / SPLV low-volatility S&P ETFs (from 2011), the off-the-shelf version of a defensive equity book.
For multi-asset trend / rotation tests: EFA developed ex-US, EEM emerging markets (from 2003-04), VNQ US REITs
(from 2004-09), QQQ Nasdaq-100, IWM Russell 2000, AGG US aggregate bonds (from 2003-09), LQD corporate bonds,
and the nine original Select Sector SPDRs (XLB ... XLY, from 1998-12).
"""
from pathlib import Path

import yfinance as yf

TICKERS = ["IEF", "TLT", "SHY", "GLD", "TIP", "DBC", "USMV", "SPLV", "EFA", "EEM", "VNQ", "QQQ", "IWM", "AGG", "LQD",
           "XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]
px = yf.download(TICKERS, start="2002-01-01", auto_adjust=True, progress=False)["Close"]
px.index = px.index.date
px.index.name = "Date"
px.to_csv(Path(__file__).parent / "etf_close.csv.gz", float_format="%.6g")
print(px.shape, {t: str(px[t].first_valid_index()) for t in TICKERS})
