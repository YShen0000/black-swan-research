"""Sector of every stock in the price files -> data/universe/sectors.csv (ticker, sector, source).

    python data/universe/fetch_sectors.py

Sources, first one that has the ticker wins:
  wikipedia  current S&P500 list, GICS sector
  swanbeta   "Sector" column of data/SwanBeta/All_SwanBeta500.xlsx (GICS, 2024 list)
  yahoo      Ticker.info sector, mapped to GICS names (only for stocks still listed)
Sectors are as of today, not point-in-time (e.g. GICS moved Google / Meta to Communication Services in 2018).
"""
import io
import urllib.request
from pathlib import Path

import pandas as pd
import yfinance as yf

HERE = Path(__file__).parent
DATA = HERE.parent
YAHOO_TO_GICS = {"Technology": "Information Technology", "Healthcare": "Health Care",
                 "Financial Services": "Financials", "Consumer Cyclical": "Consumer Discretionary",
                 "Consumer Defensive": "Consumer Staples", "Basic Materials": "Materials"}

html = urllib.request.urlopen(urllib.request.Request(
    "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", headers={"User-Agent": "Mozilla/5.0 research"})).read()
wiki = pd.read_html(io.StringIO(html.decode()))[0]
rows = {t.replace(".", "-"): (s, "wikipedia") for t, s in zip(wiki.Symbol, wiki["GICS Sector"])}

old = HERE / "sectors.csv"   # keep earlier answers: Yahoo is only asked about tickers never classified before
if old.exists():
    for t, sec, src in pd.read_csv(old).itertuples(index=False):
        rows.setdefault(t, (sec, src))

sb = pd.read_excel(DATA / "SwanBeta/All_SwanBeta500.xlsx", index_col=0)["Sector"].dropna()
for t, s in sb.items():
    rows.setdefault(str(t).replace(".", "-"), (s, "swanbeta"))

INDICES = {"^OEX", "^GSPC", "SPY", "OEF"}
tickers = set()
for u in ["SP100", "SP500"]:   # only stocks that traded in the past year need a new lookup; delisted ones are done
    px = pd.read_csv(DATA / f"prices/close_{u}.csv.gz", index_col=0).iloc[-252:]
    tickers |= set(px.columns[px.notna().any()]) - INDICES
for t in sorted(tickers - set(rows)):
    try:
        s = yf.Ticker({"BK": "BNY"}.get(t, t)).info.get("sector")
    except Exception:
        s = None
    if s:
        rows[t] = (YAHOO_TO_GICS.get(s, s), "yahoo")

out = pd.DataFrame([(t, s, src) for t, (s, src) in sorted(rows.items())], columns=["ticker", "sector", "source"])
out.to_csv(HERE / "sectors.csv", index=False)
print(out.source.value_counts().to_dict(), "| priced tickers without sector:", len(tickers - set(out.ticker)))
print(out.sector.value_counts().to_string())
