"""Bring the frozen data up to date for live use. Run before each monthly rebalance.

    python data/update_data.py

S&P500 prices: the last few weeks are downloaded for every stock that traded in the past year plus today's members,
and appended to close_SP500.csv.gz / dollar_volume_SP500.csv.gz. Yahoo re-adjusts old prices after each dividend,
so the new piece is scaled to match the stored price on the last stored day (returns stay exact).
Also refreshed in full (small files): ETFs, T-bill yield, point-in-time membership, sectors of new tickers.
The nominal price files are not updated (only the analyst-target research used them).
"""
import subprocess
import sys
import urllib.request
from pathlib import Path

import pandas as pd
import yfinance as yf

DATA = Path(__file__).parent
PRICES = DATA / "prices"
OVERLAP_DAYS = 30
INDICES = ["^OEX", "^GSPC", "SPY", "OEF"]


def refresh_membership():
    url = "https://raw.githubusercontent.com/fja05680/sp500/master/sp500_ticker_start_end.csv"
    (DATA / "universe/sp500_ticker_start_end.csv").write_bytes(urllib.request.urlopen(url).read())


def append_prices():
    close = pd.read_csv(PRICES / "close_SP500.csv.gz", index_col=0, parse_dates=True)
    dv = pd.read_csv(PRICES / "dollar_volume_SP500.csv.gz", index_col=0, parse_dates=True)
    members = pd.read_csv(DATA / "universe/sp500_ticker_start_end.csv")
    current = members[members.end_date.isna()].ticker.str.replace(".", "-").replace({"BNY": "BK"})
    recent = close.columns[close.iloc[-252:].notna().any()]
    tickers = sorted(set(recent) | set(current) | set(INDICES))
    last = close.index[-1]
    renamed = {"BK": "BNY"}
    px = yf.download([renamed.get(t, t) for t in tickers], start=str((last - pd.Timedelta(days=OVERLAP_DAYS)).date()),
                     auto_adjust=False, progress=False)
    px = px.rename(columns={v: k for k, v in renamed.items()}, level=1)
    adj = px["Adj Close"].reindex(columns=tickers)
    adj.index = pd.to_datetime(adj.index)
    if adj.index[-1] <= last:
        print("prices already up to date:", last.date())
        return
    new = adj.loc[adj.index > last].copy()
    for t in tickers:
        if t in close and pd.notna(close.at[last, t]) and last in adj.index and pd.notna(adj.at[last, t]):
            new[t] = new[t] * close.at[last, t] / adj.at[last, t]   # continue the stored series
    close = pd.concat([close, new.reindex(columns=close.columns.union(new.columns))])
    missing = [t for t in current if t not in recent]   # members with no stored history (dropped by Yahoo at freeze)
    if missing:
        full = yf.download(missing, start="1990-01-01", auto_adjust=False, progress=False)
        for t in missing:
            close[t] = full["Adj Close"][t].reindex(close.index)
        dv = dv.reindex(columns=dv.columns.union(missing))
        dv.update(full["Close"][missing] * full["Volume"][missing])
    vol = (px["Close"] * px["Volume"]).reindex(columns=tickers)
    vol.index = pd.to_datetime(vol.index)
    dv = pd.concat([dv, vol.loc[vol.index > dv.index[-1]]])
    for df, name in [(close, "close_SP500.csv.gz"), (dv, "dollar_volume_SP500.csv.gz")]:
        df.index = pd.to_datetime(df.index).date
        df.index.name = "Date"
        df.to_csv(PRICES / name, float_format="%.8g")
    print(f"prices {last.date()} -> {new.index[-1].date()}, {new.notna().any().sum()} tickers,"
          f" full history filled for: {missing}")


if __name__ == "__main__":
    refresh_membership()
    append_prices()
    for script in ["prices/freeze_etfs.py", "prices/freeze_tbill.py", "universe/fetch_sectors.py"]:
        subprocess.run([sys.executable, "-W", "ignore", str(DATA / script)], check=True)
