"""13-week T-bill yield (^IRX, % per year) -> tbill_13w.csv. Cash leg of the beta-matched "index + cash" benchmark.

    python data/prices/freeze_tbill.py
"""
from pathlib import Path

import yfinance as yf

irx = yf.download("^IRX", start="1962-01-01", auto_adjust=True, progress=False)["Close"]["^IRX"].dropna()
irx.index = irx.index.date
irx.index.name = "Date"
irx.rename("yield_pct").to_csv(Path(__file__).parent / "tbill_13w.csv", float_format="%.4f")
print(len(irx), irx.index[0], irx.index[-1])
