# Point-in-time index membership

`sp500_ticker_start_end.csv`: every S&P 500 member since 1996 with its first and last day in the index (one row per spell). From [fja05680/sp500](https://github.com/fja05680/sp500) (MIT licence), which is built on Andreas Clenow's *Trading Evolved* list up to 2019 and maintained from Wikipedia since then. Downloaded 2026-10-21. Tickers are the ones in use at the time; `v2/components.py` converts `.` to `-` (Yahoo spelling).

There is no free point-in-time source for the S&P 100, so the S&P100 universe is today's list and is survivorship-biased.
