"""Live use of the chosen allocation: risk parity of the swan-beta stock sleeve, 7-10y Treasuries (IEF) and gold (GLD).

    python data/update_data.py                                          # 1. bring the data up to date
    python v2/live.py target --capital 20000 [--stocks 30] [--whole]    # 2. target holdings for the next trading day
    python v2/live.py orders --holdings live/holdings.csv               # 3. trades from your holdings to the target
    python v2/live.py report [--holdings live/holdings.csv]             #    risk and market check, any time

Schedule, as backtested: the stock sleeve is rebuilt each quarter (first trading day of January, April, July,
October); the split between stocks, IEF and GLD is reset every month. `target` reuses this quarter's sleeve if it
already exists. Everything is saved in live/ (git-ignored: it holds your positions; the repo is public).

holdings.csv: two columns, ticker,shares (fractional allowed). Use CASH for uninvested cash, in dollars.
Backtest reference (2004-2026, see v2/README.md): ~7%/yr, max drawdown ~16%, worst calendar year about -8%.
"""
import argparse
import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import components as C
import risk_model as RM
import swan_beta

LIVE = Path(__file__).resolve().parent.parent / "live"
U = "SP500"
ETFS = ["IEF", "GLD"]


def data_end():
    return swan_beta.load_prices(U).index[-1]


def etf_prices():
    return pd.read_csv(swan_beta.DATA / "prices/etf_close.csv.gz", index_col=0, parse_dates=True)


def last_prices(tickers):
    close, etf = swan_beta.load_prices(U), etf_prices()
    return pd.Series({t: (etf[t] if t in etf else close[t]).dropna().iloc[-1] for t in tickers})


def quarter(t):
    return f"{t.year}Q{(t.month - 1) // 3 + 1}"


def sleeve(t, stocks=None, new=False):
    """This quarter's stock sleeve (weights summing to 1), built once per quarter and saved."""
    LIVE.mkdir(exist_ok=True)
    path = LIVE / f"sleeve_{quarter(t)}_{stocks or 'third'}.csv"
    if path.exists() and not new:
        return pd.read_csv(path, index_col=0).weight
    w = C.recommended(t, U, n=stocks, track="return")
    w.rename("weight").to_csv(path)
    return w


def split(s):
    """Risk parity across the sleeve, IEF and GLD: inverse of each one's past-year daily volatility."""
    close, etf = swan_beta.load_prices(U), etf_prices()
    r = close[s.index].iloc[-253:].pct_change().iloc[1:].fillna(0) @ s
    rets = pd.DataFrame({"stocks": r, **{e: etf[e].pct_change().reindex(r.index) for e in ETFS}})
    inv = 1 / rets.std()
    return inv / inv.sum(), rets


def target(capital, stocks=None, whole=False, new_sleeve=False):
    t = data_end() + datetime.timedelta(days=1)
    s = sleeve(t, stocks, new_sleeve)
    mix, _ = split(s)
    w = pd.concat([s * mix["stocks"], mix[ETFS]])
    px = last_prices(w.index)
    shares = capital * w / px
    shares = np.floor(shares) if whole else shares.round(4)
    while whole:   # leftover cash buys one more share of the most underweight position, if that gets it closer to target
        gap = capital * w - shares * px
        gap = gap[(px <= capital - (shares * px).sum()) & (px < 2 * gap)]
        if gap.empty:
            break
        shares[gap.idxmax()] += 1
    out = pd.DataFrame({"weight": w, "price": px, "shares": shares, "value": shares * px})
    out.loc["CASH"] = [np.nan, 1.0, capital - out.value.sum(), capital - out.value.sum()]
    path = LIVE / f"target_{t:%Y-%m-%d}.csv"
    out.to_csv(path, float_format="%.6g")
    print(f"prices as of {data_end().date()} | sleeve {quarter(t)} ({len(s)} stocks) | saved {path}")
    print(f"split: stocks {mix['stocks']:.1%}, IEF {mix['IEF']:.1%}, GLD {mix['GLD']:.1%}")
    print(out.sort_values("value", ascending=False).head(12).round(4).to_string())
    return out


def latest_target():
    files = sorted(LIVE.glob("target_*.csv"))
    if not files:
        raise SystemExit("no target yet: run `python v2/live.py target --capital ...` first")
    return pd.read_csv(files[-1], index_col=0), files[-1]


def orders(holdings_file, min_trade=50.0):
    tgt, path = latest_target()
    have = pd.read_csv(holdings_file, index_col=0).iloc[:, 0]
    tickers = tgt.index.union(have.index).drop("CASH", errors="ignore")
    px = last_prices(tickers)
    worth = (have.reindex(tickers).fillna(0) * px).sum() + have.get("CASH", 0.0)
    if abs(worth / tgt.value.sum() - 1) > 0.02:
        raise SystemExit(f"holdings are worth ${worth:,.0f} but {path.name} was built for ${tgt.value.sum():,.0f}:"
                         f" re-run `target --capital {worth:.0f}` (same --stocks / --whole) first")
    delta =tgt.shares.reindex(tickers).fillna(0) - have.reindex(tickers).fillna(0)
    o = pd.DataFrame({"shares": delta, "price": px, "value": delta * px})
    o = o[o.value.abs() >= min_trade].sort_values("value")   # sells first, they fund the buys
    o["side"] = np.where(o.shares > 0, "BUY", "SELL")
    out = LIVE / f"orders_{path.stem.removeprefix('target_')}.csv"
    o.to_csv(out, float_format="%.6g")
    print(f"vs {path.name}: {len(o)} trades over ${min_trade:.0f}, sells ${-o.value[o.value < 0].sum():,.0f},"
          f" buys ${o.value[o.value > 0].sum():,.0f} | saved {out}")
    print(o.round(4).to_string())


def report(holdings_file=None):
    t = data_end() + datetime.timedelta(days=1)
    close = swan_beta.load_prices(U)
    print(f"data through {data_end().date()}")

    # market state and new swans
    oex = close["^OEX"].dropna()
    dd = swan_beta.drawdown(oex).iloc[-1]
    print(f"\nmarket: ^OEX is {dd:.1%} below its 1-year high"
          + ("  <-- swan territory (>= 15%)" if dd >= 0.15 else ""))
    rule = swan_beta.detect_swans(close)
    newer = rule[rule.trough > swan_beta.SWANS.trough.max() + datetime.timedelta(days=45)]
    if len(newer):
        print("new swan(s) found by the 15% rule, not yet in data/swans.csv (add once the trough is confirmed):")
        print(newer.to_string())

    # the portfolio of the latest target (or this quarter's default sleeve if there is none yet)
    files = sorted(LIVE.glob("target_*.csv"))
    s = pd.read_csv(files[-1], index_col=0).weight.drop([*ETFS, "CASH"]) if files else sleeve(t)
    s = s / s.sum()
    mix, rets = split(s)
    cov = rets.cov() * 252
    w = mix.values
    rc = w * (cov.values @ w) / (w @ cov.values @ w)
    print(f"\nsplit (target): " + ", ".join(f"{k} {v:.1%}" for k, v in mix.items())
          + f" | expected vol {np.sqrt(w @ cov.values @ w):.1%}")
    print("risk share: " + ", ".join(f"{k} {v:.0%}" for k, v in zip(mix.index, rc)))
    sec = s.groupby(C.sectors()(s.index)).sum().sort_values(ascending=False)
    print(f"sleeve: {len(s)} stocks, top sectors " + ", ".join(f"{k} {v:.0%}" for k, v in sec.head(3).items()))
    x = RM.exposures(t, s.index, U)
    print("sleeve factor exposures (weighted): " + ", ".join(
        f"{f} {(x[f] * s).sum():+.2f}" for f in ["market", "size", "momentum", "volatility"]))

    if holdings_file:
        have = pd.read_csv(holdings_file, index_col=0).iloc[:, 0]
        px = last_prices(have.index.drop("CASH", errors="ignore"))
        value = (have.drop("CASH", errors="ignore") * px)
        total = value.sum() + have.get("CASH", 0.0)
        groups = {"stocks": value.drop(ETFS, errors="ignore").sum(), **{e: value.get(e, 0.0) for e in ETFS},
                  "CASH": have.get("CASH", 0.0)}
        print(f"\nyour holdings: ${total:,.0f} | " + ", ".join(f"{k} {v / total:.1%}" for k, v in groups.items()))
        drift = max(abs(groups[k] / total - mix.get(k, 0)) for k in ["stocks", *ETFS])
        print(f"largest drift from target split: {drift:.1%}" + ("  <-- rebalance" if drift > 0.05 else ""))

    nxt = pd.Timestamp(t.year, t.month, 1) + pd.offsets.MonthBegin(1)
    print(f"\nnext monthly rebalance: first trading day of {nxt:%Y-%m}"
          + (" (also rebuilds the stock sleeve)" if nxt.month in (1, 4, 7, 10) else ""))
    print("backtest reference: ~7%/yr, max drawdown ~16%, worst year about -8%")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["target", "orders", "report"])
    ap.add_argument("--capital", type=float)
    ap.add_argument("--stocks", type=int, help="number of stocks in the sleeve (default: lowest third, ~160)")
    ap.add_argument("--whole", action="store_true", help="whole shares only (leftover stays as cash)")
    ap.add_argument("--new-sleeve", action="store_true", help="rebuild this quarter's sleeve")
    ap.add_argument("--holdings")
    ap.add_argument("--min-trade", type=float, default=50.0)
    a = ap.parse_args()
    if a.command == "target":
        if not a.capital:
            raise SystemExit("--capital is required")
        target(a.capital, a.stocks, a.whole, a.new_sleeve)
    elif a.command == "orders":
        orders(a.holdings or LIVE / "holdings.csv", a.min_trade)
    else:
        report(a.holdings)
