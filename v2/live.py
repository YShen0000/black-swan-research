"""Live use of the chosen strategy, D1 (strategy_search.py / strategy_robustness.py):

  half the money: core   R1 swan-beta stock sleeve 70%, IEF 15%, GLD 15%   (R1 rebuilt monthly, core reset monthly;
                         --core-stocks sets the stock share, the rest split equally between IEF and GLD)
  half the money: rotation, in 4 equal tranches. Each tranche holds the 3 strongest of SPY, EFA, EEM, VNQ, DBC, IEF,
                  TLT, GLD by mean 3/6/12-month return, an asset only while above its 10-month average (else cash).
                  Tranche 0 is decided at each month end, tranches 5 / 10 / 15 that many trading days into the month.
  brake (signal_search.py, V1): checked at every close. Once SPY's 20-day realised volatility (annualised) closes above
                  30%, all the core's stocks are held in cash; they are bought back once it closes below 20% again
                  (--no-brake turns it off). IEF and GLD are not affected.

    python data/update_data.py                                          # 1. bring the data up to date
    python v2/live.py target --capital 20000 [--stocks 30] [--whole] [--core-stocks 0.7]   # 2. target holdings
    python v2/live.py orders --holdings live/holdings.csv               # 3. trades from your holdings to the target
    python v2/live.py report [--holdings live/holdings.csv]             #    signals, market and drift, any time
    python v2/live.py check                                             #    self-check of the decision-day logic

When to run 1-3 (after the close, trade the next day): the last trading day of each month, and the 6th, 11th and 16th
trading day of each month (5 / 10 / 15 days in). Only one tranche changes on each of those days, plus small drift
trades. Also every other trading day: update the data and run `report`; if it says the brake changed, run `target`
and `orders` too. `report` prints the next dates. Everything is saved in live/ (git-ignored: it holds your positions).

holdings.csv: two columns, ticker,shares (fractional allowed). Use CASH for cash in dollars: the rotation's cash slot
is meant to sit in the broker's money-market sweep (or a T-bill ETF such as SGOV / BIL, entered as CASH).
Backtest reference (2005-2026, 4 tranches): with the volatility brake (signal_search.py, core 70/15/15) 9.4%/yr, max
drawdown 13.4%, Sharpe 0.98, 1 switch a year. Without it (core_search.py): core 70/15/15 9.9% / 19.6% / -5.4%; core 50/25/25
9.6% / 17.5%; core 100/0/0 10.4% / 25.0%.
"""
import argparse
import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import allocation as A
import signal_search as SG
import components as C
import defense as DF
import risk_model as RM
import strategy_search as SS
import swan_beta

LIVE = Path(__file__).resolve().parent.parent / "live"
U = "SP500"
ROT = ["SPY", "EFA", "EEM", "VNQ", "DBC", "IEF", "TLT", "GLD"]
CORE_STOCKS = 0.7                   # rest of the core: IEF and GLD, equally
OFFSETS = (0, 5, 10, 15)            # 0 = month end
ROTATION_SHARE = 0.5


def data_end():
    return swan_beta.load_prices(U).index[-1]


def etf_prices():
    """ETF closes up to the stock data's last day, so every signal is as of the same close."""
    etf = pd.read_csv(swan_beta.DATA / "prices/etf_close.csv.gz", index_col=0, parse_dates=True)
    return etf.loc[:data_end()]


def last_prices(tickers):
    close, etf = swan_beta.load_prices(U), etf_prices()
    return pd.Series({t: (etf[t] if t in etf else close[t]).dropna().iloc[-1] for t in tickers})


def month(t):
    return f"{t:%Y-%m}"


def sleeve(t, stocks=None, new=False):
    """This month's stock sleeve (weights summing to 1), built once per month and saved."""
    LIVE.mkdir(exist_ok=True)
    path = LIVE / f"sleeve_{month(t)}_{stocks or 'third'}.csv"
    if path.exists() and not new:
        return pd.read_csv(path, index_col=0).weight
    w = C.recommended(t, U, n=stocks, track="return")
    w.rename("weight").to_csv(path)
    return w


def rotation_returns():
    etf = etf_prices()
    r = etf[[a for a in ROT if a != "SPY"]].pct_change(fill_method=None)
    r["SPY"] = swan_beta.load_prices(U)["SPY"].reindex(etf.index).pct_change(fill_method=None)
    r["cash"] = DF.tbill_daily(etf.index)
    return r.iloc[1:]


def tranche_days(index):
    """Each tranche's latest decision day in `index` (trading days up to the last data day)."""
    last = index[-1]
    n = index.to_series().groupby(index.to_period("M")).cumcount().values
    ends = [d for d in A.month_ends(index) if d < last or d == pd.offsets.BMonthEnd().rollforward(last)]
    return {o: (max(ends) if o == 0 else index[n == o].max()) for o in OFFSETS}


def next_days(last):
    """Approximate next decision days (business days, US holidays not counted)."""
    out = {}
    for o in OFFSETS:
        for m in [last.to_period("M"), last.to_period("M") + 1]:
            days = pd.bdate_range(m.start_time, m.end_time)
            d = days[-1] if o == 0 else days[o]
            if d > last:
                out[o] = d
                break
    return out


def brake_state(r):
    """Daily brake state (True = core stocks in cash) up to the last close: signal_search.vol_brake on SPY."""
    st = SG.vol_brake(r["SPY"])
    vol = r["SPY"].iloc[-SG.VOL_WINDOW:]
    if vol.isna().any():   # a missing day would freeze the brake: stop rather than trade on a stale signal
        raise SystemExit(f"SPY returns missing on {list(vol.index[vol.isna()].date)}: run data/update_data.py again")
    return st


def tranches(r):
    """Per tranche: (decision day, rotation weights)."""
    f = SS.rotation(ROT)
    return {o: (d, f(d, r.loc[:d])) for o, d in tranche_days(r.index).items()}


def weights(t, stocks=None, new_sleeve=False, core_stocks=CORE_STOCKS, brake=True):
    s = sleeve(t, stocks, new_sleeve)
    r = rotation_returns()
    tr = tranches(r)
    held = 0.0 if brake and brake_state(r).iloc[-1] else 1.0   # share of the core's stocks held
    bonds_gold = (1 - core_stocks) / 2
    core = pd.concat([s * core_stocks * held, pd.Series({"IEF": bonds_gold, "GLD": bonds_gold,
                                                          "cash": core_stocks * (1 - held)})]) * (1 - ROTATION_SHARE)
    rot = sum(w for _, w in tr.values()) / len(tr) * ROTATION_SHARE
    w = core.add(rot, fill_value=0)
    return w[w > 0], s, tr


def target(capital, stocks=None, whole=False, new_sleeve=False, core_stocks=CORE_STOCKS, brake=True):
    t = data_end() + datetime.timedelta(days=1)
    w, s, tr = weights(t, stocks, new_sleeve, core_stocks, brake)
    w = w.drop("cash", errors="ignore")
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
    out.loc["CASH"] = [1 - w.sum(), 1.0, capital - out.value.sum(), capital - out.value.sum()]
    path = LIVE / f"target_{t:%Y-%m-%d}.csv"
    out.to_csv(path, float_format="%.6g")
    print(f"prices as of {data_end().date()} | sleeve {month(t)} ({len(s)} stocks) | saved {path}")
    print_tranches(tr)
    print(f"core stocks: {'CASH (brake on)' if brake and brake_state(rotation_returns()).iloc[-1] else 'held'}")
    etfs = out.drop(s.index, errors="ignore")
    print(f"\nstocks {out.weight.reindex(s.index).sum():.1%} of the money; ETFs and cash:")
    print(etfs.sort_values("value", ascending=False).round(4).to_string())
    return out


def print_tranches(tr):
    print("rotation tranches (each 12.5% of the money):")
    for o, (d, w) in tr.items():
        name = "month end" if o == 0 else f"day {o}"
        picks = ", ".join(f"{a} {v:.0%}" for a, v in w[w > 0].sort_values(ascending=False).items())
        print(f"  {name:9s} decided {d.date()}: {picks}")


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
    delta = tgt.shares.reindex(tickers).fillna(0) - have.reindex(tickers).fillna(0)
    o = pd.DataFrame({"shares": delta, "price": px, "value": delta * px})
    opening_or_closing = (tgt.shares.reindex(tickers).fillna(0) == 0) | (have.reindex(tickers).fillna(0) == 0)
    o = o[(o.value.abs() >= min_trade) | (opening_or_closing & (o.shares != 0))]   # min size only for adjustments
    o = o.sort_values("value")   # sells first, they fund the buys
    o["side"] = np.where(o.shares > 0, "BUY", "SELL")
    out = LIVE / f"orders_{path.stem.removeprefix('target_')}.csv"
    o.to_csv(out, float_format="%.6g")
    print(f"vs {path.name}: {len(o)} trades (adjustments under ${min_trade:.0f} skipped), sells ${-o.value[o.value < 0].sum():,.0f},"
          f" buys ${o.value[o.value > 0].sum():,.0f} | saved {out}")
    print(o.round(4).to_string())


def report(holdings_file=None, core_stocks=CORE_STOCKS, brake=True):
    t = data_end() + datetime.timedelta(days=1)
    close = swan_beta.load_prices(U)
    print(f"data through {data_end().date()}")

    oex = close["^OEX"].dropna()
    dd = swan_beta.drawdown(oex).iloc[-1]
    print(f"\nmarket: ^OEX is {dd:.1%} below its 1-year high" + ("  <-- swan territory (>= 15%)" if dd >= 0.15 else ""))
    rule = swan_beta.detect_swans(close)
    newer = rule[rule.trough > swan_beta.SWANS.trough.max() + datetime.timedelta(days=45)]
    if len(newer):
        print("new swan(s) found by the 15% rule, not yet in data/swans.csv (add once the trough is confirmed):")
        print(newer.to_string())

    r = rotation_returns()
    R = SS.Rules(r, r.index[-1])
    sig = pd.DataFrame({"score (3/6/12m)": {a: R.score(a) for a in ROT}, "above 10m avg": {a: R.up(a) for a in ROT}})
    print("\nrotation signals today (top 3 by score, held only if above the average):")
    print(sig.sort_values("score (3/6/12m)", ascending=False).to_string(float_format=lambda x: f"{x:+.1%}"))
    vol = r["SPY"].rolling(SG.VOL_WINDOW).std().iloc[-1] * np.sqrt(252)
    st = brake_state(r)
    print(f"\nbrake (checked every close): SPY {SG.VOL_WINDOW}-day volatility {vol:.0%} (on above {SG.VOL_ON:.0%}, "
          f"off below {SG.VOL_OFF:.0%}) -> core stocks {'in CASH' if st.iloc[-1] else 'held'}"
          + ("" if brake else " (but --no-brake: held)"))
    if st.iloc[-1] != st.iloc[-2]:
        print(f"  <-- the brake {'went ON' if st.iloc[-1] else 'went OFF'} at this close: run target and orders now")
    w, s, tr = weights(t, core_stocks=core_stocks, brake=brake)
    print_tranches(tr)

    sec = s.groupby(C.sectors()(s.index)).sum().sort_values(ascending=False)
    print(f"\nsleeve: {len(s)} stocks, top sectors " + ", ".join(f"{k} {v:.0%}" for k, v in sec.head(3).items()))
    x = RM.exposures(t, s.index, U)
    print("sleeve factor exposures (weighted): " + ", ".join(
        f"{f} {(x[f] * s).sum():+.2f}" for f in ["market", "size", "momentum", "volatility"]))
    groups = lambda v: pd.Series({"stocks": v.drop(ROT + ["cash", "CASH"], errors="ignore").sum(),
                                  **{a: v.get(a, 0.0) for a in ROT}, "cash": v.get("cash", 0.0) + v.get("CASH", 0.0)})
    tg = groups(w)
    print("target by holding: " + ", ".join(f"{k} {v:.1%}" for k, v in tg.items() if v > 0))

    if holdings_file:
        have = pd.read_csv(holdings_file, index_col=0).iloc[:, 0]
        px = last_prices(have.index.drop("CASH", errors="ignore"))
        value = pd.concat([have.drop("CASH", errors="ignore") * px, pd.Series({"CASH": have.get("CASH", 0.0)})])
        hg = groups(value / value.sum())
        print(f"your holdings (${value.sum():,.0f}): " + ", ".join(f"{k} {v:.1%}" for k, v in hg.items() if v > 0))
        drift = (hg - tg).abs()
        print(f"largest drift from target: {drift.idxmax()} {drift.max():.1%}"
              + ("  <-- check for a missed or wrong trade" if drift.max() > 0.05 else ""))

    print("\nnext decision days (run after the close, trade the next day): " + ", ".join(
        f"{'month end' if o == 0 else f'day {o}'} {d.date()}" for o, d in sorted(next_days(data_end()).items(), key=lambda x: x[1])))
    ref = {(0.7, True): "9.4%/yr, max drawdown 13.4%", (0.5, False): "9.6%/yr, max drawdown 17.5%",
           (0.7, False): "9.9%/yr, max drawdown 19.6%", (1.0, False): "10.4%/yr, max drawdown 25.0%"}
    print(f"backtest reference, core stocks {core_stocks:.0%}, brake {'on' if brake else 'off'}: "
          + ref.get((core_stocks, brake), "not backtested"))


def check():
    """The decision days found on historical data cut-offs match the backtest's rules."""
    idx = rotation_returns().index
    cut = lambda d: idx[idx <= d]
    feb = tranche_days(cut("2020-02-28"))   # a month end: tranche 0 decides that day
    assert feb[0] == pd.Timestamp("2020-02-28"), feb
    assert feb[5] == idx[(idx.to_period("M") == "2020-02")][5], feb
    mar = tranche_days(cut("2020-03-10"))   # 7th trading day of March: day-5 tranche decided on the 6th
    assert mar[0] == pd.Timestamp("2020-02-28") and mar[5] == idx[(idx.to_period("M") == "2020-03")][5], mar
    assert mar[10] == idx[(idx.to_period("M") == "2020-02")][10], mar
    print("decision days ok:", {o: d.date() for o, d in mar.items()})
    r = rotation_returns()
    st = brake_state(r)
    on = {d: bool(st.loc[d]) for d in ["2008-10-31", "2020-03-31", "2021-06-30", "2024-06-28"]}
    assert on["2008-10-31"] and on["2020-03-31"] and not on["2021-06-30"] and not on["2024-06-28"], on
    print("brake ok:", on)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["target", "orders", "report", "check"])
    ap.add_argument("--capital", type=float)
    ap.add_argument("--stocks", type=int, help="number of stocks in the sleeve (default: lowest third, ~160)")
    ap.add_argument("--whole", action="store_true", help="whole shares only (leftover stays as cash)")
    ap.add_argument("--new-sleeve", action="store_true", help="rebuild this month's sleeve")
    ap.add_argument("--holdings")
    ap.add_argument("--min-trade", type=float, default=50.0)
    ap.add_argument("--core-stocks", type=float, default=CORE_STOCKS,
                    help="stock share of the core (default 0.7; 0.5 and 1.0 also backtested), rest IEF / GLD equally")
    ap.add_argument("--no-brake", action="store_true", help="keep the core's stocks when volatility is high (D1 without the brake)")
    a = ap.parse_args()
    if a.command == "target":
        if not a.capital:
            raise SystemExit("--capital is required")
        target(a.capital, a.stocks, a.whole, a.new_sleeve, a.core_stocks, not a.no_brake)
    elif a.command == "orders":
        orders(a.holdings or LIVE / "holdings.csv", a.min_trade)
    elif a.command == "check":
        check()
    else:
        report(a.holdings, a.core_stocks, not a.no_brake)
