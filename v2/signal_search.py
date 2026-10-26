"""Three more buy / sell signals against the live setup (D1 70/15/15, rotation in 4 tranches, daily sticky brake).
Fixed in advance; long only, no leverage. Every brake uses brake_timing.overlay: while on, the core's stocks (35% of
the money) earn T-bills, 0.3% per switch, decided at a close and effective the next day.

  S1   stock stop with daily buy-back, inside the R1 sleeve: a stock is sold (next close) once it closes 15% below its
       highest close since it was bought; while out it is bought back (next close) as soon as it closes above its
       50-day average. Money out of a stock earns T-bills. At each monthly rebuild every selected stock starts held.
       Shown with the live brake.
  W1   breadth brake: share of S&P500 members (point-in-time, month-start list) closing above their 200-day average;
       on below 40%, off above 50%
  W2   live brake OR W1
  V1   volatility brake: SPY's 20-day realised volatility (annualised); on above 30%, off below 20%
  V2   live brake OR V1
Adopted only if better than the live setup on Sharpe in 2005-14, 2015-26 and without COVID, with a max drawdown no
higher.

    python v2/signal_search.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

import brake_timing as BT
import components as C
import strategy_robustness as SR
import strategy_search as S
import swan_beta

OUT = Path(__file__).parent / "output"
U = "SP500"
COST = 0.003


def stop_rebuy_sleeve(weights, close, cash, stop=0.15, sma=50):
    """Daily returns of the monthly sleeve with per-stock trailing stop and buy-back above the `sma`-day average."""
    avg = close.rolling(sma).mean()
    dates = sorted(weights)
    out, stops, prev = [], 0, None
    for t, nxt in zip(dates, dates[1:] + [close.index[-1] + pd.Timedelta(days=1)]):
        w = weights[t]
        px = close.loc[t:nxt, w.index].ffill()
        p, ma = px.values, avg.loc[px.index, w.index].values
        c = cash.reindex(px.index).fillna(0).values
        tv = 1.0 if prev is None else w.sub(prev, fill_value=0).abs().sum()
        val = np.zeros(p.shape)
        val[0] = w.values * (1 - COST * tv)         # rebuild at t's close
        held = np.ones(p.shape[1], bool)
        pending = np.zeros(p.shape[1], int)         # +1 buy / -1 sell at the next close
        high = p[0].copy()
        for i in range(1, len(p)):
            ret = np.nan_to_num(np.where(held, p[i] / p[i - 1] - 1, c[i]))
            val[i] = val[i - 1] * (1 + ret)
            act = pending != 0
            if act.any():                           # trade at this close
                val[i, act] *= 1 - COST
                held[act] = pending[act] > 0
                high[act & held] = p[i, act & held]
                pending[:] = 0
            high = np.where(held, np.fmax(high, p[i]), high)
            sell = held & (p[i] <= (1 - stop) * high)
            buy = ~held & (p[i] > ma[i])
            stops += int(sell.sum())
            pending[sell], pending[buy] = -1, 1
        v = val.sum(axis=1)
        out.append(pd.Series(np.r_[v[1] - 1, v[2:] / v[1:-1] - 1], index=px.index[1:]))   # day 1 includes the cost
        last = pd.Series(val[-1], index=w.index)
        prev = last / last.sum()
    return pd.concat(out), stops


def breadth(close):
    """Daily share of point-in-time members closing above their 200-day average (members fixed per month)."""
    above = close > close.rolling(200, min_periods=200).mean()
    valid = close.rolling(200, min_periods=200).mean().notna()
    days = close.loc["2003-01-01":].index
    months = days.to_series().groupby(days.to_period("M")).first()
    out = []
    for t in months:
        mem = close.columns.intersection(C.members(t, U))
        block = days[days.to_period("M") == t.to_period("M")]
        a, v = above.loc[block, mem], valid.loc[block, mem]
        out.append(a.sum(axis=1) / v.sum(axis=1))
    return pd.concat(out)


def hysteresis(x, on, off, below=True):
    out, state = [], False
    for v in x:
        if np.isnan(v):
            out.append(state)
            continue
        state = (v < on if below else v > on) or (state and (v < off if below else v > off))
        out.append(state)
    return pd.Series(out, index=x.index)


def d1(r, key):
    rot = S.rotation(SR.ROT)
    fns = {"D1": lambda d, h: S.fixed({key: .7, "IEF": .15, "GLD": .15})(d, h).mul(0.5).add(rot(d, h).mul(0.5), fill_value=0)}
    copies = [S.run_all(r, fns, stock=[key])] + [S.run_all(r, fns, stock=[key], every=10 ** 6, dates=SR.mid_month(r.index, o))
                                                 for o in (5, 10, 15)]
    return pd.concat([c["D1"] for c in copies], axis=1).mean(axis=1)


if __name__ == "__main__":
    r = S.load()
    close = swan_beta.load_prices(U)
    w = {t: v["R1"] for t, v in pd.read_pickle(OUT / "search_sleeves_weights.pkl").items()}
    s1, stops = stop_rebuy_sleeve(w, close, r.cash)
    r["S1"] = s1.reindex(r.index)
    cash = r.cash
    live = BT.sticky_state(r.SPY).astype(float)
    br = breadth(close).reindex(r.index).ffill()
    vol = r.SPY.rolling(20).std() * np.sqrt(252)
    w1 = hysteresis(br, 0.40, 0.50).astype(float)
    v1 = hysteresis(vol, 0.30, 0.20, below=False).astype(float)
    base = d1(r, "R1")
    states = {"live: daily brake": live, "W1 breadth brake": w1, "W2 live OR breadth": np.maximum(live, w1),
              "V1 volatility brake": v1, "V2 live OR volatility": np.maximum(live, v1)}
    runs = {"D1, no brake": base}
    for k, st in states.items():
        runs[k] = BT.overlay(base, r.R1.fillna(0), cash, st)
    runs["S1 stop + daily buy-back, live brake"] = BT.overlay(d1(r, "S1"), r.S1.fillna(0), cash, live)
    spy, c = r.SPY.loc[S.START:], cash.loc[S.START:]
    t = pd.DataFrame({k: S.evaluate(s.loc[S.START:], spy, c) for k, s in runs.items()}).T
    t["switches/yr"] = pd.Series({k: v.loc[S.START:].diff().abs().sum() / 22 for k, v in states.items()})
    t.loc["S1 stop + daily buy-back, live brake", "switches/yr"] = t.loc["live: daily brake", "switches/yr"]
    t["time braked"] = pd.Series({k: v.loc[S.START:].mean() for k, v in states.items()})
    t.to_csv(OUT / "signal_search.csv")
    pd.set_option("display.width", 260)
    cols = ["CAGR", "Sharpe", "max DD", "worst year", "CAGR 2005-14", "CAGR 2015-26", "Sharpe 2005-14", "Sharpe 2015-26",
            "Sharpe ex-COVID", "switches/yr", "time braked"] + list(S.DF.CRISES)
    print(t[cols].astype(float).round(3).to_string())
    print(f"\nS1: {stops / 22:.0f} stock stops a year; sleeve alone:",
          S.evaluate(r.S1.loc[S.START:].dropna(), spy, c)[["CAGR", "Sharpe", "max DD"]].astype(float).round(3).to_dict(),
          "| R1 alone:", S.evaluate(r.R1.loc[S.START:].dropna(), spy, c)[["CAGR", "Sharpe", "max DD"]].astype(float).round(3).to_dict())
    print("breadth: mean %.0f%%, min %.0f%%; time below 40%%: %.0f%%" % (100 * br.loc[S.START:].mean(), 100 * br.loc[S.START:].min(),
                                                                      100 * (br.loc[S.START:] < 0.4).mean()))
    b = t.loc["live: daily brake"]
    print("adopt:", {k: bool(all(t.loc[k, x] > b[x] for x in ["Sharpe 2005-14", "Sharpe 2015-26", "Sharpe ex-COVID"])
                             and t.loc[k, "max DD"] <= b["max DD"]) for k in runs if k not in ("live: daily brake", "D1, no brake")})

    # robustness of V1: window, thresholds, one day late
    rows = {}
    for win in (10, 20, 40):
        v = r.SPY.rolling(win).std() * np.sqrt(252)
        for on, off in [(0.25, 0.15), (0.30, 0.20), (0.35, 0.25), (0.30, 0.25)]:
            st = hysteresis(v, on, off, below=False).astype(float)
            for lag in (1, 2):
                s = BT.overlay(base, r.R1.fillna(0), cash, st, lag=lag).loc[S.START:]
                e = S.evaluate(s, spy, c)
                rows[(win, f"{on:.0%}/{off:.0%}", "same close" if lag == 1 else "next close")] = {
                    **{x: e[x] for x in ["CAGR", "Sharpe", "max DD", "Sharpe 2005-14", "Sharpe 2015-26", "Sharpe ex-COVID",
                                         "COVID 2020-02→03", "2022 bear"]},
                    "switches/yr": st.loc[S.START:].diff().abs().sum() / 22}
    rb = pd.DataFrame(rows).T
    rb.to_csv(OUT / "signal_search_vol_robustness.csv")
    print("\nV1 robustness (window, on/off, fill):")
    print(rb.astype(float).round(3).to_string())
    beats = (rb["Sharpe"] > b["Sharpe"]) & (rb["max DD"] <= b["max DD"])
    print(f"beats the live brake on Sharpe and max DD in {int(beats.sum())} of {len(rb)} settings")
