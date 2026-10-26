"""The last signals on the list, plus a combination. Fixed in advance; long only, no leverage.

To vary the rotation daily, D1 is rebuilt here as half core (R1 70 / IEF 15 / GLD 15) + half rotation, each in
4 tranches, mixed 50 / 50 every day (the base row shows how close that is to the exact D1).
  R4  rotation with a daily exit: picks are still made on each tranche's decision day, but an asset held by a tranche
      is sold (cash) on any close below its 10-month average and bought back on any close above it
  E5  staged buy-back: when a brake is released the core's stocks return in 4 steps of 25%, every 5 trading days
      (selling stays immediate); for the live brake and for V1
  C1  V1 volatility brake + S1 stock stop with daily buy-back (signal_search.py)
Brakes as in signal_search.py (overlay on the core's 35% stock share).

    python v2/signal_more.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

import brake_timing as BT
import signal_search as SG
import strategy_robustness as SR
import strategy_search as S
import swan_beta

OUT = Path(__file__).parent / "output"
ROT = SR.ROT


def daily_exit_rotation(decisions):
    rot, state = S.rotation(ROT), {}
    decisions = set(decisions)

    def f(d, hist):
        if d in decisions or "w" not in state:
            state["w"] = rot(d, hist)
        w = state["w"].copy()
        R = S.Rules(hist, d)
        for a in [x for x in w.index if x != "cash" and w[x] > 0]:
            if not R.up(a):
                w["cash"] = w.get("cash", 0) + w[a]
                w[a] = 0
        return w
    return f


def tranched_rotation(r, daily):
    a = r[ROT + ["cash"]]
    plans = [(list(S.A.month_ends(a.index)), 1, ())] + [(SR.mid_month(a.index, o), 10 ** 6, SR.mid_month(a.index, o))
                                                         for o in (5, 10, 15)]
    out = []
    for decisions, every, dates in plans:
        if daily:
            fn = daily_exit_rotation(decisions)
            out.append(S.run_all(a, {"x": fn}, stock=[], every=10 ** 6, dates=list(a.index))["x"])
        else:
            out.append(S.run_all(a, {"x": S.rotation(ROT)}, stock=[], every=every, dates=dates)["x"])
    return pd.concat(out, axis=1).mean(axis=1)


def tranched_core(r, key):
    fns = {"x": S.fixed({key: .7, "IEF": .15, "GLD": .15})}
    copies = [S.run_all(r, fns, stock=[key])] + [S.run_all(r, fns, stock=[key], every=10 ** 6, dates=SR.mid_month(r.index, o))
                                                 for o in (5, 10, 15)]
    return pd.concat([c["x"] for c in copies], axis=1).mean(axis=1)


def staged(state, steps=4, gap=5):
    """Brake state (1 = core stocks out) with re-entry in `steps` equal parts, one every `gap` trading days."""
    out, level, since = [], 0.0, 0
    for on in state:
        if on:
            level, since = 1.0, 0
        elif level > 0:
            since += 1
            if since % gap == 0:
                level = max(0.0, level - 1 / steps)
        out.append(level)
    return pd.Series(out, index=state.index)


if __name__ == "__main__":
    r = S.load()
    close = swan_beta.load_prices("SP500")
    cash = r.cash
    w = {t: v["R1"] for t, v in pd.read_pickle(OUT / "search_sleeves_weights.pkl").items()}
    s1, _ = SG.stop_rebuy_sleeve(w, close, cash)
    r["S1"] = s1.reindex(r.index)
    core, core_s1 = tranched_core(r, "R1"), tranched_core(r, "S1")
    rot, rot4 = tranched_rotation(r, False), tranched_rotation(r, True)
    mix, mix4, mix_s1 = 0.5 * core + 0.5 * rot, 0.5 * core + 0.5 * rot4, 0.5 * core_s1 + 0.5 * rot
    exact = SG.d1(r, "R1")
    live = BT.sticky_state(r.SPY).astype(float)
    vol = r.SPY.rolling(20).std() * np.sqrt(252)
    v1 = SG.hysteresis(vol, 0.30, 0.20, below=False).astype(float)
    r1, s1r = r.R1.fillna(0), r.S1.fillna(0)
    runs = {
        "exact D1, no brake": exact, "base: D1 mixed daily, no brake": mix,
        "live brake": BT.overlay(mix, r1, cash, live), "V1 vol brake": BT.overlay(mix, r1, cash, v1),
        "R4 rotation daily exit, no brake": mix4, "R4 + live brake": BT.overlay(mix4, r1, cash, live),
        "R4 + V1": BT.overlay(mix4, r1, cash, v1),
        "E5 live brake, staged buy-back": BT.overlay(mix, r1, cash, staged(live)),
        "E5 V1, staged buy-back": BT.overlay(mix, r1, cash, staged(v1)),
        "C1 V1 + stock stop with daily buy-back": BT.overlay(mix_s1, s1r, cash, v1),
    }
    spy, c = r.SPY.loc[S.START:], cash.loc[S.START:]
    t = pd.DataFrame({k: S.evaluate(s.loc[S.START:], spy, c) for k, s in runs.items()}).T
    t.to_csv(OUT / "signal_more.csv")
    pd.set_option("display.width", 260)
    cols = ["CAGR", "Sharpe", "max DD", "worst year", "CAGR 2005-14", "CAGR 2015-26", "Sharpe 2005-14", "Sharpe 2015-26",
            "Sharpe ex-COVID"] + list(S.DF.CRISES)
    print(t[cols].astype(float).round(3).to_string())
    days_out = {k: (v.loc[S.START:] > 0).mean() for k, v in {"rotation cash share (R4)": rot4}.items()}
    for ref in ["live brake", "V1 vol brake"]:
        b = t.loc[ref]
        print(f"better than '{ref}' (Sharpe 05-14, 15-26, ex-COVID up; max DD not higher):",
              [k for k in runs if k not in (ref,) and all(t.loc[k, x] > b[x] for x in ["Sharpe 2005-14", "Sharpe 2015-26", "Sharpe ex-COVID"])
               and t.loc[k, "max DD"] <= b["max DD"]])
