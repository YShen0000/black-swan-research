"""Walk-forward test: predict each swan from the 5 swans before it, using only data available before it.

    python v2/walkforward.py      # prints the table, writes v2/output/walkforward.csv

Inputs are unified: Swan Beta and MarketDD (DD/T1/T2) computed from data/swans.csv + frozen prices.
Analyst targets (P_fwd) only exist for Swan9 (look-ahead: downloaded after the crash), Swan10D and Swan11,
so every swan runs without P_fwd (return = past-year trend), and those three also run with it.
"""
import datetime
from pathlib import Path

import pandas as pd

import mvo2
from swan_beta import SWANS, swan_betas

TARGETS = ["Swan5", "Swan6", "Swan7", "Swan8", "Swan9", "Swan10D", "Swan11"]
OPTDATA = {   # swan: (file, sheet, portfolio date = the date the targets were downloaded for)
    "Swan9": ("optdata/optdata_MVO2.xlsx", 1, datetime.datetime(2020, 2, 15)),
    "Swan10D": ("optdata/optdata_SP100_20220103_a.xlsx", 0, datetime.datetime(2022, 1, 3)),
    "Swan11": ("optdata/optdata_SP100_20241202.csv", 0, datetime.datetime(2024, 12, 2)),
}
LOOKBACK = datetime.timedelta(days=410)   # 1 year + ~60 trading days, as in the original notebooks
ACTUAL = swan_betas()


def params(target, with_pfwd):
    order = list(SWANS.index)
    k = order.index(target)
    w = SWANS.loc[target]
    file, sheet, end = OPTDATA.get(target, (None, 0, w.peak.to_pydatetime()))
    bench = mvo2.CLOSE[w.benchmark].loc[w.trough:]
    recovered = bench[bench >= mvo2.CLOSE[w.benchmark][w.peak]].index
    eval_end = (recovered[0] if len(recovered) else bench.index[-1]) + datetime.timedelta(days=1)
    return dict(
        beta_file=None, marketdd_file=None, optdata_file=file if with_pfwd else None, optdata_sheet=sheet,
        universe=mvo2.load_optdata(file, sheet).index if file else None,   # same universe with and without P_fwd
        start_swan=order[k - 5], end_swan=order[k - 1], start=end - LOOKBACK, end=end,
        start_swan_date=end, end_swan_date=eval_end, crash_peak=w.peak, crash_trough=w.trough,
        benchmark=w.benchmark, N=7, w1=0.17, min_weight=0.03, MA_days=30,
    )


def summarize(r, target):
    w, curve = r["weights"], r["curve"]
    return {
        "PR": r["PR"],
        "PR equal-wt preselection": r["PR_preselect_equal"],
        "full-cycle excess": (curve.portfolio_value.iloc[-1] - curve.oex.iloc[-1]) / 10000,   # portfolio date -> index back at peak
        "predicted beta": (w * r["stats"].beta[w.index]).sum(),
        "actual beta": (w * ACTUAL[target].reindex(w.index)).sum(),
        "n preselected": len(r["stats"]),
        "n held": len(w),
        "optimizer": r["optimizer"],
        "holdings": " ".join(f"{t}:{v:.2f}" for t, v in w.sort_values(ascending=False).items()),
    }


if __name__ == "__main__":
    rows = {}
    for target in TARGETS:
        for with_pfwd in ([False, True] if target in OPTDATA else [False]):
            rows[(target, "P_fwd" if with_pfwd else "no P_fwd")] = summarize(mvo2.run(params(target, with_pfwd)), target)
    same_window = {k: params("Swan9", True)[k] for k in ["end_swan_date", "crash_peak", "crash_trough"]}
    rows[("Swan9", "original input files")] = summarize(mvo2.run(dict(mvo2.SWAN9, **same_window)), "Swan9")
    table = pd.DataFrame(rows).T
    out = Path(__file__).parent / "output"
    out.mkdir(exist_ok=True)
    table.to_csv(out / "walkforward.csv")
    pd.set_option("display.width", 250, "display.max_colwidth", 70)
    num = ["PR", "PR equal-wt preselection", "full-cycle excess", "predicted beta", "actual beta"]
    table[num] = table[num].astype(float).round(3)
    print(table.drop(columns="holdings").to_string())
    print()
    print(table["holdings"].to_string())
