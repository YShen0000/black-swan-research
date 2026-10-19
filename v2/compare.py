"""Run the untouched original MVO2 notebook and v2 on the same frozen data, side by side.

    python v2/compare.py

The original notebook only gets the minimum to run offline: its yf.download is swapped for the frozen prices
(columns sorted like yfinance does), pip/quandl lines dropped, and KMeans n_init=10 (its Colab sklearn 1.2.2 default).
"""
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

import mvo2

REPO = Path(__file__).resolve().parent.parent
NOTEBOOK = REPO / "training/MVO/MVO2_0_TRAINING(run this code).ipynb"
INPUTS = ["SwanBeta/All_SwanBeta100_MVO2.xlsx", "MVO_Components/MarketDD.xlsx", "MVO_Components/trainingMarketdrop.csv",
          "optdata/optdata_MVO2.xlsx", "MVO_Components/allocation weight.xlsx"]
OFFLINE = f'''
import functools
from sklearn.cluster import KMeans as _KMeans
KMeans = functools.partial(_KMeans, n_init=10)
_PX = pd.read_csv({str(mvo2.DATA / "prices/close_SP100.csv.gz")!r}, index_col=0, parse_dates=True)
def _download(t, start, end, **kw):
    t = [t] if isinstance(t, str) else list(t)
    return pd.concat({{"Close": _PX.loc[start:end - pd.Timedelta(days=1)].reindex(columns=sorted(t))}}, axis=1)
yf.download = _download
'''


def run_original():
    work = Path(tempfile.mkdtemp())
    for f in INPUTS:
        shutil.copy(mvo2.DATA / f, work)
    cells = ["".join(c["source"]) for c in json.load(open(NOTEBOOK))["cells"] if c["cell_type"] == "code"]
    keep = lambda line: not line.strip().startswith(("!", "import quandl", "from pandas_datareader"))
    cells = ["\n".join(filter(keep, c.splitlines())) for c in cells]
    cells.insert(next(i for i, c in enumerate(cells) if "import yfinance" in c) + 1, OFFLINE)
    (work / "run.py").write_text("\n\n".join(cells))
    subprocess.run([sys.executable, "run.py"], cwd=work, check=True, capture_output=True)
    pr = {}
    for name, f in [("weight", "no_post_allocation"), ("equal", "equal_weight"),
                    ("adjusted proportion", "proportional_weight"), ("direct", "direct_weight")]:
        c = pd.read_csv(work / f"{f}_result.csv", index_col=0, parse_dates=True)
        t = c.loc[[mvo2.SWAN9["crash_peak"], mvo2.SWAN9["crash_trough"]]]
        drop = (1 - t.iloc[1] / t.iloc[0]).abs()
        pr[name] = 1 - drop.portfolio_value / drop.oex
    c = pd.read_csv(work / "Injected_result.csv", index_col=0, parse_dates=True).loc[[mvo2.SWAN9["crash_peak"], mvo2.SWAN9["crash_trough"]]]
    drop = (1 - c.iloc[1] / c.iloc[0]).abs()
    return pd.read_csv(work / "weights.csv", index_col=0).weight, 1 - drop.portfolio_value / drop.oex, pr


if __name__ == "__main__":
    w0, pr0, alloc0 = run_original()
    v2 = mvo2.run()
    nominal, mvo2.NOMINAL = mvo2.NOMINAL, mvo2.CLOSE   # v2 without the P_fwd nominal-price fix, to isolate its effect
    v2_adj = mvo2.run()
    mvo2.NOMINAL = nominal

    print("Swan9 (Swan4-8, S&P100, K_means N=7) -- weights")
    print(pd.DataFrame({"original": w0, "v2 w/o nominal fix": v2_adj["weights"], "v2": v2["weights"]}).fillna(0).round(4).to_string())
    print("\nPR (1 - portfolio drop / OEX drop, 2020-02-19 -> 2020-03-23)")
    rows = {"MVO weights (Injected_result)": (pr0, v2_adj["PR"], v2["PR"])}
    for k in alloc0:
        rows[f"post-allocation: {k}"] = (alloc0[k], v2_adj["PR_by_allocation"][k], v2["PR_by_allocation"][k])
    print(pd.DataFrame(rows, index=["original", "v2 w/o nominal fix", "v2"]).T.round(4).to_string())
