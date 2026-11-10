"""Task 1 error analysis for the report: where does the remaining error sit?

Current Task 1 configuration trained on Jun-Nov train days, scored on the Dec-Feb holdout targets.
RMSE of speed (km/h) and flow per lane (veh/h/lane) by mask regime, congestion state of the true
cell (free / transition / queued, by speed against 0.6 and 0.85 of free speed) and gap length (slots
from the target to its nearer observed anchor), for time interpolation and for the model.

    python experiments/t1_error_analysis.py
"""
import numpy as np
import pandas as pd

from tfb import t1_model as t1
from tfb.paths import CACHE, PANELS
from tfb.task1_io import days, load, meta

hold, train_days = {}, {}
for p in PANELS:
    months = np.array([x[:7] for x in days(p, "train")["days"]])
    hold[p] = np.isin(months, ["2030-12", "2031-01", "2031-02"])
    train_days[p] = ~hold[p]
models = t1.fit(t1.training_frame(PANELS, train_days, 600_000))
rows = []
for i, p in enumerate(PANELS):
    sp, fl, pdata = t1.predict_panel(models, p, "train", i, hold[p], profile_days=train_days[p])
    tr = load(p, "train", "true_speed", "true_flow", "eligible")
    d, t, l = t1.cells(pdata.target & tr["eligible"], hold[p], None, None)
    vf = np.asarray(meta(p)["free_speed_kmh"])[l]
    lanes = pdata.lanes[l]
    v_true, q_true = tr["true_speed"][d, t, l], tr["true_flow"][d, t, l]
    ratio = v_true / vf
    gap = np.minimum(t - pdata.last[d, t, l], pdata.nxt[d, t, l] - t)
    rows.append(pd.DataFrame({
        "panel": p, "regime": np.array(days(p, "train")["regime"])[d],
        "state": np.select([ratio <= 0.6, ratio <= 0.85], ["queued", "transition"], "free"),
        "gap": np.select([gap <= 1, gap <= 3], ["1", "2-3"], "4+"),
        "err_v_model": sp[d, t, l] - v_true, "err_v_interp": pdata.si[d, t, l] - v_true,
        "err_q_model": (fl[d, t, l] - q_true) / lanes, "err_q_interp": (pdata.fi[d, t, l] - q_true) / lanes}))
    print(f"  {p}: {len(d):,} cells", flush=True)
df = pd.concat(rows, ignore_index=True)
df.to_parquet(CACHE / "t1_holdout_errors.parquet", index=False)


def table(by):
    g = df.groupby(by)
    out = pd.DataFrame({
        "share": g.size() / len(df),
        "rmse_v_interp": g.err_v_interp.apply(lambda e: np.sqrt(np.mean(e ** 2))),
        "rmse_v_model": g.err_v_model.apply(lambda e: np.sqrt(np.mean(e ** 2))),
        "rmse_q_interp": g.err_q_interp.apply(lambda e: np.sqrt(np.mean(e ** 2))),
        "rmse_q_model": g.err_q_model.apply(lambda e: np.sqrt(np.mean(e ** 2))),
    })
    # share of the model's total squared speed error that sits in each group
    out["sq_err_share_v"] = g.err_v_model.apply(lambda e: np.sum(e ** 2)) / np.sum(df.err_v_model ** 2)
    return out.round(3)


for by in ("regime", "state", "gap", ["state", "regime"]):
    print(f"\nby {by}:")
    print(table(by).to_string())
