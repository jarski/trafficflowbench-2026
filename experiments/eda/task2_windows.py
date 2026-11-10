"""Official train Task 2 windows vs observed-speed queue labels (speed <= 0.6 * free speed)."""
import json
import numpy as np
import pandas as pd
from tfb.paths import RELEASE, PANELS, TASK2_EXCLUDED
from tfb.dense import tod_index
from tfb.task1_io import days, load, meta

rows = []
for panel in [p for p in PANELS if p not in TASK2_EXCLUDED]:
    m = meta(panel)
    vcut = 0.6 * np.array(m["free_speed_links_csv"])
    a = load(panel, "train", "true_speed", "eligible")
    dpos = {d: i for i, d in enumerate(days(panel, "train")["days"])}
    w = pd.read_csv(RELEASE / "task2" / panel / "train" / "window_index.csv")
    h = pd.read_parquet(RELEASE / "task2" / panel / "train" / "window_history.parquet", columns=["window_id", "timestamp"])
    hist_ts = h.groupby("window_id").timestamp.agg(["min", "max", "nunique"])
    for r in w.itertuples():
        di, T = dpos[r.date], int(tod_index(pd.Series([r.forecast_origin]))[0])
        q = (a["true_speed"][di] <= vcut) & a["eligible"][di]           # [288, L] observed queue flags
        hist = q[max(T - 11, 0):T + 1]                                  # 12 history slots ending at T
        hor = q[T + 1:T + 7]                                            # T+5 .. T+30
        per_step = hor.sum(axis=1)
        first = int(np.argmax(per_step > 0)) + 1 if per_step.any() else 0
        at_T = q[T]
        persist = np.broadcast_to(at_T, hor.shape)
        inter, union = (persist & hor).sum(), (persist | hor).sum()
        rows.append(dict(panel=panel, window=r.window_id[-3:], cond=r.condition, T=r.forecast_origin[11:16],
                         hist_first=h.loc[h.window_id == r.window_id, "timestamp"].min()[11:16],
                         hist_q=int(hist.sum()), hist_q_maxlink=int(hist.sum(axis=0).max()),
                         q_T=int(at_T.sum()), first_step=first, per_step=per_step.tolist(),
                         persist_iou=round(inter / union, 3) if union else 1.0))
d = pd.DataFrame(rows)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
print(d.to_string(index=False))
on = d[d.cond == "queue_onset"]
print("\nonset: first queued step distribution:", on.first_step.value_counts().to_dict())
og = d[d.cond == "queue_ongoing"]
print("ongoing: persistence IoU mean", og.persist_iou.mean().round(3), "| onset persistence IoU mean", on.persist_iou.mean().round(3))
