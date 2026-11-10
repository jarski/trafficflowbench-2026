"""Reproduce the organizer Task 2 window selector on train and compare with the official windows."""
import numpy as np
import pandas as pd
from tfb.paths import RELEASE, PANELS, TASK2_EXCLUDED
from tfb.dense import tod_index
from tfb.task1_io import days, load, meta
from tfb.t2_rules import vcut, iou

def candidates(q, elig, pers_slot):
    """All valid origins: (day, T, condition) in time order."""
    out = []
    for d in range(q.shape[0]):
        for T in range(12, 288 - 6):
            hor = q[d, T + 1:T + 7]
            if not hor.any():
                continue
            hist = q[d, T - 12:T]
            ongoing = hist.sum() >= 2 and hist.sum(axis=0).max() >= 2
            if elig[d, T - 12:T].mean() < 0.7:
                continue
            pers = np.broadcast_to(q[d, T + pers_slot], hor.shape)
            if iou(pers, hor) > 0.9:
                continue
            out.append((d, T, "queue_ongoing" if ongoing else "queue_onset"))
    return out

def select(cands, global_spacing):
    chosen, count = [], {"queue_onset": 0, "queue_ongoing": 0}
    for d, T, c in cands:
        if count[c] >= 5:
            continue
        t = d * 288 + T
        pool = chosen if global_spacing else [x for x in chosen if x[2] == c]
        if any(abs(t - (x[0] * 288 + x[1])) < 72 for x in pool):
            continue
        chosen.append((d, T, c)); count[c] += 1
    return sorted(chosen)

for pers_slot, label in ((-1, "persistence from T-5"), (0, "persistence from T")):
    hits = {True: 0, False: 0}; total = 0
    for panel in [p for p in PANELS if p not in TASK2_EXCLUDED]:
        a = load(panel, "train", "true_speed", "eligible")
        q = (a["true_speed"] <= vcut(panel)) & a["eligible"]
        dl = days(panel, "train")["days"]
        off = pd.read_csv(RELEASE / "task2" / panel / "train" / "window_index.csv")
        official = {(dl.index(r.date), int(tod_index(pd.Series([r.forecast_origin]))[0]), r.condition) for r in off.itertuples()}
        total += len(official)
        cands = candidates(q[:20], a["eligible"][:20], pers_slot)  # official windows sit in the first days
        for g in (True, False):
            sel = set(select(cands, g))
            hits[g] += len(sel & official)
            if panel == "D12_I5_N" and pers_slot == -1:
                print(f"  {panel} global={g}: replica {sorted(sel)}\n  official {sorted(official)}")
    print(f"{label}: exact matches  H1 global spacing {hits[True]}/{total}   H2 per-condition spacing {hits[False]}/{total}")
