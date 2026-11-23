"""The window-selector replica with the persistence cap tested on a time-filled queue state (2026-11-23).

`tfb.t2_rules.selector_windows` tests the documented cap (persistence IoU <= 0.9) on queue flags in which a
missing reading counts as "no queue". Here a missing cell carries its last value forward in time first.
Result on train: 34 of 40 official ongoing windows matched exactly and 37 within five minutes (14 and 16 with
the logged state); onset stays at 39 of 40. In 22 of the 40 official ongoing windows the origin five minutes
earlier fails the cap; there the state at T-5 has IoU 0.99 0.98 0.95 0.92 0.88 0.60 with the truth at T+5..T+30.

    python experiments/late/selector_check.py
"""
import json
import os

import numpy as np
import pandas as pd

from tfb import t2_rules as R
from tfb.paths import DENSE, RELEASE
from tfb.task1_io import load

H, Z = R.HISTORY_SLOTS, R.HORIZON_SLOTS


def tod(ts: str) -> int:
    return int(ts[11:13]) * 12 + int(ts[14:16]) // 5


def ffill_time(x: np.ndarray) -> np.ndarray:
    """Carry the last non-NaN value forward along the time axis, within each day."""
    out = x.copy()
    for t in range(1, x.shape[1]):
        m = np.isnan(out[:, t])
        out[:, t][m] = out[:, t - 1][m]
    return out


def selector(q, elig, qcap, per_condition=5, spacing=72):
    """t2_rules.selector_windows, with the cap tested on `qcap` instead of `q`."""
    S, chosen, count = q.shape[1], [], {"queue_onset": 0, "queue_ongoing": 0}
    for d in range(q.shape[0]):
        for T in range(H, S - Z):
            if min(count.values()) >= per_condition:
                return chosen
            hor = q[d, T + 1:T + 1 + Z]
            if not hor.any() or elig[d, T - H:T].mean() < 0.7:
                continue
            hist = q[d, T - H:T]
            if hist.sum() >= 2 and hist.sum(axis=0).max() >= 2:
                horc = qcap[d, T + 1:T + 1 + Z]
                if R.iou(np.broadcast_to(qcap[d, T - 1], horc.shape), horc) > 0.9:
                    continue
                c = "queue_ongoing"
            elif not hist.any() and not q[d, T].any():
                c = "queue_onset"
            else:
                continue
            t = d * S + T
            if count[c] >= per_condition or any(abs(t - (x[0] * S + x[1])) < spacing for x in chosen):
                continue
            chosen.append((d, T, c))
            count[c] += 1
    return chosen


def main() -> None:
    res = {k: dict(onset=0, ongoing=0, ongoing_within_5min=0) for k in ("logged", "filled")}
    rows = []
    for panel in sorted(os.listdir(RELEASE / "task2")):
        a = load(panel, "train", "true_speed", "eligible")
        days = json.loads((DENSE / panel / "train" / "days.json").read_text())["days"]
        dpos = {x: i for i, x in enumerate(days)}
        vc = R.vcut(panel)
        q = (a["true_speed"] <= vc) & a["eligible"]
        qf = ffill_time(a["true_speed"]) <= vc
        wi = pd.read_csv(RELEASE / "task2" / panel / "train" / "window_index.csv")
        off = {(dpos[r.date], tod(r.forecast_origin), r.condition) for r in wi.itertuples()}
        for name, qc in (("logged", q), ("filled", qf)):
            ch = selector(q, a["eligible"], qc)
            for c in ("onset", "ongoing"):
                res[name][c] += sum(1 for x in ch if x in off and x[2] == "queue_" + c)
            res[name]["ongoing_within_5min"] += sum(
                1 for x in ch if x[2] == "queue_ongoing"
                and any(o[2] == "queue_ongoing" and o[0] == x[0] and abs(o[1] - x[1]) <= 1 for o in off))
        for d, T, c in off:
            if c != "queue_ongoing":
                continue
            horp = qf[d, T:T + Z]
            rows.append(dict(cap_prev=R.iou(np.broadcast_to(qf[d, T - 2], horp.shape), horp),
                             **{f"s{k}": R.iou(qf[d, T - 1], qf[d, T + k]) for k in range(1, Z + 1)}))
    print("matches with the 40 + 40 official train windows:", res)
    df = pd.DataFrame(rows)
    b = df[df.cap_prev > 0.9]
    print("official ongoing windows whose origin five minutes earlier fails the cap:", len(b), "of", len(df))
    print("  IoU of the state at T-5 with the truth at T+5..T+30, those windows:", [round(b[f"s{k}"].mean(), 2) for k in range(1, 7)])
    print("  all official ongoing windows:", [round(df[f"s{k}"].mean(), 2) for k in range(1, 7)])


if __name__ == "__main__":
    main()
