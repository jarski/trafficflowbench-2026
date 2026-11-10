"""Task 2 onset: which links queue at T+30? Heuristics evaluated on replicated train onset events.

Replica: first time t0 on a day where some link is queued (observed speed <= 0.6*v_free, eligible)
after 17 queue-free slots (12 history + 5 horizon). T = t0 - 6 slots. Prior from other months only.
"""
import numpy as np
import pandas as pd
from tfb.paths import PANELS, TASK2_EXCLUDED
from tfb.task1_io import days, load, meta

def iou(p, t):
    u = (p | t).sum()
    return 1.0 if u == 0 else (p & t).sum() / u

res = []
for panel in [p for p in PANELS if p not in TASK2_EXCLUDED]:
    m = meta(panel); vcut = 0.6 * np.array(m["free_speed_links_csv"])
    a = load(panel, "train", "true_speed", "eligible", "speed")
    months = np.array([d[:7] for d in days(panel, "train")["days"]])
    q = (a["true_speed"] <= vcut) & a["eligible"]                       # [D, 288, L]
    events = []
    for di in range(q.shape[0]):
        any_q = q[di].any(axis=1)
        for t0 in range(17, 288):
            if any_q[t0] and not any_q[t0 - 17:t0].any():
                events.append((di, t0)); break                          # earliest onset of the day
    # prior: per-link share of being queued at onset, from other months
    for di, t0 in events:
        T = t0 - 6
        truth = q[di, t0]
        other = [(d, t) for d, t in events if months[d] != months[di]]
        prior = np.mean([q[d, t] for d, t in other], axis=0) if other else np.zeros(q.shape[2])
        sp = a["speed"][di, T]; sp = np.where(np.isnan(sp), a["speed"][di, T - 1], sp)  # state at T, else T-5
        ratio = sp / vcut
        cands = {
            "min_ratio_1": np.argsort(np.nan_to_num(ratio, nan=9))[:1],
            "min_ratio_2": np.argsort(np.nan_to_num(ratio, nan=9))[:2],
            "prior_top1": np.argsort(-prior)[:1],
            "prior_top2": np.argsort(-prior)[:2],
            "prior_x_ratio_1": np.argsort(-(prior + 1e-3) / np.nan_to_num(ratio, nan=9) ** 4)[:1],
            "prior_x_ratio_2": np.argsort(-(prior + 1e-3) / np.nan_to_num(ratio, nan=9) ** 4)[:2],
        }
        row = {"panel": panel, "n_true": int(truth.sum())}
        for k, idx in cands.items():
            p = np.zeros_like(truth); p[idx] = True; row[k] = iou(p, truth)
        res.append(row)
d = pd.DataFrame(res)
print(d.groupby("panel").mean(numeric_only=True).round(3).to_string())
print("\nmean over panels:\n", d.groupby("panel").mean(numeric_only=True).mean().round(3).to_string())
print("\nevents:", len(d))
