"""Onset: saturation-based heuristics and a time-of-day prior, same replica as t2_onset_heuristics.py."""
import numpy as np
import pandas as pd
from tfb.paths import PANELS, TASK2_EXCLUDED
from tfb.task1_io import days, load, meta

def iou(p, t):
    u = (p | t).sum()
    return 1.0 if u == 0 else (p & t).sum() / u

def pick(score, k):
    p = np.zeros(score.shape, bool); p[np.argsort(-np.nan_to_num(score, nan=-9))[:k]] = True; return p

res = []
for panel in [p for p in PANELS if p not in TASK2_EXCLUDED]:
    m = meta(panel); vcut = 0.6 * np.array(m["free_speed_links_csv"]); cap = np.array(m["capacity_vph"])
    kcrit = np.array(m["critical_density"])
    a = load(panel, "train", "true_speed", "eligible", "speed", "flow", "occ")
    months = np.array([d[:7] for d in days(panel, "train")["days"]])
    q = (a["true_speed"] <= vcut) & a["eligible"]
    events = []
    for di in range(q.shape[0]):
        any_q = q[di].any(axis=1)
        for t0 in range(17, 288):
            if any_q[t0] and not any_q[t0 - 17:t0].any():
                events.append((di, t0)); break
    ev = np.array(events)
    for di, t0 in events:
        T = t0 - 6
        truth = q[di, t0]
        other = ev[(months[ev[:, 0]] != months[di])]
        near = other[np.abs(other[:, 1] - t0) <= 24]                      # onsets within +-2 h of day time
        prior = np.mean([q[d, t] for d, t in other], axis=0)
        prior_tod = np.mean([q[d, t] for d, t in near], axis=0) if len(near) >= 5 else prior
        def last(x):  # state at T, else T-5
            return np.where(np.isnan(x[di, T]), x[di, T - 1], x[di, T])
        fl, oc = last(a["flow"]), last(a["occ"])
        sat = fl / cap                                                     # own saturation
        sat_dn = np.append(sat[1:], np.nan)                                # downstream link's saturation
        dens = 285.0 * oc / kcrit                                          # density / critical density
        row = {"panel": panel}
        for k in (1, 2, 3):
            row[f"prior_{k}"] = iou(pick(prior, k), truth)
            row[f"priorTOD_{k}"] = iou(pick(prior_tod, k), truth)
        row["sat_1"] = iou(pick(sat, 1), truth)
        row["sat_dn_1"] = iou(pick(sat_dn, 1), truth)
        row["dens_1"] = iou(pick(dens, 1), truth)
        row["priorTOD_x_dens_2"] = iou(pick((prior_tod + 1e-3) * dens ** 4, 2), truth)
        res.append(row)
d = pd.DataFrame(res)
g = d.groupby("panel").mean(numeric_only=True)
print(g.round(3).to_string()); print("\nmean over panels:\n", g.mean().round(3).to_string())
