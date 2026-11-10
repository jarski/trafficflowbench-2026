"""Does a split's own prior predict that split's counts better than other months' priors or a gravity fit?"""
import numpy as np
import pandas as pd
from tfb.paths import RELEASE, PANELS

SPLITS = ("train", "validation", "private")

def load_panel(panel):
    net = RELEASE / "corridors" / panel / "network"
    paths = pd.read_csv(net / "path_set.csv", dtype=str)
    inc = pd.read_csv(net / "path_link_incidence.csv", dtype=str)
    pid = {p: i for i, p in enumerate(paths.path_id)}
    out = {}
    for s in SPLITS:
        d = RELEASE / "task4" / panel / s
        c = pd.read_csv(d / "synthetic_link_counts.csv", dtype={"link_id": str})
        lid = {l: i for i, l in enumerate(c.link_id)}
        A = np.zeros((len(c), len(paths)))
        sub = inc[inc.link_id.isin(lid)]
        A[sub.link_id.map(lid).to_numpy(), sub.path_id.map(pid).to_numpy()] = 1.0
        b = pd.read_csv(d / "synthetic_weak_prior.csv", dtype={"path_id": str}).set_index("path_id").path_flow.reindex(paths.path_id).to_numpy()
        out[s] = (A, c["count"].to_numpy(), b)
    return paths, out

def gravity_fit(paths, logb):
    zo, zd = paths.origin_zone.to_numpy(), paths.destination_zone.to_numpy()
    gap = paths.destination_zone.str[1:].astype(int).to_numpy() - paths.origin_zone.str[1:].astype(int).to_numpy()
    X = pd.get_dummies(pd.DataFrame({"o": zo, "d": zd}), drop_first=True).astype(float)
    X["lgap"], X["gap"], X["c"] = np.log(gap), gap, 1.0
    coef, *_ = np.linalg.lstsq(X.to_numpy(), logb, rcond=None)
    return np.exp(X.to_numpy() @ coef)

def scaled_err(A, c, f):
    """Relative L1 count error after the best global scale (median ratio)."""
    pred = A @ f
    alpha = np.median(c / pred)
    return np.abs(alpha * pred - c).sum() / c.sum()

rows = []
for panel in PANELS:
    paths, data = load_panel(panel)
    logs = {s: np.log(data[s][2]) for s in SPLITS}
    g_all = gravity_fit(paths, np.mean([logs[s] for s in SPLITS], axis=0))
    geo_all = np.exp(np.mean([logs[s] for s in SPLITS], axis=0))
    for s in SPLITS:
        A, c, b = data[s]
        others = [x for x in SPLITS if x != s]
        rows.append(dict(panel=panel, split=s,
                         own_prior=scaled_err(A, c, b),
                         other_prior=np.mean([scaled_err(A, c, data[x][2]) for x in others]),
                         own_gravity=scaled_err(A, c, gravity_fit(paths, logs[s])),
                         all_gravity=scaled_err(A, c, g_all),
                         geomean_3=scaled_err(A, c, geo_all)))
d = pd.DataFrame(rows)
print(d.groupby("panel").mean(numeric_only=True).round(3).to_string())
print("\nmean:", d.mean(numeric_only=True).round(4).to_dict())
