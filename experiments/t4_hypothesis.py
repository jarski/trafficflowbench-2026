"""Hypothesis H: the hidden Task 4 truth equals the organizer NNLS projection (lambda 0.05) of the released prior.
Under H, v1 scores S_ODME = 1. Predict the S_ODME of the probed variants and compare with leaderboard deltas."""
import numpy as np
import pandas as pd
from tfb import t4
from tfb.paths import PANELS, SUBMISSIONS

FAMILY = {p: p.rsplit("_", 1)[0] for p in PANELS}

def s_odme(f, fstar, b, A, c, dest):
    eps = 1e-9
    s_od = max(0.0, 1 - np.abs(f - fstar).sum() / max(fstar.sum(), eps))
    s_link = max(0.0, 1 - np.abs(A @ f - c).sum() / max(c.sum(), eps))
    dhat, dstar = np.abs(f - b).sum(), np.abs(fstar - b).sum()
    s_dev = float(np.exp(-abs(dhat / dstar - 1)))
    a_hat = pd.Series(f).groupby(dest).sum(); a_hat /= a_hat.sum()
    a_star = pd.Series(fstar).groupby(dest).sum(); a_star /= a_star.sum()
    s_attr = max(0.0, 1 - 0.5 * np.abs(a_hat - a_star).sum())
    return 0.45 * s_od + 0.25 * s_link + 0.15 * s_dev + 0.15 * s_attr

variants = {v: pd.read_csv(SUBMISSIONS / v / "odme.csv", dtype={"path_id": str}) for v in ("v1", "v2a", "v2b", "v2c")}
rows = []
for panel in PANELS:
    d = t4.load(panel, "validation")
    ids = d["paths"].path_id.to_numpy()
    fstar = variants["v1"].query("panel == @panel and departure_time == @d['departure_time']").set_index("path_id").path_flow.reindex(ids).to_numpy()
    dest = d["paths"].destination_zone.to_numpy()
    r = {"panel": panel}
    for v, df in variants.items():
        f = df.query("panel == @panel and departure_time == @d['departure_time']").set_index("path_id").path_flow.reindex(ids).to_numpy()
        r[v] = s_odme(f, fstar, d["b"], d["A"], d["c"], dest)
    rows.append(r)
r = pd.DataFrame(rows)
r["family"] = r.panel.map(FAMILY)
pred = r.groupby("family")[["v1", "v2a", "v2b", "v2c"]].mean().mean()
observed = {"v2a": (0.76712 - 0.84035) / 0.2, "v2b": (0.81010 - 0.84035) / 0.2, "v2c": (0.78575 - 0.84035) / 0.2}
print("predicted S_ODME under H:", pred.round(4).to_dict())
for v, o in observed.items():
    print(f"{v}: predicted delta vs v1 {pred[v] - pred['v1']:+.4f}   observed delta {o:+.4f}")
