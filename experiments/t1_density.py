"""Task 1 D4: density-aware adjustment for Task 3, on the Dec-Feb holdout.

Task 3 checks vehicle conservation on N = length * q / v, so what matters there is density, while
Task 1 scores q and v separately by RMSE. A third LightGBM model predicts log density relative to
the interpolation (L1 loss, i.e. the median, which is what the absolute-residual score rewards).
The speed/flow predictions are then pulled toward that density:
    k* = k_model^(1-beta) * k_density^beta,   v' = v * (k_model/k*)^(1-alpha),   q' = q * (k*/k_model)^alpha
so q'/v' = k*. beta = 0 is the current submission. Scored with the exact Task 1/3 scorer replicas.
Second run (2026-10-09): flow-only moves (alpha = 1) with the factor capped to [1/cap, cap].

    python experiments/t1_density.py
"""
import lightgbm as lgb
import numpy as np
import pandas as pd

from tfb import t1_model as t1
from tfb.metrics import s_physics, s_state
from tfb.paths import PANELS
from tfb.task1_io import days

hold, train_days = {}, {}
for p in PANELS:
    months = np.array([x[:7] for x in days(p, "train")["days"]])
    hold[p] = np.isin(months, ["2030-12", "2031-01", "2031-02"])
    train_days[p] = ~hold[p]
df = t1.training_frame(PANELS, train_days, 600_000)
lanes = df.lanes.to_numpy()
si, fi = df.si.to_numpy(), df.fi_lane.to_numpy() * lanes
v_true, q_true = df.y_s.to_numpy() + si, df.y_f.to_numpy() * lanes + fi
df["y_k"] = np.log(np.clip(q_true, 1, None) / np.clip(v_true, 1, None)) - np.log(np.clip(fi, 1, None) / np.clip(si, 1, None))
models = t1.fit(df.drop(columns=["y_k"]))
feats = models["y_s"][1]
m_k = lgb.train({**t1.PARAMS, "objective": "l1"}, lgb.Dataset(df[feats], df.y_k), t1.ROUNDS).free_dataset()
del df
print("trained speed, flow and density models", flush=True)

grid = [(0.0, 1.0, None)] + [(0.5, 1.0, c) for c in (None, 1.1, 1.25, 1.5)] + [(0.75, 1.0, c) for c in (1.1, 1.25)]
rows = []
for i, p in enumerate(PANELS):
    sp, fl, pdata = t1.predict_panel(models, p, "train", i, hold[p], profile_days=train_days[p])
    d, t, l = t1.cells(pdata.target, hold[p], None, None)
    x = pdata.features(d, t, l)
    x["panel_code"] = i
    k_dens = np.clip(pdata.fi[d, t, l], 1, None) / np.clip(pdata.si[d, t, l], 1, None) * np.exp(m_k.predict(x[feats]))
    v0, q0 = sp[d, t, l].astype(float), fl[d, t, l].astype(float)
    k_model = q0 / v0
    for beta, alpha, cap in grid:
        factor = (k_dens / k_model) ** beta
        if cap is not None:   # bound the flow change: extreme density predictions are costly under RMSE
            factor = np.clip(factor, 1 / cap, cap)
        v = v0
        q = q0 * factor
        s2, f2 = sp.copy(), fl.copy()
        s2[d, t, l], f2[d, t, l] = v, q
        s2 = t1.plausible_speed(p, s2, f2)
        st, ph = s_state(p, "train", s2, f2, hold[p]), s_physics(p, "train", s2, f2, hold[p])
        rows.append(dict(panel=p, beta=beta, cap=str(cap), S_state=st["S_state"], S_physics=ph["S_physics"]))
    print(f"  {p}: done", flush=True)
r = pd.DataFrame(rows)
r["family"] = r.panel.map(t1.FAMILY)
fam = r.groupby(["beta", "cap", "family"])[["S_state", "S_physics"]].mean().groupby(["beta", "cap"]).mean()
fam["composite"] = 0.35 * fam.S_state + 0.15 * fam.S_physics
fam["vs_base"] = fam.composite - fam.loc[(0.0, "None"), "composite"]
print(fam.round(5).to_string())
