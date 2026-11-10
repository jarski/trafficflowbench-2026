"""Task 1: average with a second model family (tfb.t1_public: a public notebook's feature set, retrained here).

Dec-Feb holdout with the exact scorer replicas. Ours: the production configuration with 1 seed, trained on
Jun-Nov with self-supervised rows from the held-out months. Public-feature model: trained on Jun-Nov
synthetic masks. Averages of the two with public weight 0, 0.25, 0.5 and 1. Composite = 0.35 S_state +
0.15 S_physics, family-averaged as on the leaderboard. Probe v38 (50/50 with the 0.8714 notebook's Task 1)
gained +0.0011 public.

    python experiments/t1_public_holdout.py
"""
import numpy as np
import pandas as pd

from tfb import t1_model as t1
from tfb import t1_public
from tfb.metrics import s_physics, s_state
from tfb.paths import PANELS
from tfb.task1_io import days

t1.SEEDS = 1
hold, train_days = {}, {}
for p in PANELS:
    months = np.array([x[:7] for x in days(p, "train")["days"]])
    hold[p] = np.isin(months, ["2030-12", "2031-01", "2031-02"])
    train_days[p] = ~hold[p]
df = t1.training_frame(PANELS, train_days, 600_000)
rng = np.random.default_rng(1)
df = pd.concat([df] + [t1.pseudo_frame(p, "train", i, 0.15, 300_000, rng, hold[p], train_days[p])
                       for i, p in enumerate(PANELS)], ignore_index=True)
models = t1.fit(df)
del df
print("our models trained", flush=True)
pub = t1_public.fit(("2030-06", "2030-07", "2030-08", "2030-09", "2030-10", "2030-11"))
print("public-feature models trained", flush=True)

rows = []
for i, p in enumerate(PANELS):
    sp, fl, pdata = t1.predict_panel(models, p, "train", i, hold[p], profile_days=train_days[p])
    ps, pf = t1_public.predict(pub, p, "train", hold[p])
    d, t, l = t1.cells(pdata.target, hold[p], None, None)
    assert np.isfinite(ps[d, t, l]).all() and np.isfinite(pf[d, t, l]).all(), p
    for w in (0.0, 0.25, 0.5, 1.0):
        s2, f2 = sp.copy(), fl.copy()
        s2[d, t, l] = (1 - w) * sp[d, t, l] + w * ps[d, t, l]
        f2[d, t, l] = (1 - w) * fl[d, t, l] + w * pf[d, t, l]
        s2 = t1.plausible_speed(p, s2, f2)
        st, ph = s_state(p, "train", s2, f2, hold[p]), s_physics(p, "train", s2, f2, hold[p])
        rows.append(dict(panel=p, w=w, S_state=st["S_state"], S_physics=ph["S_physics"],
                         rmse_v_R3=st["R3_rmse_speed"], rmse_q_R3=st["R3_rmse_flow_lane"]))
    print(f"  {p}: " + ", ".join(f"w={x['w']}: {0.35 * x['S_state'] + 0.15 * x['S_physics']:.5f}" for x in rows[-4:]),
          flush=True)
r = pd.DataFrame(rows)
r["family"] = r.panel.map(t1.FAMILY)
fam = r.groupby(["w", "family"])[["S_state", "S_physics", "rmse_v_R3", "rmse_q_R3"]].mean().groupby("w").mean()
fam["composite"] = 0.35 * fam.S_state + 0.15 * fam.S_physics
fam["vs_ours"] = fam.composite - fam.loc[0.0, "composite"]
print(fam.round(5).to_string())
