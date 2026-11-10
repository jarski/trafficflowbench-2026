"""Task 1: the U-Net family (tfb.t1_unet) averaged with the production pair on the Dec-Feb holdout.

Base = the cached holdout predictions of ours (1 seed, trained Jun-Nov) blended 0.75 / 0.25 with the public
feature set (cache/t1_holdout_base/<panel>.npz, written by the cache script of 2026-10-20). The U-Net is the
Jun-Nov model (cache/bench/t1_unet_holdout.pt or --model). Averages at the target cells with U-Net weight
0 / 0.2 / 0.33 / 0.5 / 1, then plausible_speed; exact scorer replicas; composite = 0.35 S_state + 0.15 S_physics,
family-averaged as on the leaderboard.

    python experiments/t1_unet_holdout.py [--model cache/bench/t1_unet_holdout.pt]
"""
import argparse

import numpy as np
import pandas as pd

from tfb import t1_model as t1
from tfb import t1_unet
from tfb.metrics import s_physics, s_state
from tfb.paths import CACHE, PANELS
from tfb.task1_io import load

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=str(CACHE / "bench" / "t1_unet_holdout.pt"), help="comma-separated: a seed average")
ap.add_argument("--weights", default="0,0.2,0.33,0.5,1")
args = ap.parse_args()
weights = [float(w) for w in args.weights.split(",")]
model = [t1_unet.load_model(m).to(t1_unet.DEVICE) for m in args.model.split(",")]

rows = []
for p in PANELS:
    base = np.load(CACHE / "t1_holdout_base" / f"{p}.npz")
    hold = base["hold"]
    target = load(p, "train", "target")["target"]
    sp = np.full(target.shape, np.nan, np.float32)
    fl = np.full(target.shape, np.nan, np.float32)
    sp[hold], fl[hold] = base["speed"], base["flow"]
    us, uf = t1_unet.predict_split(model, p, "train", hold)
    d, t, l = t1.cells(target, hold, None, None)
    assert np.isfinite(us[d, t, l]).all() and np.isfinite(uf[d, t, l]).all(), p
    for w in weights:
        s2, f2 = sp.copy(), fl.copy()
        s2[d, t, l] = (1 - w) * sp[d, t, l] + w * us[d, t, l]
        f2[d, t, l] = (1 - w) * fl[d, t, l] + w * uf[d, t, l]
        s2 = t1.plausible_speed(p, s2, f2)
        st, ph = s_state(p, "train", s2, f2, hold), s_physics(p, "train", s2, f2, hold)
        rows.append(dict(panel=p, w=w, S_state=st["S_state"], S_physics=ph["S_physics"],
                         rmse_v_R3=st["R3_rmse_speed"], rmse_q_R3=st["R3_rmse_flow_lane"]))
    print(f"  {p}: " + ", ".join(f"w={x['w']}: {0.35 * x['S_state'] + 0.15 * x['S_physics']:.5f}" for x in rows[-len(weights):]),
          flush=True)
r = pd.DataFrame(rows)
r["family"] = r.panel.map(t1.FAMILY)
fam = r.groupby(["w", "family"])[["S_state", "S_physics", "rmse_v_R3", "rmse_q_R3"]].mean().groupby("w").mean()
fam["composite"] = 0.35 * fam.S_state + 0.15 * fam.S_physics
fam["vs_base"] = fam.composite - fam.loc[0.0, "composite"]
print(fam.round(5).to_string())
