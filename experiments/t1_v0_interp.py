"""Task 1 v0: historical mean vs time interpolation vs a simple space-time blend, all panels (train)."""
import json
import numpy as np
from tfb.paths import PANELS
from tfb.task1_io import days, load, meta
from tfb.metrics import s_physics, s_state

FAMILY = {p: p.rsplit("_", 1)[0] for p in PANELS}

def interp_time(x):
    D, T, L = x.shape
    valid = ~np.isnan(x)
    idx = np.broadcast_to(np.arange(T)[None, :, None], x.shape)
    last = np.maximum.accumulate(np.where(valid, idx, -1), axis=1)
    nxt = np.minimum.accumulate(np.where(valid, idx, T)[:, ::-1], axis=1)[:, ::-1]
    xl = np.take_along_axis(x, np.clip(last, 0, T - 1), axis=1)
    xn = np.take_along_axis(x, np.clip(nxt, 0, T - 1), axis=1)
    w = (idx - last) / np.maximum(nxt - last, 1)
    out = np.where(last < 0, xn, np.where(nxt >= T, xl, xl + w * (xn - xl)))
    return np.where(valid, x, out)

def profile(x, weekdays, month, months):
    """Leave-month-out weekday x time-of-day mean per link."""
    prof = np.full((7,) + x.shape[1:], np.nan, np.float32)
    keep = months != month
    for wd in range(7):
        sel = keep & (weekdays == wd)
        prof[wd] = np.nanmean(x[sel], axis=0)
    return prof[weekdays]

def spatial(x):
    up = np.pad(x, ((0, 0), (0, 0), (1, 0)), constant_values=np.nan)[:, :, :-1]
    dn = np.pad(x, ((0, 0), (0, 0), (0, 1)), constant_values=np.nan)[:, :, 1:]
    return np.nanmean(np.stack([up, dn]), axis=0)

rows = []
for panel in PANELS:
    a = load(panel, "train", "speed", "flow", "eligible", "true_speed", "true_flow")
    info = days(panel, "train")
    dates = np.array(info["days"], dtype="datetime64[D]")
    weekdays = ((dates.astype(int) + 3) % 7)  # 1970-01-01 was a Thursday -> Monday=0
    months = np.array([d[:7] for d in info["days"]])
    res = {}
    for ch in ("speed", "flow"):
        x = a[ch]
        # inputs: every published (non-blank) value, eligible or not
        prof = np.empty_like(x)
        for mo in np.unique(months):
            sel = months == mo
            prof[sel] = profile(a[f"true_{ch}"] * 0 + x, weekdays, mo, months)[sel]  # masked layer only
        t = interp_time(x)
        s = spatial(np.where(np.isnan(x), np.nan, x))
        res[ch] = {"hist": prof, "time": np.where(np.isnan(t), prof, t),
                   "blend": np.where(np.isnan(t), prof, np.where(np.isnan(s), t, 0.6 * t + 0.4 * s))}
    for m in ("hist", "time", "blend"):
        sp, fl = res["speed"][m], res["flow"][m]
        st, ph = s_state(panel, "train", sp, fl), s_physics(panel, "train", sp, fl)
        rows.append({"panel": panel, "method": m, "S_state": st["S_state"], "S_physics": ph["S_physics"],
                     "rmse_v_R3": st["R3_rmse_speed"], "rmse_q_R3": st["R3_rmse_flow_lane"],
                     "LWR": [round(ph[f"{r}_S_LWR"], 3) for r in ("R1", "R2", "R3")]})
        print(rows[-1], flush=True)

import pandas as pd
df = pd.DataFrame(rows)
df["family"] = df.panel.map(FAMILY)
fam = df.groupby(["method", "family"])[["S_state", "S_physics"]].mean().groupby("method").mean()
print(fam.round(4))
