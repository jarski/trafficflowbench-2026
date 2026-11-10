"""Measurement noise: size, scaling and temporal correlation.

Adjacent links with no ramp attached carry the same true 5-minute flow (a
vehicle crosses a ~0.2 km link in seconds), so their observed flow difference
is almost pure measurement noise.
"""
import json
import numpy as np
from tfb.paths import DENSE, PANELS

def load(panel, split, names):
    return {n: np.load(DENSE / panel / split / f"{n}.npy", mmap_mode="r") for n in names}

print(f"{'panel':11s} pairs  sd(dq)/sqrt2  rel_sd  ac1(dq)  ac1(dq,night)  sd(dv)/sqrt2 freeflow  ac1(dv)")
for panel in PANELS:
    m = json.loads((DENSE / panel / "meta.json").read_text())
    a = load(panel, "train", ["true_flow", "true_speed", "pct"])
    ramp = np.array([bool(o) or bool(f) for o, f in zip(m["on_ramps"], m["off_ramps"])])
    ramp_link = np.zeros(len(m["links"]), bool)
    ramp_link[[i for i in m["ramp_link_index"] if i >= 0]] = True
    ramp |= ramp_link
    pairs = [i for i in range(len(ramp) - 1) if not ramp[i] and not ramp[i + 1]]
    if not pairs:
        print(panel, "no ramp-free pairs"); continue
    days = slice(0, 60)
    q = np.asarray(a["true_flow"][days]); v = np.asarray(a["true_speed"][days]); pct = np.asarray(a["pct"][days])
    i = np.array(pairs)
    ok = (pct[:, :, i] == 100) & (pct[:, :, i + 1] == 100)
    dq = q[:, :, i] - q[:, :, i + 1]
    qm = 0.5 * (q[:, :, i] + q[:, :, i + 1])
    sd = np.nanstd(dq[ok]) / np.sqrt(2)
    rel = np.nanstd((dq / qm)[ok]) / np.sqrt(2)
    # lag-1 autocorrelation over time of the pair difference (noise persistence)
    x = np.where(ok, dq, np.nan); x = x - np.nanmean(x, axis=1, keepdims=True)
    ac = np.nansum(x[:, 1:] * x[:, :-1]) / np.nansum(x[:, :-1] ** 2)
    xn = x[:, :60]  # 00:00-05:00
    acn = np.nansum(xn[:, 1:] * xn[:, :-1]) / np.nansum(xn[:, :-1] ** 2)
    free = ok & (v[:, :, i] > 90) & (v[:, :, i + 1] > 90)
    dv = v[:, :, i] - v[:, :, i + 1]
    y = np.where(free, dv, np.nan); y = y - np.nanmean(y, axis=1, keepdims=True)
    acv = np.nansum(y[:, 1:] * y[:, :-1]) / np.nansum(y[:, :-1] ** 2)
    print(f"{panel:11s} {len(pairs):5d} {sd:12.1f} {rel:8.3f} {ac:8.3f} {acn:13.3f} {np.nanstd(dv[free]) / np.sqrt(2):21.2f} {acv:8.3f}")
