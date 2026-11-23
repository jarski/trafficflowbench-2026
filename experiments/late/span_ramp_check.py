"""Do ramp flows carry information about the mainline inside a blank span? (2026-11-23)

Ramp flows are not blanked in the spans of the scored months (span_facts: observed share 0.80 inside and outside).
For every emulated Dec-Feb span and every link with an attached on- or off-ramp: the ramp flow's departure from its
own straight line across the span, against the mainline speed's and flow's departure from their interpolation
(the truth minus the interpolation of the emulated layer). Correlations pooled over span cells, and the share of
the squared interpolation error a per-panel linear fit on the ramp departure removes (in-sample, an upper bound
for a linear use of one ramp).

    python experiments/late/span_ramp_check.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import span_lib as L  # noqa: E402

from tfb.t1_interp import interp_time  # noqa: E402
from tfb.task1_io import meta  # noqa: E402

origins = L.emulated_origins()
for kind in ("OR", "FR"):
    X, YV, YQ, tot = [], [], [], []
    for p in L.T2_PANELS:
        m = meta(p)
        lay = L.emulated_layer(p, origins[p])
        a = L._ORIG_LOAD(p, "train", "true_speed", "true_flow", "ramp_flow", "ramp_pct", "ramp_eligible")
        si, fi = interp_time(lay["speed"]), interp_time(lay["flow"])
        lanes = np.clip(np.asarray(m["lanes"], float), 1, None)
        li, typ = np.asarray(m["ramp_link_index"]), np.asarray(m["ramp_type"])
        rf = np.where(a["ramp_eligible"] & (a["ramp_pct"] >= 75), a["ramp_flow"], np.nan)
        xs, yv, yq = [], [], []
        for d, T, c in origins[p]:
            if T + L.SPAN + 1 >= L.S:
                continue
            sl = slice(T + 1, T + 1 + L.SPAN)
            w = (np.arange(1, L.SPAN + 1) / (L.SPAN + 1))[:, None]
            for j in np.flatnonzero((typ == kind) & (li >= 0)):
                r = rf[d, :, j]
                if not (np.isfinite(r[T]) and np.isfinite(r[T + L.SPAN + 1])):
                    continue
                dev = r[sl] - (r[T] + w[:, 0] * (r[T + L.SPAN + 1] - r[T]))
                dv = a["true_speed"][d, sl, li[j]] - si[d, sl, li[j]]
                dq = (a["true_flow"][d, sl, li[j]] - fi[d, sl, li[j]]) / lanes[li[j]]
                ok = np.isfinite(dev) & np.isfinite(dv) & np.isfinite(dq)
                xs.append(dev[ok]); yv.append(dv[ok]); yq.append(dq[ok])
        x, v, q = np.concatenate(xs), np.concatenate(yv), np.concatenate(yq)
        bv, bq = (x * v).sum() / (x * x).sum(), (x * q).sum() / (x * x).sum()
        tot.append((len(x), (v ** 2).sum(), ((v - bv * x) ** 2).sum(), (q ** 2).sum(), ((q - bq * x) ** 2).sum()))
        print(f"  {kind} {p}: n {len(x):6d}  corr(ramp dev, speed dev) {np.corrcoef(x, v)[0, 1]:+.3f}  corr(ramp dev, flow dev) {np.corrcoef(x, q)[0, 1]:+.3f}"
              f"  speed RMSE {np.sqrt((v ** 2).mean()):.2f} -> {np.sqrt(((v - bv * x) ** 2).mean()):.2f}"
              f"  flow RMSE {np.sqrt((q ** 2).mean()):.1f} -> {np.sqrt(((q - bq * x) ** 2).mean()):.1f}")
        X.append(x); YV.append(v); YQ.append(q)
    t = np.array(tot).sum(axis=0)
    x, v, q = np.concatenate(X), np.concatenate(YV), np.concatenate(YQ)
    print(f"{kind} all panels: n {len(x)}, pooled corr speed {np.corrcoef(x, v)[0, 1]:+.3f}, flow {np.corrcoef(x, q)[0, 1]:+.3f}; "
          f"interpolation speed RMSE at ramp links {np.sqrt(t[1] / t[0]):.2f} -> {np.sqrt(t[2] / t[0]):.2f} with the per-panel linear fit; "
          f"flow {np.sqrt(t[3] / t[0]):.1f} -> {np.sqrt(t[4] / t[0]):.1f}")
