"""Sanity check of span predictions on the scored months against the emulated holdout (2026-11-23).

No truth exists for the scored spans, so this compares what the fix predicts there with what it predicts on the
emulated Dec-Feb spans, where the truth is known: the share of span targets predicted queued (speed <= 0.6 x free
speed), by window condition and by thirds of the span, next to the same share under plain interpolation and, on
the holdout, in the truth. A fix that behaves on March and April as on Dec-Feb predicts similar shares.

    python experiments/late/span_fix_check.py --holdout x --scored <dir with <split>_<panel>.npz>
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import span_eval as E  # noqa: E402
import span_lib as L  # noqa: E402

from tfb import t2_rules as R  # noqa: E402
from tfb.t1_interp import interp_time  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--holdout", default="x")
ap.add_argument("--scored", type=Path, required=True)
args = ap.parse_args()
rows = []


def add(period, panel, cond, pos, at, vc, **arrays):
    for c, cname in ((1, "onset"), (2, "ongoing")):
        for lo, hi, part in ((1, 6, "T+1..6"), (7, 12, "T+7..12"), (13, 18, "T+13..18")):
            m = at & ((cond == c) & (pos >= lo) & (pos <= hi))[:, :, None]
            if m.any():
                vcm = np.broadcast_to(vc, m.shape)[m]
                rows.append(dict(period=period, panel=panel, cond=cname, part=part, n=int(m.sum()),
                                 **{k: int((v[m] <= vcm).sum()) for k, v in arrays.items()}))


origins = L.emulated_origins()
L.install(None)
for p in L.T2_PANELS:
    vc = R.vcut(p)
    h = E.Holdout(p, origins[p])
    fs, _ = E.stored("B", args.holdout, p)
    si = interp_time(h.lay["speed"])[h.hold]
    add("Dec-Feb", p, h.cond, h.pos, h.target & h.span, vc, fix=fs, interp=si, truth=h.ts)
    for split, period in (("validation", "March"), ("private", "April")):
        a = L._ORIG_LOAD(p, split, "speed", "target")
        o = L.official_origins(p, split)
        span, _ = L.slot_masks(a["speed"].shape[0], o)
        cond, pos = np.zeros(span.shape, np.int8).reshape(-1), np.zeros(span.shape, np.int8).reshape(-1)
        for d, T, c in o:
            t = d * L.S + T
            cond[t + 1:t + 1 + L.SPAN] = 1 if c == "queue_onset" else 2
            pos[t + 1:t + 1 + L.SPAN] = np.arange(1, L.SPAN + 1)
        z = np.load(args.scored / f"{split}_{p}.npz")
        add(period, p, cond.reshape(span.shape), pos.reshape(span.shape), a["target"] & span[:, :, None], vc,
            fix=z["speed"], interp=interp_time(a["speed"]))
r = pd.DataFrame(rows)
g = r.groupby(["cond", "part", "period"])[["n", "fix", "interp", "truth"]].sum(min_count=1)
for c in ("fix", "interp", "truth"):
    g[c] = g[c] / g.n
pd.set_option("display.width", 200)
print("Share of span targets at or below the queue threshold:")
print(g.round(3).to_string())
t = r.groupby("period")[["n", "fix", "interp", "truth"]].sum(min_count=1)
print((t[["fix", "interp", "truth"]].div(t.n, axis=0)).round(3).to_string())
