"""Holdout levels of the production blend as the project measured them, next to the emulated scored month.

    release   the released train layer and its own target set (no spans; the windows' histories carry targets):
              the cached pair (cache/t1_holdout_base) averaged with the 2-seed holdout U-Net, scored (a) on the
              three months pooled, as experiments/t1_unet_holdout.py does, and (b) per month
    A         the same predictions scored on the emulated month's target set (history cells are not targets)
    B         the emulated month: spans blank, histories unmasked (span_eval.py `blend`)

    python experiments/late/span_levels.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import span_eval as E  # noqa: E402
import span_lib as L  # noqa: E402

from tfb.paths import PANELS  # noqa: E402

origins = L.emulated_origins()
L.install(None)
rows = []
for p in PANELS:
    h = E.Holdout(p, origins[p])
    release_target = L._ORIG_LOAD(p, "train", "target")["target"]
    cs, cf = E.plain("A", "blendc", p, h)
    bs, bf = E.plain("B", "blend", p, h)
    for name, (s, f), target in (("release", (cs, cf), release_target), ("A", (cs, cf), h.target_full), ("B", (bs, bf), h.target_full)):
        s, f = h.full(np.nan_to_num(s)), h.full(np.nan_to_num(f))
        rows.append(dict(panel=p, kind=name, month="pooled", **L.score(p, s, f, h.hold, target)))
        for m in L.HOLD_MONTHS:
            rows.append(dict(panel=p, kind=name, month=m, **L.score(p, s, f, L.day_months(p) == m, target)))
r = pd.DataFrame(rows)
r.to_csv(L.LATE / "span_levels.csv", index=False)
cols = ["S_state", "S_physics", "S_FD", "S_LWR"]
out = []
for (kind, month), df in r.groupby(["kind", "month"]):
    out.append(dict(kind=kind, month=month, **L.macro(df, cols).to_dict()))
o = pd.DataFrame(out)
o["total"] = 0.35 * o.S_state + 0.15 * o.S_physics
print(o.round(5).to_string(index=False))
pm = o[o.month != "pooled"].groupby("kind")[cols + ["total"]].mean()
print("\nmean of the three months:")
print(pm.round(5).to_string())
