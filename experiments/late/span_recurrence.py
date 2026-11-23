"""How much of what happens inside a span does train recurrence predict? (2026-11-23)

The scored months are shifted against train, so a span model that leans on train recurrence would not transfer.
On the emulated Dec-Feb spans, three shifted-profile estimators (profile(t) + interpolated anomaly of the day):
    month    the month's own profile from its masked layer (same day type, the day left out)   [as span_simple.py]
    train    the Jun-Nov profile of the truth by weekday (tfb.t1_interp.weekday_profile_by_day)
    train2   the Jun-Nov profile by day type (weekday / weekend)
and the best constant blend with interpolation for each, in-sample (an upper bound for that estimator).

    python experiments/late/span_recurrence.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import span_lib as L  # noqa: E402
import span_simple as SS  # noqa: E402

from tfb.t1_interp import interp_time, weekday, weekday_profile_by_day  # noqa: E402
from tfb.task1_io import days, meta  # noqa: E402

origins = L.emulated_origins()
acc = {}
for p in L.T2_PANELS:
    lay = L.emulated_layer(p, origins[p])
    a = L._ORIG_LOAD(p, "train", "true_speed", "true_flow")
    info = days(p, "train")
    wd = weekday(info["days"])
    train_days = np.isin(L.day_months(p), L.TRAIN_MONTHS)
    sp = lay["speed"]
    si = interp_time(sp)
    ms, _ = SS.month_profiles(p, sp, lay["flow"])
    tp = weekday_profile_by_day(a["true_speed"], wd, train_days, wd, True)
    wk = (wd >= 5).astype(int)
    tp2 = weekday_profile_by_day(a["true_speed"], wk, train_days, wk, True)
    m = lay["target"] & lay["span"][:, :, None]
    truth = a["true_speed"][m].astype(float)
    base = si[m].astype(float)
    for name, prof in (("month", ms), ("train", tp), ("train2", tp2)):
        sh = prof + interp_time(sp - prof)
        sh = np.where(np.isfinite(sh), sh, si)[m].astype(float)
        x, y = sh - base, truth - base
        s = acc.setdefault(name, dict(n=0, xx=0.0, xy=0.0, yy=0.0, e1=0.0))
        s["n"] += len(y); s["xx"] += (x * x).sum(); s["xy"] += (x * y).sum(); s["yy"] += (y * y).sum(); s["e1"] += ((y - x) ** 2).sum()
rows = []
for name, s in acc.items():
    b = s["xy"] / s["xx"]
    rows.append(dict(estimator=name, n=s["n"], rmse_interp=np.sqrt(s["yy"] / s["n"]), rmse_shift=np.sqrt(s["e1"] / s["n"]),
                     best_weight=b, rmse_best_blend=np.sqrt((s["yy"] - b * s["xy"]) / s["n"]),
                     corr=s["xy"] / np.sqrt(s["xx"] * s["yy"])))
print(pd.DataFrame(rows).round(3).to_string(index=False))
