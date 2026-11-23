"""Simple estimators inside the emulated blank spans (no training): interpolation, the month's own profile shifted
to the values around the span, and blends of the two.

    interp       linear in time between the link's last value before and first value after the span
    shift        the month's time-of-day profile of the link (same day type, the day itself left out, observed
                 cells of the emulated layer: tfb.t1_model.PanelData.mprof_*) plus the interpolated anomaly
                 (value - profile) of the day: profile(t) + interp(x - profile)(t)
    a = 0..1     (1 - a) interp + a shift

Evaluated on the span targets of the emulated Dec-Feb months and, for choosing a, on the emulated spans of
Jun-Nov (same selector, per month).

    python experiments/late/span_simple.py
    python experiments/late/span_simple.py --store    # also store two families for span_eval.py (world B, Dec-Feb):
                                                      # `shift` (a = 1) and `simple` (speed a = 0.25, flow a = 0.75,
                                                      # the best values on the Jun-Nov spans)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import span_lib as L  # noqa: E402

from tfb.paths import PANELS  # noqa: E402
from tfb.t1_interp import group_profile_by_day, interp_time, weekday  # noqa: E402
from tfb.task1_io import days, meta  # noqa: E402

ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)


def month_profiles(panel: str, speed: np.ndarray, flow: np.ndarray):
    info = days(panel, "train")
    wd = weekday(info["days"])
    groups = np.array([f"{dd[:7]}|{w >= 5}" for dd, w in zip(info["days"], wd)])
    return group_profile_by_day(speed, groups), group_profile_by_day(flow, groups)


def estimators(panel: str, lay: dict) -> dict:
    """{name: (speed, flow)} on all train days of the emulated layer."""
    sp, fl = lay["speed"], lay["flow"]
    si, fi = interp_time(sp), interp_time(fl)
    ms, mf = month_profiles(panel, sp, fl)
    ss = ms + interp_time(sp - ms)
    sf = mf + interp_time(fl - mf)
    ss, sf = np.where(np.isfinite(ss), ss, si), np.where(np.isfinite(sf), sf, fi)
    return {"interp": (si, fi), "shift": (ss, sf)}


def store() -> None:
    origins = L.emulated_origins()
    for p in PANELS:
        lay = L.emulated_layer(p, origins[p])
        hold = np.isin(L.day_months(p), L.HOLD_MONTHS)
        est = estimators(p, lay)
        (si, fi), (ss, sf) = est["interp"], est["shift"]
        np.savez_compressed(L.LATE / "pred" / f"B_shift_{p}.npz", speed=ss[hold], flow=sf[hold])
        np.savez_compressed(L.LATE / "pred" / f"B_simple_{p}.npz", speed=(0.75 * si + 0.25 * ss)[hold], flow=(0.25 * fi + 0.75 * sf)[hold])


def main() -> None:
    if "--store" in sys.argv:
        store()
        return
    rows = []
    for period, months in (("Jun-Nov", L.TRAIN_MONTHS), ("Dec-Feb", L.HOLD_MONTHS)):
        origins = L.emulated_origins(months)
        for p in L.T2_PANELS:
            lay = L.emulated_layer(p, origins[p])
            a = L._ORIG_LOAD(p, "train", "true_speed", "true_flow")
            lanes = np.clip(np.asarray(meta(p)["lanes"], float), 1.0, None)
            est = estimators(p, lay)
            m = lay["target"] & lay["span"][:, :, None]
            ts, tf = a["true_speed"][m], a["true_flow"][m]
            ln = np.broadcast_to(lanes, m.shape)[m]
            (si, fi), (ss, sf) = est["interp"], est["shift"]
            for al in ALPHAS:
                ev = (1 - al) * si[m] + al * ss[m] - ts
                eq = ((1 - al) * fi[m] + al * sf[m] - tf) / ln
                rows.append(dict(period=period, panel=p, alpha=al, n=int(m.sum()), sse_v=float((ev.astype(float) ** 2).sum()),
                                 sse_q=float((eq.astype(float) ** 2).sum()), nan=int(np.isnan(ev).sum())))
    r = pd.DataFrame(rows)
    g = r.groupby(["period", "alpha"])[["n", "sse_v", "sse_q", "nan"]].sum()
    g["rmse_v"], g["rmse_q_lane"] = np.sqrt(g.sse_v / g.n), np.sqrt(g.sse_q / g.n)
    print(g[["n", "nan", "rmse_v", "rmse_q_lane"]].round(3).to_string())
    r["rmse_v"] = np.sqrt(r.sse_v / r.n)
    print(r[r.period == "Dec-Feb"].pivot(index="panel", columns="alpha", values="rmse_v").round(2).to_string())


if __name__ == "__main__":
    main()
