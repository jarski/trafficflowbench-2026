"""Transfer test on the scored months themselves: pseudo-spans on released observations (2026-11-23).

The scored months are shifted against train, and the train holdout cannot show whether a span model transfers.
The released observations of March and April can: blank a further 18-slot all-link span in a month's masked layer
at an origin chosen as the selector would choose it, predict it, and compare with the observations that were
blanked. No hidden truth is needed. The same is done on the emulated Dec-Feb months with the same models (all
trained on Jun-Nov only), so the three periods are unseen months of the same kind for the models and differ only
in the shift.

    * queue flags come from the masked layer with each missing cell carrying its last value forward in time;
    * candidate origins: on every day and every full hour, the first origin that qualifies as an onset and the
      first that qualifies as an ongoing window (span_windows.hourly_windows on those flags), at least 6 h apart
      and clear of the month's real spans and their histories;
    * a pseudo-span differs from a real one in its history, which keeps the ordinary cell mask; that is the same
      in the three periods.

    python experiments/late/span_pseudo.py --plain a.pt[,b.pt] --span x.pt[,y.pt] [--more name=z.pt ...] [--days odd|even]
Prints, per period and window condition, the speed and per-lane flow RMSE over the blanked observed eligible
cells for interpolation, the production U-Net (--plain), the span-aware U-Net (--span) and any further models
(--more). --days keeps the pseudo-windows on odd or even days of the month only (span_adapt.py trains on the
even days of March and April and is tested on the odd ones).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import selector_check as SC  # noqa: E402
import span_lib as L  # noqa: E402
import span_unet_x as X  # noqa: E402
import span_windows as W  # noqa: E402

from tfb import t1_unet as U  # noqa: E402
from tfb import t2_rules as R  # noqa: E402
from tfb.t1_interp import interp_time  # noqa: E402
from tfb.task1_io import days, meta  # noqa: E402

H, Z = SC.H, SC.Z


def hourly_from(q: np.ndarray, elig: np.ndarray, qf: np.ndarray, day_ok: np.ndarray) -> list:
    """span_windows.hourly_windows on given flags: [(day, T, condition)]."""
    found = set()
    for d in np.flatnonzero(day_ok):
        ok = {"queue_onset": [], "queue_ongoing": []}
        for T in range(H, L.S - Z):
            hor = q[d, T + 1:T + 1 + Z]
            if not hor.any() or elig[d, T - H:T].mean() < 0.7:
                continue
            hist = q[d, T - H:T]
            if hist.sum() >= 2 and hist.sum(axis=0).max() >= 2:
                horc = qf[d, T + 1:T + 1 + Z]
                if R.iou(np.broadcast_to(qf[d, T - 1], horc.shape), horc) <= 0.9:
                    ok["queue_ongoing"].append(T)
            elif not hist.any() and not q[d, T].any():
                ok["queue_onset"].append(T)
        for c, ts in ok.items():
            ts = np.asarray(ts)
            for o in range(H, L.S - Z, 12):
                later = ts[ts >= o]
                if len(later):
                    found.add((int(d), int(later[0]), c))
    return sorted(found)


def pseudo_windows(panel: str, period: str, parity: str = "all") -> tuple:
    """(layer tuple of period_layer, candidate pseudo-windows [(day, T, condition)] clear of the real spans)."""
    lay = period_layer(panel, period)
    split, sp, fl, oc, elig, real, sel, _ = lay
    sf = SC.ffill_time(sp)
    vc = R.vcut(panel)
    q, qf = (sf <= vc) & elig, sf <= vc
    cand = hourly_from(q, elig, qf, sel)
    # clear of the real spans and histories: the pseudo-history and pseudo-span must not touch them
    cand = [(d, T, c) for d, T, c in cand if T + L.SPAN < L.S and not real[d, T - L.HIST:T + L.SPAN + 1].any()]
    if parity != "all":
        dom = np.array([int(x[8:10]) for x in days(panel, split)["days"]])
        cand = [(d, T, c) for d, T, c in cand if (dom[d] % 2 == 1) == (parity == "odd")]
    return lay, cand


def period_layer(panel: str, period: str):
    """(split, speed, flow, occ, eligible, real-span-or-history slots [day, 288], day selection, real origins)."""
    if period == "Dec-Feb":
        origins = L.emulated_origins()[panel]
        lay = L.emulated_layer(panel, origins)
        elig = L._ORIG_LOAD(panel, "train", "eligible")["eligible"]
        sel = np.isin(L.day_months(panel), L.HOLD_MONTHS)
        return "train", lay["speed"], lay["flow"], lay["occ"], elig, lay["span"] | lay["hist"], sel, origins
    split = {"March": "validation", "April": "private"}[period]
    a = L._ORIG_LOAD(panel, split, "speed", "flow", "occ", "eligible")
    origins = L.official_origins(panel, split)
    span, hist = L.slot_masks(a["speed"].shape[0], origins)
    return split, a["speed"], a["flow"], a["occ"], a["eligible"], span | hist, np.ones(a["speed"].shape[0], bool), origins


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plain", required=True)
    ap.add_argument("--span", required=True)
    ap.add_argument("--out", default="span_pseudo")
    ap.add_argument("--more", action="append", default=[], help="name=weights[,weights]: a further model to score")
    ap.add_argument("--days", choices=("all", "odd", "even"), default="all")
    args = ap.parse_args()
    U.THREADS = 1
    torch.set_num_threads(1)
    plain, pex = X.load_models(args.plain)
    spanm, sex = X.load_models(args.span)
    more = {m.split("=")[0]: X.load_models(m.split("=")[1]) for m in args.more}
    rng = np.random.default_rng(0)
    rows = []
    for period in ("Dec-Feb", "March", "April"):
        for p in L.T2_PANELS:
            (split, sp, fl, oc, elig, real, sel, _), cand = pseudo_windows(p, period, args.days)
            wins = [(d, T, c) for d in sorted({x[0] for x in cand}) for T, c in W.pick([(t, c) for dd, t, c in cand if dd == d], rng)]
            pspan, _ = L.slot_masks(sp.shape[0], wins)
            assert not (pspan & real).any()
            b = pspan[:, :, None]
            sp2, fl2, oc2 = np.where(b, np.nan, sp), np.where(b, np.nan, fl), np.where(b, np.nan, oc)
            dsel = pspan.any(axis=1)
            cond = X.cond_of_day(sp.shape[0], wins)
            lanes = np.clip(np.asarray(meta(p)["lanes"], float), 1, None)
            preds = {"interp": (interp_time(sp2), interp_time(fl2)),
                     "plain": X.predict(plain, pex, p, split, sp2, fl2, oc2, wins, dsel),
                     "span": X.predict(spanm, sex, p, split, sp2, fl2, oc2, wins, dsel)}
            for name, (mm, ex) in more.items():
                preds[name] = X.predict(mm, ex, p, split, sp2, fl2, oc2, wins, dsel)
            m = b & np.isfinite(sp) & np.isfinite(fl) & elig
            for c, cname in ((1, "onset"), (2, "ongoing")):
                mc = m & (cond == c)[:, :, None]
                for name, (ps, pf) in preds.items():
                    ev = (ps[mc] - sp[mc]).astype(float)
                    eq = ((pf - fl) / lanes)[mc].astype(float)
                    rows.append(dict(period=period, panel=p, cond=cname, method=name, n=int(mc.sum()), windows=sum(1 for w in wins if (w[2] == "queue_onset") == (c == 1)),
                                     sse_v=float(np.nansum(ev ** 2)), sse_q=float(np.nansum(eq ** 2)), nan=int(np.isnan(ev).sum())))
            print(f"  {period} {p}: {len(wins)} pseudo-windows on {int(dsel.sum())} days", flush=True)
    r = pd.DataFrame(rows)
    r.to_csv(L.LATE / f"{args.out}.csv", index=False)
    pd.set_option("display.width", 250)
    for by in (["period"], ["period", "cond"]):
        g = r.groupby(by + ["method"])[["n", "sse_v", "sse_q"]].sum()
        g["rmse_v"], g["rmse_q_lane"] = np.sqrt(g.sse_v / g.n), np.sqrt(g.sse_q / g.n)
        t = g[["rmse_v", "rmse_q_lane"]].unstack("method")
        for name in ["span"] + list(more):
            t[("gain_mse_v", name)] = 1 - (t[("rmse_v", name)] / t[("rmse_v", "interp")]) ** 2
            t[("gain_mse_q", name)] = 1 - (t[("rmse_q_lane", name)] / t[("rmse_q_lane", "interp")]) ** 2
        t[("cells", "n")] = g.n.unstack("method")["interp"]
        print(t.round(3).to_string(), "\n")
    w = r[r.method == "interp"].groupby(["period", "cond"]).windows.sum().unstack("cond")
    print("pseudo-windows:", w.to_dict())
    g = r.groupby(["period", "panel", "method"])[["n", "sse_v"]].sum()
    print(np.sqrt(g.sse_v / g.n).unstack("method").round(2).unstack("period").to_string())


if __name__ == "__main__":
    main()
