"""Ongoing dissipation physics on the train split.

1. Does the synthetic generator follow its own triangular fundamental diagram? For queued cells the
   density should sit on the congested branch at the observed flow (k = k_jam - q (k_jam - k_crit) / cap),
   for free cells on the free branch (k = q / v_free).
2. How well do kinematic-wave estimates at T predict the queue tail over T+5..T+30, against persistence
   and the tail's own recent motion?
3. Which queued-at-T cells clear in the horizon: tail side, head side or interior, and how far above
   the queue threshold do they end up?

    python experiments/eda/t2_dissipation.py [--panels D7_I10_W,D7_I10_E] [--max-windows 1500]
"""
import argparse

import numpy as np
import pandas as pd

from tfb import t2_ongoing as og
from tfb.t2_features import K_OCC
from tfb.t2_onset import T2_PANELS
from tfb.t2_rules import vcut
from tfb.task1_io import load, meta

DT = 5 / 60
NEAR = 12


def fd_check(panel: str, a: dict, q: np.ndarray) -> dict:
    m = meta(panel)
    cap, kc, kj, vf = (np.asarray(m[k], float) for k in ("capacity_vph", "critical_density", "k_jam", "free_speed_kmh"))
    k_obs = a["true_occ"] * K_OCC
    flow = a["true_flow"]
    el = a["eligible"] & np.isfinite(k_obs) & np.isfinite(flow)
    k_cong = kj - flow * (kj - kc) / cap
    k_free = flow / vf
    qd, fr = el & q, el & ~q
    rel_q = ((k_obs - k_cong) / kc)[qd]
    rel_f = ((k_obs - k_free) / kc)[fr]
    # q/k identity against the released speed
    with np.errstate(invalid="ignore", divide="ignore"):
        v_id = flow / k_obs
    dv = (v_id - a["true_speed"])[el & (k_obs > 1)]
    return dict(panel=panel, n_queued=int(qd.sum()), q_med=np.nanmedian(rel_q), q_iqr=np.nanpercentile(rel_q, 75) - np.nanpercentile(rel_q, 25),
                f_med=np.nanmedian(rel_f), f_iqr=np.nanpercentile(rel_f, 75) - np.nanpercentile(rel_f, 25),
                v_id_med=np.nanmedian(dv), v_id_iqr=np.nanpercentile(dv, 75) - np.nanpercentile(dv, 25))


def main_segment(flags: np.ndarray) -> tuple[int, int] | None:
    segs = og._segment_list(flags)
    if not segs:
        return None
    return max(segs, key=lambda s: (s[1] - s[0], s[1]))


def tail_in(flags: np.ndarray, lo: int, hi: int) -> int | None:
    idx = np.flatnonzero(flags[lo:hi + 1])
    return int(idx[0] + lo) if idx.size else None


def shock(q_a, v_a, q_q, k_q):
    """Tail shockwave speed (km/h along travel): (q_a - q_q) / (k_a - k_q); NaN when undefined."""
    with np.errstate(invalid="ignore", divide="ignore"):
        dk = q_a / v_a - k_q
        return np.clip((q_a - q_q) / dk, -60, 60) if np.isfinite(dk) and abs(dk) > 1e-6 else np.nan


def window_record(panel: str, a: dict, q: np.ndarray, d: int, T: int, m: dict) -> tuple[dict | None, list]:
    cap, kc, kj, vf = (np.asarray(m[k], float) for k in ("capacity_vph", "critical_density", "k_jam", "free_speed_kmh"))
    lengths = np.asarray(m["length_km"], float)
    pos = np.cumsum(lengths) - lengths / 2
    vc = vcut(panel)
    n = len(cap)
    sp, fl, oc = a["true_speed"][d], a["true_flow"][d], a["true_occ"][d]
    flags_T = og._bridge(q[d, T], np.isnan(sp[T]))
    seg = main_segment(flags_T)
    if seg is None:
        return None, []
    s0, e0 = seg
    lo, hi = max(s0 - NEAR, 0), min(e0 + 2, n - 1)
    # states at T (truth; the production features see T-5 complete and T imputed)
    up3 = np.arange(max(s0 - 3, 0), s0)
    q3 = np.arange(s0, min(s0 + 3, e0 + 1))
    with np.errstate(invalid="ignore"):
        q_a = np.nanmean(fl[T, up3]) if up3.size else np.nan
        v_a = np.nanmean(sp[T, up3]) if up3.size else np.nan
        q_q3, k_q3 = np.nanmean(fl[T, q3]), np.nanmean(oc[T, q3]) * K_OCC
        qall = np.arange(s0, e0 + 1)
        q_qm, k_qm = np.nanmean(fl[T, qall]), np.nanmean(oc[T, qall]) * K_OCC
        k_fd3 = np.nanmean(kj[q3] - fl[T, q3] * (kj[q3] - kc[q3]) / cap[q3])
        # arriving state averaged over the last 3 slots
        q_a3 = np.nanmean(fl[T - 2:T + 1][:, up3]) if up3.size else np.nan
        v_a3 = np.nanmean(sp[T - 2:T + 1][:, up3]) if up3.size else np.nan
        q_q33 = np.nanmean(fl[T - 2:T + 1][:, q3])
        k_q33 = np.nanmean(oc[T - 2:T + 1][:, q3]) * K_OCC
        q_a30 = np.nanmean(fl[T - 6, up3]) if up3.size else np.nan
    w_cur = shock(q_a, v_a, q_q3, k_q3)              # production w_tail
    w_fd = shock(q_a, v_a, q_q3, k_fd3)              # FD density instead of the measured one
    w_avg = shock(q_a, v_a, q_qm, k_qm)              # whole-queue state
    w_t3 = shock(q_a3, v_a3, q_q33, k_q33)           # 15-min averaged states
    # tail motion over the last 30 min (truth history)
    tail_m30 = tail_in(og._bridge(q[d, T - 6], np.isnan(sp[T - 6])), lo, hi)
    w_emp = (pos[s0] - pos[tail_m30]) / (6 * DT) if tail_m30 is not None else 0.0
    slope = (q_a - q_a30) / (6 * DT) if np.isfinite(q_a30) else 0.0   # vph per hour
    rec = dict(panel=panel, d=d, T=T, s0=s0, e0=e0, seg_len=e0 - s0 + 1, q_a=q_a, q_q=q_q3, k_q=k_q3, k_fd=k_fd3,
               w_cur=w_cur, w_fd=w_fd, w_avg=w_avg, w_t3=w_t3, w_emp=w_emp, slope=slope)
    cells = []
    for h in range(1, 7):
        flags_h = q[d, T + h]
        t_h = tail_in(flags_h, lo, hi)
        rec[f"d_true{h}"] = (pos[t_h] - pos[s0]) if t_h is not None else (pos[e0] - pos[s0] + lengths[e0])
        rec[f"gone{h}"] = t_h is None
        # trend-adjusted shock: arriving flow drifts with its 30-min slope
        if np.isfinite(w_fd):
            wsum = 0.0
            for s in range(1, h + 1):
                wsum += shock(q_a + slope * s * DT, v_a, q_q3, k_fd3) * DT
            rec[f"d_trend{h}"] = wsum
        else:
            rec[f"d_trend{h}"] = np.nan
        # queued-at-T cells of the main segment that are free at T+h
        with np.errstate(invalid="ignore"):
            for i in range(s0, e0 + 1):
                if a["eligible"][d, T + h, i] and not flags_h[i]:
                    cells.append(dict(panel=panel, h=h, tail_d=i - s0, head_d=e0 - i, seg_len=e0 - s0 + 1,
                                      margin_T=sp[T, i] / vc[i], margin_h=sp[T + h, i] / vc[i],
                                      requeue=bool(q[d, T + h + 1:T + 7, i].any()) if h < 6 else False))
    return rec, cells


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panels", default=",".join(T2_PANELS))
    ap.add_argument("--max-windows", type=int, default=1500)
    args = ap.parse_args()
    fd, recs, cells = [], [], []
    for panel in args.panels.split(","):
        a = load(panel, "train", "true_speed", "true_flow", "true_occ", "eligible")
        q = (a["true_speed"] <= vcut(panel)) & a["eligible"]
        fd.append(fd_check(panel, a, q))
        m = meta(panel)
        cands = og.candidates(q, a["eligible"])
        if len(cands) > args.max_windows:
            cands = [cands[i] for i in np.linspace(0, len(cands) - 1, args.max_windows).astype(int)]
        for d, T in cands:
            r, c = window_record(panel, a, q, d, T, m)
            if r is not None:
                recs.append(r)
                cells += c
        print(f"{panel}: {len(cands)} windows", flush=True)
    fd = pd.DataFrame(fd)
    print("\n1. Fundamental-diagram check (density residual relative to k_crit; speed identity q/k - v in km/h):")
    print(fd.round(3).to_string(index=False))

    r = pd.DataFrame(recs)
    print(f"\n2. Tail motion over the horizon: {len(r)} windows. Positive = tail moved downstream (queue shrank).")
    rows = []
    for h in (1, 3, 6):
        true = r[f"d_true{h}"].to_numpy()
        preds = {"persistence": np.zeros(len(r)), "w_cur (3-link states, prod)": r.w_cur * h * DT,
                 "w_fd (FD density)": r.w_fd * h * DT, "w_avg (whole queue)": r.w_avg * h * DT,
                 "w_t3 (15-min states)": r.w_t3 * h * DT, "w_emp (tail last 30 min)": r.w_emp * h * DT,
                 "trend-adjusted w_fd": r[f"d_trend{h}"].to_numpy()}
        for name, p in preds.items():
            p = np.nan_to_num(np.asarray(p, float), nan=0.0)
            err = p - true
            rows.append(dict(h=h, predictor=name, mae_km=np.abs(err).mean(), rmse_km=np.sqrt((err ** 2).mean()),
                             corr=np.corrcoef(p, true)[0, 1] if p.std() > 0 else np.nan,
                             sign_acc=((p > 0.1) == (true > 0.1)).mean()))
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print("\nshare of windows whose main queue is gone by step h:", {h: round(float(r[f'gone{h}'].mean()), 3) for h in (1, 3, 6)})
    print("true tail move at h=6 (km) quantiles:", np.quantile(r.d_true6, [0.1, 0.25, 0.5, 0.75, 0.9]).round(2))
    print("w_fd quantiles (km/h):", np.nanquantile(r.w_fd, [0.1, 0.25, 0.5, 0.75, 0.9]).round(1),
          " w_cur:", np.nanquantile(r.w_cur, [0.1, 0.25, 0.5, 0.75, 0.9]).round(1))
    print("corr(w_fd, w_cur):", round(float(r[["w_fd", "w_cur"]].dropna().corr().iloc[0, 1]), 3),
          " corr(w_fd, w_emp):", round(float(r[["w_fd", "w_emp"]].dropna().corr().iloc[0, 1]), 3))

    c = pd.DataFrame(cells)
    print(f"\n3. Queued-at-T cells of the main segment that are free at T+h: {len(c)} cells")
    c["tail_b"] = pd.cut(c.tail_d, [-1, 0, 1, 2, 5, 1000], labels=["tail", "tail+1", "tail+2", "tail+3-5", "tail+6+"])
    c["head_b"] = pd.cut(c.head_d, [-1, 0, 1, 2, 5, 1000], labels=["head", "head-1", "head-2", "head-3-5", "head-6+"])
    print("by distance from the tail at T (share of cleared cells):")
    print((c.groupby("h").tail_b.value_counts(normalize=True).unstack().fillna(0)).round(3).to_string())
    print("by distance from the head at T:")
    print((c.groupby("h").head_b.value_counts(normalize=True).unstack().fillna(0)).round(3).to_string())
    print("speed margin at T+h (true speed / threshold) of cleared cells, quantiles:",
          np.quantile(c.margin_h, [0.1, 0.25, 0.5, 0.75, 0.9]).round(2))
    print("share of cleared cells with margin <= 1.15 (borderline):", round(float((c.margin_h <= 1.15).mean()), 3),
          "; re-queued later in the horizon:", round(float(c.requeue.mean()), 3))
    print("speed margin at T of cleared cells, quantiles:", np.quantile(c.margin_T, [0.1, 0.25, 0.5, 0.75, 0.9]).round(2))
    # interior cells: how do they clear, from the tail side in order?
    inner = c[(c.tail_d > 0) & (c.head_d > 0)]
    print(f"interior cleared cells: {len(inner)} ({len(inner) / max(len(c), 1):.1%}); their median tail_d {inner.tail_d.median()}, head_d {inner.head_d.median()}")


if __name__ == "__main__":
    main()
