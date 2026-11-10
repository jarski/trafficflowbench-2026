"""Task 2 ongoing windows: forecast an established queue over T+5..T+30.

Training windows are replicas of the organizer selector's ongoing candidates on
train: an established queue in the history (>= 2 queued observations, >= 2 on one
link), a queue in the horizon, persistence IoU <= 0.9 and history coverage >= 0.7.
Rows are (window, link, step) for links within NEAR links of the queue; links
further away are predicted free.

    python -m tfb.t2_ongoing --cv
"""
from __future__ import annotations

import argparse

import lightgbm as lgb
import numpy as np
import pandas as pd

from tfb.paths import CACHE
from tfb.t1_interp import weekday
from tfb.t2_features import K_OCC, at_origin, impute_origin, ramp_slots, static_features
from tfb.t2_onset import T2_PANELS, ramp_link_flows
from tfb.t2_rules import iou, vcut
from tfb.task1_io import days, load, meta

NEAR = 12
STEPS = 6
BRIDGE_SEGMENTS = True  # segment features from bridged queue flags (A3b, LOG 2026-10-06)
# Also model links beyond NEAR where queues recur in the horizon (train frequency >= FAR_QFREQ): new
# bottlenecks activating far from the current queue are 18-20% of the remaining errors. None = off.
FAR_QFREQ: float | None = None  # 0.02 gained +0.025 on the bench but -0.0012 public (v12): see LOG 2026-10-07
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=100,
              feature_fraction=0.8, bagging_fraction=0.7, bagging_freq=1, lambda_l2=1.0,
              verbose=-1, seed=7, deterministic=True, num_threads=8)
ROUNDS = 400
# qfreq_*: train-month queue recurrence. Train months are near copies of each other while the scored months
# were drawn independently (LOG 2026-10-07), so recurrence features mislead there: dropping them gained +0.0029 public (v13).
FEATURES_DROP = {"y", "w", "key", "panel", "month", "link", "persist", "outside_true", "qfreq_h", "qfreq_T", "qfreq_trend"}


def is_established(hist_q: np.ndarray) -> bool:
    return hist_q.sum() >= 2 and hist_q.sum(axis=0).max() >= 2


CAND_STRIDE, CAND_CAP = 6, 6  # training windows: every CAND_STRIDE-th valid origin, at most CAND_CAP per day


def candidates(q: np.ndarray, elig: np.ndarray, stride: int | None = None, cap_per_day: int | None = None,
               seed: int = 0) -> list[tuple]:
    stride, cap_per_day = stride or CAND_STRIDE, cap_per_day or CAND_CAP
    rng = np.random.default_rng(seed)
    out = []
    for d in range(q.shape[0]):
        valid = []
        for T in range(12, 288 - STEPS):
            hor = q[d, T + 1:T + 1 + STEPS]
            if not hor.any() or not is_established(q[d, T - 12:T]) or elig[d, T - 12:T].mean() < 0.7:
                continue
            if iou(np.broadcast_to(q[d, T - 1], hor.shape), hor) > 0.9:
                continue
            valid.append(T)
        valid = valid[::stride]
        if len(valid) > cap_per_day:
            valid = sorted(rng.choice(valid, cap_per_day, replace=False))
        out += [(d, T) for T in valid]
    return out


def _segments(flags: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per link: distance (links) to the queue tail (upstream end) and head of its segment; segment length."""
    n = len(flags)
    tail_d, head_d, seg_len = np.full(n, np.nan), np.full(n, np.nan), np.zeros(n)
    i = 0
    while i < n:
        if flags[i]:
            j = i
            while j + 1 < n and flags[j + 1]:
                j += 1
            idx = np.arange(i, j + 1)
            tail_d[idx], head_d[idx], seg_len[idx] = idx - i, j - idx, j - i + 1
            i = j + 1
        else:
            i += 1
    return tail_d, head_d, seg_len


def _nearest(flags: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Links to the nearest queued link downstream (>0) and upstream (>0); inf if none."""
    n = len(flags)
    pos = np.flatnonzero(flags)
    idx = np.arange(n)
    if pos.size == 0:
        return np.full(n, np.inf), np.full(n, np.inf)
    j = np.searchsorted(pos, idx)
    dn = np.where(j < pos.size, pos[np.minimum(j, pos.size - 1)] - idx, np.inf)
    up = np.where(j > 0, idx - pos[np.maximum(j - 1, 0)], np.inf)
    dn = np.where(flags, 0, dn)
    up = np.where(flags, 0, up)
    return dn.astype(float), up.astype(float)


DT_H = 5 / 60  # one step in hours


def _segment_dynamics(q_last: np.ndarray, speed: np.ndarray, flow: np.ndarray, flow_m30: np.ndarray,
                      pos_km: np.ndarray, cap: np.ndarray, vf: np.ndarray) -> dict[str, np.ndarray]:
    """Per link, for the queue segment at or downstream of it: tail shockwave speed and head discharge.

    Tail speed w = (q_arr - q_q) / (k_arr - k_q) from the arriving state (3 links upstream of the tail)
    and the queued state (first 3 queued links), in km/h along the travel direction: negative means the
    tail moves upstream (queue grows), positive means it retreats (queue clears from upstream).
    """
    n = len(q_last)
    out = {k: np.full(n, np.nan) for k in ("w_tail", "tail_pos", "arr_flow", "arr_trend", "dis_flow", "dis_speed")}
    prev_end = -1
    for s_, e_ in _segment_list(q_last):
        up_l, q_l = np.arange(max(s_ - 3, 0), s_), np.arange(s_, min(s_ + 3, e_ + 1))
        with np.errstate(invalid="ignore", divide="ignore"):
            q_arr, v_arr = np.nanmean(flow[up_l]) if up_l.size else np.nan, np.nanmean(speed[up_l]) if up_l.size else np.nan
            q_q, v_q = np.nanmean(flow[q_l]), np.nanmean(speed[q_l])
            dk = q_arr / v_arr - q_q / v_q
            w = (q_arr - q_q) / dk if np.isfinite(dk) and abs(dk) > 1e-6 else np.nan
        region = np.arange(prev_end + 1, e_ + 1)
        out["w_tail"][region] = np.clip(w, -60, 60)
        out["tail_pos"][region] = pos_km[s_]
        if up_l.size:
            out["arr_flow"][region] = np.nanmean(flow[up_l] / cap[up_l])
            out["arr_trend"][region] = np.nanmean((flow[up_l] - flow_m30[up_l]) / cap[up_l])
        if e_ + 1 < n:
            out["dis_flow"][region] = flow[e_ + 1] / cap[e_ + 1]
            out["dis_speed"][region] = speed[e_ + 1] / vf[e_ + 1]
        prev_end = e_
    return out


def _bridge(flags: np.ndarray, unknown: np.ndarray, max_gap: int = 2) -> np.ndarray:
    """Mark runs of up to max_gap unknown links as queued when queued links sit on both sides."""
    out = flags.copy()
    for s_, e_ in _segment_list(unknown):
        if e_ - s_ < max_gap and s_ > 0 and e_ + 1 < len(flags) and flags[s_ - 1] and flags[e_ + 1]:
            out[s_:e_ + 1] = True
    return out


def _segment_list(flags: np.ndarray) -> list[tuple[int, int]]:
    """(upstream end, downstream end) of each run of queued links, in travel order."""
    idx = np.flatnonzero(flags)
    if idx.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.r_[idx[0], idx[breaks + 1]]
    ends = np.r_[idx[breaks], idx[-1]]
    return list(zip(starts.tolist(), ends.tolist()))


def window_rows(panel: str, hist: dict, at_t: dict, r_on: np.ndarray, qfreq: np.ndarray, T: int, wd: int,
                st: dict, fprof: np.ndarray | None = None) -> pd.DataFrame:
    """Features for every (link, step) of one window. hist[ch]: [12, L]; at_t[ch]: [L]; qfreq: [288, L].

    fprof: [288, L] train time-of-day flow / capacity, for the demand-anomaly features (NaN without it).
    """
    m = meta(panel)
    vc, vf, cap = vcut(panel), np.asarray(m["free_speed_kmh"]), np.asarray(m["capacity_vph"])
    kcrit = np.asarray(m["critical_density"])
    sp_last = np.where(np.isnan(at_t["speed"]), hist["speed"][-1], at_t["speed"])
    hq = hist["speed"] <= vc                                           # [12, L] history queue flags
    q_last = np.nan_to_num(sp_last, nan=np.inf) <= vc
    q_m15 = hq[-3] if hq.shape[0] >= 3 else hq[0]
    q_m30 = hq[-6]
    # links without a reading (masked at T and ineligible at T-5) between queued links are usually queued too
    q_seg = _bridge(q_last, np.isnan(sp_last))
    qs, q30s = (q_seg, _bridge(q_m30, np.isnan(hist["speed"][-6]))) if BRIDGE_SEGMENTS else (q_last, q_m30)
    tail_d, head_d, seg_len = _segments(qs)
    dn, up = _nearest(qs)
    dn30, up30 = _nearest(q30s)
    links = np.arange(len(vc))
    queued_now = np.flatnonzero(qs)
    tails_now = queued_now.min() if queued_now.size else -1
    q30 = np.flatnonzero(q30s)
    tails_30 = q30.min() if q30.size else -1
    near = np.isfinite(np.minimum(dn, up)) & (np.minimum(dn, up) <= NEAR)
    near |= np.isfinite(np.minimum(dn30, up30)) & (np.minimum(dn30, up30) <= NEAR)
    # Every ongoing window has a queue in the horizon, so never leave the candidate set empty:
    # also cover links near any queue seen anywhere in the history.
    dn_h, up_h = _nearest(hq.any(axis=0))
    near |= np.isfinite(np.minimum(dn_h, up_h)) & (np.minimum(dn_h, up_h) <= NEAR)
    if FAR_QFREQ is not None:
        near |= qfreq[T + 1:T + 1 + STEPS].max(axis=0) >= FAR_QFREQ
    base = pd.DataFrame({
        "link": links, "q_last": q_last, "q_m15": q_m15, "q_m30": q_m30, "q_age": hq.sum(axis=0),
        "speed_last": sp_last / vf, "speed_m30": hist["speed"][-6] / vf,
        "flow_last": np.where(np.isnan(at_t["flow"]), hist["flow"][-1], at_t["flow"]) / cap,
        "dens_last": np.where(np.isnan(at_t["occ"]), hist["occ"][-1], at_t["occ"]) * K_OCC / kcrit,
        "dist_dn": dn, "dist_up": up, "dist_dn_30": dn30, "dist_up_30": up30, "dist_dn_h": dn_h, "dist_up_h": up_h,
        "tail_d": tail_d, "head_d": head_d, "seg_len": seg_len,
        "n_queued": qs.sum(), "n_queued_30": q30s.sum(), "n_queued_60": hq[0].sum(),
        "tail_shift_30": (tails_30 - tails_now) if (tails_now >= 0 and tails_30 >= 0) else 0,
        "ramp_on_frac": r_on / cap, "lanes": st["lanes"], "cap_dn_ratio": st["cap_dn_ratio"],
        "pos": st["pos"], "weekday": wd, "tod": T,
    })
    for k in (1, 2, 3):
        base[f"speed_up{k}"] = np.append(np.full(k, np.nan), base.speed_last.to_numpy()[:-k])
        base[f"flow_up{k}"] = np.append(np.full(k, np.nan), base.flow_last.to_numpy()[:-k])
        base[f"speed_dn{k}"] = np.append(base.speed_last.to_numpy()[k:], np.full(k, np.nan))
    # A3 (LOG 2026-10-06): recovery, demand and tail/head dynamics, aimed at queues that clear while we keep them
    lengths = np.asarray(m["length_km"], float)
    pos_km = np.cumsum(lengths) - lengths / 2
    fl_last = np.where(np.isnan(at_t["flow"]), hist["flow"][-1], at_t["flow"])
    base["speed_slope15"] = (sp_last - hist["speed"][-3]) / vf
    base["speed_slope30"] = (sp_last - hist["speed"][-6]) / vf
    base["q_recent"] = hq[-3:].sum(axis=0) + q_last
    base["flow_anom"] = fl_last / cap - fprof[T] if fprof is not None else np.nan
    with np.errstate(invalid="ignore"):
        i_dn, i_up = links + np.nan_to_num(dn, posinf=0).astype(int), links - np.nan_to_num(up, posinf=0).astype(int)
        base["dist_dn_km"] = np.where(np.isfinite(dn), pos_km[np.clip(i_dn, 0, len(links) - 1)] - pos_km, np.nan)
        base["dist_up_km"] = np.where(np.isfinite(up), pos_km - pos_km[np.clip(i_up, 0, len(links) - 1)], np.nan)
    base["q_bridge"] = q_seg
    dyn = _segment_dynamics(q_seg, sp_last, fl_last, hist["flow"][-6], pos_km, cap, vf)
    for k in ("w_tail", "arr_flow", "arr_trend", "dis_flow", "dis_speed"):
        base[k] = dyn[k]
    base["arr_anom"] = dyn["arr_flow"] - (np.nanmean(fprof[T]) if fprof is not None else np.nan)
    base["_tail_pos"], base["_pos_km"] = dyn["tail_pos"], pos_km
    base = base[near].reset_index(drop=True)
    rows = []
    for h in range(1, STEPS + 1):
        r = base.copy()
        r["h"] = h
        r["qfreq_h"] = qfreq[min(T + h, 287), r.link.to_numpy()]
        r["qfreq_T"] = qfreq[T, r.link.to_numpy()]
        r["qfreq_trend"] = r.qfreq_h - r.qfreq_T
        # signed km from the shockwave-predicted tail at step h (> 0: downstream of it, i.e. inside the queue extent)
        r["tail_gap_km"] = r._pos_km - (r._tail_pos + r.w_tail * h * DT_H)
        rows.append(r.drop(columns=["_tail_pos", "_pos_km"]))
    return pd.concat(rows, ignore_index=True)


class TrainPanel:
    """Train arrays, labels and month-excluded queue frequencies for building labelled ongoing windows."""

    def __init__(self, panel: str):
        self.panel = panel
        self.a = load(panel, "train", "true_speed", "true_flow", "true_occ", "speed", "flow", "occ", "eligible")
        self.q = (self.a["true_speed"] <= vcut(panel)) & self.a["eligible"]
        dl = days(panel, "train")["days"]
        self.months = np.array([d[:7] for d in dl])
        self.wd = weekday(dl)
        self.on, _ = ramp_link_flows(panel, "train")
        self.st = static_features(panel)
        self.qfreq_by_month = {mo: self.q[self.months != mo].mean(axis=0) for mo in np.unique(self.months)}
        flow = self.a["true_flow"] / np.asarray(meta(panel)["capacity_vph"], np.float32)
        self.fprof_by_month = {mo: np.nanmean(flow[self.months != mo], axis=0) for mo in np.unique(self.months)}

    def window(self, d: int, T: int) -> pd.DataFrame:
        """Feature rows and labels of the window at (day, T); history from truth, slot T from the masked layer."""
        a, q = self.a, self.q
        hist = {ch: a[f"true_{ch}"][d, T - 12:T] for ch in ("speed", "flow", "occ")}
        at_t = impute_origin(at_origin(a, d, T), hist, "ongoing", self.panel, T, self.months[d])
        r = window_rows(self.panel, hist, at_t, np.nanmean(self.on[d, ramp_slots(T)], axis=0),
                        self.qfreq_by_month[self.months[d]], T, self.wd[d], self.st, self.fprof_by_month[self.months[d]])
        r["y"] = q[d, T + r.h.to_numpy(), r.link.to_numpy()].astype(int)
        r["w"] = a["eligible"][d, T + r.h.to_numpy(), r.link.to_numpy()].astype(float)
        r["key"], r["panel"], r["month"] = f"{self.panel}|{d}|{T}", self.panel, self.months[d]
        # queued eligible cells outside the modelled neighbourhood still count in the union
        r["outside_true"] = int(q[d, T + 1:T + 1 + STEPS].sum() - r.y.sum())
        return r


def train_frame() -> pd.DataFrame:
    frames = []
    for panel in T2_PANELS:
        tp = TrainPanel(panel)
        cands = candidates(tp.q, tp.a["eligible"])
        frames += [tp.window(d, T) for d, T in cands]
        print(f"  {panel}: {len(cands)} ongoing windows", flush=True)
    return pd.concat(frames, ignore_index=True)


STEP_SHRINK = 0.0  # 0.2 helped the bench, about neutral on the leaderboard (v11 vs v12); LOG 2026-10-07


def choose_window(p: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Cells to predict queued: p shrunk linearly by up to STEP_SHRINK at the last step, then choose_cells."""
    return choose_cells(p * (1 - STEP_SHRINK * (np.asarray(h) - 1) / (STEPS - 1)))


def choose_cells(p: np.ndarray) -> np.ndarray:
    """Top-k cells by p, k maximizing sum_in p / (k + sum_out p)."""
    order = np.argsort(-p)
    cs = np.cumsum(p[order])
    k = np.arange(1, len(p) + 1)
    val = cs / (k + p.sum() - cs)
    pred = np.zeros(len(p), bool)
    pred[order[:int(np.argmax(val)) + 1]] = True
    return pred


def evaluate(df: pd.DataFrame, pcol: str) -> pd.Series:
    res = []
    for key, g in df.groupby("key", sort=False):
        el = g.w.to_numpy() > 0
        truth = g.y.to_numpy().astype(bool) & el
        pred = (choose_window(g[pcol].to_numpy(), g.h.to_numpy()) if pcol != "persist" else g.q_last.to_numpy().astype(bool)) & el
        union = (pred | truth).sum() + g.outside_true.iloc[0]
        res.append((g.panel.iloc[0], (pred & truth).sum() / union if union else 1.0))
    return pd.DataFrame(res, columns=["panel", "iou"]).groupby("panel").iou.mean()


def cv(df: pd.DataFrame) -> pd.DataFrame:
    feats = [c for c in df.columns if c not in FEATURES_DROP]
    df = df.copy()
    df["panel_code"] = df.panel.astype("category").cat.codes
    feats.append("panel_code")
    df["p"] = np.nan
    for month in sorted(df.month.unique()):
        tr, va = df[df.month != month], df.month == month
        model = lgb.train(PARAMS, lgb.Dataset(tr[feats], tr.y, weight=tr.w), ROUNDS)
        df.loc[va, "p"] = model.predict(df.loc[va, feats])
        print(f"  fold {month}: done", flush=True)
    out = pd.DataFrame({"persistence": evaluate(df, "persist"), "lgbm": evaluate(df, "p")})
    print(out.round(3).to_string())
    print("mean over panels:", out.mean().round(4).to_dict())
    imp = pd.Series(model.feature_importance("gain"), index=feats).sort_values(ascending=False)
    print("top features:", imp.head(12).round(0).to_dict())
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cv", action="store_true")
    ap.add_argument("--out", default=None, help="build the training frame into this file (always rebuilds)")
    args = ap.parse_args()
    if args.out:
        train_frame().to_parquet(args.out, index=False)
        print("wrote", args.out)
        return
    path = CACHE / "t2_ongoing_train.parquet"
    if path.exists():
        df = pd.read_parquet(path)
    else:
        df = train_frame()
        df.to_parquet(path, index=False)
    print(f"{len(df):,} rows, {df.key.nunique():,} windows")
    if args.cv:
        cv(df)


if __name__ == "__main__":
    main()
