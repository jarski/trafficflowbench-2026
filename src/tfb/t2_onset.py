"""Task 2 onset windows: which links are queued at T+30?

Training events are every onset in train: a slot t0 with a queued link after 18
queue-free slots (the 12-slot history, T itself and the 5 horizon slots before
T+30), so T = t0 - 6. Inputs mimic the released window: complete history for
T-60..T-5 and the masked layer at T. Priors are computed without the event's own
month (and without the held-out month in cross-validation).

    python -m tfb.t2_onset --cv        # leave-one-month-out evaluation on train
"""
from __future__ import annotations

import argparse
import json

import lightgbm as lgb
import numpy as np
import pandas as pd

from tfb.paths import CACHE, PANELS, TASK2_EXCLUDED
from tfb.t1_interp import weekday
from tfb.t2_features import K_OCC, _shift, at_origin, dynamic_features, impute_origin, ramp_slots, static_features
from tfb.t2_rules import iou, vcut
from tfb.task1_io import days, load, meta

T2_PANELS = [p for p in PANELS if p not in TASK2_EXCLUDED]
QUIET_SLOTS = 18
TOD_WINDOW = 24
# Optional feature groups, off in production: "anomaly" (state against the train profile) hurt validation by
# about -0.006 (v6, LOG 2026-10-06); "context" (sharper priors, same-day and same-month history) was mixed on the bench.
EXTRA_FEATURES: tuple[str, ...] = ()
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=40,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              verbose=-1, seed=7, deterministic=True, num_threads=8)
ROUNDS = 400


def ramp_link_flows(panel: str, split: str) -> tuple[np.ndarray, np.ndarray]:
    """[day, tod, link] summed valid on-/off-ramp flow attached to each link (NaN if none valid)."""
    m = meta(panel)
    r = load(panel, split, "ramp_flow", "ramp_eligible")
    flow = np.where(r["ramp_eligible"], r["ramp_flow"], np.nan)
    shape = flow.shape[:2] + (len(m["links"]),)
    on, off = np.full(shape, np.nan, np.float32), np.full(shape, np.nan, np.float32)
    for j, (li, typ) in enumerate(zip(m["ramp_link_index"], m["ramp_type"])):
        if li < 0:
            continue
        tgt = on if typ.upper() in ("OR", "ON") else off
        tgt[:, :, li] = np.where(np.isnan(tgt[:, :, li]), flow[:, :, j], np.nansum([tgt[:, :, li], flow[:, :, j]], axis=0))
    return on, off


def onset_events(q: np.ndarray) -> np.ndarray:
    """All (day, t0) with a queue at t0 and none in the QUIET_SLOTS slots before."""
    any_q = q.any(axis=2)
    ev = []
    for d in range(q.shape[0]):
        for t0 in range(QUIET_SLOTS, q.shape[1]):
            if any_q[d, t0] and not any_q[d, t0 - QUIET_SLOTS:t0].any():
                ev.append((d, t0))
    return np.array(ev, dtype=int).reshape(-1, 2)


def train_profiles(panel: str) -> dict[str, np.ndarray]:
    """[day, tod, link] train truth normalized like the dynamic features (speed/vf, flow/capacity, density/k_crit)."""
    m = meta(panel)
    a = load(panel, "train", "true_speed", "true_flow", "true_occ")
    return {"speed": a["true_speed"] / np.asarray(m["free_speed_kmh"], np.float32),
            "flow": a["true_flow"] / np.asarray(m["capacity_vph"], np.float32),
            "occ": a["true_occ"] * np.float32(K_OCC) / np.asarray(m["critical_density"], np.float32)}


class Priors:
    """Onset-location and recurrent-congestion priors from train, excluding given months.

    With profiles (train_profiles), also the time-of-day profile used for the anomaly features.
    """

    def __init__(self, q: np.ndarray, events: np.ndarray, months: np.ndarray, profiles: dict | None = None,
                 weekdays: np.ndarray | None = None):
        self.q, self.events, self.months, self.profiles = q, events, months, profiles
        self.daytype = None if weekdays is None else day_type(weekdays)
        self._prof = {}

    def freq(self, t0: int, exclude: set[str], width: int = TOD_WINDOW) -> np.ndarray:
        """[link] train queue frequency within +-width slots of t0 (days of excluded months left out)."""
        keep = ~np.isin(self.months, list(exclude))
        return self.q[keep, max(t0 - width, 0):t0 + width + 1].mean(axis=(0, 1))

    def profile(self, exclude: set[str]) -> dict[str, np.ndarray]:
        key = frozenset(exclude)
        if key not in self._prof:
            keep = ~np.isin(self.months, list(exclude))
            self._prof[key] = {ch: np.nanmean(x[keep], axis=0) for ch, x in self.profiles.items()}
        return self._prof[key]

    def compute(self, t0: int, exclude: set[str], daytype: int | None = None) -> dict[str, np.ndarray]:
        ev = self.events[~np.isin(self.months[self.events[:, 0]], list(exclude))]
        near = ev[np.abs(ev[:, 1] - t0) <= TOD_WINDOW]
        near = near if len(near) >= 5 else ev
        on_tod = self.q[near[:, 0], near[:, 1]].mean(axis=0)
        on_all = self.q[ev[:, 0], ev[:, 1]].mean(axis=0)
        keep = ~np.isin(self.months, list(exclude))
        qmap = self.q[keep, max(t0 - 1, 0):t0 + 2].mean(axis=(0, 1))
        out = {"prior_tod": on_tod, "prior_all": on_all, "qfreq_t0": qmap,
               "prior_tod_up1": np.append(np.nan, on_tod[:-1]), "prior_tod_dn1": np.append(on_tod[1:], np.nan)}
        if daytype is not None and self.daytype is not None:
            # B2: sharper priors. Narrower onset-time windows and the same day type (weekday / Sat / Sun).
            same = ev[self.daytype[ev[:, 0]] == daytype]
            for name, pool, width in (("prior_tod6", ev, 6), ("prior_tod12", ev, 12),
                                      ("prior_dt", same, TOD_WINDOW), ("prior_dt12", same, 12)):
                sel = pool[np.abs(pool[:, 1] - t0) <= width]
                out[name] = self.q[sel[:, 0], sel[:, 1]].mean(axis=0) if len(sel) >= 5 else np.full(self.q.shape[2], np.nan)
            out["onset_rate_tod6"] = np.full(self.q.shape[2], (np.abs(ev[:, 1] - t0) <= 6).sum() / max(keep.sum(), 1))
        return out


def day_type(weekdays: np.ndarray) -> np.ndarray:
    """0 Monday-Friday, 1 Saturday, 2 Sunday."""
    return np.where(weekdays == 5, 1, np.where(weekdays == 6, 2, 0))


ADAPT_PSEUDO_DAYS = 2.0  # shrinkage of the month's own bottleneck activity toward train, in days of evidence


def month_activity_ratio(speed: np.ndarray, vc: np.ndarray, month_days: np.ndarray, d: int, t0: int,
                         f_train: np.ndarray) -> np.ndarray:
    """[link] how active each bottleneck is this month relative to train, from the month's earlier days.

    Scored months keep the train bottlenecks but activate fewer of them (LOG 2026-10-07). The month's own
    queue frequency within +-2 h of t0 on days before d (masked layer, at or before T) is shrunk toward the
    train frequency with ADAPT_PSEUDO_DAYS days of pseudo-evidence; the ratio to train rescales the priors.
    """
    lo, hi = max(t0 - TOD_WINDOW, 0), min(t0 + TOD_WINDOW + 1, speed.shape[1])
    earlier = month_days[month_days < d]
    alpha = ADAPT_PSEUDO_DAYS * (hi - lo)
    if earlier.size:
        win = speed[earlier, lo:hi]
        obs = ~np.isnan(win)
        k, n = ((win <= vc) & obs).sum(axis=(0, 1)), obs.sum(axis=(0, 1))
    else:
        k, n = np.zeros(speed.shape[2]), np.zeros(speed.shape[2])
    post = (k + alpha * f_train) / (n + alpha)
    return (post + 0.01) / (f_train + 0.01)


def context_features(speed: np.ndarray, vc: np.ndarray, month_days: np.ndarray, d: int, T: int) -> dict[str, np.ndarray]:
    """B2: same-day and same-month context from the masked layer, all at or before T.

    speed: [day, tod, link] masked layer of the split; month_days: indices of the split's days in d's month.
    """
    obs = ~np.isnan(speed[d, :T - 12])
    q_today = (speed[d, :T - 12] <= vc) & obs                     # earlier today, before the window history
    last_q = np.flatnonzero(q_today.any(axis=1))
    earlier = month_days[month_days < d]
    lo, hi = max(T + 6 - TOD_WINDOW, 0), min(T + 6 + TOD_WINDOW + 1, speed.shape[1])
    if earlier.size:
        win = speed[earlier, lo:hi]
        n_obs = (~np.isnan(win)).sum(axis=(0, 1))
        with np.errstate(invalid="ignore", divide="ignore"):
            month_q = np.where(n_obs > 0, ((win <= vc) & ~np.isnan(win)).sum(axis=(0, 1)) / n_obs, np.nan)
    else:
        month_q = np.full(speed.shape[2], np.nan)
    L = speed.shape[2]
    return {"today_q_link": q_today.any(axis=0).astype(float), "today_q_any": np.full(L, float(q_today.any())),
            "today_since_q": np.full(L, float(T - last_q[-1]) if last_q.size else np.nan),
            "month_q_tod": month_q, "month_days_before": np.full(L, float(earlier.size))}


def panel_frame(panel: str, split: str, items: list[tuple], priors: Priors, exclude_fn) -> pd.DataFrame:
    """Rows = (item, link). items: (day, T, key, label_or_None). History from truth on train, masked layer otherwise."""
    st = static_features(panel)
    a = load(panel, split, "speed", "flow", "occ")
    hist_src = load(panel, "train", "true_speed", "true_flow", "true_occ") if split == "train" else None
    on, off = ramp_link_flows(panel, split)
    dl = days(panel, split)["days"]
    wd = weekday(dl)
    month = np.array([x[:7] for x in dl])
    vc = vcut(panel)
    frames = []
    for d, T, key, label in items:
        if hist_src is not None:
            hist = {ch: hist_src[f"true_{ch}"][d, T - 12:T] for ch in ("speed", "flow", "occ")}
        else:
            hist = {ch: a[ch][d, T - 12:T] for ch in ("speed", "flow", "occ")}
        at_t = impute_origin(at_origin(a, d, T), hist, "onset", panel, T, next(iter(exclude_fn(d)), None))
        r_on = np.nanmean(on[d, ramp_slots(T)], axis=0)
        r_off = np.nanmean(off[d, ramp_slots(T)], axis=0)
        f = {**st, **dynamic_features(panel, hist, at_t, r_on, r_off),
             **priors.compute(T + 6, exclude_fn(d), int(day_type(wd[d:d + 1])[0]))}
        if priors.profiles is not None:
            f.update(anomaly_features(f, priors.profile(exclude_fn(d)), T))
        if priors.daytype is not None:
            f.update(context_features(a["speed"], vc, np.flatnonzero(month == month[d]), d, T))
        if "adapt" in EXTRA_FEATURES:   # replace the train priors by month-adapted ones (no fallback feature)
            ratio = month_activity_ratio(a["speed"], vc, np.flatnonzero(month == month[d]), d, T + 6,
                                         priors.freq(T + 6, exclude_fn(d)))
            for k in ("prior_tod", "prior_all", "qfreq_t0"):
                f[k] = np.clip(f[k] * ratio, 0, 1)
            f["prior_tod_up1"] = np.append(np.nan, f["prior_tod"][:-1])
            f["prior_tod_dn1"] = np.append(f["prior_tod"][1:], np.nan)
        df = pd.DataFrame(f)
        df["tod"], df["weekday"], df["link"], df["key"] = T + 6, wd[d], np.arange(len(df)), key
        if label is not None:
            df["y"], df["w"] = label[0].astype(int), label[1].astype(float)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["panel"] = panel
    return out


def anomaly_features(f: dict, prof: dict, T: int) -> dict[str, np.ndarray]:
    """State at T against the time-of-day profile; the downstream density anomaly is the bottleneck signature.

    Out-of-fold 0.9071 -> 0.9101 (LOG 2026-10-02, experiments/t2_onset_dynamics.py).
    """
    out = {}
    for ch in ("speed", "flow", "occ"):
        out[f"{ch}_anom"] = f[f"{ch}_last"] - prof[ch][T]
        out[f"{ch}_anom_m15"] = f[f"{ch}_m15"] - prof[ch][T]
    occ = out["occ_anom"]
    for k in (1, 2, -1):
        out[f"occ_anom_n{k:+d}"] = _shift(occ, k)
    out["occ_anom_rank"] = pd.Series(occ).rank(ascending=False, pct=True).to_numpy()
    out["occ_anom_max_corr"] = np.full_like(occ, np.nanmax(occ) if np.isfinite(occ).any() else np.nan)
    return out


def make_priors(panel: str, q: np.ndarray, events: np.ndarray, months: np.ndarray) -> Priors:
    """Priors with the inputs the enabled EXTRA_FEATURES need."""
    return Priors(q, events, months, train_profiles(panel) if "anomaly" in EXTRA_FEATURES else None,
                  weekday(days(panel, "train")["days"]) if "context" in EXTRA_FEATURES else None)


def train_frame() -> pd.DataFrame:
    frames = []
    for panel in T2_PANELS:
        a = load(panel, "train", "true_speed", "eligible")
        q = (a["true_speed"] <= vcut(panel)) & a["eligible"]
        months = np.array([d[:7] for d in days(panel, "train")["days"]])
        ev = onset_events(q)
        ev = ev[ev[:, 1] - 6 >= 12]
        pri = make_priors(panel, q, ev, months)
        items = [(d, t0 - 6, f"{panel}|{d}|{t0}", (q[d, t0], a["eligible"][d, t0])) for d, t0 in ev]
        df = panel_frame(panel, "train", items, pri, lambda d, m=months: {m[d]})
        df["month"] = months[df.key.str.split("|").str[1].astype(int)]
        frames.append(df)
        print(f"  {panel}: {len(ev)} onset events", flush=True)
    return pd.concat(frames, ignore_index=True)


FEATURES_DROP = {"y", "w", "key", "panel", "month", "link"}


def choose_set(p: np.ndarray, kmax: int = 8) -> np.ndarray:
    """Top-k by p, k maximizing the ratio-of-expectations IoU approximation."""
    order = np.argsort(-p)
    total = p.sum()
    best_k, best = 1, -1.0
    for k in range(1, kmax + 1):
        inside = p[order[:k]].sum()
        val = inside / (k + total - inside)
        if val > best:
            best_k, best = k, val
    pred = np.zeros(p.shape, bool)
    pred[order[:best_k]] = True
    return pred


def evaluate(df: pd.DataFrame, pcol: str) -> pd.Series:
    """Mean IoU per panel over events; prediction and truth restricted to eligible links at T+30."""
    res = []
    for key, g in df.groupby("key", sort=False):
        pred = choose_set(g[pcol].to_numpy())
        el = g.w.to_numpy() > 0
        res.append((g.panel.iloc[0], iou(pred & el, g.y.to_numpy().astype(bool) & el)))
    r = pd.DataFrame(res, columns=["panel", "iou"])
    return r.groupby("panel").iou.mean()


def cv(df: pd.DataFrame) -> None:
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
    model_iou = evaluate(df, "p")
    prior_iou = evaluate(df.assign(pp=df.prior_tod), "pp")
    out = pd.DataFrame({"prior_tod": prior_iou, "lgbm": model_iou})
    print(out.round(3).to_string())
    print("mean over panels:", out.mean().round(4).to_dict())
    imp = pd.Series(model.feature_importance("gain"), index=feats).sort_values(ascending=False)
    print("top features:", imp.head(15).round(0).to_dict())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cv", action="store_true")
    ap.add_argument("--out", default=None, help="build the training frame into this file (always rebuilds)")
    args = ap.parse_args()
    if args.out:
        train_frame().to_parquet(args.out, index=False)
        print("wrote", args.out)
        return
    path = CACHE / "t2_onset_train.parquet"
    if path.exists():
        df = pd.read_parquet(path)
    else:
        df = train_frame()
        CACHE.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
    print(f"{len(df):,} rows, {df.key.nunique():,} events")
    if args.cv:
        cv(df)


if __name__ == "__main__":
    main()
