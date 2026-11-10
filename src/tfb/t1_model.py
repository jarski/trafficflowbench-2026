"""Task 1 v1: LightGBM correction of time interpolation using neighbouring links.

The key signal is a neighbour's deviation from its own straight-line interpolation
across the target cell's gap (same anchor times): a congestion wave passing
through shows there first. Targets are the residuals true - interpolation, for
speed (km/h) and flow per lane.

    python -m tfb.t1_model --cv     # train Jun-Nov, score Dec-Feb with tfb.metrics
"""
from __future__ import annotations

import argparse

import lightgbm as lgb
import numpy as np
import pandas as pd

from tfb.metrics import s_physics, s_state
from tfb.paths import CACHE, PANELS
from tfb.t1_interp import group_profile_by_day, weekday, weekday_profile_by_day
from tfb.task1_io import days, load, meta

FAMILY = {p: p.rsplit("_", 1)[0] for p in PANELS}
OFFSETS = (-5, -4, -3, -2, -1, 1, 2, 3, 4, 5)  # tuned 2026-10-01: +-5 beat +-3
K_OCC = 285.0
PARAMS = dict(objective="regression", learning_rate=0.05, num_leaves=255, min_data_in_leaf=200,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              verbose=-1, seed=11, deterministic=True, num_threads=10)
ROUNDS = 1000


def interp_anchors(x: np.ndarray):
    """Time interpolation plus the previous/next observed slot of every cell (-1 / T when absent)."""
    D, T, L = x.shape
    valid = ~np.isnan(x)
    idx = np.broadcast_to(np.arange(T, dtype=np.int16)[None, :, None], x.shape)
    last = np.maximum.accumulate(np.where(valid, idx, -1).astype(np.int16), axis=1)
    nxt = np.minimum.accumulate(np.where(valid, idx, T).astype(np.int16)[:, ::-1], axis=1)[:, ::-1]
    xl = np.take_along_axis(x, np.clip(last, 0, T - 1).astype(np.int64), axis=1)
    xn = np.take_along_axis(x, np.clip(nxt, 0, T - 1).astype(np.int64), axis=1)
    w = (idx - last) / np.maximum(nxt - last, 1)
    out = np.where(last < 0, xn, np.where(nxt >= T, xl, xl + w * (xn - xl)))
    return np.where(valid, x, out).astype(np.float32), last, nxt


DAY_ANOM_HALF = 12  # slots on each side for the day-anomaly features


def _window_mean(x: np.ndarray, half: int) -> np.ndarray:
    """NaN-aware mean over +-half slots along time, per day and link."""
    valid = ~np.isnan(x)
    c = np.cumsum(np.pad(np.where(valid, x, 0.0), ((0, 0), (1, 0), (0, 0))), axis=1)
    n = np.cumsum(np.pad(valid.astype(np.int32), ((0, 0), (1, 0), (0, 0))), axis=1)
    T = x.shape[1]
    lo, hi = np.clip(np.arange(T) - half, 0, T), np.clip(np.arange(T) + half + 1, 0, T)
    s, k = c[:, hi] - c[:, lo], n[:, hi] - n[:, lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(k > 0, s / k, np.nan).astype(np.float32)


class PanelData:
    """Masked inputs, their interpolations and profiles for one panel and split."""

    def __init__(self, panel: str, split: str, profile_days: np.ndarray | None = None, masked: dict | None = None):
        """masked: replacement speed/flow/occ/target arrays (pseudo-masked layers for self-supervised rows)."""
        self.panel, self.split = panel, split
        m = meta(panel)
        a = {**load(panel, split, "speed", "flow", "occ", "pct", "target"), **(masked or {})}
        self.target, self.pct = a["target"], a["pct"]
        self.L = len(m["links"])
        self.lanes = np.clip(np.asarray(m["lanes"]), 1, None)
        self.vf, self.cap = np.asarray(m["free_speed_kmh"]), np.asarray(m["capacity_vph"])
        self.kcrit = np.asarray(m["critical_density"])
        info = days(panel, split)
        self.regime_rate = np.array([{"R1": .2, "R2": .3, "R3": .5}[r] for r in info["regime"]])
        self.wd = weekday(info["days"])
        tr = load(panel, "train", "true_speed", "true_flow")
        tr_wd = weekday(days(panel, "train")["days"])
        keep = np.ones(len(tr_wd), bool) if profile_days is None else profile_days
        # Profiles per day of this split. On train each day is left out of its own profile:
        # otherwise training rows see their own answer in the profile and test rows do not.
        own = split == "train"
        self.prof_s = weekday_profile_by_day(tr["true_speed"], tr_wd, keep, self.wd, own)
        self.prof_f = weekday_profile_by_day(tr["true_flow"], tr_wd, keep, self.wd, own)
        self.raw = {"s": a["speed"], "f": a["flow"], "o": a["occ"]}
        self.si, self.last, self.nxt = interp_anchors(a["speed"])
        self.fi, _, _ = interp_anchors(a["flow"])
        self.oi, _, _ = interp_anchors(a["occ"])
        for name, prof in (("si", self.prof_s), ("fi", self.prof_f)):
            arr = getattr(self, name)
            setattr(self, name, np.where(np.isnan(arr), prof, arr))
        self.oi = np.where(np.isnan(self.oi), np.nanmean(self.oi), self.oi)
        # D2: profile of the scored month itself (its observed cells, same day type, target day left out).
        # Each month has its own demand draw; Task 1 is offline, so the whole month may be used.
        groups = np.array([f"{dd[:7]}|{w >= 5}" for dd, w in zip(info["days"], self.wd)])
        self.mprof_s = group_profile_by_day(a["speed"], groups)
        self.mprof_f = group_profile_by_day(a["flow"], groups)
        # D2b: how far today runs from this month's usual day on the same link, averaged over +-1 h of
        # observed cells (the target cell itself is blank in the masked layer)
        self.dayanom_s = _window_mean(a["speed"] - self.mprof_s, DAY_ANOM_HALF)
        self.dayanom_f = _window_mean(a["flow"] - self.mprof_f, DAY_ANOM_HALF)
        self.si_nb, self.fi_nb = self.si, self.fi   # what neighbour features read (see set_neighbour_fill)

    def set_neighbour_fill(self, speed: np.ndarray, flow: np.ndarray) -> None:
        """Let neighbour features read these values at masked target cells instead of the interpolation.

        A second pass fills them with first-pass predictions; an oracle test fills them with the truth.
        A cell's own features are unchanged, so its own answer never enters its features.
        """
        fill = self.target & ~np.isnan(speed)
        self.si_nb = np.where(fill, speed, self.si).astype(np.float32)
        self.fi_nb = np.where(fill & ~np.isnan(flow), flow, self.fi).astype(np.float32)

    def features(self, d: np.ndarray, t: np.ndarray, l: np.ndarray) -> pd.DataFrame:
        T = self.si.shape[1]
        tp = self.last[d, t, l].astype(np.int64)
        tn = self.nxt[d, t, l].astype(np.int64)
        tp_c, tn_c = np.where(tp < 0, tn, tp).clip(0, T - 1), np.where(tn >= T, tp, tn).clip(0, T - 1)
        w = np.where(tn_c > tp_c, (t - tp_c) / np.maximum(tn_c - tp_c, 1), 0.5)
        lanes, vf = self.lanes[l], self.vf[l]
        f = {
            "tod": t, "regime_rate": self.regime_rate[d], "pct": self.pct[d, t, l], "lanes": lanes, "vf": vf,
            "cap_lane": self.cap[l] / lanes, "pos": l / max(self.L - 1, 1),
            "gap_prev": np.where(tp < 0, 99, t - tp), "gap_next": np.where(tn >= T, 99, tn - t),
            "si": self.si[d, t, l], "fi_lane": self.fi[d, t, l] / lanes, "oi_dens": self.oi[d, t, l] * K_OCC / self.kcrit[l],
            "s_prev": self.si[d, tp_c, l], "s_next": self.si[d, tn_c, l],
            "prof_s": self.prof_s[d, t, l], "prof_f_lane": self.prof_f[d, t, l] / lanes,
            "anom_s_prev": self.si[d, tp_c, l] - self.prof_s[d, tp_c, l],
            "anom_s_next": self.si[d, tn_c, l] - self.prof_s[d, tn_c, l],
            "mprof_s": self.mprof_s[d, t, l], "mprof_f_lane": self.mprof_f[d, t, l] / lanes,
            "manom_s_prev": self.si[d, tp_c, l] - self.mprof_s[d, tp_c, l],
            "manom_s_next": self.si[d, tn_c, l] - self.mprof_s[d, tn_c, l],
            "manom_f_prev": (self.fi[d, tp_c, l] - self.mprof_f[d, tp_c, l]) / lanes,
            "manom_f_next": (self.fi[d, tn_c, l] - self.mprof_f[d, tn_c, l]) / lanes,
            "dayanom_s": self.dayanom_s[d, t, l], "dayanom_f_lane": self.dayanom_f[d, t, l] / lanes,
        }
        mins = np.full(len(d), np.inf)
        for k in OFFSETS:
            n = l + k
            ok = (n >= 0) & (n < self.L)
            nn = np.clip(n, 0, self.L - 1)
            for ch, arr, scale in (("s", self.si_nb, 1.0), ("f", self.fi_nb, self.lanes[nn]), ("o", self.oi, self.kcrit[nn] / K_OCC)):
                vt, vp, vn = arr[d, t, nn], arr[d, tp_c, nn], arr[d, tn_c, nn]
                f[f"dev_{ch}{k:+d}"] = np.where(ok, (vt - (vp + w * (vn - vp))) / scale, np.nan)
            f[f"lvl_s{k:+d}"] = np.where(ok, self.si_nb[d, t, nn] / self.vf[nn], np.nan)
            f[f"obs{k:+d}"] = np.where(ok, ~np.isnan(self.raw["s"][d, t, nn]), False).astype(np.int8)
            mins = np.minimum(mins, np.where(ok, self.si_nb[d, t, nn] / self.vf[nn], np.inf))
        f["min_lvl_s_nb"] = mins
        return pd.DataFrame(f)


def cells(target: np.ndarray, day_sel: np.ndarray, n: int | None, rng) -> tuple:
    d, t, l = np.nonzero(target & day_sel[:, None, None])
    if n is not None and len(d) > n:
        i = rng.choice(len(d), n, replace=False)
        d, t, l = d[i], t[i], l[i]
    return d, t, l


CONGESTED_EXTRA = 300_000  # extra congested training cells per panel, importance-weighted: +0.0008 public (v35)


def training_frame(panels, train_days: dict, n_per_panel: int, seed: int = 0, fill=None) -> pd.DataFrame:
    """fill: optional function(pdata) that sets neighbour fills before features are built (second pass)."""
    rng = np.random.default_rng(seed)
    frames = []
    for i, panel in enumerate(panels):
        pdata = PanelData(panel, "train", profile_days=train_days[panel])
        if fill is not None:
            fill(pdata)
        tr = load(panel, "train", "true_speed", "true_flow")
        pool = pdata.target & train_days[panel][:, None, None]
        d, t, l = cells(pdata.target, train_days[panel], n_per_panel, rng)
        w = np.ones(len(d))
        if CONGESTED_EXTRA:
            # Congested cells are ~4% of targets but ~21% of the squared speed error (LOG 2026-10-10). Add more
            # of them, flagged by a slow interpolated speed, and weight so that the loss stays unbiased.
            cong = pool & (pdata.si < 0.85 * pdata.vf[None, None, :])
            r_u = len(d) / max(pool.sum(), 1)
            taken = np.zeros_like(pool)
            taken[d, t, l] = True
            dc, tc, lc = np.nonzero(cong & ~taken)
            if len(dc) > CONGESTED_EXTRA:
                k = rng.choice(len(dc), CONGESTED_EXTRA, replace=False)
                dc, tc, lc = dc[k], tc[k], lc[k]
            r_c = len(dc) / max((cong & ~taken).sum(), 1)
            w_cong = r_u / (r_u + (1 - r_u) * r_c)
            w = np.where(cong[d, t, l], w_cong, 1.0)
            d, t, l = np.r_[d, dc], np.r_[t, tc], np.r_[l, lc]
            w = np.r_[w, np.full(len(dc), w_cong)]
        x = pdata.features(d, t, l)
        x["w"] = w
        x["y_s"] = tr["true_speed"][d, t, l] - x.si
        x["y_f"] = (tr["true_flow"][d, t, l] - pdata.fi[d, t, l]) / pdata.lanes[l]
        x["panel_code"] = i
        frames.append(x)
        print(f"  {panel}: {len(x):,} training cells", flush=True)
    return pd.concat(frames, ignore_index=True)


# Train-profile features are left out: the scored months are shifted from train, and dropping them in favour
# of the month-adapted profiles gained +0.0011 public (v16, LOG 2026-10-08). The columns stay in the frame.
FEATURES_DROP: set[str] = {"prof_s", "prof_f_lane", "anom_s_prev", "anom_s_next"}


def pseudo_frame(panel: str, split: str, code: int, rate: float, n: int, rng, day_sel: np.ndarray | None = None,
                 profile_days: np.ndarray | None = None) -> pd.DataFrame:
    """D5: self-supervised Task 1 rows from a month's own observed cells.

    Hides a random share `rate` of the observed, eligible, non-target cells (all channels, like the real
    mask), rebuilds the features on that layer and uses the hidden observations as labels. Task 1 is
    offline, so the scored months' released observations may be used this way.
    """
    a = load(panel, split, "speed", "flow", "occ", "target", "eligible")
    sel = np.ones(a["speed"].shape[0], bool) if day_sel is None else day_sel
    cand = ~np.isnan(a["speed"]) & ~np.isnan(a["flow"]) & a["eligible"] & ~a["target"] & sel[:, None, None]
    pseudo = cand & (rng.random(cand.shape) < rate)
    masked = {ch: np.where(pseudo, np.nan, a[ch]).astype(np.float32) for ch in ("speed", "flow", "occ")}
    masked["target"] = pseudo
    pdata = PanelData(panel, split, profile_days, masked)
    d, t, l = cells(pseudo, sel, n, rng)
    x = pdata.features(d, t, l)
    x["y_s"] = a["speed"][d, t, l] - x.si
    x["y_f"] = (a["flow"][d, t, l] - pdata.fi[d, t, l]) / pdata.lanes[l]
    x["panel_code"] = code
    x["w"] = 1.0
    return x


SEEDS = 2  # D3: models per target, averaged over seeds (2 seeds: +0.0003 public, v21)
# D4: density adjustment for Task 3. A third model predicts log density against the interpolation (L1);
# flow is moved so q/v goes `beta` of the way (in logs) to it. Holdout +0.0006 at beta 0.5 (LOG 2026-10-08).
DENSITY: dict | None = {"beta": 0.5, "cap": 1.25}  # +0.0003 public on top of seeds and D5 (v26)


class SeedAverage:
    """Mean prediction of several boosters; saves one file per seed."""

    def __init__(self, boosters: list):
        self.boosters = boosters

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return np.mean([b.predict(x) for b in self.boosters], axis=0)

    def save_model(self, path: str) -> None:
        for i, b in enumerate(self.boosters):
            b.save_model(path.replace(".txt", f"_s{i}.txt"))

    def feature_importance(self, kind: str) -> np.ndarray:
        return np.mean([b.feature_importance(kind) for b in self.boosters], axis=0)


def density_target(df: pd.DataFrame) -> np.ndarray:
    """log(true q / true v) minus log(interpolated q / v), from the residual targets in a training frame."""
    lanes = df.lanes.to_numpy()
    si, fi = df.si.to_numpy(), df.fi_lane.to_numpy() * lanes
    v, q = df.y_s.to_numpy() + si, df.y_f.to_numpy() * lanes + fi
    return np.log(np.clip(q, 1, None) / np.clip(v, 1, None)) - np.log(np.clip(fi, 1, None) / np.clip(si, 1, None))


def fit(df: pd.DataFrame) -> dict:
    feats = [c for c in df.columns if not c.startswith("y_") and c not in FEATURES_DROP and c != "w"]
    w = df["w"] if "w" in df else None
    out = {}
    if DENSITY:
        out["y_k"] = (lgb.train({**PARAMS, "objective": "l1"}, lgb.Dataset(df[feats], density_target(df), weight=w),
                                ROUNDS).free_dataset(), feats)
    for y in ("y_s", "y_f"):
        boosters = [lgb.train({**PARAMS, "seed": PARAMS["seed"] + i}, lgb.Dataset(df[feats], df[y], weight=w), ROUNDS).free_dataset()
                    for i in range(SEEDS)]
        out[y] = (boosters[0] if SEEDS == 1 else SeedAverage(boosters), feats)
    return out


def predict_panel(models: dict, panel: str, split: str, code: int, day_sel: np.ndarray | None = None,
                  profile_days: np.ndarray | None = None, fill=None) -> tuple[np.ndarray, np.ndarray, PanelData]:
    """Speed and flow arrays: interpolation everywhere, corrected at target cells."""
    pdata = PanelData(panel, split, profile_days)
    if fill is not None:
        fill(pdata)
    sel = np.ones(pdata.target.shape[0], bool) if day_sel is None else day_sel
    d, t, l = cells(pdata.target, sel, None, None)
    x = pdata.features(d, t, l)
    x["panel_code"] = code
    sp, fl = pdata.si.copy(), pdata.fi.copy()
    m_s, feats = models["y_s"]
    m_f, _ = models["y_f"]
    sp[d, t, l] = np.clip(x.si + m_s.predict(x[feats]), 1.0, None)
    fl[d, t, l] = np.clip(pdata.fi[d, t, l] + m_f.predict(x[feats]) * pdata.lanes[l], 1.0, None)
    if DENSITY and "y_k" in models:
        k_dens = (np.clip(pdata.fi[d, t, l], 1, None) / np.clip(pdata.si[d, t, l], 1, None)
                  * np.exp(models["y_k"][0].predict(x[feats])))
        v = np.maximum(sp[d, t, l], SPEED_FLOOR).astype(float)
        k_model = fl[d, t, l] / v
        factor = (k_dens / k_model) ** DENSITY["beta"]   # q/v moves beta of the way, v unchanged
        if DENSITY.get("cap"):   # bound the flow change: rare extreme density predictions are costly under RMSE
            factor = np.clip(factor, 1 / DENSITY["cap"], DENSITY["cap"])
        fl[d, t, l] = fl[d, t, l] * factor
    return plausible_speed(panel, sp, fl), fl, pdata


SPEED_FLOOR = 5.0  # observed speeds never go below about 5 km/h (train minimum 4.9)


def plausible_speed(panel: str, speed: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """Speed floor and density cap q/v <= k_jam: a near-zero speed implies an absurd accumulation in Task 3."""
    kjam = np.asarray(meta(panel)["k_jam"])
    return np.maximum(np.maximum(speed, SPEED_FLOOR), flow / kjam).astype(np.float32)


def cv(n_per_panel: int = 250_000, pseudo_rate: float = 0.0, pseudo_n: int = 200_000) -> None:
    hold = {}
    train_days = {}
    for p in PANELS:
        months = np.array([x[:7] for x in days(p, "train")["days"]])
        hold[p] = np.isin(months, ["2030-12", "2031-01", "2031-02"])
        train_days[p] = ~hold[p]
    df = training_frame(PANELS, train_days, n_per_panel)
    if pseudo_rate > 0:   # D5: self-supervised rows from the held-out months' own observed cells
        rng = np.random.default_rng(1)
        extra = [pseudo_frame(p, "train", i, pseudo_rate, pseudo_n, rng, hold[p], train_days[p]) for i, p in enumerate(PANELS)]
        df = pd.concat([df] + extra, ignore_index=True)
        print(f"  + {sum(len(e) for e in extra):,} self-supervised rows from the held-out months", flush=True)
    models = fit(df)
    rows = []
    for i, p in enumerate(PANELS):
        sp, fl, pdata = predict_panel(models, p, "train", i, hold[p], profile_days=train_days[p])
        base = (pdata.si, pdata.fi)
        for name, (s, f) in (("interp", base), ("lgbm", (sp, fl))):
            st, ph = s_state(p, "train", s, f, hold[p]), s_physics(p, "train", s, f, hold[p])
            rows.append({"panel": p, "method": name, "S_state": st["S_state"], "S_physics": ph["S_physics"],
                         "rmse_v_R3": st["R3_rmse_speed"], "rmse_q_R3": st["R3_rmse_flow_lane"]})
        print(rows[-2], "\n", rows[-1], flush=True)
    r = pd.DataFrame(rows)
    r["family"] = r.panel.map(FAMILY)
    print(r.groupby(["method", "family"])[["S_state", "S_physics"]].mean().groupby("method").mean().round(4))
    imp = pd.Series(models["y_s"][0].feature_importance("gain"), index=models["y_s"][1]).sort_values(ascending=False)
    print("speed model top features:", imp.head(12).round(0).to_dict())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cv", action="store_true")
    ap.add_argument("--n", type=int, default=250_000)
    args = ap.parse_args()
    if args.cv:
        cv(args.n)


if __name__ == "__main__":
    main()
