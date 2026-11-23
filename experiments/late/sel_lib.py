"""Shared code for the selection-conditioned decoding experiment (2026-11-23). Nothing here edits src/tfb.

Three parts:
  1. the corrected replica of the organizers' Task 2 window selector (train analysis; it reads the truth);
  2. the "cap-bound" flag of a window, computed only from data stamped <= T (what a forecast may use);
  3. the production ongoing forecast for a train window out of fold, and the official window IoU.

Notation: slots are five minutes, H = 12 history slots (T-60..T-5), Z = 6 horizon slots (T+5..T+30).
Arrays are [day, 288, link] in travel order, as tfb.dense writes them.
"""
from __future__ import annotations

import os

import lightgbm as lgb
import numpy as np
import pandas as pd

from tfb import t2_ongoing as og
from tfb import t2_public
from tfb.dense import tod_index
from tfb.paths import CACHE, RELEASE
from tfb.t2_onset import T2_PANELS
from tfb.t2_rules import vcut
from tfb.task1_io import days, load, meta

H, Z, S = 12, 6, 288
CAP = 0.9            # config/synthetic_release_v1.json: maximum_persistence_iou
COVERAGE = 0.7       # minimum_history_coverage
SPACING = 72         # minimum_spacing_minutes 360
PER_CONDITION = 5
THREADS = 4          # the machine is shared: LightGBM and BLAS stay at 4 threads
LATE = CACHE / "late"
CODES = {p: i for i, p in enumerate(sorted(T2_PANELS))}

os.environ.setdefault("OMP_NUM_THREADS", str(THREADS))


# ---------------------------------------------------------------------------------------------- states
def fill_forward(x: np.ndarray) -> np.ndarray:
    """Carry the last finite value forward along time on one continuous timeline (across midnight)."""
    D, S_, L = x.shape
    f = x.reshape(D * S_, L)
    idx = np.where(np.isfinite(f), np.arange(D * S_)[:, None], 0)
    np.maximum.accumulate(idx, axis=0, out=idx)
    return f[idx, np.arange(L)[None, :]].reshape(D, S_, L)


def interp_time(x: np.ndarray) -> np.ndarray:
    """Linear interpolation along time between finite readings (two-sided: an analysis proxy, not causal)."""
    D, S_, L = x.shape
    f = x.reshape(D * S_, L).astype(float)
    t = np.arange(D * S_)
    out = np.full_like(f, np.nan)
    for l in range(L):
        ok = np.isfinite(f[:, l])
        if ok.any():
            out[:, l] = np.interp(t, t[ok], f[ok, l])
    return out.reshape(D, S_, L)


def iou(pred: np.ndarray, truth: np.ndarray) -> float:
    union = (pred | truth).sum()
    return 1.0 if union == 0 else float((pred & truth).sum() / union)


def cap_iou(state: np.ndarray, ref: int, elig: np.ndarray | None = None) -> np.ndarray:
    """[D, 288] IoU of the state at T + ref repeated over T+5..T+30 against the state there (NaN off range).

    ref = 0 repeats the state at the origin (the README's persistence), ref = -1 the state at T-5 (the
    baseline script's). With `elig`, only eligible horizon cells count, as in score_task2.score_window.
    """
    D, S_, L = state.shape
    out = np.full((D, S_), np.nan)
    for T in range(H, S_ - Z):
        r = state[:, T + ref][:, None, :]
        hor = state[:, T + 1:T + 1 + Z]
        if elig is not None:
            m = elig[:, T + 1:T + 1 + Z]
            inter, union = (r & hor & m).sum(axis=(1, 2)), ((r | hor) & m).sum(axis=(1, 2))
        else:
            inter, union = (r & hor).sum(axis=(1, 2)), (r | hor).sum(axis=(1, 2))
        out[:, T] = np.where(union > 0, inter / np.maximum(union, 1), 1.0)
    return out


def established(q: np.ndarray) -> np.ndarray:
    """[D, 288]: the history T-60..T-5 holds >= 2 queued cells and >= 2 on one link (tfb.t2_ongoing.is_established)."""
    D, S_, L = q.shape
    c = np.cumsum(np.concatenate([np.zeros((D, 1, L), np.int32), q.astype(np.int32)], axis=1), axis=1)
    out = np.zeros((D, S_), bool)
    per_link = c[:, H:S_] - c[:, :S_ - H]           # [D, S-H, L]: sum over slots T-H..T-1 for T = H..S-1
    out[:, H:] = (per_link.sum(axis=2) >= 2) & (per_link.max(axis=2) >= 2)
    return out


def coverage(elig: np.ndarray) -> np.ndarray:
    """[D, 288] share of eligible cells in the history T-60..T-5: window_index.history_coverage to 4 decimals."""
    D, S_, L = elig.shape
    e = elig.sum(axis=2).astype(np.int64)
    c = np.cumsum(np.concatenate([np.zeros((D, 1), np.int64), e], axis=1), axis=1)
    out = np.full((D, S_), np.nan)
    out[:, H:] = (c[:, H:S_] - c[:, :S_ - H]) / (H * L)
    return out


def any_ahead(x: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """[D, 288]: any of x[D, 288] over slots T+lo..T+hi inclusive, within the day."""
    out = np.zeros(x.shape, bool)
    for k in range(lo, hi + 1):
        if k > 0:
            out[:, :-k] |= x[:, k:]
        elif k < 0:
            out[:, -k:] |= x[:, :k]
        else:
            out |= x
    return out


# ---------------------------------------------------------------------------------------------- selector replica
class TrainTruth:
    """Train arrays of one panel and the states the replica needs."""

    def __init__(self, panel: str, n_days: int | None = None):
        a = load(panel, "train", "true_speed", "eligible")
        self.panel = panel
        self.speed = a["true_speed"][:n_days]
        self.elig = a["eligible"][:n_days]
        self.vc = vcut(panel)
        self.days = days(panel, "train")["days"][:n_days]
        self.months = np.array([d[:7] for d in self.days])
        self.q = (self.speed <= self.vc) & self.elig            # the logged queue flag: a missing reading is "no queue"
        self._cache: dict = {}

    def state(self, name: str) -> np.ndarray:
        if name not in self._cache:
            sp = self.speed
            self._cache[name] = {
                "logged": lambda: self.q,
                "ffill": lambda: fill_forward(sp) <= self.vc,
                "ffill_el": lambda: fill_forward(np.where(self.elig, sp, np.nan)) <= self.vc,
                "interp": lambda: interp_time(sp) <= self.vc,
                "interp_el": lambda: interp_time(np.where(self.elig, sp, np.nan)) <= self.vc,
            }[name]()
        return self._cache[name]

    def base(self) -> np.ndarray:
        """[D, 288]: every ongoing condition except the cap, spacing and count."""
        if "base" not in self._cache:
            valid = np.zeros(self.q.shape[:2], bool)
            valid[:, H:S - Z] = True
            self._cache["base"] = (valid & any_ahead(self.q.any(axis=2), 1, Z) & (coverage(self.elig) >= COVERAGE)
                                   & established(self.q))
        return self._cache["base"]

    def onset_ok(self) -> np.ndarray:
        if "onset" not in self._cache:
            anyq = self.q.any(axis=2)
            valid = np.zeros(self.q.shape[:2], bool)
            valid[:, H:S - Z] = True
            self._cache["onset"] = (valid & any_ahead(anyq, 1, Z) & (coverage(self.elig) >= COVERAGE)
                                    & ~established(self.q) & ~any_ahead(anyq, -H, 0))
        return self._cache["onset"]

    def cap(self, state: str = "interp", ref: int = 0, eligible_only: bool = False) -> np.ndarray:
        key = ("cap", state, ref, eligible_only)
        if key not in self._cache:
            self._cache[key] = cap_iou(self.state(state), ref, self.elig if eligible_only else None)
        return self._cache[key]


def scan(ok_onset: np.ndarray, ok_ongoing: np.ndarray, per_condition: int | None = PER_CONDITION,
         spacing: int = SPACING, start: int = 0) -> list[tuple]:
    """The forward scan: the first qualifying origins, >= `spacing` slots from every chosen one (both conditions).

    Stops when both conditions hold `per_condition` windows (None: no limit). `start`: first slot scanned.
    """
    D = ok_onset.shape[0]
    chosen, count, last = [], {"queue_onset": 0, "queue_ongoing": 0}, -10 ** 9
    for d in range(D):
        cand = np.flatnonzero(ok_onset[d] | ok_ongoing[d])
        for T in cand:
            t = d * S + int(T)
            if t < start or t - last < spacing:
                continue
            c = "queue_ongoing" if ok_ongoing[d, T] else "queue_onset"
            if per_condition is not None and count[c] >= per_condition:
                continue
            chosen.append((d, int(T), c))
            count[c] += 1
            last = t
            if per_condition is not None and min(count.values()) >= per_condition:
                return chosen
    return chosen


def replica(tt: TrainTruth, state: str = "interp", ref: int = 0, eligible_only: bool = False,
            day_idx: np.ndarray | None = None, per_condition: int | None = PER_CONDITION, start: int = 0) -> list[tuple]:
    """Corrected selector replica on a panel (or on the days `day_idx`, e.g. one month): [(day, T, condition)]."""
    ok_og = tt.base() & (tt.cap(state, ref, eligible_only) <= CAP)
    ok_on = tt.onset_ok()
    if day_idx is None:
        return scan(ok_on, ok_og, per_condition, start=start)
    return [(int(day_idx[d]), T, c) for d, T, c in scan(ok_on[day_idx], ok_og[day_idx], per_condition, start=start)]


def month_replica(tt: TrainTruth, **kw) -> list[tuple]:
    """The replica run on each train month separately, as if every month were a scored split."""
    out = []
    for mo in np.unique(tt.months):
        out += replica(tt, day_idx=np.flatnonzero(tt.months == mo), **kw)
    return out


def official(panel: str, split: str = "train") -> pd.DataFrame:
    """window_index.csv of one panel and split with day index d, slot T, and the gap to the previous origin."""
    w = pd.read_csv(RELEASE / "task2" / panel / split / "window_index.csv")
    dl = days(panel, split)["days"]
    w["d"] = [dl.index(x) for x in w.date]
    w["T"] = tod_index(w.forecast_origin)
    w["t"] = w.d * S + w["T"]
    w = w.sort_values("t").reset_index(drop=True)
    w["gap"] = w.t.diff()            # slots since the previous origin of either condition; NaN for the first
    return w


# ---------------------------------------------------------------------------------------------- data visible at T
def visible(masked: np.ndarray, d: int, T: int, history: np.ndarray | None = None, back: int = 36) -> np.ndarray:
    """[back + 1, L] speeds a forecast for origin (d, T) may read, slots T-back..T: the masked layer, with the
    window history T-60..T-5 complete. Nothing after T is touched.

    On the scored splits the masked layer already holds the complete history (it equals window_history.parquet
    there), so `history` is None. On train, pass the unmasked history [12, L] to emulate that.
    """
    flat = masked.reshape(-1, masked.shape[2])
    t = d * S + T
    lo = max(t - back, 0)
    v = np.full((back + 1, masked.shape[2]), np.nan, np.float32)
    v[back - (t - lo):] = flat[lo:t + 1]
    if history is not None:
        v[back - H:back] = history
    return v


def last_reading(v: np.ndarray) -> np.ndarray:
    """[L] the latest finite speed at or before T in the visible block (NaN if none)."""
    idx = np.where(np.isfinite(v), np.arange(len(v))[:, None], -1).max(axis=0)
    return np.where(idx >= 0, v[np.maximum(idx, 0), np.arange(v.shape[1])], np.nan)


def flag_window(v: np.ndarray, elig: np.ndarray, vc: np.ndarray, gap: float, lag: int = 1) -> dict:
    """Was the origin `lag` steps before T a qualifying ongoing origin in everything but the cap? Data <= T only.

    v: [back + 1, L] visible speeds for slots T-back..T; elig: [back + 1, L] eligibility flags of the same slots
    (published for blank cells too); gap: slots since the previous window origin of this panel and split (NaN: none).
    The earlier origin T' = T - lag qualifies when
      spacing   T' is at least 72 slots after the previous origin (or there is none);
      coverage  its history T'-60..T'-5 is >= 0.7 eligible;
      queue     that history holds an established queue (>= 2 queued readings, >= 2 on one link).
    A queue in its horizon is not tested: the window at T is ongoing, so T' sees the same queue.
    """
    back = len(v) - 1
    hist = slice(back - lag - H, back - lag)            # slots T'-12..T'-1
    q = (v[hist] <= vc) & elig[hist]
    spacing = bool(np.isnan(gap) or gap - lag >= SPACING)
    cov = float(elig[hist].mean())
    est = bool(q.sum() >= 2 and q.sum(axis=0).max() >= 2)
    return {"spacing_ok": spacing, "coverage": cov, "established": est,
            "flag": bool(spacing and cov >= COVERAGE and est)}


# ---------------------------------------------------------------------------------------------- production forecast, train windows
class Production:
    """The production ongoing forecast for train windows, out of fold (the window's month left out of both models).

    ours:   LightGBM on cache/t2_ongoing_train.parquet (tfb.t2_ongoing features), `seeds` models per fold;
    public: tfb.t2_public refitted without the month (cache/bench/t2_public_without_<month>.txt, LOG 2026-10-19);
    combined as pipeline_v1.predict_window does: on our candidate cells (1 - w) ours + w public, off them
    off_scale * public, then the expected-IoU set over every link and step (choose_window).
    """

    def __init__(self, seeds: int = 1, frame: str | None = None, tag: str = "prod"):
        self.seeds, self.tag = seeds, tag
        self.frame_path = frame or str(CACHE / "t2_ongoing_train.parquet")
        self._folds: dict = {}
        self._public: dict = {}
        self._panels: dict = {}
        self._pub_panels: dict = {}
        self._frame = None
        self.feats: list | None = None

    def frame(self) -> pd.DataFrame:
        if self._frame is None:
            df = pd.read_parquet(self.frame_path)
            df["panel_code"] = df.panel.map(CODES)
            self._frame = df
            self.feats = [c for c in df.columns if c not in og.FEATURES_DROP]
        return self._frame

    def features(self) -> list:
        if self.feats is None:
            import pyarrow.parquet as pq
            cols = pq.read_schema(self.frame_path).names + ["panel_code"]
            self.feats = [c for c in cols if c not in og.FEATURES_DROP]
        return self.feats

    def fold(self, month: str) -> list:
        """Boosters trained without `month` (saved under cache/late so later runs only load them)."""
        if month not in self._folds:
            paths = [LATE / f"sel_{self.tag}_without_{month}_s{i}.txt" for i in range(self.seeds)]
            if all(p.exists() for p in paths):
                self._folds[month] = [lgb.Booster(model_file=str(p)) for p in paths]
            else:
                df = self.frame()
                tr = df[df.month != month]
                boosters = []
                for i, p in enumerate(paths):
                    if p.exists():
                        boosters.append(lgb.Booster(model_file=str(p)))
                        continue
                    params = {**og.PARAMS, "seed": og.PARAMS["seed"] + i, "num_threads": THREADS}
                    b = lgb.train(params, lgb.Dataset(tr[self.feats], tr.y, weight=tr.w), og.ROUNDS).free_dataset()
                    LATE.mkdir(parents=True, exist_ok=True)
                    b.save_model(str(p))
                    boosters.append(b)
                self._folds[month] = boosters
        return self._folds[month]

    def public(self, month: str) -> lgb.Booster:
        if month not in self._public:
            path = CACHE / "bench" / f"t2_public_without_{month}.txt"
            self._public[month] = lgb.Booster(model_file=str(path))
        return self._public[month]

    def panel(self, name: str) -> og.TrainPanel:
        if name not in self._panels:
            self._panels[name] = og.TrainPanel(name)
            self._pub_panels[name] = t2_public.PubPanel(name)
        return self._panels[name]

    def probabilities(self, panel: str, d: int, T: int, p_ours: np.ndarray | None = None) -> dict:
        """{"ours": [6, L] (0 off candidates), "cand": [6, L] bool, "public": [6, L], "x": feature rows}."""
        tp = self.panel(panel)
        pp = self._pub_panels[panel]
        month = tp.months[d]
        L = len(pp.links)
        x = tp.window(d, T)
        ours, cand = np.zeros((Z, L)), np.zeros((Z, L), bool)
        if len(x):
            x["panel_code"] = CODES[panel]
            hh, ll = x.h.to_numpy() - 1, x.link.to_numpy()
            if p_ours is None:
                p_ours = np.mean([b.predict(x[self.features()], num_threads=THREADS) for b in self.fold(month)], axis=0)
            ours[hh, ll], cand[hh, ll] = p_ours, True
        a = tp.a
        origin = pd.Timestamp(tp_day(panel, d), tz="UTC") + pd.Timedelta(minutes=5 * int(T))
        X = pp.features(a["true_speed"][d, T - H:T][:, pp.ours], a["true_flow"][d, T - H:T][:, pp.ours], origin, "queue_ongoing")
        pub_their = self.public(month).predict(X, num_threads=THREADS).reshape(Z, L)
        pub = np.empty_like(pub_their)
        pub[:, pp.ours] = pub_their
        return {"ours": ours, "cand": cand, "public": pub, "x": x}


def tp_day(panel: str, d: int) -> str:
    return days(panel, "train")["days"][d]


def combine(ours: np.ndarray, cand: np.ndarray, public: np.ndarray, weight: float | None = None,
            off_scale: float | None = None) -> np.ndarray:
    """[6, L] the production probability: pipeline_v1.predict_window with tfb.t2_public's two constants."""
    w = t2_public.WEIGHT if weight is None else weight
    off = t2_public.OFF_SCALE if off_scale is None else off_scale
    if off is None:
        off = w
    return np.where(cand, (1 - w) * ours + w * public, off * public)


def choose(p: np.ndarray) -> np.ndarray:
    """[6, L] bool: the expected-IoU set over every link and step (t2_ongoing.choose_window)."""
    steps = np.repeat(np.arange(1, Z + 1), p.shape[1])
    return og.choose_window(p.ravel(), steps).reshape(p.shape)


def window_iou(pred: np.ndarray, truth: np.ndarray, elig: np.ndarray) -> float:
    """score_task2.score_window on arrays: IoU over the eligible target cells of the window ([6, L] each)."""
    if not elig.any():
        return 0.0
    return iou(pred & elig, truth & elig)


def step_iou(pred: np.ndarray, truth: np.ndarray, elig: np.ndarray) -> list:
    return [iou(pred[j] & elig[j], truth[j] & elig[j]) for j in range(Z)]
