"""Shared helpers for the blank-span experiments (2026-11-23).

In the scored months the 90 minutes after every Task 2 window origin T (slots T+1..T+18) are blank on every
link, the 60 minutes before it (T-12..T-1) are shown unmasked and hold no Task 1 target, and slot T carries
the ordinary cell mask (experiments/late/span_facts.py). The train split has no such span. This module

  * reads the official origins of a split (`official_origins`),
  * chooses emulated origins on train months with the corrected selector replica of
    experiments/late/selector_check.py, per panel and month, 5 onset + 5 ongoing (`emulated_origins`),
  * builds the emulated masked layer of the train split (`emulated_layer`: spans blanked, histories unmasked
    and removed from the targets), and
  * makes the production code read it (`install`): tfb.task1_io.load is wrapped in every module that imported
    it, in this process only. Nothing on disk changes.

Scores: `score` runs the exact replicas tfb.metrics.s_state / s_physics on a chosen target set.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import selector_check as SC  # noqa: E402  (the corrected selector replica)

from tfb import metrics, t1_model, t1_public, t1_unet, task1_io  # noqa: E402
from tfb import t2_rules as R  # noqa: E402
from tfb.paths import CACHE, PANELS, RELEASE, TASK2_EXCLUDED  # noqa: E402

SPAN, HIST, S = 18, 12, 288
LATE = CACHE / "late"
LATE.mkdir(parents=True, exist_ok=True)
HOLD_MONTHS = ("2030-12", "2031-01", "2031-02")
TRAIN_MONTHS = ("2030-06", "2030-07", "2030-08", "2030-09", "2030-10", "2030-11")
T2_PANELS = tuple(p for p in PANELS if p not in TASK2_EXCLUDED)
FAMILY = {p: p.rsplit("_", 1)[0] for p in PANELS}
_ORIG_LOAD = task1_io.load


def tod(ts: str) -> int:
    return int(ts[11:13]) * 12 + int(ts[14:16]) // 5


def day_months(panel: str, split: str = "train") -> np.ndarray:
    return np.array([d[:7] for d in task1_io.days(panel, split)["days"]])


def official_origins(panel: str, split: str) -> list[tuple[int, int, str]]:
    """[(day index, origin slot T, condition)] of the released Task 2 windows; empty for the two panels without Task 2."""
    path = RELEASE / "task2" / panel / split / "window_index.csv"
    if not path.exists():
        return []
    dpos = {x: i for i, x in enumerate(task1_io.days(panel, split)["days"])}
    wi = pd.read_csv(path)
    return [(dpos[r.date], tod(r.forecast_origin), r.condition) for r in wi.itertuples()]


def queue_state(panel: str):
    """Queue flags on the train truth, eligibility, and the flags on the time-filled state (the persistence cap)."""
    a = _ORIG_LOAD(panel, "train", "true_speed", "eligible")
    vc = R.vcut(panel)
    return (a["true_speed"] <= vc) & a["eligible"], a["eligible"], SC.ffill_time(a["true_speed"]) <= vc


def emulated_origins(months=HOLD_MONTHS, per_condition: int = 5) -> dict:
    """{panel: [(day, T, condition)]}: the corrected selector run separately on each month, as on a scored month."""
    path = LATE / f"span_origins_{'_'.join(months)}_{per_condition}.json"
    if path.exists():
        return {p: [tuple(x) for x in v] for p, v in json.loads(path.read_text()).items()}
    out = {}
    for panel in PANELS:
        out[panel] = []
        if panel not in T2_PANELS:
            continue
        q, elig, qf = queue_state(panel)
        mo = day_months(panel)
        for m in months:
            idx = np.flatnonzero(mo == m)
            out[panel] += [(int(idx[d]), int(T), c) for d, T, c in SC.selector(q[idx], elig[idx], qf[idx], per_condition)]
    path.write_text(json.dumps(out))
    return out


def slot_masks(n_days: int, origins) -> tuple[np.ndarray, np.ndarray]:
    """([day, 288] blank-span slots T+1..T+18, [day, 288] unmasked history slots T-12..T-1). A span that runs past
    midnight continues into the next day of the array (none does in the scored months)."""
    span, hist = np.zeros(n_days * S, bool), np.zeros(n_days * S, bool)
    for d, T, _ in origins:
        t = d * S + T
        span[t + 1:t + 1 + SPAN] = True
        hist[max(t - HIST, 0):t] = True
    hist &= ~span          # a later window's history inside an earlier span stays blank
    return span.reshape(n_days, S), hist.reshape(n_days, S)


def emulated_layer(panel: str, origins, unmask_history: bool = True) -> dict:
    """The train split's masked layer as a scored month would show it: speed / flow / occ / target plus the masks."""
    a = _ORIG_LOAD(panel, "train", "speed", "flow", "occ", "target", "true_speed", "true_flow", "true_occ")
    span, hist = slot_masks(a["speed"].shape[0], origins)
    out = {"span": span, "hist": hist}
    for ch in ("speed", "flow", "occ"):
        x = a[ch]
        if unmask_history:
            x = np.where(hist[:, :, None], a["true_" + ch], x)
        out[ch] = np.where(span[:, :, None], np.nan, x).astype(np.float32)
    out["target"] = a["target"] & ~hist[:, :, None] if unmask_history else a["target"]
    return out


_LAYERS: dict = {}


def _load(panel: str, split: str, *names: str) -> dict:
    out = _ORIG_LOAD(panel, split, *names)
    lay = _LAYERS.get((panel, split))
    if lay:
        for n in names:
            if n in lay:
                out[n] = lay[n]
    return out


def install(layers: dict | None = None, split: str = "train") -> None:
    """Make the production modules read these layers ({panel: {speed, flow, occ, target}}); install() restores the release."""
    _LAYERS.clear()
    for p, lay in (layers or {}).items():
        _LAYERS[(p, split)] = {k: v for k, v in lay.items() if k in ("speed", "flow", "occ", "target")}
    for mod in (task1_io, t1_model, t1_unet, t1_public, metrics):
        mod.load = _load


def set_target(panel: str, target: np.ndarray | None, split: str = "train") -> None:
    """Score on this target set (None: back to the installed layer's)."""
    key = (panel, split)
    if target is None:
        _LAYERS.get(key, {}).pop("_t", None)
        if key in _LAYERS and "_target0" in _LAYERS[key]:
            _LAYERS[key]["target"] = _LAYERS[key].pop("_target0")
        return
    lay = _LAYERS.setdefault(key, {})
    if "_target0" not in lay and "target" in lay:
        lay["_target0"] = lay["target"]
    lay["target"] = target


def score(panel: str, speed: np.ndarray, flow: np.ndarray, day_sel: np.ndarray, target: np.ndarray | None = None,
          physics: bool = True) -> dict:
    """S_state and S_physics (exact replicas) of [day, 288, link] predictions on the selected train days."""
    if target is not None:
        set_target(panel, target)
    try:
        st = metrics.s_state(panel, "train", speed, flow, day_sel)
        out = {"S_state": st["S_state"]}
        if physics:
            ph = metrics.s_physics(panel, "train", speed, flow, day_sel)
            out.update(S_physics=ph["S_physics"], S_FD=float(np.mean([ph.get(f"{r}_S_FD", 0.0) for r in metrics.REGIMES])),
                       S_LWR=float(np.mean([ph.get(f"{r}_S_LWR", 0.0) for r in metrics.REGIMES])))
    finally:
        if target is not None:
            set_target(panel, None)
    return out


def macro(df: pd.DataFrame, cols) -> pd.Series:
    """Leaderboard aggregation: panels -> corridor family -> overall, equal weights."""
    return df.assign(family=df.panel.map(FAMILY)).groupby("family")[list(cols)].mean().mean()
