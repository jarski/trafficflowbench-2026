"""Per-link Task 2 features at a forecast origin T, using only data timestamped at or before T.

Inputs per window: the 12-slot history T-60..T-5 (complete, as the released window
history is) and the masked-layer slot at T (Task 1 targets blank). Links are in
travel order, so +k is k links downstream.
"""
from __future__ import annotations

import numpy as np

from tfb.task1_io import meta

K_OCC = 285.0  # density_occ_linear_vehpkm = 285 * occupancy (total over lanes)
# Task 2 information set. True: timestamps <= T, including the masked-layer observation at T (rules: "at
# or before T"; decided 2026-10-01). False: timestamps < T only, in case the organizers rule so (thread 745700).
USE_T_SLOT = True


def at_origin(a: dict, d: int, T: int) -> dict:
    """Readings at slot T, or all-NaN when the T slot may not be used (features then fall back to T-5)."""
    return {ch: (a[ch][d, T] if USE_T_SLOT else np.full(a[ch].shape[2], np.nan, np.float32)) for ch in ("speed", "flow", "occ")}


# Families whose links masked at T are filled from their T-5 value plus their observed neighbours' change.
# Ongoing: bench +0.003 / +0.001 / +0.003, public +0.0010 (v31). Onset: worse on the bench (LOG 2026-10-10).
IMPUTE_T_SLOT: set[str] = {"ongoing"}
IMPUTE_MODEL = False  # learned estimate (tfb.t2_impute) instead of the neighbour-change rule


def impute_origin(at_t: dict, hist: dict, family: str, panel: str | None = None, T: int | None = None,
                  exclude_month: str | None = None, reach: int = 2) -> dict:
    """Causal estimate at T for links blank at T: own T-5 value + mean T-5 -> T change of observed links within +-reach.

    Uses only data at or before T. Links without an observed neighbour stay blank (the features fall back to T-5).
    """
    if family not in IMPUTE_T_SLOT:
        return at_t
    if IMPUTE_MODEL and panel is not None:
        from tfb import t2_impute   # imported here: t2_impute builds on this module
        return t2_impute.impute(panel, at_t, hist, T, exclude_month)
    out = {}
    for ch, x in at_t.items():
        prev = hist[ch][-1]
        delta = x - prev                                   # NaN where blank at T or at T-5
        num = np.zeros_like(delta, dtype=float)
        cnt = np.zeros_like(delta, dtype=float)
        for k in range(-reach, reach + 1):
            if k == 0:
                continue
            nb = _shift(delta, k)
            ok = ~np.isnan(nb)
            num += np.where(ok, nb, 0.0)
            cnt += ok
        est = np.where(cnt > 0, prev + num / np.maximum(cnt, 1), np.nan)
        out[ch] = np.where(np.isnan(x), est, x).astype(np.float32)
    return out


def ramp_slots(T: int) -> slice:
    """Ramp-flow slots averaged at the origin: T-5 and T, or only T-5."""
    return slice(T - 1, T + 1) if USE_T_SLOT else slice(T - 1, T)
NEIGHBOURS = (-3, -2, -1, 1, 2, 3)


def static_features(panel: str) -> dict[str, np.ndarray]:
    m = meta(panel)
    cap = np.asarray(m["capacity_vph"])
    lanes = np.asarray(m["lanes"])
    n = len(cap)
    nxt = lambda x: np.append(x[1:], x[-1])
    on = np.array([len([s for s in r.split(";") if s]) for r in m["on_ramps"]], float)
    off = np.array([len([s for s in r.split(";") if s]) for r in m["off_ramps"]], float)
    return {
        "lanes": lanes, "cap_lane": cap / np.clip(lanes, 1, None), "vf": np.asarray(m["free_speed_kmh"]),
        "length": np.asarray(m["length_km"]), "cap_dn_ratio": nxt(cap) / cap, "lanes_dn_diff": nxt(lanes) - lanes,
        "min_cap_dn5_ratio": np.array([cap[i + 1:i + 6].min() / cap[i] if i + 1 < n else 1.0 for i in range(n)]),
        "on_ramps": on, "off_ramps": off, "pos": np.arange(n) / max(n - 1, 1),
    }


def _shift(x: np.ndarray, k: int) -> np.ndarray:
    """Value of the link k positions downstream (k > 0) or upstream (k < 0); NaN off the corridor."""
    out = np.full_like(x, np.nan, dtype=float)
    if k > 0:
        out[:-k] = x[k:]
    elif k < 0:
        out[-k:] = x[:k]
    else:
        out[:] = x
    return out


def _window_max(x: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """Max over links at offsets lo..hi (inclusive), NaN-aware."""
    return np.nanmax(np.stack([_shift(x, k) for k in range(lo, hi + 1)]), axis=0)


def dynamic_features(panel: str, hist: dict[str, np.ndarray], at_t: dict[str, np.ndarray],
                     ramp_on: np.ndarray, ramp_off: np.ndarray) -> dict[str, np.ndarray]:
    """hist[ch]: [12, L] for T-60..T-5; at_t[ch]: [L] at T (NaN where blank); ramp_*: [L] flow at T-5..T."""
    m = meta(panel)
    vf, cap = np.asarray(m["free_speed_kmh"]), np.asarray(m["capacity_vph"])
    kcrit = np.asarray(m["critical_density"])
    f = {}
    for ch, scale in (("speed", vf), ("flow", cap), ("occ", kcrit / K_OCC)):
        h = hist[ch] / scale
        last = np.where(np.isnan(at_t[ch]), hist[ch][-1], at_t[ch]) / scale
        f[f"{ch}_last"] = last
        f[f"{ch}_m15"] = np.nanmean(np.vstack([h[-2:], last[None]]), axis=0)
        f[f"{ch}_m60"] = np.nanmean(h, axis=0)
        f[f"{ch}_slope"] = f[f"{ch}_m15"] - np.nanmean(h[:3], axis=0)
        f[f"{ch}_hist_min"] = np.nanmin(h, axis=0)
        f[f"{ch}_hist_max"] = np.nanmax(h, axis=0)
        for k in NEIGHBOURS:
            f[f"{ch}_last_n{k:+d}"] = _shift(last, k)
    dens = f["occ_last"]
    f["dens_max_dn5"] = _window_max(dens, 1, 5)
    f["dens_max_up5"] = _window_max(dens, -5, -1)
    f["speed_min_dn5"] = -_window_max(-f["speed_last"], 1, 5)
    f["dens_corr_max"] = np.full_like(dens, np.nanmax(dens))
    f["dens_rank"] = (np.argsort(np.argsort(-np.nan_to_num(dens, nan=-1))) / len(dens)).astype(float)
    f["dist_to_densest"] = np.arange(len(dens)) - np.nanargmax(np.nan_to_num(dens, nan=-1))
    f["ramp_on_frac"] = ramp_on / cap
    f["ramp_off_frac"] = ramp_off / cap
    return f
