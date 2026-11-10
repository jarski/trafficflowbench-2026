"""Fast local re-implementations of the Task 1 and Task 3 scorers on dense arrays.

Both mirror the official scorers in organizer-repo/src (score_task1.py and
score_task3.py) and must agree with them; experiments/eda/task3_proxy_check.py
holds the comparison. Task 3 uses the observed accumulation change as the
conservation right-hand side (see tfb.task3_proxy), so it is only computable
where the unmasked truth is released: the train split.

Arrays are [day, tod, link] predictions; only target cells are read.
"""
from __future__ import annotations

import datetime as dt
from functools import lru_cache

import numpy as np
import pandas as pd

from tfb.paths import RELEASE
from tfb.task1_io import days, load, meta

REGIMES = ("R1", "R2", "R3")
SPEED_SCALE, FLOW_SCALE, SPEED_W, FLOW_W = 25.0, 600.0, 0.54, 0.46
EMPTY_FLOOR_VPH, EMPTY_SHARE = 50.0, 0.20


def s_state(panel: str, split: str, speed: np.ndarray, flow: np.ndarray, day_sel: np.ndarray | None = None) -> dict:
    """Task 1 score per regime and the panel mean (score_task1.score_panel); day_sel restricts the days scored."""
    a = load(panel, split, "target", "true_speed", "true_flow")
    lanes = np.clip(np.asarray(meta(panel)["lanes"], float), 1.0, None)
    regime = np.asarray(days(panel, split)["regime"])
    if day_sel is None:
        day_sel = np.ones(len(regime), bool)
    out = {}
    for r in REGIMES:
        m = a["target"] & ((regime == r) & day_sel)[:, None, None]
        if not m.any():
            out[r] = 0.0
            continue
        ps, pf = np.nan_to_num(speed[m], nan=0.0), np.nan_to_num(flow[m], nan=0.0)
        lane = np.broadcast_to(lanes, m.shape)[m]
        rs = np.sqrt(np.mean((ps - a["true_speed"][m]) ** 2))
        rq = np.sqrt(np.mean(((pf - a["true_flow"][m]) / lane) ** 2))
        out[r] = SPEED_W * max(0.0, 1 - rs / SPEED_SCALE) + FLOW_W * max(0.0, 1 - rq / FLOW_SCALE)
        out[f"{r}_rmse_speed"], out[f"{r}_rmse_flow_lane"] = float(rs), float(rq)
    out["S_state"] = float(np.mean([out[r] for r in REGIMES]))
    return out


@lru_cache(maxsize=None)
def _physics_static(panel: str, split: str) -> dict:
    """Per-link FD parameters and ramp validity, as score_task3 derives them."""
    m = meta(panel)
    net = RELEASE / "corridors" / panel / "network"
    links = pd.read_csv(net / "links.csv", dtype={"link_id": str}).drop_duplicates("link_id").set_index("link_id")
    fd = pd.read_csv(net / "fd_parameters.csv", dtype={"link_id": str}).groupby("link_id").mean(numeric_only=True)
    ids = m["links"]
    length = pd.to_numeric(links.length_km.reindex(ids), errors="coerce").fillna(1.0).to_numpy()
    vf = fd.free_speed_kmh.reindex(ids).fillna(pd.to_numeric(links.free_speed_kmh.reindex(ids), errors="coerce").fillna(100.0)).to_numpy()
    lanes = fd.lanes.reindex(ids).fillna(pd.to_numeric(links.lanes.reindex(ids), errors="coerce").fillna(1.0).clip(lower=1.0)).to_numpy()
    cap = fd.capacity_vph.reindex(ids).fillna(pd.to_numeric(links.capacity_vph.reindex(ids), errors="coerce")).to_numpy()
    kcrit = cap / np.clip(vf, 1, None)
    kjam = fd.k_jam.reindex(ids).to_numpy() if "k_jam" in fd else np.full(len(ids), np.nan)
    kjam = np.where(np.isnan(kjam), 100.0 * np.clip(lanes, 1, None), kjam)
    kjam = np.maximum(kjam, kcrit * 1.05)
    # Ramp validity at each (day, tod, link): score_task3.load_ramp_flows keeps valid ramp
    # records only and maps them through ramp_attachment_map; a link with no ramp of a type
    # in the topology is structurally valid for that type.
    r = load(panel, split, "ramp_flow", "ramp_pct", "ramp_eligible")
    valid = r["ramp_eligible"] & (r["ramp_pct"] >= 75) & np.isfinite(r["ramp_flow"])
    shape = valid.shape[:2] + (len(ids),)
    on_ok, off_ok = np.zeros(shape, bool), np.zeros(shape, bool)
    for j, (li, typ) in enumerate(zip(m["ramp_link_index"], m["ramp_type"])):
        if li < 0:
            continue
        tgt = on_ok if typ.upper() in ("OR", "ON") else off_ok if typ.upper() in ("FR", "OFF") else None
        if tgt is not None:
            tgt[:, :, li] |= valid[:, :, j]
    on_ok |= ~np.array([bool(s) for s in m["on_ramps"]])[None, None, :]
    off_ok |= ~np.array([bool(s) for s in m["off_ramps"]])[None, None, :]
    return {"length": length, "vf": vf, "lanes": lanes, "cap": cap, "kcrit": kcrit, "kjam": kjam, "ramp_ok": on_ok & off_ok}


def s_physics(panel: str, split: str, speed: np.ndarray, flow: np.ndarray, day_sel: np.ndarray | None = None) -> dict:
    """Task 3 score per regime and the panel mean, with the observed-accumulation flux; day_sel restricts the days."""
    st = _physics_static(panel, split)
    a = load(panel, split, "target", "eligible", "true_speed", "true_flow")
    info = days(panel, split)
    regime = np.asarray(info["regime"])
    dates = [dt.date.fromisoformat(d) for d in info["days"]]
    L = st["length"][None, None, :]
    lanes = np.clip(st["lanes"], 1.0, None)[None, None, :]
    if day_sel is None:
        day_sel = np.ones(len(regime), bool)
    out = {}
    for r in REGIMES:
        sel = np.flatnonzero((regime == r) & day_sel)
        if sel.size == 0:
            out[r] = 0.0
            continue
        tgt, elig = a["target"][sel], a["eligible"][sel]
        sp = np.where(tgt, speed[sel], a["true_speed"][sel])
        fl = np.where(tgt, flow[sel], a["true_flow"][sel])
        k = fl / np.clip(sp, 1.0, None)
        # S_FD over every eligible row of the regime's days, in per-lane units.
        kl, ql = k / lanes, fl / lanes
        capl, kcl, kjl = (st["cap"] / st["lanes"])[None, None, :], (st["kcrit"] / st["lanes"])[None, None, :], (st["kjam"] / st["lanes"])[None, None, :]
        qfd = np.where(kl <= kcl, st["vf"][None, None, :] * kl, capl / np.maximum(kjl - kcl, 1e-9) * np.maximum(kjl - kl, 0))
        s_fd = max(0.0, 1.0 - np.abs(ql - qfd)[elig].sum() / (np.abs(ql)[elig].sum() + 1e-9))
        sub = tgt & elig
        if sub.any() and float((fl[sub] < EMPTY_FLOOR_VPH).mean()) > EMPTY_SHARE:
            s_fd = 0.0
        # Transitions: next 5-min slot, across midnight only into the next calendar day of the same regime.
        n_sub = k * L
        n_obs = a["true_flow"][sel] / np.clip(a["true_speed"][sel], 1.0, None) * L
        ok = st["ramp_ok"][sel]
        num = den = 0.0
        e_now, e_next = elig[:, :-1], elig[:, 1:]
        m = e_now & e_next & ok[:, :-1]
        dsub, dobs = n_sub[:, 1:] - n_sub[:, :-1], n_obs[:, 1:] - n_obs[:, :-1]
        num += np.abs(dsub - dobs)[m].sum()
        den += np.abs(dobs)[m].sum()
        for i in range(len(sel) - 1):
            if (dates[sel[i + 1]] - dates[sel[i]]).days == 1:
                mm = elig[i, -1] & elig[i + 1, 0] & ok[i, -1]
                num += np.abs((n_sub[i + 1, 0] - n_sub[i, -1]) - (n_obs[i + 1, 0] - n_obs[i, -1]))[mm].sum()
                den += np.abs(n_obs[i + 1, 0] - n_obs[i, -1])[mm].sum()
        e_lwr = num / (den + 1e-9)
        s_lwr = max(0.0, 1.0 - min(1.0, e_lwr))
        out[r] = (s_fd + 2 * s_lwr) / 3.0
        out[f"{r}_S_FD"], out[f"{r}_S_LWR"], out[f"{r}_E_LWR"] = float(s_fd), float(s_lwr), float(e_lwr)
    out["S_physics"] = float(np.mean([out[r] for r in REGIMES]))
    return out
