"""Second Task 1 model family: a public notebook's feature set (CC0), retrained here with LightGBM.

Source: Kaggle dataset anubisyy/trafficflowbench-086517-public-assets (CC0-1.0), files state_model.py
(compute_features, simulated_mask, sampled_targets) and prepare_state.py (lag and lead features), from the
public notebook "CatBoost + LGBM + NNLS Pipeline | Top 50 | 0.86517" by anubisyy.

Each hidden cell gets 94 features:
  - time interpolation of speed and flow, and a flow baseline blending it with the topology neighbours;
  - the neighbours' current values and their differences;
  - the link's previous and next observation and the gaps to them;
  - the milepost neighbours at +-1 and +-2: their distance and same-time and interpolated values;
  - values at +-5 to +-60 minutes with quality flags.
The targets are the residuals of speed and per-lane flow against the two baselines. Training cells come from
synthetic masks on the train split's unmasked states: 20/30/50% of cells by a hashed day regime, two
realizations per month, 750 cells per day, as in the original.

Changes from the original:
  - LightGBM with our Task 1 parameters instead of GPU CatBoost, so no new dependency;
  - inputs come from our dense arrays;
  - training months are a parameter, for the Dec-Feb holdout.
"""
from __future__ import annotations

import hashlib
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import sparse

from tfb.paths import PANELS, RELEASE
from tfb.task1_io import days, load, meta

SEED = 20261001
RATE = np.array([.20, .30, .50], dtype=np.float32)
PER_DAY = 750
PARAMS = dict(objective="regression", learning_rate=0.05, num_leaves=255, min_data_in_leaf=200,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              verbose=-1, seed=13, deterministic=True, num_threads=10)
ROUNDS = 1000
REGIME = {"R1": 0, "R2": 1, "R3": 2}


def all_panels() -> list[str]:
    return [x["corridor_id"] for x in json.loads((RELEASE / "config" / "corridors.json").read_text())["panels"]]


def _adjacency(topo: pd.DataFrame, column: str, index: dict) -> sparse.csr_matrix:
    rows, cols = [], []
    for row in topo.itertuples(index=False):
        link = str(row.link_id)
        if link not in index:
            continue
        for other in str(getattr(row, column)).split(";"):
            if other in index:
                rows.append(index[link])
                cols.append(index[other])
    result = sparse.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(len(index), len(index)))
    result.data[:] = 1.0
    return result


class Panel:
    """Link data in the original's order (fd_parameters.csv); `ours[j]` is our index of its link j."""

    def __init__(self, name: str):
        net = RELEASE / "corridors" / name / "network"
        fd = pd.read_csv(net / "fd_parameters.csv", dtype={"link_id": str}).drop_duplicates("link_id")
        topo = pd.read_csv(net / "lwr_mainline_topology.csv", dtype=str)
        self.name, self.panel_no = name, all_panels().index(name)
        self.links = fd.link_id.astype(str).tolist()
        index = {k: i for i, k in enumerate(self.links)}
        self.lanes = fd.lanes.to_numpy(dtype=np.float32)
        self.vf = fd.free_speed_kmh.to_numpy(dtype=np.float32)
        self.cap = fd.capacity_vph.to_numpy(dtype=np.float32)
        self.length = fd.length_km.to_numpy(dtype=np.float32)
        self.incoming = _adjacency(topo, "incoming_link_ids", index)
        self.outgoing = _adjacency(topo, "outgoing_link_ids", index)
        m = meta(name)
        pos = {l: i for i, l in enumerate(m["links"])}
        assert set(pos) == set(self.links), name
        self.ours = np.array([pos[l] for l in self.links])
        self.milepost = np.asarray(m["milepost"], np.float32)[self.ours]
        self.sorted_links = np.argsort(self.milepost)
        self.link_rank = np.empty(len(self.links), dtype=np.int32)
        self.link_rank[self.sorted_links] = np.arange(len(self.links))


def _stamps(day_list: list[str]) -> list[str]:
    return [f"{d}T{s // 12:02d}:{s % 12 * 5:02d}:00Z" for d in day_list for s in range(288)]


def simulated_mask(panel_name, times, speed, flow, eligible, seed):
    dates = pd.Index([text[:10] for text in times])
    result = np.zeros(speed.shape, dtype=bool)
    regimes = np.empty(speed.shape[0], dtype=np.uint8)
    for date in dates.unique():
        rows = np.flatnonzero(dates == date)
        s = int.from_bytes(hashlib.blake2b(f"{seed}|{panel_name}|{date}".encode(), digest_size=8).digest(), "big")
        regime = s % 3
        rng = np.random.default_rng(s)
        regimes[rows] = regime
        result[rows] = (rng.random((len(rows), speed.shape[1])) < RATE[regime]) & eligible[rows] \
            & np.isfinite(speed[rows]) & np.isfinite(flow[rows])
    return result, regimes


def sampled_targets(mask, times, per_day, seed):
    dates = pd.Index([text[:10] for text in times])
    row_out, col_out = [], []
    for date in dates.unique():
        rows = np.flatnonzero(dates == date)
        coords = np.argwhere(mask[rows])
        if not len(coords):
            continue
        s = int.from_bytes(hashlib.blake2b(f"{seed}|sample|{date}|{mask.shape[1]}".encode(), digest_size=8).digest(), "big")
        chosen = coords[np.random.default_rng(s).choice(len(coords), size=min(per_day, len(coords)), replace=False)]
        row_out.append(rows[chosen[:, 0]])
        col_out.append(chosen[:, 1])
    return np.concatenate(row_out), np.concatenate(col_out)


def _graph_mean(graph, values):
    observed = np.isfinite(values).astype(np.float32)
    count = (graph @ observed.T).T
    mean = (graph @ np.nan_to_num(values, nan=0.0).T).T / np.maximum(count, 1)
    return np.where(count > 0, mean, np.nan), count


def compute_features(panel: Panel, times, observed_speed, observed_flow, rows, cols, regime):
    """The original's 54 base features and the two interpolation baselines (speed, per-lane flow)."""
    n_times, n_links = observed_speed.shape
    temporal_speed = pd.DataFrame(observed_speed).interpolate(axis=0, limit_direction="both").to_numpy(dtype=np.float32)
    temporal_flow = pd.DataFrame(observed_flow).interpolate(axis=0, limit_direction="both").to_numpy(dtype=np.float32)
    temporal_speed = np.where(np.isfinite(temporal_speed), temporal_speed, panel.vf[None, :])
    temporal_flow = np.where(np.isfinite(temporal_flow), temporal_flow, .6 * panel.cap[None, :])

    graph = (panel.incoming + panel.outgoing).tocsr()
    graph.data[:] = 1.0
    spatial_flow, spatial_count = _graph_mean(graph, observed_flow)
    flow_base = np.where(spatial_count > 0, .5 * temporal_flow + .5 * spatial_flow, temporal_flow)
    up_speed, up_count = _graph_mean(panel.incoming, observed_speed)
    down_speed, down_count = _graph_mean(panel.outgoing, observed_speed)
    up_flow, _ = _graph_mean(panel.incoming, observed_flow)
    down_flow, _ = _graph_mean(panel.outgoing, observed_flow)

    valid = np.isfinite(observed_speed)
    positions = np.arange(n_times, dtype=np.int32)[:, None]
    prev_index = np.maximum.accumulate(np.where(valid, positions, -1), axis=0)
    next_index = np.minimum.accumulate(np.where(valid, positions, n_times)[::-1], axis=0)[::-1]
    p = prev_index[rows, cols]
    n = next_index[rows, cols]
    prev_v = observed_speed[np.maximum(p, 0), cols].copy()
    next_v = observed_speed[np.minimum(n, n_times - 1), cols].copy()
    prev_q = observed_flow[np.maximum(p, 0), cols].copy()
    next_q = observed_flow[np.minimum(n, n_times - 1), cols].copy()
    prev_v[p < 0] = np.nan
    prev_q[p < 0] = np.nan
    next_v[n >= n_times] = np.nan
    next_q[n >= n_times] = np.nan
    prev_gap = np.where(p >= 0, rows - p, 300).astype(np.float32)
    next_gap = np.where(n < n_times, n - rows, 300).astype(np.float32)

    stamp = pd.to_datetime(pd.Index(times), utc=True)
    tod = stamp.hour.to_numpy(dtype=np.float32) + stamp.minute.to_numpy(dtype=np.float32) / 60
    dow = stamp.dayofweek.to_numpy(dtype=np.float32)
    lanes = np.maximum(panel.lanes[cols], 1)
    base_speed = temporal_speed[rows, cols]
    base_flow = flow_base[rows, cols] / lanes
    temporal_flow_lane = temporal_flow[rows, cols] / lanes
    spatial_flow_lane = spatial_flow[rows, cols] / lanes
    up_q = up_flow[rows, cols] / lanes
    down_q = down_flow[rows, cols] / lanes
    regime_vector = regime if len(regime) == len(rows) else regime[rows]
    features = [
        np.full(len(rows), panel.panel_no), cols, regime_vector,
        tod[rows], np.sin(2 * np.pi * tod[rows] / 24), np.cos(2 * np.pi * tod[rows] / 24), dow[rows],
        panel.vf[cols], panel.cap[cols] / lanes, lanes, panel.length[cols],
        base_speed, base_flow, temporal_flow_lane, spatial_flow_lane,
        base_speed / panel.vf[cols], base_flow * lanes / panel.cap[cols],
        prev_v, next_v, prev_q / lanes, next_q / lanes, prev_gap, next_gap,
        next_v - prev_v, (next_q - prev_q) / lanes,
        up_speed[rows, cols], down_speed[rows, cols], up_q, down_q,
        up_count[rows, cols], down_count[rows, cols], spatial_count[rows, cols],
        up_speed[rows, cols] - down_speed[rows, cols], up_q - down_q,
    ]
    for offset in (-2, -1, 1, 2):
        ranks = panel.link_rank[cols] + offset
        valid_neighbor = (ranks >= 0) & (ranks < n_links)
        neighbor = panel.sorted_links[np.clip(ranks, 0, n_links - 1)]
        neighbor_lanes = np.maximum(panel.lanes[neighbor], 1)
        features.extend((
            np.where(valid_neighbor, np.abs(panel.milepost[neighbor] - panel.milepost[cols]), np.nan),
            np.where(valid_neighbor, observed_speed[rows, neighbor], np.nan),
            np.where(valid_neighbor, observed_flow[rows, neighbor] / neighbor_lanes, np.nan),
            np.where(valid_neighbor, temporal_speed[rows, neighbor], np.nan),
            np.where(valid_neighbor, temporal_flow[rows, neighbor] / neighbor_lanes, np.nan),
        ))
    X = np.column_stack(features).astype(np.float32)
    X[~np.isfinite(X)] = np.nan
    return X, base_speed, base_flow


def features(panel: Panel, times, v, q, e, rows, cols, regime):
    """All 94 features: the base set plus values at -60..+60 minutes (prepare_state.py)."""
    X, bv, bq = compute_features(panel, times, v, q, rows, cols, regime)
    extra = []
    lanes = np.maximum(panel.lanes[cols], 1)
    for offset in (-12, -6, -3, -2, -1, 1, 2, 3, 6, 12):
        rr = rows + offset
        valid = (rr >= 0) & (rr < len(times))
        rr = np.clip(rr, 0, len(times) - 1)
        vv, qq = v[rr, cols], q[rr, cols] / lanes
        extra.extend([np.where(valid, vv, np.nan), np.where(valid, qq, np.nan),
                      np.where(valid, qq / np.maximum(vv, 1), np.nan),
                      np.where(valid & e[rr, cols] & np.isfinite(vv), 1, 0)])
    return np.column_stack([X, *extra]).astype(np.float32), bv, bq


def _month_matrices(panel: Panel, a: dict, names: tuple, day_idx: list[int]) -> list[np.ndarray]:
    """[time, link] matrices (days concatenated, the original's link order) of the given dense arrays."""
    return [a[n][day_idx].reshape(-1, a[n].shape[2])[:, panel.ours] for n in names]


def training_frame(months: tuple, panels=PANELS) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """X, speed residual and per-lane flow residual on synthetic masks of the train split's given months."""
    xs, yv, yq = [], [], []
    for name in panels:
        p = Panel(name)
        a = load(name, "train", "true_speed", "true_flow", "eligible")
        dl = days(name, "train")["days"]
        for month in months:
            idx = [d for d, x in enumerate(dl) if x[:7] == month]
            times = _stamps([dl[d] for d in idx])
            v, q, e = _month_matrices(p, a, ("true_speed", "true_flow", "eligible"), idx)
            for rep in (0, 1):
                seed = SEED + rep * 7919
                mask, regime = simulated_mask(name, times, v, q, e, seed)
                rows, cols = sampled_targets(mask, times, PER_DAY, seed)
                X, bv, bq = features(p, times, np.where(mask, np.nan, v), np.where(mask, np.nan, q), e, rows, cols, regime)
                xs.append(X)
                yv.append(v[rows, cols] - bv)
                yq.append(q[rows, cols] / p.lanes[cols] - bq)
        print(f"  {name}: synthetic-mask rows built", flush=True)
    return np.vstack(xs), np.concatenate(yv), np.concatenate(yq)


def fit(months: tuple) -> dict:
    X, yv, yq = training_frame(months)
    return {k: lgb.train(PARAMS, lgb.Dataset(X, y), ROUNDS).free_dataset() for k, y in (("speed", yv), ("flow", yq))}


def predict(models: dict, name: str, split: str, day_sel: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """[day, 288, link] speed and flow (our link order) at the Task 1 target cells of the selected days, NaN elsewhere.

    Days are processed by calendar month, as the original processes a month (a split) at a time."""
    p = Panel(name)
    a = load(name, split, "speed", "flow", "eligible", "target")
    info = days(name, split)
    dl, regime_of_day = info["days"], np.array([REGIME[r] for r in info["regime"]])
    sel = np.ones(len(dl), bool) if day_sel is None else day_sel
    speed = np.full(a["speed"].shape, np.nan, np.float32)
    flow = np.full(a["flow"].shape, np.nan, np.float32)
    for month in sorted({x[:7] for x, s in zip(dl, sel) if s}):
        idx = [d for d, x in enumerate(dl) if x[:7] == month and sel[d]]
        times = _stamps([dl[d] for d in idx])
        v, q, e, tgt = _month_matrices(p, a, ("speed", "flow", "eligible", "target"), idx)
        rows, cols = np.nonzero(tgt)
        regime = np.repeat(regime_of_day[idx], 288)[rows]   # per target cell, as the original's template regime
        X, bv, bq = features(p, times, v, q, e, rows, cols, regime)
        sv = np.clip(bv + models["speed"].predict(X), 1, 180)
        sq = np.clip((bq + models["flow"].predict(X)) * p.lanes[cols], 0, 150000)
        d_loc, slot = rows // 288, rows % 288
        dd = np.asarray(idx)[d_loc]
        speed[dd, slot, p.ours[cols]] = sv
        flow[dd, slot, p.ours[cols]] = sq
    return speed, flow
