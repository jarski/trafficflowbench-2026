"""All-link ongoing queue classifier ported from a public notebook (CC0) and retrained here on the train split.

Source: Kaggle dataset anubisyy/trafficflowbench-086517-public-assets (CC0-1.0), file queue_ongoing.py,
from the public notebook "CatBoost + LGBM + NNLS Pipeline | Top 50 | 0.86517" by anubisyy. The feature
code and the training procedure below follow that file. The model scores every link at T+5..T+30 from its
last 60 minutes, T-60..T-5 (strictly before T), with 36 features:
  - last observed speed and flow and their age;
  - recent and earlier means and their change;
  - mean and minimum speed, mean and maximum flow, coverage and queued steps;
  - the topology neighbours' last values;
  - time of day, day of week, link data and the step.
Training uses up to two origins per day and condition, at least 6 h apart, drawn with a seeded hash from
the train split's unmasked states. The last train month is the early-stopping set.

Changes from the original:
  - inputs come from our dense arrays, which equal window_history at float32 (LOG 2026-10-19);
  - LightGBM runs deterministically;
  - months can be left out of training for the bench.

Ongoing forecasts average its probabilities with our ongoing model's over every link, weight WEIGHT, then
take the expected-IoU set (pipeline_v1.predict_window). Public leaderboard: +0.0027 with the original
checkpoint (v40, LOG 2026-10-19).
"""
from __future__ import annotations

import hashlib
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import sparse

from tfb.paths import RELEASE, TASK2_EXCLUDED
from tfb.task1_io import days, load, meta

WEIGHT = 0.35   # probability weight of this model on our candidate cells in ongoing windows; 0 turns it off.
#                 Public leaderboard (LOG 2026-10-29/30, off-scale 0.5): 0.2 +0.0004, 0.3 +0.0006, 0.35 +0.0007, 0.5 0; 0.65 with
#                 off-scale 0.65 -0.0026.
OFF_SCALE: float | None = 0.45   # scale of this model's probability off our candidate cells (ours is 0 there); None = WEIGHT.
#                 At weight 0.35: 0.45 +0.0001 over 0.5, 0.65 -0.0007; at weight 0.5, 0.35 cost 0.0008.
HISTORY, HORIZON = 12, 6
SEED = 20260930
VALID_MONTH = "2031-02"
PARAMS = {"objective": "binary", "metric": "binary_logloss", "learning_rate": 0.045, "num_leaves": 31,
          "max_depth": 8, "min_data_in_leaf": 250, "feature_fraction": .85, "bagging_fraction": .85,
          "bagging_freq": 1, "lambda_l2": 8.0, "max_bin": 127, "num_threads": 8, "verbosity": -1, "seed": SEED,
          "deterministic": True, "force_row_wise": True}
ROUNDS, PATIENCE = 700, 60


def panel_names() -> list[str]:
    """Task 2 panels in the original's order, which defines its panel-number feature."""
    return [e["corridor_id"] for e in json.loads((RELEASE / "config" / "corridors.json").read_text())["panels"]
            if e["corridor_id"] not in TASK2_EXCLUDED]


def _avg(values):
    valid = np.isfinite(values)
    count = valid.sum(axis=0)
    return np.where(count > 0, np.nansum(values, axis=0) / np.maximum(count, 1), np.nan)


def _min_finite(values):
    result = np.where(np.isfinite(values), values, np.inf).min(axis=0)
    return np.where(np.isfinite(result), result, np.nan)


class PubPanel:
    """Link data in the original's order (fd_parameters.csv); `ours[j]` is our index of its link j."""

    def __init__(self, name: str):
        net = RELEASE / "corridors" / name / "network"
        fd = pd.read_csv(net / "fd_parameters.csv", dtype={"link_id": str}).drop_duplicates("link_id")
        topo = pd.read_csv(net / "lwr_mainline_topology.csv", dtype=str)
        self.name, self.panel_no = name, panel_names().index(name)
        self.links = fd.link_id.astype(str).tolist()
        self.link_index = {k: i for i, k in enumerate(self.links)}
        self.lanes = fd.lanes.to_numpy(dtype=np.float32)
        self.vf = fd.free_speed_kmh.to_numpy(dtype=np.float32)
        self.cap = fd.capacity_vph.to_numpy(dtype=np.float32)
        self.length = fd.length_km.to_numpy(dtype=np.float32)
        self.cut = (fd.v_cut if "v_cut" in fd else 0.6 * fd.free_speed_kmh).to_numpy(dtype=np.float32)
        self.incoming = self._adjacency(topo, "incoming_link_ids")
        self.outgoing = self._adjacency(topo, "outgoing_link_ids")
        pos = {l: i for i, l in enumerate(meta(name)["links"])}
        assert set(pos) == set(self.links), name
        self.ours = np.array([pos[l] for l in self.links])

    def _adjacency(self, topo, column):
        rows, cols = [], []
        for row in topo.itertuples(index=False):
            link = str(row.link_id)
            if link not in self.link_index:
                continue
            for other in str(getattr(row, column)).split(";"):
                if other in self.link_index:
                    rows.append(self.link_index[link])
                    cols.append(self.link_index[other])
        result = sparse.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                                   shape=(len(self.links), len(self.links)))
        result.data[:] = 1.0
        return result

    def _neighboring(self, graph, values):
        valid = np.isfinite(values).astype(np.float32)
        count = graph @ valid
        mean = graph @ np.nan_to_num(values, nan=0.0)
        return np.where(count > 0, mean / np.maximum(count, 1), np.nan)

    def features(self, speed, flow, origin: pd.Timestamp, condition: str) -> np.ndarray:
        """Rows (step, link), step-major, from the [12, link] history T-60..T-5 in this panel's link order."""
        if speed.shape != (HISTORY, len(self.links)) or flow.shape != speed.shape:
            raise ValueError((self.name, speed.shape, flow.shape))
        speed = speed.astype(np.float32) / self.vf[None, :]
        flow = flow.astype(np.float32) / self.cap[None, :]
        speed = np.where(np.isfinite(speed), speed, np.nan)
        flow = np.where(np.isfinite(flow), flow, np.nan)
        valid = np.isfinite(speed)
        positions = np.where(valid, np.arange(HISTORY)[:, None], -1).max(axis=0)
        columns = np.arange(len(self.links))
        last_speed = np.where(positions >= 0, speed[np.maximum(positions, 0), columns], np.nan)
        last_flow = np.where(positions >= 0, flow[np.maximum(positions, 0), columns], np.nan)
        age = np.where(positions >= 0, HISTORY - 1 - positions, 12)
        recent_speed, prior_speed = _avg(speed[-3:]), _avg(speed[-6:-3])
        recent_flow, prior_flow = _avg(flow[-3:]), _avg(flow[-6:-3])
        queue_history = np.where(valid, speed <= (self.cut / self.vf)[None, :], False)
        last_queue = last_speed <= self.cut / self.vf
        hour = origin.hour + origin.minute / 60
        constants = lambda value: np.full(len(self.links), value, dtype=np.float32)
        with np.errstate(invalid="ignore"):
            max_flow = np.nanmax(np.where(np.isfinite(flow), flow, -np.inf), axis=0)
        features = [
            constants(self.panel_no), np.arange(len(self.links), dtype=np.float32),
            self.vf, self.cap / np.maximum(self.lanes, 1), self.lanes, self.length,
            constants(hour), constants(np.sin(2 * np.pi * hour / 24)), constants(np.cos(2 * np.pi * hour / 24)),
            constants(origin.dayofweek), constants(1 if condition == "queue_ongoing" else 0),
            last_speed, last_flow, constants(0) + age, recent_speed, prior_speed,
            recent_speed - prior_speed, recent_flow, prior_flow, recent_flow - prior_flow,
            _avg(speed), _min_finite(speed), _avg(flow), max_flow,
            valid.mean(axis=0), queue_history.sum(axis=0).astype(np.float32), last_queue.astype(np.float32),
            self._neighboring(self.incoming, last_speed), self._neighboring(self.outgoing, last_speed),
            self._neighboring(self.incoming, last_flow), self._neighboring(self.outgoing, last_flow),
            self._neighboring(self.incoming, last_queue.astype(np.float32)),
            self._neighboring(self.outgoing, last_queue.astype(np.float32)),
            constants(np.isfinite(speed).mean()), constants(np.nanmin(speed) if np.isfinite(speed).any() else 1.0),
        ]
        base = np.column_stack(features).astype(np.float32)
        X = np.column_stack([np.tile(base, (HORIZON, 1)), np.repeat(np.arange(1, HORIZON + 1), len(self.links))])
        X = X.astype(np.float32)
        X[~np.isfinite(X)] = np.nan
        return X


# ------------------------------------------------------------------ training (the original's collect / fit_model)
def _candidates(pp: PubPanel, speed, eligible) -> dict:
    queued = (speed <= pp.cut[None, :]) & eligible & np.isfinite(speed)
    active = queued.any(axis=1)
    out = {"queue_onset": [], "queue_ongoing": []}
    for t in range(HISTORY, len(speed) - HORIZON):
        if np.isfinite(speed[t - HISTORY:t]).mean() < .7:
            continue
        if not active[t + 1:t + HORIZON + 1].any():
            continue
        if not active[t - HISTORY:t].any() and not active[t]:
            out["queue_onset"].append(t)
        if active[t]:
            out["queue_ongoing"].append(t)
    return out


def _choose(candidates: dict, panel: str, date: str) -> list[tuple]:
    out = []
    for condition, values in candidates.items():
        seed = int.from_bytes(hashlib.blake2b(f"{SEED}|{panel}|{date}|{condition}".encode(), digest_size=8).digest(), "big")
        chosen = []
        for t in np.random.default_rng(seed).permutation(values):
            if all(abs(t - other) >= 72 for other in chosen):
                chosen.append(int(t))
            if len(chosen) >= 2:
                break
        out.extend((condition, t) for t in chosen)
    return out


def windows(exclude: tuple = (), valid_month: str = VALID_MONTH) -> tuple[list, list]:
    """(X, y, weight) per training window and per early-stopping window; months in `exclude` are left out."""
    train, valid = [], []
    for name in panel_names():
        pp = PubPanel(name)
        a = load(name, "train", "true_speed", "true_flow", "eligible")
        for d, date in enumerate(days(name, "train")["days"]):
            month = date[:7]
            if month in exclude:
                continue
            speed, flow = a["true_speed"][d][:, pp.ours], a["true_flow"][d][:, pp.ours]
            eligible = a["eligible"][d][:, pp.ours]
            times = pd.date_range(date, periods=288, freq="5min", tz="UTC")
            for condition, t in _choose(_candidates(pp, speed, eligible), name, date):
                X = pp.features(speed[t - HISTORY:t], flow[t - HISTORY:t], times[t], condition)
                future = speed[t + 1:t + HORIZON + 1]
                y = ((future <= pp.cut[None, :]) & np.isfinite(future)).reshape(-1)
                ok = (eligible[t + 1:t + HORIZON + 1] & np.isfinite(future)).reshape(-1)
                if ok.sum() < .6 * len(ok):
                    continue
                (valid if month == valid_month else train).append(
                    (X[ok], y[ok].astype(np.uint8), np.full(ok.sum(), 1000 / ok.sum(), dtype=np.float32)))
    return train, valid


def fit(exclude: tuple = (), valid_month: str = VALID_MONTH) -> lgb.Booster:
    train, valid = windows(exclude, valid_month)
    stack = lambda part, i: np.vstack([w[i] for w in part]) if i == 0 else np.concatenate([w[i] for w in part])
    ds = lgb.Dataset(stack(train, 0), stack(train, 1), weight=stack(train, 2), free_raw_data=True)
    vs = lgb.Dataset(stack(valid, 0), stack(valid, 1), weight=stack(valid, 2), reference=ds, free_raw_data=True)
    model = lgb.train(PARAMS, ds, ROUNDS, valid_sets=[vs], callbacks=[lgb.early_stopping(PATIENCE, verbose=False)])
    return model.free_dataset()


# ------------------------------------------------------------------ scored windows
def window_input(ctx: dict, w, T: int) -> np.ndarray:
    """Feature rows of a released window, from the dense history T-60..T-5 (the window history)."""
    pp, a = ctx["pub"], ctx["a"]
    d = ctx["days"].index(w.date)
    return pp.features(a["speed"][d, T - HISTORY:T][:, pp.ours], a["flow"][d, T - HISTORY:T][:, pp.ours],
                       pd.Timestamp(w.forecast_origin), w.condition)


def window_probs(model: lgb.Booster, ctx: dict, w, T: int) -> np.ndarray:
    """[step, link] queue probabilities in our link order."""
    pp = ctx["pub"]
    p = model.predict(window_input(ctx, w, T)).reshape(HORIZON, len(pp.links))
    out = np.empty_like(p)
    out[:, pp.ours] = p
    return out
