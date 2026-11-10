"""Learned causal estimate of the links masked at the Task 2 origin T.

At a window's origin T about a third of the links are Task 1 targets, blank in the masked layer. This
model predicts their speed, flow and occupancy at T from data at or before T only: the link's own last
30 minutes (the window history), its neighbours at T and T-5 where observed, the simple neighbour-change
rule (t2_features.impute_origin), link data and time of day. Training rows mirror that situation exactly:
train cells that are Task 1 targets at their slot, with the true value as label.

Models are fitted with each train month left out (for the Task 2 training frames of that month) and on
all train months (for the scored splits), so no Task 2 training window uses an estimate that saw its
own month's answers.

    python -m tfb.t2_impute --eval     # held-out-month accuracy against the simple rule
"""
from __future__ import annotations

import argparse
from functools import lru_cache

import lightgbm as lgb
import numpy as np
import pandas as pd

from tfb.paths import CACHE, PANELS, TASK2_EXCLUDED
from tfb.t2_features import K_OCC, _shift
from tfb.t2_rules import vcut
from tfb.task1_io import days, load, meta

CHANNELS = ("speed", "flow", "occ")
NEIGHBOURS = (-3, -2, -1, 1, 2, 3)
PARAMS = dict(objective="regression", learning_rate=0.05, num_leaves=63, min_data_in_leaf=100,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              verbose=-1, seed=3, deterministic=True, num_threads=8)
ROUNDS = 400
N_PER_PANEL = 150_000
T2_PANELS = [p for p in PANELS if p not in TASK2_EXCLUDED]


def _scales(panel: str) -> dict[str, np.ndarray]:
    m = meta(panel)
    return {"speed": np.asarray(m["free_speed_kmh"], float), "flow": np.asarray(m["capacity_vph"], float),
            "occ": np.asarray(m["critical_density"], float) / K_OCC}


def rule_estimate(at_t: np.ndarray, prev: np.ndarray, reach: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Own T-5 value plus the mean T-5 -> T change of observed links within +-reach; also the count used."""
    delta = at_t - prev
    num, cnt = np.zeros_like(delta, float), np.zeros_like(delta, float)
    for k in range(-reach, reach + 1):
        if k:
            nb = _shift(delta, k)
            ok = ~np.isnan(nb)
            num += np.where(ok, nb, 0.0)
            cnt += ok
    return np.where(cnt > 0, prev + num / np.maximum(cnt, 1), np.nan), cnt


def features(panel: str, at_t: dict, hist: dict, T: int, links: np.ndarray) -> pd.DataFrame:
    """Rows for the given links (masked at T). at_t[ch]: [L] (NaN where masked); hist[ch]: [12, L] (T-60..T-5)."""
    sc = _scales(panel)
    m = meta(panel)
    f = {"tod": np.full(len(links), T), "pos": links / max(len(m["links"]) - 1, 1),
         "lanes": np.asarray(m["lanes"], float)[links], "vf": sc["speed"][links]}
    for ch in CHANNELS:
        s = sc[ch]
        now, prev = at_t[ch] / s, hist[ch][-1] / s
        for lag in (1, 2, 3, 6):
            f[f"{ch}_m{lag}"] = (hist[ch][-lag] / s)[links]
        est, cnt = rule_estimate(now, prev)
        f[f"{ch}_rule_delta"] = (est - prev)[links]
        if ch == "speed":
            f["n_obs_nb"] = cnt[links]
        for k in NEIGHBOURS:
            nb_now, nb_prev = _shift(now, k), _shift(prev, k)
            f[f"{ch}_nb{k:+d}"] = nb_now[links]
            f[f"{ch}_nbd{k:+d}"] = (nb_now - nb_prev)[links]
    return pd.DataFrame(f)


def training_rows(panel: str, rng, n: int) -> pd.DataFrame:
    """Train cells masked at their slot (Task 1 targets) near a queue, with history from truth."""
    a = load(panel, "train", "true_speed", "true_flow", "true_occ", "speed", "flow", "occ", "target", "eligible")
    sc = _scales(panel)
    q = (a["true_speed"] <= vcut(panel)) & a["eligible"]
    near_q = np.zeros_like(q)
    for k in range(-12, 13):   # a queue within 12 links five minutes earlier: the ongoing-window situation
        near_q |= np.roll(q, k, axis=2)
    near_q = np.roll(near_q, 1, axis=1)
    cand = a["target"] & a["eligible"] & ~np.isnan(a["true_speed"]) & near_q
    cand[:, :12] = False
    d, t, l = np.nonzero(cand)
    if len(d) > n:
        i = rng.choice(len(d), n, replace=False)
        d, t, l = d[i], t[i], l[i]
    months = np.array([x[:7] for x in days(panel, "train")["days"]])
    frames = []
    for dd, tt in pd.DataFrame({"d": d, "t": t}).drop_duplicates().itertuples(index=False):
        sel = (d == dd) & (t == tt)
        at_t = {ch: a[ch][dd, tt] for ch in CHANNELS}
        hist = {ch: a[f"true_{ch}"][dd, tt - 12:tt] for ch in CHANNELS}
        x = features(panel, at_t, hist, tt, l[sel])
        for ch in CHANNELS:
            x[f"y_{ch}"] = (a[f"true_{ch}"][dd, tt, l[sel]] - hist[ch][-1][l[sel]]) / sc[ch][l[sel]]
        x["month"], x["panel"] = months[dd], panel
        frames.append(x)
    return pd.concat(frames, ignore_index=True)


def train_frame() -> pd.DataFrame:
    path = CACHE / "t2_impute_train.parquet"
    if path.exists():
        return pd.read_parquet(path)
    rng = np.random.default_rng(0)
    df = pd.concat([training_rows(p, rng, N_PER_PANEL) for p in T2_PANELS], ignore_index=True)
    df.to_parquet(path, index=False)
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if not c.startswith("y_") and c not in ("month", "panel")]


@lru_cache(maxsize=16)
def models(exclude_month: str | None) -> dict:
    """{channel: booster} trained without `exclude_month` (None: all train months)."""
    df = train_frame()
    if exclude_month is not None:
        df = df[df.month != exclude_month]
    feats = feature_columns(df)
    out = {}
    for ch in CHANNELS:
        ok = df[f"y_{ch}"].notna()
        out[ch] = lgb.train(PARAMS, lgb.Dataset(df.loc[ok, feats], df.loc[ok, f"y_{ch}"]), ROUNDS).free_dataset()
    return out


def impute(panel: str, at_t: dict, hist: dict, T: int, exclude_month: str | None) -> dict:
    """at_t with masked links filled by the learned estimate (links without any usable input stay blank)."""
    blank = np.isnan(at_t["speed"])
    if not blank.any():
        return at_t
    links = np.flatnonzero(blank)
    x = features(panel, at_t, hist, T, links)
    sc = _scales(panel)
    out = {}
    for ch in CHANNELS:
        prev = hist[ch][-1][links]
        pred = models(exclude_month)[ch].predict(x[feature_columns(x)])
        est = prev + pred * sc[ch][links]
        filled = at_t[ch].astype(np.float32).copy()
        filled[links] = np.where(np.isnan(at_t[ch][links]), est, at_t[ch][links])
        out[ch] = filled
    return out


def evaluate() -> None:
    """Held-out-month RMSE at T of: last value (T-5), the simple rule, and the learned model."""
    df = train_frame()
    feats = feature_columns(df)
    res = []
    for mo in sorted(df.month.unique()):
        tr, va = df[df.month != mo], df[df.month == mo]
        for ch in CHANNELS:
            ok_tr, ok_va = tr[f"y_{ch}"].notna(), va[f"y_{ch}"].notna()
            m = lgb.train(PARAMS, lgb.Dataset(tr.loc[ok_tr, feats], tr.loc[ok_tr, f"y_{ch}"]), ROUNDS)
            y = va.loc[ok_va, f"y_{ch}"].to_numpy()
            rule = va.loc[ok_va, f"{ch}_rule_delta"].fillna(0).to_numpy()
            res.append(dict(month=mo, ch=ch, last=np.sqrt(np.mean(y ** 2)), rule=np.sqrt(np.mean((y - rule) ** 2)),
                            model=np.sqrt(np.mean((y - m.predict(va.loc[ok_va, feats])) ** 2))))
    r = pd.DataFrame(res)
    print("RMSE at T in normalized units (speed/vf, flow/capacity, density/k_crit), mean over held-out months:")
    print(r.groupby("ch")[["last", "rule", "model"]].mean().round(4).to_string())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--eval", action="store_true")
    args = ap.parse_args()
    df = train_frame()
    print(f"{len(df):,} training rows")
    if args.eval:
        evaluate()


if __name__ == "__main__":
    main()
