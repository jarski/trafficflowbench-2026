"""v1 submission: v0 with Task 2 replaced by the learned onset and ongoing models.

Tasks 1 and 4 are copied from v0 so the leaderboard delta isolates Task 2
(0.30 * delta S_queue).

    python -m tfb.pipeline_v1
"""
from __future__ import annotations

import shutil
import subprocess
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

from tfb import t2_cnn
from tfb import t2_ongoing as og
from tfb import t2_onset as on
from tfb import t2_public
from tfb.dense import tod_index
from tfb.kaggle_submit import package
from tfb.paths import CACHE, ORGANIZER_REPO, RELEASE, SUBMISSIONS
from tfb.pipeline_v0 import SCORED, validate
from tfb.t1_interp import weekday
from tfb.t2_features import at_origin, impute_origin, ramp_slots, static_features
from tfb.t2_rules import vcut
from tfb.task1_io import days, load, meta


def train_models(seeds: int | dict = 1) -> tuple:
    """Onset and ongoing models on the cached training frames; seeds > 1 averages seed models.

    seeds: one count for both, or {"onset": n, "ongoing": m}. The ongoing tuple is (model, features,
    public all-link model or None, convolutional model or None); see tfb.t2_public and tfb.t2_cnn.
    """
    n_seeds = seeds if isinstance(seeds, dict) else {"onset": seeds, "ongoing": seeds}
    d_on = pd.read_parquet(CACHE / "t2_onset_train.parquet")
    d_og = pd.read_parquet(CACHE / "t2_ongoing_train.parquet")
    codes = {p: i for i, p in enumerate(sorted(on.T2_PANELS))}
    f_on = [c for c in d_on.columns if c not in on.FEATURES_DROP] + ["panel_code"]
    f_og = [c for c in d_og.columns if c not in og.FEATURES_DROP] + ["panel_code"]
    d_on["panel_code"], d_og["panel_code"] = d_on.panel.map(codes), d_og.panel.map(codes)

    def fit(df, feats, mod, n):
        boosters = [lgb.train({**mod.PARAMS, "seed": mod.PARAMS["seed"] + i},
                              lgb.Dataset(df[feats], df.y, weight=df.w), mod.ROUNDS).free_dataset() for i in range(n)]
        return boosters[0] if n == 1 else Ensemble(boosters)

    public = t2_public.fit() if t2_public.WEIGHT > 0 else None
    cnn = t2_cnn.fit(t2_cnn.train_data()) if t2_cnn.WEIGHT > 0 else None
    return codes, (fit(d_on, f_on, on, n_seeds["onset"]), f_on), (fit(d_og, f_og, og, n_seeds["ongoing"]), f_og, public, cnn)


class Ensemble:
    """Mean probability of several boosters (seed bagging); behaves like one booster."""

    def __init__(self, boosters: list):
        self.boosters = boosters

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return np.mean([b.predict(x) for b in self.boosters], axis=0)


def panel_context(panel: str, split: str) -> dict:
    """Everything Task 2 features need for one panel and scored split: train priors and the split's inputs."""
    tr = load(panel, "train", "true_speed", "eligible")
    q_tr = (tr["true_speed"] <= vcut(panel)) & tr["eligible"]
    months = np.array([d[:7] for d in days(panel, "train")["days"]])
    dl = days(panel, split)["days"]
    r_on, _ = on.ramp_link_flows(panel, split)
    fl = load(panel, "train", "true_flow")["true_flow"] / np.asarray(meta(panel)["capacity_vph"], np.float32)
    return {"panel": panel, "split": split, "fprof": np.nanmean(fl, axis=0), "priors": on.make_priors(panel, q_tr, on.onset_events(q_tr), months),
            "qfreq": q_tr.mean(axis=0), "a": load(panel, split, "speed", "flow", "occ"), "days": dl,
            "wd": weekday(dl), "r_on": r_on, "st": static_features(panel), "L": len(meta(panel)["links"]),
            "pub": t2_public.PubPanel(panel), "cnn_static": t2_cnn.link_static(panel)}


def window_features(ctx: dict, w) -> tuple[int, pd.DataFrame]:
    """(T, feature rows) for one released window. Uses only data at or before T (see tfb.t2_bench causality)."""
    panel, split, a = ctx["panel"], ctx["split"], ctx["a"]
    T = int(tod_index(pd.Series([w.forecast_origin]))[0])
    d = ctx["days"].index(w.date)
    if w.condition == "queue_onset":
        x = on.panel_frame(panel, split, [(d, T, w.window_id, None)], ctx["priors"], lambda _: set())
    else:
        hist = {ch: a[ch][d, T - 12:T] for ch in ("speed", "flow", "occ")}
        at_t = impute_origin(at_origin(a, d, T), hist, "ongoing", panel, T, None)
        r_on = np.nanmean(ctx["r_on"][d, ramp_slots(T)], axis=0)
        x = og.window_rows(panel, hist, at_t, r_on, ctx["qfreq"], T, ctx["wd"][d], ctx["st"], ctx["fprof"])
    return T, x


def predict_window(ctx: dict, w, code: int, onset_model: tuple, ongoing_model: tuple) -> tuple[int, np.ndarray]:
    """(T, [step, link] queue prediction) for one window."""
    T, x = window_features(ctx, w)
    x["panel_code"] = code
    pred = np.zeros((og.STEPS, ctx["L"]), bool)
    if w.condition == "queue_onset":
        model, feats = onset_model
        pred[og.STEPS - 1] = on.choose_set(model.predict(x[feats]))
        return T, pred
    model, feats = ongoing_model[:2]
    public = ongoing_model[2] if len(ongoing_model) > 2 else None
    cnn = ongoing_model[3] if len(ongoing_model) > 3 else None
    if public is None and cnn is None:
        if len(x):
            sel = og.choose_window(model.predict(x[feats]), x.h.to_numpy())
            pred[x.h.to_numpy()[sel] - 1, x.link.to_numpy()[sel]] = True
        return T, pred
    # average over every link with the all-link models (ours is 0 off its candidate cells), then the set:
    # p = (1 - w_pub - w_cnn) ours + w_pub public + w_cnn cnn
    w_pub = t2_public.WEIGHT if public is not None else 0.0
    w_cnn = t2_cnn.WEIGHT if cnn is not None else 0.0
    p = np.zeros((og.STEPS, ctx["L"]))
    if public is not None:
        pp = t2_public.window_probs(public, ctx, w, T)
        if t2_public.OFF_SCALE is None:
            p += w_pub * pp
        else:   # off our candidate cells the public model stands alone, scaled by OFF_SCALE instead of w_pub
            scale = np.full(pp.shape, t2_public.OFF_SCALE)
            if len(x):
                scale[x.h.to_numpy() - 1, x.link.to_numpy()] = w_pub
            p += scale * pp
    if cnn is not None:
        p += w_cnn * t2_cnn.window_probs(cnn, ctx, w, T)
    if len(x):
        p[x.h.to_numpy() - 1, x.link.to_numpy()] += (1 - w_pub - w_cnn) * model.predict(x[feats])
    steps = np.repeat(np.arange(1, og.STEPS + 1), ctx["L"])
    return T, og.choose_window(p.ravel(), steps).reshape(og.STEPS, ctx["L"])


def task2(split: str, codes: dict, onset_model: tuple, ongoing_model: tuple) -> pd.DataFrame:
    frames = []
    for panel in on.T2_PANELS:
        path = RELEASE / "task2" / panel / split / "window_index.csv"
        if not path.exists():   # only in a release cut at some time (the organizers' Task 2 check); merging would fail loudly
            continue
        ctx = panel_context(panel, split)
        win = pd.read_csv(path)
        tmpl = pd.read_csv(RELEASE / "task2" / panel / split / "sample_submission_queue.csv", dtype=str)
        pos = {l: i for i, l in enumerate(meta(panel)["links"])}
        for w in win.itertuples():
            T, pred = predict_window(ctx, w, codes[panel], onset_model, ongoing_model)
            rows = tmpl[tmpl.window_id == w.window_id].copy()
            step = tod_index(rows.timestamp) - T
            rows["queue_pred"] = pred[step - 1, rows.link_id.map(pos).to_numpy()].astype(int)
            frames.append(rows[["window_id", "timestamp", "link_id", "queue_pred"]])
        print(f"  {panel}/{split}: {len(win)} windows", flush=True)
    if not frames:
        return pd.DataFrame(columns=["window_id", "timestamp", "link_id", "queue_pred"])
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    work = SUBMISSIONS / "v1"
    work.mkdir(parents=True, exist_ok=True)
    v0 = SUBMISSIONS / "v0"
    for f in ("state.csv", "odme.csv"):
        shutil.copy(v0 / f, work / f)
    codes, m_on, m_og = train_models()
    pd.concat([task2(s, codes, m_on, m_og) for s in SCORED]).to_csv(work / "queue.csv", index=False)
    subprocess.run([sys.executable, str(ORGANIZER_REPO / "src/merge_submissions.py"),
                    "--state", str(work / "state.csv"), "--queue", str(work / "queue.csv"),
                    "--odme", str(work / "odme.csv"), "--key", str(RELEASE / "submission_key.csv"),
                    "--output", str(work / "submission.csv")], check=True)
    validate(work / "submission.csv")
    print("wrote", package(work / "submission.csv"))


if __name__ == "__main__":
    main()
