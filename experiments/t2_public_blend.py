"""Ongoing probe: average our ongoing probabilities with a different public model's, then our expected-IoU set.

The public model is the all-link ongoing classifier of the anubisyy notebook (assets dataset
anubisyy/trafficflowbench-086517-public-assets, CC0; file ongoing_lgb.txt, code queue_ongoing.py). It
scores every link from the 12-step window history strictly before T with 36 simple features. The 0.8714
notebook gained +0.0023 public from a 50/50 probability average of its own ongoing models and this one,
and probe v37 (that notebook's ongoing rows) beat ours by +0.0025. This tests whether the same average
helps our model:
    v39  p = 0.5 p_ours + 0.5 p_public on our candidate rows, then choose_window as in production
    v40  the same over every link (p_ours = 0 off our candidates)
Both replace only the ongoing rows of v35. Weight 0 reproduces v35 exactly (checked).

    python experiments/t2_public_blend.py
"""
import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

from tfb import pipeline_v1 as pv1
from tfb import t2_ongoing as og
from tfb import t2_onset as on
from tfb.kaggle_submit import package
from tfb.paths import DATA_ROOT, RELEASE, SUBMISSIONS
from tfb.pipeline_final import T2_SEEDS
from tfb.pipeline_v0 import validate
from tfb.task1_io import meta

ASSETS = DATA_ROOT / "public_nb" / "assets"
os.environ.update(TFB_DATA_ROOT=str(RELEASE), TFB_OUTPUT_ROOT=str(DATA_ROOT / "public_nb" / "out"), TFB_THREADS="8")
sys.path.insert(0, str(ASSETS))
from queue_ongoing import HISTORY, Panel, panel_names  # noqa: E402  public code (CC0)

pub_model = lgb.Booster(model_file=str(ASSETS / "ongoing_lgb.txt"))
assert pub_model.num_feature() == 36
codes, _, ongoing = pv1.train_models({"onset": 1, "ongoing": T2_SEEDS["ongoing"]})
m_og, f_og = ongoing[:2]
pub_no = {name: i for i, name in enumerate(panel_names())}
base_q = pd.read_csv(SUBMISSIONS / "v35" / "queue.csv", dtype={"link_id": str})
base_q = base_q.set_index(["window_id", "timestamp", "link_id"]).queue_pred

new = {"v39": {}, "v40": {}}
for split in ("validation", "private"):
    for panel in on.T2_PANELS:
        ctx = pv1.panel_context(panel, split)
        links = meta(panel)["links"]
        pp = Panel(panel, pub_no[panel])
        pub_col = np.array([pp.link_index[l] for l in links])   # our link index -> public column
        folder = RELEASE / "task2" / panel / split
        win = pd.read_csv(folder / "window_index.csv")
        hist = pd.read_parquet(folder / "window_history.parquet")
        hist["link_id"] = hist.link_id.astype(str)
        for w in win[win.condition == "queue_ongoing"].itertuples():
            T, x = pv1.window_features(ctx, w)
            x["panel_code"] = codes[panel]
            p_ours = m_og.predict(x[f_og])
            hh, ll = x.h.to_numpy(), x.link.to_numpy()
            origin = pd.Timestamp(w.forecast_origin)
            times = pd.date_range(end=origin - pd.Timedelta(minutes=5), periods=HISTORY, freq="5min",
                                  tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
            h = hist[hist.window_id == w.window_id]
            assert h.timestamp.max() < w.forecast_origin   # the public model reads only T-60..T-5
            mat = lambda c: h.pivot(index="timestamp", columns="link_id", values=c).reindex(
                index=times, columns=pp.links).to_numpy(np.float32)
            X = pp.features(mat("speed_kmh"), mat("flow_vph"), np.nan_to_num(mat("is_score_eligible")).astype(bool),
                            origin, w.condition)
            p_pub = pub_model.predict(X).reshape(og.STEPS, len(pp.links))[:, pub_col]   # [step, our link]
            stamps = [(origin + pd.Timedelta(minutes=5 * s)).strftime("%Y-%m-%dT%H:%M:%SZ") for s in range(1, og.STEPS + 1)]

            def emit(version, pred):
                for s in range(og.STEPS):
                    for li in range(len(links)):
                        new[version][(w.window_id, stamps[s], links[li])] = int(pred[s, li])

            pred = np.zeros((og.STEPS, len(links)), bool)   # weight 0: must equal v35
            sel = og.choose_window(p_ours, hh)
            pred[hh[sel] - 1, ll[sel]] = True
            for s in range(og.STEPS):
                for li in range(len(links)):
                    k = (w.window_id, stamps[s], links[li])
                    if k in base_q.index:
                        assert base_q[k] == int(pred[s, li]), f"{w.window_id}: weight 0 differs from v35"
            pa = 0.5 * p_ours + 0.5 * p_pub[hh - 1, ll]
            pred = np.zeros((og.STEPS, len(links)), bool)
            sel = og.choose_window(pa, hh)
            pred[hh[sel] - 1, ll[sel]] = True
            emit("v39", pred)
            pb = 0.5 * p_pub.copy()
            pb[hh - 1, ll] += 0.5 * p_ours
            steps = np.repeat(np.arange(1, og.STEPS + 1), len(links))
            emit("v40", og.choose_window(pb.ravel(), steps).reshape(og.STEPS, len(links)))
        print(f"  {panel}/{split} done", flush=True)

sub = pd.read_csv(SUBMISSIONS / "v35" / "submission.csv")
key = pd.read_csv(RELEASE / "submission_key.csv", usecols=["task", "window_id", "timestamp", "link_id"], dtype=str)
probe = pd.read_csv(SUBMISSIONS / "v37" / "submission.csv", usecols=["queue_pred"]).queue_pred.to_numpy()
qi = np.flatnonzero((key.task == "queue").to_numpy())   # submission rows are in key order (checked in public_probe.py)
kq = list(zip(key.window_id.to_numpy()[qi], key.timestamp.to_numpy()[qi], key.link_id.to_numpy()[qi]))
for version, preds in new.items():
    hit = np.array([x in preds for x in kq])
    m = np.zeros(len(sub), bool)
    m[qi[hit]] = True
    out = sub.copy()
    out.loc[m, "queue_pred"] = [preds[x] for x, keep in zip(kq, hit) if keep]
    a, b, c = sub.queue_pred.to_numpy()[m] == 1, out.queue_pred.to_numpy()[m] == 1, probe[m] == 1
    print(f"{version}: ongoing cells queued {b.sum()} (v35 {a.sum()}, public {c.sum()}); "
          f"Jaccard vs v35 {(a & b).sum() / (a | b).sum():.3f}, vs public {(b & c).sum() / (b | c).sum():.3f}")
    work = SUBMISSIONS / version
    work.mkdir(parents=True, exist_ok=True)
    out.to_csv(work / "submission.csv", index=False)
    validate(work / "submission.csv")
    print(version, "->", package(work / "submission.csv"))
