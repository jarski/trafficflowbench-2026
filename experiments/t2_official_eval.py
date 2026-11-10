"""Score the onset and ongoing models on the 80 official train windows (model trained without the window's month)."""
import numpy as np
import pandas as pd
import lightgbm as lgb
from tfb.paths import CACHE, RELEASE
from tfb.dense import tod_index
from tfb.t1_interp import weekday
from tfb.t2_features import static_features
from tfb.t2_onset import T2_PANELS, ramp_link_flows
from tfb import t2_ongoing as og
from tfb.t2_rules import iou, vcut
from tfb.task1_io import days, load

df = pd.read_parquet(CACHE / "t2_ongoing_train.parquet")
feats = [c for c in df.columns if c not in og.FEATURES_DROP] + ["panel_code"]
codes = {p: i for i, p in enumerate(sorted(df.panel.unique()))}
df["panel_code"] = df.panel.map(codes)
models = {}
rows = []
for panel in T2_PANELS:
    a = load(panel, "train", "true_speed", "true_flow", "true_occ", "speed", "flow", "occ", "eligible")
    q = (a["true_speed"] <= vcut(panel)) & a["eligible"]
    dl = days(panel, "train")["days"]; months = np.array([d[:7] for d in dl]); wd = weekday(dl)
    on, _ = ramp_link_flows(panel, "train"); st = static_features(panel)
    w = pd.read_csv(RELEASE / "task2" / panel / "train" / "window_index.csv")
    for r in w[w.condition == "queue_ongoing"].itertuples():
        d = dl.index(r.date); T = int(tod_index(pd.Series([r.forecast_origin]))[0]); mo = months[d]
        if mo not in models:
            tr = df[df.month != mo]
            models[mo] = lgb.train(og.PARAMS, lgb.Dataset(tr[feats], tr.y, weight=tr.w), og.ROUNDS)
        qfreq = q[months != mo].mean(axis=0)
        hist = {ch: a[f"true_{ch}"][d, T - 12:T] for ch in ("speed", "flow", "occ")}
        at_t = {ch: a[ch][d, T] for ch in ("speed", "flow", "occ")}
        x = og.window_rows(panel, hist, at_t, np.nanmean(on[d, T - 1:T + 1], axis=0), qfreq, T, wd[d], st)
        x["panel_code"] = codes[panel]
        p = models[mo].predict(x[feats])
        L = q.shape[2]
        pred = np.zeros((og.STEPS, L), bool)
        sel = og.choose_cells(p)
        pred[x.h.to_numpy()[sel] - 1, x.link.to_numpy()[sel]] = True
        el = a["eligible"][d, T + 1:T + 1 + og.STEPS]
        truth = q[d, T + 1:T + 1 + og.STEPS]
        pers = np.broadcast_to(x.groupby("link").q_last.first().reindex(range(L), fill_value=False).to_numpy().astype(bool), truth.shape)
        rows.append((panel, iou(pred & el, truth), iou(pers & el, truth)))
r = pd.DataFrame(rows, columns=["panel", "lgbm", "persistence"]).groupby("panel").mean()
print(r.round(3).to_string()); print("mean over panels:", r.mean().round(4).to_dict())
