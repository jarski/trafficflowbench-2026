"""Task 2: does the model know which windows are hard?

Self-assessed expected IoU (the ratio-of-expectations value the set selector
maximizes) on train out-of-fold predictions, the official train windows, and the
validation and private windows (v4 models, inputs at or before T only). The
realized validation IoUs (onset 0.764, ongoing 0.741) come from the
leaderboard component isolation in LOG 2026-10-02.

    python experiments/eda/t2_confidence.py [--ongoing-oof]
"""
import argparse

import lightgbm as lgb
import numpy as np
import pandas as pd

from tfb import t2_ongoing as og
from tfb import t2_onset as on
from tfb.dense import tod_index
from tfb.paths import CACHE, RELEASE, SUBMISSIONS
from tfb.t1_interp import weekday
from tfb.t2_features import static_features
from tfb.t2_rules import iou, vcut
from tfb.task1_io import days, load

ap = argparse.ArgumentParser()
ap.add_argument("--ongoing-oof", action="store_true", help="also run the 9-fold ongoing CV (slow)")
args = ap.parse_args()


def exp_iou(p: np.ndarray, sel: np.ndarray) -> float:
    inside = p[sel].sum()
    return float(inside / (sel.sum() + p.sum() - inside))


def summarize(r: pd.DataFrame, label: str) -> None:
    r = r.assign(weekend=r.weekday >= 5)
    cols = [c for c in ("exp", "real", "pmax", "k") if c in r]
    print(f"\n{label}: {len(r)} windows")
    print(r.groupby("weekend")[cols].mean().round(3).to_string())
    print("all:", r[cols].mean().round(3).to_dict())


# ---------------------------------------------------------------- onset, train OOF
oof = pd.read_parquet(CACHE / "t2_onset_oof.parquet", columns=["key", "panel", "y", "w", "p", "weekday", "tod"])
rows = []
for key, g in oof.groupby("key", sort=False):
    p = g.p.to_numpy()
    sel = on.choose_set(p)
    el = g.w.to_numpy() > 0
    y = g.y.to_numpy().astype(bool) & el
    rows.append(dict(key=key, panel=g.panel.iloc[0], weekday=g.weekday.iloc[0], exp=exp_iou(p, sel),
                     real=iou(sel & el, y), pmax=p.max(), k=sel.sum()))
tr_on = pd.DataFrame(rows)
summarize(tr_on, "onset, train out-of-fold, all replicated events")

official = []
for panel in on.T2_PANELS:
    dl = days(panel, "train")["days"]
    w = pd.read_csv(RELEASE / "task2" / panel / "train" / "window_index.csv")
    for r in w[w.condition == "queue_onset"].itertuples():
        T = int(tod_index(pd.Series([r.forecast_origin]))[0])
        official.append(f"{panel}|{dl.index(r.date)}|{T + 6}")
summarize(tr_on[tr_on.key.isin(official)], "onset, train out-of-fold, official train windows")

# ---------------------------------------------------------------- v4 models on validation / private
m_on = lgb.Booster(model_file=str(SUBMISSIONS / "v4" / "t2_onset.txt"))
m_og = lgb.Booster(model_file=str(SUBMISSIONS / "v4" / "t2_ongoing.txt"))
codes = {p: i for i, p in enumerate(sorted(on.T2_PANELS))}
res = {"queue_onset": [], "queue_ongoing": []}
for panel in on.T2_PANELS:
    tr = load(panel, "train", "true_speed", "eligible")
    q_tr = (tr["true_speed"] <= vcut(panel)) & tr["eligible"]
    months = np.array([d[:7] for d in days(panel, "train")["days"]])
    priors = on.Priors(q_tr, on.onset_events(q_tr), months)
    qfreq = q_tr.mean(axis=0)
    st = static_features(panel)
    for split in ("validation", "private"):
        a = load(panel, split, "speed", "flow", "occ")
        dl = days(panel, split)["days"]
        wd = weekday(dl)
        r_on, _ = on.ramp_link_flows(panel, split)
        win = pd.read_csv(RELEASE / "task2" / panel / split / "window_index.csv")
        for w in win.itertuples():
            T = int(tod_index(pd.Series([w.forecast_origin]))[0])
            d = dl.index(w.date)
            if w.condition == "queue_onset":
                x = on.panel_frame(panel, split, [(d, T, w.window_id, None)], priors, lambda _: set())
                x["panel_code"] = codes[panel]
                p = m_on.predict(x[m_on.feature_name()])
                sel = on.choose_set(p)
                extra = dict(prior_max=float(x.prior_tod.max()), prior_at_pick=float(x.prior_tod.to_numpy()[sel].max()))
            else:
                hist = {ch: a[ch][d, T - 12:T] for ch in ("speed", "flow", "occ")}
                at_t = {ch: a[ch][d, T] for ch in ("speed", "flow", "occ")}
                x = og.window_rows(panel, hist, at_t, np.nanmean(r_on[d, T - 1:T + 1], axis=0), qfreq, T, wd[d], st)
                x["panel_code"] = codes[panel]
                p = m_og.predict(x[m_og.feature_name()])
                sel = og.choose_cells(p)
                extra = dict(n_queued_T=int(x.groupby("link").q_last.first().sum()))
            res[w.condition].append(dict(split=split, panel=panel, window=w.window_id, date=w.date, weekday=wd[d],
                                         T=T, exp=exp_iou(p, sel), pmax=p.max(), k=int(sel.sum()), **extra))
for cond, rows in res.items():
    r = pd.DataFrame(rows)
    for split, g in r.groupby("split"):
        summarize(g, f"{cond}, {split} (v4 model)")
    r.to_csv(CACHE / f"t2_confidence_{cond}.csv", index=False)

# ---------------------------------------------------------------- ongoing, train OOF (slow)
if args.ongoing_oof:
    df = pd.read_parquet(CACHE / "t2_ongoing_train.parquet")
    df = og.cv(df)
    rows = []
    for key, g in df.groupby("key", sort=False):
        p = g.p.to_numpy()
        sel = og.choose_cells(p)
        el = g.w.to_numpy() > 0
        truth = g.y.to_numpy().astype(bool) & el
        union = ((sel & el) | truth).sum() + g.outside_true.iloc[0]
        rows.append(dict(key=key, panel=g.panel.iloc[0], weekday=g.weekday.iloc[0], exp=exp_iou(p, sel),
                         real=((sel & el) & truth).sum() / union if union else 1.0, pmax=p.max(), k=sel.sum()))
    tr_og = pd.DataFrame(rows)
    tr_og.to_csv(CACHE / "t2_confidence_ongoing_oof.csv", index=False)
    summarize(tr_og, "ongoing, train out-of-fold, replicated windows")
