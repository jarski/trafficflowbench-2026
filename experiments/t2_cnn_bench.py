"""Train-split evidence for the convolutional ongoing model (tfb.t2_cnn) as a third member of the ongoing average.

Bench windows of tfb.t2_bench (official / selector / all), scored on every link and step with the expected-IoU
set (choose_window), as in production:
  - ours: out-of-fold probabilities of the production features (cache/bench/ongoing_imp.parquet, 1 seed);
  - public: tfb.t2_public refitted without the window's month (cache/bench/t2_public_without_<m>.txt);
  - cnn: tfb.t2_cnn, either the Jun-Nov model on Dec-Feb windows (--mode holdout, cache/bench/t2_cnn_holdout.pt)
    or leave-one-month-out models (--mode oof, cache/bench/t2_cnn_without_<m>.pt);
  - prod: 0.5 ours + 0.5 public (v41); prod+cnn at weights 0.25 / 0.33 / 0.5; ours+cnn 0.5.
Mean IoU over panels per window set, on all days and on atypical days (the 20% of each panel's days least like
the other months, as in experiments/t2_public_bench.py). Per-window scores go to cache/t2_cnn_bench_<mode>.parquet.

    python experiments/t2_cnn_bench.py --mode holdout
    python experiments/t2_cnn_bench.py --mode oof
"""
import argparse

import lightgbm as lgb
import numpy as np
import pandas as pd

from tfb import t2_cnn
from tfb import t2_ongoing as og
from tfb import t2_public
from tfb.paths import CACHE
from tfb.t2_onset import T2_PANELS
from tfb.t2_rules import vcut
from tfb.task1_io import days, load

ap = argparse.ArgumentParser()
ap.add_argument("--mode", default="holdout", choices=["holdout", "oof"])
ap.add_argument("--max-all", type=int, default=1500, help="windows of the 'all' set to score (sampled)")
ap.add_argument("--model", default="", help="holdout: suffix of the model file, cache/bench/t2_cnn_holdout<suffix>.pt")
args = ap.parse_args()

bench = pd.read_parquet(CACHE / "bench" / "ongoing_imp.parquet")
if args.mode == "holdout":
    bench = bench[bench.month >= "2030-12"]
sets = bench.drop_duplicates(["key", "set"])[["key", "set"]]
keep = set(sets[sets.set != "all"].key)
all_keys = sets[sets.set == "all"].key.drop_duplicates()
keep |= set(all_keys.sample(min(args.max_all, len(all_keys)), random_state=0))
bench = bench[bench.key.isin(keep)]
months = sorted(bench.month.unique())
pub = {m: lgb.Booster(model_file=str(CACHE / "bench" / f"t2_public_without_{m}.txt")) for m in months}
cnn = {m: t2_cnn.load_model(CACHE / "bench" / (f"t2_cnn_holdout{args.model}.pt" if args.mode == "holdout" else f"t2_cnn_without_{m}.pt"))
       for m in months}


def atypical_days(q: np.ndarray, month_of_day: np.ndarray, share: float = 0.2) -> set:
    maps = q[:, 60:264].reshape(len(q), -1).astype(np.float32)
    corr = np.full(len(q), np.nan)
    for mo in np.unique(month_of_day):
        ref = maps[month_of_day != mo].mean(axis=0)
        for d in np.flatnonzero(month_of_day == mo):
            if maps[d].std() > 0:
                corr[d] = np.corrcoef(maps[d], ref)[0, 1]
    ok = np.isfinite(corr)
    return set(np.flatnonzero(ok & (corr <= np.nanquantile(corr, share))).tolist())


VARIANTS = ("ours", "public", "cnn", "prod", "prod+cnn.25", "prod+cnn.33", "prod+cnn.5", "ours+cnn.5", "public+cnn.5")
rows = []
for panel in T2_PANELS:
    pp = t2_public.PubPanel(panel)
    tp = t2_cnn.TrainPanel(panel)
    a, q = tp.a, tp.q
    dl = days(panel, "train")["days"]
    odd = atypical_days(q, tp.months)
    L = len(pp.links)
    sub = bench[bench.panel == panel]
    by_key = {k: g.drop_duplicates(["link", "h"]) for k, g in sub.groupby("key")}
    windows = sub.drop_duplicates("key")[["key", "month"]]
    for month, grp in windows.groupby("month"):
        X_pub, X_cnn, meta_rows = [], [], []
        for key in grp.key:
            _, d, T = key.split("|")
            d, T = int(d), int(T)
            origin = pd.Timestamp(dl[d], tz="UTC") + pd.Timedelta(minutes=5 * T)
            X_pub.append(pp.features(a["true_speed"][d, T - 12:T][:, pp.ours], a["true_flow"][d, T - 12:T][:, pp.ours],
                                     origin, "queue_ongoing"))
            X_cnn.append(tp.window(d, T)[0])
            meta_rows.append((key, d, T))
        p_pub_all = pub[month].predict(np.vstack(X_pub)).reshape(len(X_pub), og.STEPS, L)
        p_cnn_all = t2_cnn.predict(cnn[month], np.stack(X_cnn))
        for (key, d, T), p_pub_their, p_cnn in zip(meta_rows, p_pub_all, p_cnn_all):
            p_pub = np.empty_like(p_pub_their)
            p_pub[:, pp.ours] = p_pub_their
            g = by_key[key]
            hh, ll, po = g.h.to_numpy(), g.link.to_numpy(), g.p.to_numpy()
            p_ours = np.zeros((og.STEPS, L))
            p_ours[hh - 1, ll] = po
            truth = q[d, T + 1:T + 1 + og.STEPS]
            el = a["eligible"][d, T + 1:T + 1 + og.STEPS]
            steps = np.repeat(np.arange(1, og.STEPS + 1), L)

            def iou(p):
                pred = og.choose_window(p.ravel(), steps).reshape(og.STEPS, L)
                t, pr = truth & el, pred & el
                union = (t | pr).sum()
                return (t & pr).sum() / union if union else 1.0

            prod = 0.5 * p_ours + 0.5 * p_pub
            res = {"key": key, "panel": panel, "month": dl[d][:7], "atypical": d in odd,
                   "ours": iou(p_ours), "public": iou(p_pub), "cnn": iou(p_cnn), "prod": iou(prod),
                   "prod+cnn.25": iou(0.75 * prod + 0.25 * p_cnn), "prod+cnn.33": iou(0.67 * prod + 0.33 * p_cnn),
                   "prod+cnn.5": iou(0.5 * prod + 0.5 * p_cnn), "ours+cnn.5": iou(0.5 * p_ours + 0.5 * p_cnn),
                   "public+cnn.5": iou(0.5 * p_pub + 0.5 * p_cnn)}
            rows.append(res)
    print(f"  {panel}: {len(windows)} windows", flush=True)

scores = pd.DataFrame(rows).drop_duplicates("key")
scored = scores.merge(sets, on="key")
scored.to_parquet(CACHE / f"t2_cnn_bench_{args.mode}{args.model}.parquet", index=False)
for label, part in (("all days", scored), ("atypical days", scored[scored.atypical])):
    table = (part.groupby(["set", "panel"])[list(VARIANTS)].mean().groupby("set").mean()
             .reindex(["official", "selector", "all"]))
    table["windows"] = part.groupby("set").key.nunique()
    print(f"\nMean ongoing IoU over panels, {label} ({args.mode}; every link and step):")
    print(table.round(4).to_string())
for name in ("cnn", "prod+cnn.33", "prod+cnn.5"):
    diff = scored.assign(d=scored[name] - scored["prod"]).groupby("set").d.agg(["mean", "std", "count"])
    diff["se"] = diff["std"] / np.sqrt(diff["count"])
    print(f"\nPer-window IoU difference, {name} minus prod:")
    print(diff.round(4).to_string())
