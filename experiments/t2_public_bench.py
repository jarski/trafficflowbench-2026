"""Train-split evidence for averaging our ongoing model with the public all-link model (tfb.t2_public).

Leave-one-month-out on the bench windows of tfb.t2_bench (official / selector / all):
  - ours: out-of-fold probabilities of the production ongoing features (cache/bench/ongoing_imp.parquet, 1 seed);
  - public: tfb.t2_public refitted without the window's month (early stopping on Feb, or Jan for Feb).
Every window is scored on all links and steps with the expected-IoU set (choose_window), as in production:
ours alone, public alone, averages with public weight 0.25 / 0.5 / 0.75 over every link, and the 0.5 average
on our candidate cells only (links within 12 of a history queue). Mean IoU over panels, per window set,
on all days and on atypical days: the 20% of each panel's days whose congestion map (queued share by link
and slot, 05:00-22:00) correlates least with the mean map of the other months, a stand-in for the shifted
scored months (LOG 2026-10-10). Per-window scores go to cache/t2_public_bench_windows.parquet.

    python experiments/t2_public_bench.py
"""
import lightgbm as lgb
import numpy as np
import pandas as pd

from tfb import t2_ongoing as og
from tfb import t2_public
from tfb.paths import CACHE
from tfb.t2_onset import T2_PANELS
from tfb.t2_rules import vcut
from tfb.task1_io import days, load

bench = pd.read_parquet(CACHE / "bench" / "ongoing_imp.parquet")
months = sorted(bench.month.unique())
models = {}
for m in months:
    path = CACHE / "bench" / f"t2_public_without_{m}.txt"
    if path.exists():
        models[m] = lgb.Booster(model_file=str(path))
    else:
        models[m] = t2_public.fit(exclude=(m,), valid_month="2031-01" if m == "2031-02" else "2031-02")
        models[m].save_model(str(path))
    print(f"  public model without {m}: {models[m].num_trees()} trees", flush=True)


def atypical_days(q: np.ndarray, month_of_day: np.ndarray, share: float = 0.2) -> set:
    """Days whose 05:00-22:00 congestion map correlates least with the other months' mean map."""
    maps = q[:, 60:264].reshape(len(q), -1).astype(np.float32)
    corr = np.full(len(q), np.nan)
    for mo in np.unique(month_of_day):
        ref = maps[month_of_day != mo].mean(axis=0)
        for d in np.flatnonzero(month_of_day == mo):
            if maps[d].std() > 0:
                corr[d] = np.corrcoef(maps[d], ref)[0, 1]
    ok = np.isfinite(corr)
    return set(np.flatnonzero(ok & (corr <= np.nanquantile(corr, share))).tolist())

VARIANTS = ("ours", "public", "avg.25", "avg.5", "avg.75", "avg.5 candidates")
rows = []
for panel in T2_PANELS:
    pp = t2_public.PubPanel(panel)
    a = load(panel, "train", "true_speed", "true_flow", "eligible")
    q = (a["true_speed"] <= vcut(panel)) & a["eligible"]
    dl = days(panel, "train")["days"]
    odd = atypical_days(q, np.array([x[:7] for x in dl]))
    L = len(pp.links)
    sub = bench[bench.panel == panel]
    by_key = {k: g.drop_duplicates(["link", "h"]) for k, g in sub.groupby("key")}
    windows = sub.drop_duplicates("key")[["key", "month"]]
    keys = windows.key.tolist()
    for month, grp in windows.groupby("month"):
        X, meta_rows = [], []
        for key in grp.key:
            _, d, T = key.split("|")
            d, T = int(d), int(T)
            origin = pd.Timestamp(dl[d], tz="UTC") + pd.Timedelta(minutes=5 * T)
            X.append(pp.features(a["true_speed"][d, T - 12:T][:, pp.ours], a["true_flow"][d, T - 12:T][:, pp.ours],
                                 origin, "queue_ongoing"))
            meta_rows.append((key, d, T))
        p_all = models[month].predict(np.vstack(X)).reshape(len(X), og.STEPS, L)
        for (key, d, T), p_pub_their in zip(meta_rows, p_all):
            p_pub = np.empty_like(p_pub_their)
            p_pub[:, pp.ours] = p_pub_their
            g = by_key[key]
            hh, ll, po = g.h.to_numpy(), g.link.to_numpy(), g.p.to_numpy()
            p_ours = np.zeros((og.STEPS, L))
            p_ours[hh - 1, ll] = po
            truth = q[d, T + 1:T + 1 + og.STEPS]
            el = a["eligible"][d, T + 1:T + 1 + og.STEPS]
            steps = np.repeat(np.arange(1, og.STEPS + 1), L)

            def iou(pred):
                t, pr = truth & el, pred & el
                union = (t | pr).sum()
                return (t & pr).sum() / union if union else 1.0

            res = {"key": key, "panel": panel, "month": dl[d][:7], "atypical": d in odd}
            pred = np.zeros((og.STEPS, L), bool)
            sel = og.choose_window(po, hh)
            pred[hh[sel] - 1, ll[sel]] = True
            res["ours"] = iou(pred)
            for name, wt in (("public", 1.0), ("avg.25", .25), ("avg.5", .5), ("avg.75", .75)):
                p = wt * p_pub + (1 - wt) * p_ours
                res[name] = iou(og.choose_window(p.ravel(), steps).reshape(og.STEPS, L))
            pc = 0.5 * po + 0.5 * p_pub[hh - 1, ll]
            pred = np.zeros((og.STEPS, L), bool)
            sel = og.choose_window(pc, hh)
            pred[hh[sel] - 1, ll[sel]] = True
            res["avg.5 candidates"] = iou(pred)
            rows.append(res)
    print(f"  {panel}: {len(keys)} windows", flush=True)

scores = pd.DataFrame(rows).drop_duplicates("key")
sets = bench.drop_duplicates(["key", "set"])[["key", "set"]]
scored = scores.merge(sets, on="key")
scored.to_parquet(CACHE / "t2_public_bench_windows.parquet", index=False)
for label, part in (("all days", scored), ("atypical days", scored[scored.atypical])):
    table = (part.groupby(["set", "panel"])[list(VARIANTS)].mean().groupby("set").mean()
             .reindex(["official", "selector", "all"]))
    table["windows"] = part.groupby("set").key.nunique()
    print(f"\nMean ongoing IoU over panels, {label} (leave-one-month-out, every link and step):")
    print(table.round(4).to_string())
diff = scored.assign(d=scored["avg.5"] - scored["ours"]).groupby("set").d.agg(["mean", "std", "count"])
diff["se"] = diff["std"] / np.sqrt(diff["count"])
print("\nPer-window IoU difference, 0.5 average minus ours:")
print(diff.round(4).to_string())
