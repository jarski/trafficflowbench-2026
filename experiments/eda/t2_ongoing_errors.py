"""Task 2 ongoing: where does the IoU loss come from?

Uses the out-of-fold predictions cached by `python -m tfb.t2_bench ongoing --tag <tag>` (each window
predicted by a model trained without its month). Each error cell is classified by its position
relative to the (gap-bridged) queue at T:
  in_queue         queued at T: FN = queue persisted but we dropped it, FP = it cleared but we kept it
  upstream 1-3     1..3 links upstream of the queue (tail growth or retreat)
  upstream 4-12    further upstream (fast tail growth)
  downstream 1-3   1..3 links downstream (head movement)
  downstream 4-12  further downstream (a new queue forming)

    python experiments/eda/t2_ongoing_errors.py --tag a3b [--set official]
"""
import argparse

import numpy as np
import pandas as pd

from tfb import t2_ongoing as og
from tfb.t2_bench import cache_path

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="a3b")
ap.add_argument("--set", default="official", choices=["official", "selector"])
args = ap.parse_args()

cache = pd.read_parquet(cache_path("ongoing", args.tag))
cache = cache[cache.set == args.set]
panels = {}
cells, wins = [], []
for key, g in cache.groupby("key", sort=False):
    panel, d, T = key.split("|")
    if panel not in panels:
        panels[panel] = og.TrainPanel(panel)
    r = panels[panel].window(int(d), int(T)).merge(g[["link", "h", "p"]], on=["link", "h"])
    sel = og.choose_window(r.p.to_numpy(), r.h.to_numpy())
    el = r.w.to_numpy() > 0
    y = (r.y.to_numpy() > 0) & el
    pred = sel & el
    dn, up = r.dist_dn.to_numpy(), r.dist_up.to_numpy()   # links to the queue downstream / upstream of each link
    kind = np.select([r.q_bridge.to_numpy(), dn <= 3, dn <= 12, up <= 3, up <= 12],
                     ["in_queue", "upstream 1-3", "upstream 4-12", "downstream 1-3", "downstream 4-12"], "other")
    err = np.where(y & ~pred, "FN", np.where(pred & ~y, "FP", ""))
    union = (pred | y).sum() + r.outside_true.iloc[0]
    wins.append(dict(key=key, panel=panel, iou=(pred & y).sum() / union if union else 1.0, union=union,
                     n_err=(err != "").sum() + r.outside_true.iloc[0]))
    for k, e, h in zip(kind, err, r.h):
        if e:
            cells.append((k, e, h))
    cells += [("beyond 12 links", "FN", 0)] * int(r.outside_true.iloc[0])
w = pd.DataFrame(wins)
c = pd.DataFrame(cells, columns=["kind", "err", "h"])
print(f"{args.tag} on {args.set} windows: mean IoU {w.groupby('panel').iou.mean().mean():.4f}, {len(w)} windows, {len(c)} error cells")
print("\nshare of error cells by position and type:")
print((pd.crosstab(c.kind, c.err) / len(c)).round(3).to_string())
print("\nerror cells by step (1..6):")
print(pd.crosstab(c.err, c.h).to_string())
print("\nworst windows:")
print(w.sort_values("iou").head(8).round(3).to_string(index=False))
