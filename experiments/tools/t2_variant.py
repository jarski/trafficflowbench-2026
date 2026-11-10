"""Build a submission that changes one Task 2 family only.

Retrains the onset or ongoing model from a given training frame (seed-averaged) and predicts that
family's windows; every other row (the other family, Tasks 1 and 4) is copied from a base version.
The leaderboard delta against the base then isolates one family (0.15 of the total each).

    python experiments/tools/t2_variant.py --out v7 --base v5 --family onset --frame ~/data/tfb/cache/t2_onset_train_v1.parquet --seeds 5
"""
import argparse
import shutil
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from tfb import pipeline_v1 as pv1
from tfb import t2_ongoing as og
from tfb import t2_onset as on
from tfb.compose import compose
from tfb.dense import tod_index
from tfb.paths import RELEASE, SUBMISSIONS
from tfb.pipeline_v0 import SCORED
from tfb.task1_io import meta

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--base", required=True)
ap.add_argument("--family", choices=["onset", "ongoing"], required=True)
ap.add_argument("--frame", required=True)
ap.add_argument("--seeds", type=int, default=5)
ap.add_argument("--shrink", type=float, default=None, help="override t2_ongoing.STEP_SHRINK")
ap.add_argument("--drop", default="", help="comma-separated feature columns to leave out")
ap.add_argument("--onset-extra", default="", help="t2_onset.EXTRA_FEATURES for test-time features, e.g. context")
args = ap.parse_args()
if args.shrink is not None:
    og.STEP_SHRINK = args.shrink
on.EXTRA_FEATURES = tuple(x for x in args.onset_extra.split(",") if x)

mod = on if args.family == "onset" else og
cond = f"queue_{args.family}"
codes = {p: i for i, p in enumerate(sorted(on.T2_PANELS))}
df = pd.read_parquet(Path(args.frame).expanduser())
df["panel_code"] = df.panel.map(codes)
drop = {c for c in args.drop.split(",") if c}
feats = [c for c in df.columns if c not in mod.FEATURES_DROP and c not in drop]
model = pv1.Ensemble([lgb.train({**mod.PARAMS, "seed": mod.PARAMS["seed"] + i},
                                lgb.Dataset(df[feats], df.y, weight=df.w), mod.ROUNDS).free_dataset()
                      for i in range(args.seeds)])
del df
print(f"trained {args.family}: {args.seeds} seed(s), {len(feats)} features", flush=True)

work = SUBMISSIONS / args.out
work.mkdir(parents=True, exist_ok=True)
queue = pd.read_csv(SUBMISSIONS / args.base / "queue.csv", dtype=str)
new = []
for split in SCORED:
    for panel in on.T2_PANELS:
        ctx = pv1.panel_context(panel, split)
        win = pd.read_csv(RELEASE / "task2" / panel / split / "window_index.csv")
        pos = {l: i for i, l in enumerate(meta(panel)["links"])}
        for w in win[win.condition == cond].itertuples():
            T, x = pv1.window_features(ctx, w)
            x["panel_code"] = codes[panel]
            pred = np.zeros((og.STEPS, ctx["L"]), bool)
            if args.family == "onset":
                pred[og.STEPS - 1] = on.choose_set(model.predict(x[feats]))
            elif len(x):
                sel = og.choose_window(model.predict(x[feats]), x.h.to_numpy())
                pred[x.h.to_numpy()[sel] - 1, x.link.to_numpy()[sel]] = True
            rows = queue[queue.window_id == w.window_id].copy()
            step = tod_index(rows.timestamp) - T
            rows["queue_pred"] = pred[step - 1, rows.link_id.map(pos).to_numpy()].astype(int).astype(str)
            new.append(rows)
    print(f"  {split}: predicted", flush=True)
new = pd.concat(new)
keep = queue[~queue.window_id.isin(new.window_id)]
out = pd.concat([keep, new]).sort_values(["window_id", "timestamp", "link_id"])
assert len(out) == len(queue)
out.to_csv(work / "queue.csv", index=False)
for name in ("state.csv", "odme.csv"):
    shutil.copy(SUBMISSIONS / args.base / name, work / name)
compose(args.out, {"state": args.out, "queue": args.out, "odme": args.out})
