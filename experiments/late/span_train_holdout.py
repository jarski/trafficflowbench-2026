"""Holdout models for the blank-span emulation: our LightGBM family and the public-feature family, trained on
Jun-Nov only (2026-11-23).

The Dec-Feb months are read through the emulated layer of span_lib (spans blanked, histories unmasked), so the
self-supervised rows come from the months as a scored month would show them, as in production.

Production configuration (tfb.t1_model, pipeline_final) with three reductions that only save CPU time on a
shared machine, all stated in the report:
    1 seed instead of 2; 200k + 100k congested training cells per panel instead of 600k + 300k;
    100k self-supervised rows per panel instead of 300k per scored month.
The public-feature family is trained exactly as `tfb.t1_public.fit` on the six months.

    python experiments/late/span_train_holdout.py [--threads 3] [--skip-ours] [--skip-public]
Writes cache/late/hold_ours_{y_k,y_s,y_f}.txt and cache/late/hold_public_{speed,flow}.txt.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import span_lib as L  # noqa: E402

from tfb import t1_model as t1  # noqa: E402
from tfb import t1_public  # noqa: E402
from tfb.paths import PANELS  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--threads", type=int, default=3)
ap.add_argument("--cells", type=int, default=200_000)
ap.add_argument("--congested", type=int, default=100_000)
ap.add_argument("--pseudo", type=int, default=100_000)
ap.add_argument("--skip-ours", action="store_true")
ap.add_argument("--skip-public", action="store_true")
args = ap.parse_args()

t1.PARAMS["num_threads"] = args.threads
t1_public.PARAMS["num_threads"] = args.threads
t1.SEEDS = 1
t1.CONGESTED_EXTRA = args.congested
t0 = time.time()


def log(msg: str) -> None:
    print(f"[{(time.time() - t0) / 60:6.1f} min] {msg}", flush=True)


origins = L.emulated_origins()
L.install({p: L.emulated_layer(p, origins[p]) for p in PANELS})
hold = {p: np.isin(L.day_months(p), L.HOLD_MONTHS) for p in PANELS}
train_days = {p: ~hold[p] for p in PANELS}

if not args.skip_ours:
    df = t1.training_frame(PANELS, train_days, args.cells)
    rng = np.random.default_rng(1)
    df = pd.concat([df] + [t1.pseudo_frame(p, "train", i, 0.15, args.pseudo, rng, hold[p], train_days[p])
                           for i, p in enumerate(PANELS)], ignore_index=True)
    log(f"training frame: {len(df):,} rows x {df.shape[1]} columns")
    models = t1.fit(df)
    for name, (m, feats) in models.items():
        m.save_model(str(L.LATE / f"hold_ours_{name}.txt"))
        log(f"saved hold_ours_{name}.txt ({len(feats)} features)")
    del df, models

if not args.skip_public:
    pub = t1_public.fit(L.TRAIN_MONTHS)
    for name, m in pub.items():
        m.save_model(str(L.LATE / f"hold_public_{name}.txt"))
    log("saved hold_public_{speed,flow}.txt")
log("done")
