"""Leave-one-month-out boosters of the production ongoing model, saved for the sel_ scripts.

The same training as tfb.t2_bench.fold_models on cache/t2_ongoing_train.parquet (og.PARAMS, og.ROUNDS, seed 7 + i),
at 4 threads. Files: ~/data/tfb/cache/late/sel_prod_without_<month>_s<i>.txt

    python experiments/late/sel_folds.py --seeds 1 [--months 2030-06,2030-07]
"""
import argparse
import time

import sel_lib as sl

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=1)
ap.add_argument("--months", default="")
args = ap.parse_args()
prod = sl.Production(seeds=args.seeds)
months = [m for m in args.months.split(",") if m] or sorted(prod.frame().month.unique())
for m in months:
    t0 = time.time()
    prod.fold(m)
    prod._folds.clear()
    print(f"fold without {m}: {args.seeds} booster(s) in {time.time() - t0:.0f} s", flush=True)
