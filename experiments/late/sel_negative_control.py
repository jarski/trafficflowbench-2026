"""A control for the bench of the selection-conditioned model: windows that the selector did NOT choose.

The model of sel_model.py gains on the official and replica windows. If that gain came from something other
than the selection (more or different training data, a leak in the features, the scoring), it would show on any
ongoing window. Here both models forecast ordinary origins: a seeded sample of the production training windows
(every sixth valid origin of a day, cache/t2_ongoing_train.parquet), each with the fold that never saw its month.
The selection features are computed as at test time, as if the origin had been reached without a spacing limit.
Expected: the selection-conditioned model is no better, or worse, here, because the change it has learned to
expect at T+30 does not come.

    python experiments/late/sel_negative_control.py [--per-panel 60]
"""
import argparse

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import sel_frame as sf
import sel_lib as sl
import sel_model as sm
from tfb.paths import CACHE
from tfb.t2_onset import T2_PANELS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-panel", type=int, default=60)
    args = ap.parse_args()
    keys = pq.read_table(CACHE / "t2_ongoing_train.parquet", columns=["key"]).to_pandas().key.drop_duplicates()
    faithful = set(pq.read_table(sf.OUT, columns=["key"]).to_pandas().key.str.rsplit("|", n=1).str[0].unique())
    rng = np.random.default_rng(20261123)
    prod = sl.Production(seeds=1)
    boosters = {}
    rows = []
    for panel in T2_PANELS:
        ks = keys[keys.str.startswith(panel + "|")].to_numpy()
        ks = np.array([k for k in ks if k not in faithful])            # ordinary origins only: not a selector-faithful one
        for key in rng.choice(ks, min(args.per_panel, len(ks)), replace=False):
            _, d, T = key.split("|")
            d, T = int(d), int(T)
            tp = prod.panel(panel)
            month = tp.months[d]
            pr = prod.probabilities(panel, d, T)
            if month not in boosters:
                boosters[month] = lgb.Booster(model_file=str(sm.model_path(month)))
            x = sf.window_rows(tp, d, T, np.nan, "control")
            new = np.zeros(pr["ours"].shape)
            if len(x):
                x["panel_code"] = sl.CODES[panel]
                new[x.h.to_numpy() - 1, x.link.to_numpy()] = boosters[month].predict(x[boosters[month].feature_name()], num_threads=sl.THREADS)
            truth, el = tp.q[d, T + 1:T + 1 + sl.Z], tp.a["eligible"][d, T + 1:T + 1 + sl.Z]
            flag = bool(len(x) and x.sel_flag.iloc[0])
            rows.append(dict(panel=panel, flag=flag,
                             production=sl.window_iou(sl.choose(sl.combine(pr["ours"], pr["cand"], pr["public"])), truth, el),
                             new_blend=sl.window_iou(sl.choose(sl.combine(new, pr["cand"], pr["public"])), truth, el),
                             ours_alone=sl.window_iou(sl.choose(sl.combine(pr["ours"], pr["cand"], pr["public"], weight=0.0)), truth, el),
                             new_alone=sl.window_iou(sl.choose(sl.combine(new, pr["cand"], pr["public"], weight=0.0)), truth, el)))
        print(f"  {panel}: done", flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(sl.LATE / "sel_negative_control.parquet", index=False)
    print(f"\n{len(df)} ordinary ongoing origins (not selector-faithful), out of fold:")
    for label, part in (("all", df), ("flag computed as 1", df[df.flag]), ("flag computed as 0", df[~df.flag])):
        d1, d2 = part.new_blend - part.production, part.new_alone - part.ours_alone
        print(f"  {label:20s} n={len(part):4d}  production {part.production.mean():.4f}  new in the blend {part.new_blend.mean():.4f} "
              f"({d1.mean():+.4f} ± {d1.sem():.4f})   ours alone {part.ours_alone.mean():.4f}  new alone {part.new_alone.mean():.4f} ({d2.mean():+.4f} ± {d2.sem():.4f})")


if __name__ == "__main__":
    main()
