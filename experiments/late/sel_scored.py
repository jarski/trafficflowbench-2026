"""The production ongoing probabilities of v64 for the 80 scored ongoing windows, rebuilt from v64's saved models.

v64/queue.csv holds only the final 0/1 cells. Rules that change probabilities before the expected-IoU set is
chosen need the probabilities, so this script recomputes them exactly as tfb.pipeline_v1.predict_window does:
  ours    mean of the ten boosters ~/data/tfb/submissions/v64/t2_ongoing_s0..s9.txt on pipeline_v1.window_features
  public  ~/data/tfb/submissions/v64/t2_public.txt on tfb.t2_public.window_input
and checks that the set chosen from them equals v64's queue.csv in every cell of all 80 windows (positive control).

Writes ~/data/tfb/cache/late/sel_scored_cache.pkl: per window the probabilities, the feature rows, the visible
block (slots <= T) and the selection features. Nothing stamped after T is read (pipeline_v1.window_features and
t2_public.window_input passed the award check; the visible block is sel_lib.visible).

    python experiments/late/sel_scored.py
"""
import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd

import sel_frame as sf
import sel_lib as sl
import sel_submit as ss
from tfb import pipeline_v1 as pv1
from tfb import t2_ongoing as og
from tfb import t2_public
from tfb.dense import tod_index
from tfb.paths import RELEASE, SUBMISSIONS
from tfb.pipeline_final import T2_SEEDS
from tfb.t2_onset import T2_PANELS
from tfb.task1_io import load, meta

OUT = sl.LATE / "sel_scored_cache.pkl"
BACK = 36


def main() -> None:
    v64 = SUBMISSIONS / "v64"
    boosters = [lgb.Booster(model_file=str(v64 / f"t2_ongoing_s{i}.txt")) for i in range(T2_SEEDS["ongoing"])]
    public = lgb.Booster(model_file=str(v64 / "t2_public.txt"))
    feats = boosters[0].feature_name()
    base = pd.read_csv(v64 / "queue.csv", dtype={"link_id": str})
    by_window = dict(tuple(base.groupby("window_id", sort=False)))
    codes = sl.CODES
    recs, cells, differ = [], 0, 0
    for split in ("validation", "private"):
        for panel in T2_PANELS:
            ctx = pv1.panel_context(panel, split)
            a = load(panel, split, "speed", "eligible")
            vc = sl.vcut(panel)
            pos = {l: i for i, l in enumerate(meta(panel)["links"])}
            L = len(pos)
            index = pd.read_csv(RELEASE / "task2" / panel / split / "window_index.csv")
            for w in ss.window_table(index, ctx["days"]).itertuples():
                T, x = pv1.window_features(ctx, w)
                assert T == w.T
                x["panel_code"] = codes[panel]
                ours, cand = np.zeros((sl.Z, L)), np.zeros((sl.Z, L), bool)
                if len(x):
                    hh, ll = x.h.to_numpy() - 1, x.link.to_numpy()
                    ours[hh, ll] = np.mean([b.predict(x[feats], num_threads=sl.THREADS) for b in boosters], axis=0)
                    cand[hh, ll] = True
                pub = t2_public.window_probs(public, ctx, w, T)
                pred = sl.choose(sl.combine(ours, cand, pub))
                rows = by_window[w.window_id]
                step = tod_index(rows.timestamp) - T - 1
                li = rows.link_id.map(pos).to_numpy()
                got = rows.queue_pred.to_numpy() == 1
                cells += len(rows)
                differ += int((pred[step, li] != got).sum())
                v = sl.visible(a["speed"], int(w.d), T, None, BACK)
                e = sl.visible(a["eligible"].astype(np.float32), int(w.d), T, None, BACK) > 0.5
                f = sl.flag_window(v, e, vc, w.gap, lag=1)
                recs.append(dict(split=split, panel=panel, window_id=w.window_id, d=int(w.d), T=T, gap=w.gap,
                                 ours=ours.astype(np.float32), cand=cand, public=pub.astype(np.float32), x=x,
                                 v=v, e=e, vc=vc.astype(np.float32), flag=f["flag"], weekday=int(ctx["wd"][int(w.d)]),
                                 **sf.sel_features_visible(v, e, vc, w.gap)))
            print(f"  {panel}/{split}: done", flush=True)
    print(f"positive control: the set chosen from the rebuilt probabilities differs from v64/queue.csv in {differ} of {cells:,} cells "
          f"({len(recs)} ongoing windows)")
    assert differ == 0
    with open(OUT, "wb") as fh:
        pickle.dump(recs, fh)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
