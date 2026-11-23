"""1c (iv). The production ongoing LightGBM retrained on selector-faithful windows with the selection flags as features.

Training frame: ~/data/tfb/cache/late/sel_faithful_train.parquet (sel_frame.py). Features: the production
features (tfb.t2_ongoing, FEATURES_DROP applied) plus sel_flag, sel_sb, sel_flag_h. Parameters and
rounds are the production model's (og.PARAMS, og.ROUNDS), 4 threads; nothing is tuned.

Leave-one-month-out: a window of month m is predicted by the booster trained on the faithful windows of the
other eight months (~/data/tfb/cache/late/sel_faith_without_<month>_s0.txt). The production probabilities for the
same windows come from sel_eval_cache.pkl (also without month m).

Tags (ablations of what the gain comes from):
  faith     faithful windows, with the three selection features (the candidate)
  faithnf   faithful windows, without the selection features: only the choice of training windows changes
  faithdyn  faithful windows and selection features, without time of day, weekday, position and panel:
            what is left when the model cannot learn where and when train queues recur

    python experiments/late/sel_model.py --fit [--tag faithnf]       # the nine folds
    python experiments/late/sel_model.py --predict [--tag faithnf]   # p_new for the official and replica windows
    python experiments/late/sel_model.py --final --seeds 5           # all nine months, for the scored windows
"""
import argparse
import pickle
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

import sel_frame as sf
import sel_lib as sl
from tfb import t2_ongoing as og
from tfb.t2_onset import T2_PANELS

SEL = ["sel_flag", "sel_sb", "sel_flag_h"]
# sel_kback is left out: in the frame it is not limited by the spacing to the previous window, at test time it is
DROP = set(og.FEATURES_DROP) | {"role", "sel_kback"}
RECUR = {"tod", "weekday", "pos", "panel_code"}     # the features that can carry train-month recurrence (time and place)
TAG_DROP = {"faith": set(), "faithnf": set(SEL), "faithdyn": RECUR}


def pred_path(tag: str = "faith"):
    return sl.LATE / ("sel_model_pred.pkl" if tag == "faith" else f"sel_model_pred_{tag}.pkl")


PRED = pred_path()


def load_frame(tag: str = "faith") -> tuple[pd.DataFrame, list]:
    df = pd.read_parquet(sf.OUT)
    df["panel_code"] = df.panel.map(sl.CODES)
    feats = [c for c in df.columns if c not in DROP and c not in TAG_DROP[tag]]
    return df, feats


def model_path(month: str | None, seed: int = 0, tag: str = "faith"):
    return sl.LATE / (f"sel_{tag}_all_s{seed}.txt" if month is None else f"sel_{tag}_without_{month}_s{seed}.txt")


def train(df: pd.DataFrame, feats: list, seed: int = 0) -> lgb.Booster:
    params = {**og.PARAMS, "seed": og.PARAMS["seed"] + seed, "num_threads": sl.THREADS}
    return lgb.train(params, lgb.Dataset(df[feats], df.y, weight=df.w), og.ROUNDS).free_dataset()


def fit(final: bool, seeds: int = 1, tag: str = "faith") -> None:
    df, feats = load_frame(tag)
    print(f"[{tag}] {len(df):,} rows, {df.key.nunique():,} windows, {len(feats)} features")
    for month in ([None] if final else sorted(df.month.unique())):
        for s in range(seeds):
            path = model_path(month, s, tag)
            if path.exists():
                continue
            t0 = time.time()
            train(df if month is None else df[df.month != month], feats, s).save_model(str(path))
            print(f"  {path.name} in {time.time() - t0:.0f} s", flush=True)


def predict(tag: str = "faith") -> None:
    """p_new [6, L] (0 off the candidate cells) for every window of sel_eval_cache.pkl, out of fold."""
    with open(sl.LATE / "sel_eval_cache.pkl", "rb") as f:
        recs = pickle.load(f)
    boosters = {}
    by_panel = {}
    for i, r in enumerate(recs):
        by_panel.setdefault(r["panel"], []).append(i)
    preds = [None] * len(recs)
    for panel in T2_PANELS:
        tp = og.TrainPanel(panel)
        for i in by_panel.get(panel, []):
            r = recs[i]
            month = r["month"]
            if month not in boosters:
                boosters[month] = lgb.Booster(model_file=str(model_path(month, 0, tag)))
            b = boosters[month]
            x = sf.window_rows(tp, int(r["d"]), int(r["T"]), r["gap"], "eval")
            p = np.zeros(r["ours"].shape, np.float32)
            if len(x):
                x["panel_code"] = sl.CODES[panel]
                hh, ll = x.h.to_numpy() - 1, x.link.to_numpy()
                assert r["cand"][hh, ll].all() and r["cand"].sum() == len(x)
                assert int(x.sel_flag.iloc[0]) == int(r["flag"]), (panel, r["d"], r["T"])
                p[hh, ll] = b.predict(x[b.feature_name()], num_threads=sl.THREADS)
            preds[i] = p
        print(f"  {panel}: {len(by_panel.get(panel, []))} windows", flush=True)
    with open(pred_path(tag), "wb") as f:
        pickle.dump(preds, f)
    print("wrote", pred_path(tag))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--predict", action="store_true")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--tag", default="faith", choices=sorted(TAG_DROP))
    args = ap.parse_args()
    if args.fit:
        fit(False, args.seeds, args.tag)
    if args.final:
        fit(True, args.seeds, args.tag)
    if args.predict:
        predict(args.tag)
