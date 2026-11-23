"""1c/1d. Out-of-fold production forecasts for the train windows, with a positive control, cached for sel_decode.py.

For every official ongoing train window (40) and every per-month replica window (sel_replica.py) this builds
  truth, elig   [6, L] the queue label (observed speed <= 0.6 free speed on eligible cells) and eligibility at T+5..T+30
  ours, cand    [6, L] our LightGBM's probability out of fold and its candidate cells (links within 12 of a queue)
  public        [6, L] the public all-link model's probability out of fold
  v, e          [37, L] the speeds and eligibility flags a forecast may read (slots T-180..T): sel_lib.visible
  truth_i       [6, L] the interpolated state at T+5..T+30 (a stand-in for the noise-free label), analysis only
and writes ~/data/tfb/cache/late/sel_eval_cache.pkl.

Positive control (--control): the same code path scored on the bench's cached out-of-fold probabilities
(cache/bench/ongoing_imp.parquet) must reproduce LOG 2026-10-19: official 0.8452 (ours), 0.7926 (public alone),
0.8239 (0.5 average); selector 0.8834 / 0.8371 / 0.8712. Then the folds retrained here (sel_folds.py) must
reproduce the cached probabilities on the official windows.

    python experiments/late/sel_eval.py --control
    python experiments/late/sel_eval.py --build [--seeds 1]
"""
import argparse
import pickle

import numpy as np
import pandas as pd

import sel_lib as sl
from tfb.paths import CACHE
from tfb.t2_onset import T2_PANELS

CACHE_FILE = sl.LATE / "sel_eval_cache.pkl"
OFFICIAL_FILE = sl.LATE / "sel_eval_cache_official.pkl"   # --build --only-official: needs the June fold only
BACK = 36


def panel_mean(df: pd.DataFrame, cols: list) -> pd.Series:
    """Mean over panels of the mean window IoU, as tfb.t2_bench.summary (equal to the window mean at 5 per panel)."""
    return df.groupby("panel")[cols].mean().mean()


def control(prod: sl.Production) -> None:
    bench = pd.read_parquet(CACHE / "bench" / "ongoing_imp.parquet")
    rows = []
    for wset in ("official", "selector"):
        part = bench[bench.set == wset]
        for key, g in part.groupby("key", sort=False):
            panel, d, T = key.split("|")
            d, T = int(d), int(T)
            g = g.drop_duplicates(["link", "h"])
            tp = prod.panel(panel)
            L = tp.q.shape[2]
            ours, cand = np.zeros((sl.Z, L)), np.zeros((sl.Z, L), bool)
            ours[g.h.to_numpy() - 1, g.link.to_numpy()] = g.p.to_numpy()
            cand[g.h.to_numpy() - 1, g.link.to_numpy()] = True
            pr, dmax = None, np.nan
            if wset == "official":   # the fold retrained here against the cached probabilities
                pr = prod.probabilities(panel, d, T)
                dmax = float(np.abs(pr["ours"] - ours).max())
                assert np.array_equal(pr["cand"], cand), key
            public = pr["public"] if pr else _public_only(prod, panel, d, T)
            truth, el = tp.q[d, T + 1:T + 1 + sl.Z], tp.a["eligible"][d, T + 1:T + 1 + sl.Z]
            rows.append(dict(set=wset, key=key, panel=panel, dmax=dmax,
                             ours=sl.window_iou(sl.choose(ours), truth, el),
                             public=sl.window_iou(sl.choose(public), truth, el),
                             avg5=sl.window_iou(sl.choose(0.5 * ours + 0.5 * public), truth, el),
                             ours_refit=sl.window_iou(sl.choose(pr["ours"]), truth, el) if pr else np.nan))
    df = pd.DataFrame(rows)
    print("Positive control: this code path on the bench's cached out-of-fold probabilities (1 seed)")
    print("  LOG 2026-10-19: official ours 0.8452, public 0.7926, avg.5 0.8239; selector 0.8834, 0.8371, 0.8712")
    for wset in ("official", "selector"):
        part = df[df.set == wset]
        m = panel_mean(part, ["ours", "public", "avg5"])
        print(f"  {wset:8s} ({len(part)} windows): ours {m.ours:.4f}, public {m.public:.4f}, avg.5 {m.avg5:.4f}")
    off = df[df.set == "official"]
    print(f"  folds retrained here vs cached probabilities, official windows: max |dp| = {off.dmax.max():.2e}; "
          f"IoU with the retrained fold {panel_mean(off, ['ours_refit']).ours_refit:.4f}")


def _public_only(prod: sl.Production, panel: str, d: int, T: int) -> np.ndarray:
    tp = prod.panel(panel)
    pp = prod._pub_panels[panel]
    a = tp.a
    origin = pd.Timestamp(sl.tp_day(panel, d), tz="UTC") + pd.Timedelta(minutes=5 * int(T))
    X = pp.features(a["true_speed"][d, T - sl.H:T][:, pp.ours], a["true_flow"][d, T - sl.H:T][:, pp.ours], origin, "queue_ongoing")
    p = prod.public(tp.months[d]).predict(X, num_threads=sl.THREADS).reshape(sl.Z, len(pp.links))
    out = np.empty_like(p)
    out[:, pp.ours] = p
    return out


def build(prod: sl.Production, only_official: bool = False) -> None:
    flags = pd.read_parquet(sl.LATE / "sel_flags.parquet")
    flags = flags[flags.split == "train"]
    if only_official:
        flags = flags[flags.set == "official"]
    recs = []
    for panel in T2_PANELS:
        tp = prod.panel(panel)
        a = tp.a
        vc = sl.vcut(panel)
        interp = sl.interp_time(a["true_speed"]) <= vc
        for r in flags[flags.panel == panel].itertuples():
            d, T = int(r.d), int(r.T)
            pr = prod.probabilities(panel, d, T)
            rec = {k: getattr(r, k) for k in ("set", "panel", "window_id", "d", "T", "month", "gap", "kind", "cap_m5", "cap_0",
                                              "cap_m5_ffill", "spacing_ok", "coverage", "established", "flag", "k_back")}
            rec.update(truth=tp.q[d, T + 1:T + 1 + sl.Z].copy(), elig=a["eligible"][d, T + 1:T + 1 + sl.Z].copy(),
                       truth_i=interp[d, T + 1:T + 1 + sl.Z].copy(),
                       ours=pr["ours"].astype(np.float32), cand=pr["cand"], public=pr["public"].astype(np.float32),
                       v=sl.visible(a["speed"], d, T, a["true_speed"][d, T - sl.H:T], BACK),
                       e=sl.visible(a["eligible"].astype(np.float32), d, T, None, BACK) > 0.5, vc=vc.astype(np.float32),
                       tod=T, weekday=int(tp.wd[d]))
            recs.append(rec)
        print(f"  {panel}: {int((flags.panel == panel).sum())} windows", flush=True)
    path = OFFICIAL_FILE if only_official else CACHE_FILE
    with open(path, "wb") as f:
        pickle.dump(recs, f)
    print("wrote", path, len(recs), "windows")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--only-official", action="store_true")
    ap.add_argument("--seeds", type=int, default=1)
    args = ap.parse_args()
    prod = sl.Production(seeds=args.seeds)
    if args.control:
        control(prod)
    if args.build:
        build(prod, args.only_official)


if __name__ == "__main__":
    main()
