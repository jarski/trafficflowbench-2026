"""Holdout predictions of every Task 1 family on the Dec-Feb months, with and without the emulated blank spans.

World B: the emulated layer of span_lib (spans blanked, histories unmasked). World A: the released train layer.
Families (all trained on Jun-Nov only):
    interp   plain time interpolation (tfb.t1_model.PanelData.si / fi = tfb.t1_interp.interp_time, with the
             production fallback for links without any observation that day)
    ours     tfb.t1_model, cache/late/hold_ours_*.txt (span_train_holdout.py)
    public   tfb.t1_public, cache/late/hold_public_*.txt
    unet5/6  tfb.t1_unet seeds 5 and 6, cache/bench/t1_unet_holdout{,_s6}.pt (the project's holdout weights);
    unet     their production average (corrections averaged before the clip)
Each is stored as cache/late/pred/<world>_<family>_<panel>.npz: speed and flow, [holdout day, 288, link].

--days window (the two LightGBM families; prediction is the slow step on a shared machine) predicts only the days
that carry a window, NaN elsewhere. The two worlds differ on those days only, apart from a small change of the
month profiles. span_eval.py takes the other days from the project's cached holdout predictions of the same pair
(cache/t1_holdout_base, full training configuration, released layer). The public family reads values up to an
hour away across midnight, so it is run on each window day with its two neighbours and kept for the window day.

    python experiments/late/span_predict.py --families interp,unet --worlds A,B
    python experiments/late/span_predict.py --families ours,public --worlds B,A --days window --threads 4
"""
import argparse
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import span_lib as L  # noqa: E402

from tfb import t1_model as t1  # noqa: E402
from tfb import t1_public, t1_unet  # noqa: E402
from tfb.paths import CACHE, PANELS  # noqa: E402

PRED = L.LATE / "pred"
PRED.mkdir(exist_ok=True)


def save(world: str, family: str, panel: str, speed, flow, hold) -> None:
    np.savez_compressed(PRED / f"{world}_{family}_{panel}.npz", speed=speed[hold].astype(np.float32), flow=flow[hold].astype(np.float32))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", default="interp,unet,ours,public")
    ap.add_argument("--worlds", default="A,B")
    ap.add_argument("--unet", default=f"{CACHE / 'bench' / 't1_unet_holdout.pt'},{CACHE / 'bench' / 't1_unet_holdout_s6.pt'}")
    ap.add_argument("--unet-name", default="unet", help="family name of the U-Net average; single seeds get the suffix 5 / 6")
    ap.add_argument("--ours-prefix", default="hold_ours")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--days", choices=("all", "window"), default="all")
    args = ap.parse_args()
    fams = args.families.split(",")
    t1.PARAMS["num_threads"] = args.threads
    t1_unet.THREADS = args.threads
    import torch
    torch.set_num_threads(args.threads)
    t0 = time.time()
    origins = L.emulated_origins()
    hold = {p: np.isin(L.day_months(p), L.HOLD_MONTHS) for p in PANELS}
    train_days = {p: ~hold[p] for p in PANELS}
    ours = pub = unets = None
    if "ours" in fams:
        ours = {}
        for y in ("y_k", "y_s", "y_f"):
            b = lgb.Booster(model_file=str(L.LATE / f"{args.ours_prefix}_{y}.txt"), params={"num_threads": args.threads})
            ours[y] = (b, b.feature_name())
    if "public" in fams:
        pub = {k: lgb.Booster(model_file=str(L.LATE / f"hold_public_{k}.txt"), params={"num_threads": args.threads}) for k in ("speed", "flow")}
    if "unet" in fams:
        unets = [t1_unet.load_model(m).to(t1_unet.DEVICE) for m in args.unet.split(",")]
    for world in args.worlds.split(","):
        L.install({p: L.emulated_layer(p, origins[p]) for p in PANELS} if world == "B" else None)
        for i, p in enumerate(PANELS):
            if "interp" in fams:
                pd_ = t1.PanelData(p, "train", profile_days=train_days[p])
                save(world, "interp", p, pd_.si, pd_.fi, hold[p])
            sel = hold[p]
            if args.days == "window":     # the days of this panel that carry a window (none on the two panels without Task 2)
                sel = np.zeros(len(hold[p]), bool)
                sel[sorted({d for d, _, _ in origins[p]})] = True
                sel &= hold[p]
            if ours and sel.any():
                sp, fl, _ = t1.predict_panel(ours, p, "train", i, sel, profile_days=train_days[p])
                save(world, "ours", p, np.where(sel[:, None, None], sp, np.nan), np.where(sel[:, None, None], fl, np.nan), hold[p])
            if pub and sel.any():
                near = sel.copy()
                near[1:] |= sel[:-1]
                near[:-1] |= sel[1:]
                sp, fl = t1_public.predict(pub, p, "train", near & hold[p] if args.days == "window" else sel)
                save(world, "public", p, np.where(sel[:, None, None], sp, np.nan), np.where(sel[:, None, None], fl, np.nan), hold[p])
            if unets:
                if len(unets) > 1:
                    for m, tag in zip(unets, ("5", "6")):
                        save(world, args.unet_name + tag, p, *t1_unet.predict_split(m, p, "train", hold[p]), hold[p])
                save(world, args.unet_name, p, *t1_unet.predict_split(unets, p, "train", hold[p]), hold[p])
            print(f"[{(time.time() - t0) / 60:5.1f} min] world {world} {p} done", flush=True)


if __name__ == "__main__":
    main()
