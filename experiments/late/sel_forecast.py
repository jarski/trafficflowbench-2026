"""3. Forecasts for the two candidates against v64, from train rates applied to the scored windows' own mix.

Every ongoing window is put in one of four strata that can be told at test time (data stamped <= T):
  flagged, persistent   flagged, and the production forecast is (nearly) the persistence of the state at T:
                        IoU between the production cells and the state repeated over six steps > 0.9
  flagged, deviating    flagged, and the production forecast departs from persistence (it clears or grows cells)
  spacing               not flagged, exactly 6 h after the previous window
  new                   not flagged, later than 6 h: no established queue yet at T-5
For each stratum the mean change in window IoU against production is measured on train, out of fold, on the 40
official windows and on the 359 replica windows (rule A: steps 1-5 = state in flagged windows; rule B: the
selection-conditioned model in the production blend). The forecast for a scored month is
    sum over strata (windows of the month in the stratum x train mean change) / 40 x 0.15.
These are train rates, before any discount for the shift of the scored months; the bands in the report are wider.

    python experiments/late/sel_forecast.py
"""
import pickle

import numpy as np
import pandas as pd

import sel_lib as sl
import sel_model as sm

STRATA = ["flagged, persistent", "flagged, deviating", "spacing", "new"]


def stratum(flag: bool, gap: float, prod: np.ndarray, state: np.ndarray) -> str:
    if flag:
        return "flagged, persistent" if sl.iou(prod, np.broadcast_to(state, prod.shape)) > 0.9 else "flagged, deviating"
    return "spacing" if gap == sl.SPACING else "new"


def main() -> None:
    pd.set_option("display.width", 250)
    with open(sl.LATE / "sel_eval_cache.pkl", "rb") as f:
        recs = pickle.load(f)
    with open(sm.PRED, "rb") as f:
        preds = pickle.load(f)
    rows = []
    for r, new in zip(recs, preds):
        ours, cand, pub = r["ours"].astype(float), r["cand"], r["public"].astype(float)
        prod = sl.choose(sl.combine(ours, cand, pub))
        state = sl.last_reading(r["v"]) <= r["vc"]
        a = prod.copy()
        if r["flag"]:
            a[:5] = state[None, :]
        b = sl.choose(sl.combine(new.astype(float), cand, pub))
        base = sl.window_iou(prod, r["truth"], r["elig"])
        rows.append(dict(set=r["set"], stratum=stratum(bool(r["flag"]), r["gap"], prod, state), production=base,
                         dA=sl.window_iou(a, r["truth"], r["elig"]) - base, dB=sl.window_iou(b, r["truth"], r["elig"]) - base,
                         cellsA=int((a != prod).sum()), cellsB=int((b != prod).sum())))
    tr = pd.DataFrame(rows)
    rates = {}
    for wset in ("official", "replica"):
        part = tr[tr.set == wset]
        g = part.groupby("stratum").agg(n=("dA", "size"), production=("production", "mean"), dA=("dA", "mean"), seA=("dA", "sem"),
                                        dB=("dB", "mean"), seB=("dB", "sem"), cellsA=("cellsA", "mean"), cellsB=("cellsB", "mean")).reindex(STRATA)
        rates[wset] = g
        print(f"\ntrain rates, {wset} windows (mean change in window IoU against production, out of fold):")
        print(g.round(4).to_string())

    with open(sl.LATE / "sel_scored_cache.pkl", "rb") as f:
        sc = pickle.load(f)
    srows = []
    for r in sc:
        prod = sl.choose(sl.combine(r["ours"].astype(float), r["cand"], r["public"].astype(float)))
        state = sl.last_reading(r["v"]) <= r["vc"]
        srows.append(dict(split=r["split"], stratum=stratum(bool(r["flag"]), r["gap"], prod, state)))
    s = pd.DataFrame(srows)
    counts = pd.crosstab(s.stratum, s.split).reindex(STRATA).fillna(0).astype(int)
    print("\nscored ongoing windows per stratum:")
    print(counts.to_string())
    print("\nforecast of the change in TOTAL score against v64 at train rates (one window = 0.00375):")
    for rule, col in (("A", "dA"), ("B", "dB")):
        for split in ("validation", "private"):
            vals = {}
            for wset in ("official", "replica"):
                d = float((counts[split] * rates[wset][col].fillna(0)).sum()) / 40
                var = float((counts[split] ** 2 * rates[wset]["se" + rule].fillna(0) ** 2).sum()) / 40 ** 2
                vals[wset] = (0.15 * d, 0.15 * np.sqrt(var))
            print(f"  rule {rule}, {split:10s}: official rates {vals['official'][0]:+.4f} ± {vals['official'][1]:.4f}; "
                  f"replica rates {vals['replica'][0]:+.4f} ± {vals['replica'][1]:.4f}")
    print("  (± is the standard error of the train rates only; it leaves out the sampling of 40 scored windows and the month shift)")
    for wset in ("official", "replica"):
        part = tr[tr.set == wset]
        print(f"  spread of the per-window change on train, {wset}: rule A sd {part.dA.std():.4f}, rule B sd {part.dB.std():.4f} "
              f"-> one month of 40 windows: ± {0.15 * part.dA.std() / np.sqrt(40):.4f} (A), ± {0.15 * part.dB.std() / np.sqrt(40):.4f} (B) in total score")


if __name__ == "__main__":
    main()
