"""1c (iv). A training frame of selector-faithful ongoing windows, with the selection flags as features.

The production frame (cache/t2_ongoing_train.parquet) samples every sixth valid origin of a day. The scored
windows are not such a sample: the selector takes the first origin at which the cap passes, so half of them sit
at the end of a stable spell. This frame holds origins as the corrected replica (sel_replica.py) selects them:

  role cap      every origin that passes the cap while the origin five minutes earlier qualified in everything
                but the cap ("first passage"): what a scan arrives at when the spacing does not bind;
  role new      every qualifying origin whose predecessor had no established queue in its history;
  role spacing  origins the replica selects exactly 6 h after the previous window, from scans started on
                every day of the train split with the 5 + 5 count limit.

Rows and production features come from tfb.t2_ongoing.TrainPanel.window, unchanged. Added columns, all computed
as at test time from data stamped <= T (sel_lib.flag_window on the emulated visible block):
  sel_flag    the cap-bound flag
  sel_sb      the window is exactly 6 h after the previous one (spacing-bound; from window_index at test time)
  sel_kback   consecutive earlier origins that qualified (0..12)
  sel_flag_h  sel_flag * step

    python experiments/late/sel_frame.py            # -> ~/data/tfb/cache/late/sel_faithful_train.parquet
"""
import time

import numpy as np
import pandas as pd

import sel_lib as sl
from tfb import t2_ongoing as og
from tfb.t2_onset import T2_PANELS

BACK = 36
OUT = sl.LATE / "sel_faithful_train.parquet"
CHOSEN = dict(state="interp", ref=0)


def sel_features(a: dict, vc: np.ndarray, d: int, T: int, gap: float) -> dict:
    """The selection features of a train window, from the view a forecast has at T (history complete, rest masked)."""
    v = sl.visible(a["speed"], d, T, a["true_speed"][d, T - sl.H:T], BACK)
    e = sl.visible(a["eligible"].astype(np.float32), d, T, None, BACK) > 0.5
    return sel_features_visible(v, e, vc, gap)


def sel_features_visible(v: np.ndarray, e: np.ndarray, vc: np.ndarray, gap: float) -> dict:
    f = sl.flag_window(v, e, vc, gap, lag=1)
    k = 0
    while k < 12 and sl.flag_window(v, e, vc, gap, lag=k + 1)["flag"]:
        k += 1
    return {"sel_flag": int(f["flag"]), "sel_sb": int(gap == sl.SPACING), "sel_kback": k}


def origins(tt: sl.TrainTruth) -> pd.DataFrame:
    """(d, T, role, gap) of the faithful training origins of one panel."""
    base, cap = tt.base(), tt.cap(**CHOSEN)
    ok = base & (cap <= sl.CAP)
    prev_base = np.zeros_like(base)
    prev_base[:, 1:] = base[:, :-1]
    prev_cap = np.full(cap.shape, np.nan)
    prev_cap[:, 1:] = cap[:, :-1]
    rows = [(d, T, "cap", np.nan) for d, T in zip(*np.nonzero(ok & prev_base & (prev_cap > sl.CAP)))]
    rows += [(d, T, "new", np.nan) for d, T in zip(*np.nonzero(ok & ~prev_base))]
    ok_on = tt.onset_ok()
    seen = set()
    for start in range(base.shape[0]):
        prev = None
        for d, T, c in sl.scan(ok_on[start:], ok[start:], per_condition=sl.PER_CONDITION):
            t = (d + start) * sl.S + T
            gap = np.nan if prev is None else t - prev
            prev = t
            if c == "queue_ongoing" and gap == sl.SPACING:
                seen.add((d + start, T))
    rows += [(d, T, "spacing", float(sl.SPACING)) for d, T in sorted(seen)]
    return pd.DataFrame(rows, columns=["d", "T", "role", "gap"])


def window_rows(tp: og.TrainPanel, d: int, T: int, gap: float, role: str) -> pd.DataFrame:
    r = tp.window(d, T)
    if r.empty:
        return r
    f = sel_features(tp.a, sl.vcut(tp.panel), d, T, gap)
    for k, x in f.items():
        r[k] = x
    r["sel_flag_h"] = r.sel_flag * r.h
    r["key"] = f"{tp.panel}|{d}|{T}|{role}"
    r["role"] = role
    return r


def main() -> None:
    frames = []
    for panel in T2_PANELS:
        t0 = time.time()
        tt = sl.TrainTruth(panel)
        o = origins(tt)
        del tt
        tp = og.TrainPanel(panel)
        for r in o.itertuples():
            w = window_rows(tp, int(r.d), int(r.T), r.gap, r.role)
            if len(w):
                frames.append(w)
        print(f"  {panel}: {o.role.value_counts().to_dict()} windows in {time.time() - t0:.0f} s", flush=True)
    df = pd.concat(frames, ignore_index=True)
    df.to_parquet(OUT, index=False)
    w = df.drop_duplicates("key")
    print(f"wrote {OUT}: {len(df):,} rows, {len(w):,} windows")
    print("flag by role:", pd.crosstab(w.role, w.sel_flag).to_dict())


if __name__ == "__main__":
    main()
