"""1a. The corrected replica of the Task 2 window selector, against the 40 + 40 official train windows.

The replica keeps every documented condition of tfb.t2_rules.selector_windows (forward scan, 6 h spacing across
both conditions, 5 per condition, history coverage >= 0.7, a queue in the horizon, an established queue in the
history) and changes only the state on which the persistence cap (IoU <= 0.9) is tested:

  state      logged     a missing reading is "no queue" (tfb.t2_rules, LOG 2026-10-06: 14 of 40)
             ffill      missing readings carry the last value forward
             interp     missing readings are interpolated in time between the neighbouring readings
             *_el       the same from eligible readings only
  reference  T-5        the state at the last history step is repeated (the baseline script)
             T          the state at the origin is repeated (README: "repeats the queue state at T")
  cells      all        every link counts
             eligible   only eligible horizon cells count (as score_task2.score_window does)

Prints the match table, the official ongoing windows one by one with what bound the selector at T-5, the
windows the chosen replica misses, and writes the window lists used by the other sel_ scripts:

    ~/data/tfb/cache/late/sel_windows.parquet     official + per-month replica ongoing windows on train

    python experiments/late/sel_replica.py
"""
import itertools

import numpy as np
import pandas as pd

import sel_lib as sl
from tfb.t2_onset import T2_PANELS

CHOSEN = dict(state="interp", ref=0, eligible_only=False)   # the replica used from here on
N_DAYS = 30                                                 # the official train windows end on day 20


def matches(tts: dict, **kw) -> dict:
    res = dict(onset=0, ongoing=0, ongoing_5min=0)
    for panel, tt in tts.items():
        off = sl.official(panel)
        offs = {(r.d, r.T, r.condition) for r in off.itertuples()}
        ch = sl.replica(tt, **kw)
        res["onset"] += sum(1 for x in ch if x in offs and x[2] == "queue_onset")
        res["ongoing"] += sum(1 for x in ch if x in offs and x[2] == "queue_ongoing")
        res["ongoing_5min"] += sum(1 for x in ch if x[2] == "queue_ongoing"
                                   and any(o[2] == x[2] and o[0] == x[0] and abs(o[1] - x[1]) <= 1 for o in offs))
    return res


def binding(tt: sl.TrainTruth, off: pd.DataFrame, state: str, ref: int, eligible_only: bool = False) -> pd.DataFrame:
    """What kept the selector from taking the origin five minutes before each official ongoing window."""
    cap, base = tt.cap(state, ref, eligible_only), tt.base()
    rows = []
    for r in off[off.condition == "queue_ongoing"].itertuples():
        if r.gap == sl.SPACING:
            kind = "spacing"            # T-5 is closer than 6 h to the previous window
        elif not base[r.d, r.T - 1]:
            kind = "new"                # T-5 has no established queue in its history (or coverage, or horizon)
        elif cap[r.d, r.T - 1] > sl.CAP:
            kind = "cap"                # T-5 met everything but the cap
        else:
            kind = "unexplained"        # the replica would have taken T-5
        rows.append(dict(panel=tt.panel, window_id=r.window_id, d=r.d, T=r.T, gap_min=5 * r.gap, kind=kind,
                         cap_m10=cap[r.d, r.T - 2], cap_m5=cap[r.d, r.T - 1], cap_0=cap[r.d, r.T], cap_p5=cap[r.d, r.T + 1],
                         queued_at_T=int(tt.state(state)[r.d, r.T].sum())))
    return pd.DataFrame(rows)


def main() -> None:
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    tts = {p: sl.TrainTruth(p, N_DAYS) for p in T2_PANELS}

    print("Matches with the official train windows (40 onset, 40 ongoing):")
    rows = []
    for state, ref, el in itertools.product(("logged", "ffill", "ffill_el", "interp", "interp_el"), (-1, 0), (False, True)):
        rows.append(dict(cap_state=state, reference="T" if ref == 0 else "T-5", cells="eligible" if el else "all",
                         **matches(tts, state=state, ref=ref, eligible_only=el)))
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))

    print("\nOfficial ongoing train windows under the chosen replica", CHOSEN)
    b = pd.concat([binding(tt, sl.official(p), CHOSEN["state"], CHOSEN["ref"]) for p, tt in tts.items()], ignore_index=True)
    reps = {p: sl.replica(tt, **CHOSEN) for p, tt in tts.items()}
    b["replica_dT"] = [next((x[1] - r.T for x in reps[r.panel] if x[2] == "queue_ongoing" and x[0] == r.d and abs(x[1] - r.T) <= 12), np.nan)
                       for r in b.itertuples()]
    print(b.round(3).to_string(index=False))
    print("\nwhat bound the selector at T-5:", b.kind.value_counts().to_dict())
    print("mean cap IoU by kind (origin T-10, T-5, T, T+5):")
    print(b.groupby("kind")[["cap_m10", "cap_m5", "cap_0", "cap_p5"]].mean().round(3).to_string())
    miss = b[b.replica_dT != 0]
    print(f"\nwindows the chosen replica misses: {len(miss)}")
    print(miss.round(3).to_string(index=False))

    # the task description's definition (ffill, reference T-5): the same windows?
    b2 = pd.concat([binding(tt, sl.official(p), "ffill", -1) for p, tt in tts.items()], ignore_index=True)
    print("\ncap fails at T-5, per definition:  interp/T:", int((b.kind == "cap").sum()), " ffill/T-5:", int((b2.kind == "cap").sum()),
          " both:", int(((b.kind == "cap") & (b2.kind == "cap")).sum()))
    print("cap IoU at T-5 > 0.9 regardless of the other conditions:  interp/T:", int((b.cap_m5 > sl.CAP).sum()),
          " ffill/T-5:", int((b2.cap_m5 > sl.CAP).sum()))

    # window lists for the other scripts: the whole train split, month by month
    out = []
    for p in T2_PANELS:
        tt = sl.TrainTruth(p)
        off = sl.official(p)
        bb = binding(tt, off, CHOSEN["state"], CHOSEN["ref"])
        bf = binding(tt, off, "ffill", -1)
        for r, rf in zip(bb.itertuples(), bf.itertuples()):
            out.append(dict(set="official", panel=p, d=r.d, T=r.T, month=tt.months[r.d], window_id=r.window_id, gap=r.gap_min / 5,
                            kind=r.kind, cap_m5=r.cap_m5, cap_0=r.cap_0, cap_m5_ffill=rf.cap_m5, kind_ffill=rf.kind))
        cap, capf, base = tt.cap(**CHOSEN), tt.cap("ffill", -1), tt.base()
        for mo in np.unique(tt.months):
            idx = np.flatnonzero(tt.months == mo)
            ch = sl.replica(tt, day_idx=idx, **CHOSEN)
            prev = None
            for d, T, c in ch:
                t = d * sl.S + T
                gap = np.nan if prev is None else t - prev
                prev = t
                if c != "queue_ongoing":
                    continue
                kind = ("spacing" if gap == sl.SPACING else "new" if not base[d, T - 1] else
                        "cap" if cap[d, T - 1] > sl.CAP else "unexplained")
                kf = ("spacing" if gap == sl.SPACING else "new" if not base[d, T - 1] else
                      "cap" if capf[d, T - 1] > sl.CAP else "unexplained")
                out.append(dict(set="replica", panel=p, d=d, T=T, month=mo, window_id=None, gap=gap, kind=kind,
                                cap_m5=cap[d, T - 1], cap_0=cap[d, T], cap_m5_ffill=capf[d, T - 1], kind_ffill=kf))
    w = pd.DataFrame(out)
    sl.LATE.mkdir(parents=True, exist_ok=True)
    w.to_parquet(sl.LATE / "sel_windows.parquet", index=False)
    print("\nwindow lists written:", sl.LATE / "sel_windows.parquet")
    print(w.groupby(["set", "kind"]).size().unstack(fill_value=0).to_string())
    rep = w[w.set == "replica"]
    print("replica ongoing windows per month:", rep.groupby("month").size().to_dict())
    june = rep[rep.month == "2030-06"]
    offw = w[w.set == "official"]
    both = june.merge(offw, on=["panel", "d", "T"])
    print(f"June replica windows that are official windows: {len(both)} of {len(june)}")


if __name__ == "__main__":
    main()
