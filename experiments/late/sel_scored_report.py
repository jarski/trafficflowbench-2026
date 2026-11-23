"""Descriptive checks on the 80 scored ongoing windows, all from data stamped <= T, set against the train windows.

  1. Do flagged windows look the same on the scored months as on train? Queue stability before the origin
     (IoU of the time-filled state one step earlier with the readings of the next slot), by window type.
  2. How often is the production forecast of a flagged window (nearly) pure persistence, and how often does it
     contradict the flag (c <= 0.9, sel_decode.consistency)?
  3. What rule A and candidate B change, window by window (from the candidates' sel_windows.csv), and the
     consistency of v64, A and B with the flag.
  4. Rule A's forecast from the size strata of sel_decode.py: train replica rates x the scored windows' counts.
  5. The D7_I210_E boundary queue: what follows a first-passage origin on train, by hour; what B does there.
  6. Cells changed per window by the model swap: train (1 seed against 1 seed) and scored (B against v64).
  7. Does the D7_I210_E boundary queue end the same way in March as on train? From the masked layer of March
     days 6-31, which hold no scored D7_I210_E window. Every one of those days is stamped before every April
     origin, so this is admissible evidence for the private windows (thread 748789); it is not used by any
     candidate, and it says nothing admissible about the March windows themselves.

    python experiments/late/sel_scored_report.py
"""
import pickle

import numpy as np
import pandas as pd

import sel_decode as sd
import sel_lib as sl
import sel_model as sm
from tfb.dense import tod_index
from tfb.paths import RELEASE, SUBMISSIONS
from tfb.t2_onset import T2_PANELS
from tfb.task1_io import days, load, meta


def wtype(flag: bool, gap: float) -> str:
    return "flagged" if flag else "spacing" if gap == sl.SPACING else "new"


def stability(r: dict) -> tuple[float, float]:
    v, vc = r["v"], r["vc"]
    s2, s1 = sl.last_reading(v[:-2]) <= vc, sl.last_reading(v[:-1]) <= vc
    o1, o0 = np.isfinite(v[-2]), np.isfinite(v[-1])
    return sl.iou(s2 & o1, (v[-2] <= vc) & o1), sl.iou(s1 & o0, (v[-1] <= vc) & o0)


def grid(q: pd.DataFrame, r: dict) -> np.ndarray:
    rows = q[q.window_id == r["window_id"]]
    pos = {l: i for i, l in enumerate(meta(r["panel"])["links"])}
    g = np.zeros((sl.Z, len(pos)), bool)
    g[tod_index(rows.timestamp) - r["T"] - 1, rows.link_id.map(pos).to_numpy()] = rows.queue_pred.to_numpy() == 1
    return g


def main() -> None:
    pd.set_option("display.width", 260)
    pd.set_option("display.max_rows", 300)
    with open(sl.LATE / "sel_eval_cache.pkl", "rb") as f:
        tr = pickle.load(f)
    with open(sl.LATE / "sel_scored_cache.pkl", "rb") as f:
        sc = pickle.load(f)
    with open(sm.PRED, "rb") as f:
        preds = pickle.load(f)

    # 1, 2
    rows = []
    for r in tr + sc:
        prod, s = sd.production(r), sd.state_last(r)
        a, b = stability(r)
        rows.append(dict(split=("train " + r["set"]) if "set" in r else r["split"], type=wtype(bool(r["flag"]), r["gap"]),
                         stab_m10_m5=a, stab_m5_0=b, n_state=int(s.sum()), c=sd.consistency(s, prod)))
    d = pd.DataFrame(rows)
    print("1. queue stability before the origin and size of the state, by window type (data <= T only):")
    print(d.groupby(["split", "type"]).agg(windows=("c", "size"), stab_m10_m5=("stab_m10_m5", "mean"), stab_m5_0=("stab_m5_0", "mean"),
                                           median_state=("n_state", "median")).round(3).to_string())
    f = d[d.type == "flagged"]
    print("\n2. flagged windows: production against the persistence of the state at steps 1-5")
    print(f.groupby("split").agg(windows=("c", "size"), mean_c=("c", "mean"), pure_persistence=("c", lambda x: int((x > 0.9).sum())),
                                 contradicts_flag_090=("c", lambda x: int((x <= 0.9).sum())), contradicts_flag_085=("c", lambda x: int((x <= 0.85).sum()))).round(3).to_string())

    # 3
    q0 = pd.read_csv(SUBMISSIONS / "v64" / "queue.csv", dtype={"link_id": str})
    qa = pd.read_csv(SUBMISSIONS / "late_sel_a" / "queue.csv", dtype={"link_id": str})
    qb = pd.read_csv(SUBMISSIONS / "late_sel_b" / "queue.csv", dtype={"link_id": str})
    idx = pd.concat([pd.read_csv(RELEASE / "task2" / p / sp / "window_index.csv") for sp in ("validation", "private") for p in T2_PANELS])
    origin = dict(zip(idx.window_id, idx.forecast_origin.str[11:16]))
    rows = []
    for r in sc:
        s = sd.state_last(r)
        g0, ga, gb = grid(q0, r), grid(qa, r), grid(qb, r)
        size, direction = sd.strata(int(s.sum()), int((np.broadcast_to(s, g0.shape)[:5] & ~g0[:5]).sum()), int((g0[:5] & ~np.broadcast_to(s, g0.shape)[:5]).sum()))
        rows.append(dict(split=r["split"], window=r["window_id"][3:], origin=origin[r["window_id"]], type=wtype(bool(r["flag"]), r["gap"]),
                         state=int(s.sum()), v64_cells=int(g0.sum()), v64_step6=int(g0[5].sum()),
                         A_add=int((ga & ~g0).sum()), A_rem=int((g0 & ~ga).sum()), B_add=int((gb & ~g0).sum()), B_rem=int((g0 & ~gb).sum()),
                         B_step6=int(gb[5].sum()), c_v64=sd.consistency(s, g0), c_B=sd.consistency(s, gb), size=size, direction=direction,
                         flag=bool(r["flag"])))
    w = pd.DataFrame(rows)
    print("\n3. scored ongoing windows that A or B changes (cells added / removed against v64; c = consistency with the flag):")
    ch = w[(w.A_add + w.A_rem + w.B_add + w.B_rem) > 0]
    print(ch.drop(columns=["size", "direction", "flag"]).round(3).to_string(index=False))
    print("\n   totals per split and type:")
    print(w.groupby(["split", "type"]).agg(windows=("state", "size"), A_changed=("A_add", lambda x: 0), A_add=("A_add", "sum"), A_rem=("A_rem", "sum"),
                                            B_add=("B_add", "sum"), B_rem=("B_rem", "sum")).drop(columns="A_changed").to_string())
    fw = w[w.flag]
    print("   flagged windows contradicting the flag (c <= 0.9): v64", fw[fw.c_v64 <= 0.9].groupby("split").size().to_dict(),
          " B", fw[fw.c_B <= 0.9].groupby("split").size().to_dict())

    # 4
    scores = pd.read_parquet(sl.LATE / "sel_decode_scores.parquet")
    rep = scores[(scores.set == "replica") & scores.flag]
    rate = rep.assign(dA=rep["A (k=5)"] - rep["production"]).groupby(["size", "direction"]).dA.agg(["mean", "sem", "count"]).rename(columns={"count": "train_n"})
    counts = fw.groupby(["split", "size", "direction"]).size().rename("windows").reset_index()
    counts = counts.merge(rate.reset_index(), on=["size", "direction"], how="left")
    print("\n4. rule A: scored flagged windows per size stratum with the train replica rate of each stratum")
    print(counts.round(4).to_string(index=False))
    for split, g in counts.groupby("split"):
        tot = float((g.windows * g["mean"].fillna(0)).sum())
        se = float(np.sqrt((g.windows ** 2 * g["sem"].fillna(0) ** 2).sum()))
        print(f"   {split}: expected change {tot:+.3f} ± {se:.3f} window-IoU in total -> {0.15 * tot / 40:+.4f} ± {0.15 * se / 40:.4f} of the total score")

    # 5
    tt = sl.TrainTruth("D7_I210_E")
    qi = tt.state("interp")
    base, cap = tt.base(), tt.cap("interp", 0)
    ok = base & (cap <= sl.CAP)
    first = ok.copy()
    first[:, 1:] &= base[:, :-1] & (cap[:, :-1] > sl.CAP)
    first[:, 0] = False
    out = []
    for dd, T in zip(*np.nonzero(first)):
        s0, s5, s6 = qi[dd, T], qi[dd, T + 5], qi[dd, T + 6]
        if s0.sum() != 5 or not s0[:5].all():
            continue
        out.append(dict(hour=T // 12, half_hour_phase=int(T % 6), boundary_queue_gone=not s6[:5].any(), others_at_step6=int(s6[5:].sum()),
                        boundary_kept_through_step5=bool(s5[:5].sum() >= 4)))
    o = pd.DataFrame(out)
    print(f"\n5. D7_I210_E, train: {len(o)} first-passage origins whose state is exactly the 5-link queue at the upstream end")
    print(o.groupby("hour").agg(n=("boundary_queue_gone", "size"), gone_at_step6=("boundary_queue_gone", "mean"),
                                kept_through_step5=("boundary_kept_through_step5", "mean"), other_links_at_step6=("others_at_step6", "mean")).round(2).to_string())
    print("   origin slot modulo 30 minutes (0 = :00/:30, 1 = :05/:35 ...):", o.half_hour_phase.value_counts().sort_index().to_dict())
    print("   scored D7_I210_E windows:")
    print(w[w.window.str.startswith("D7_I210_E")][["split", "window", "origin", "type", "state", "v64_cells", "v64_step6", "B_add", "B_rem", "B_step6", "c_v64"]].round(3).to_string(index=False))

    # 6
    rows = []
    for r, new in zip(tr, preds):
        ours, cand, pub = r["ours"].astype(float), r["cand"], r["public"].astype(float)
        a, b = sl.choose(sl.combine(ours, cand, pub)), sl.choose(sl.combine(new.astype(float), cand, pub))
        rows.append(dict(set="train " + r["set"], cells=int((a != b).sum()), step6=int((a[5] != b[5]).sum()), changed=bool((a != b).any())))
    t = pd.DataFrame(rows)
    print("\n6. cells changed per window by the model swap:")
    print(t.groupby("set").agg(windows=("cells", "size"), share_changed=("changed", "mean"), cells=("cells", "mean"), step6=("step6", "mean")).round(2).to_string())
    print(f"   scored, B against v64: {len(w)} windows, share changed {((w.B_add + w.B_rem) > 0).mean():.2f}, cells {(w.B_add + w.B_rem).mean():.2f} per window")

    # 7
    vc5 = sl.vcut("D7_I210_E")[:5]
    print("\n7. D7_I210_E: the evening end of the 5-link queue at the upstream end, from the masked layer")
    for split, first_day in (("train", 0), ("validation", 5)):
        sp = load("D7_I210_E", split, "speed")["speed"][:, :, :5]
        dl = days("D7_I210_E", split)["days"]
        q, obs = sp <= vc5, np.isfinite(sp)
        ends, at_once, phase = [], [], []
        for dd in range(first_day, len(dl)):
            on = np.flatnonzero((q[dd].sum(axis=1) >= 2) & (np.arange(sl.S) >= 144))      # >= 2 of the 5 links seen queued, after noon
            if len(on) == 0:
                continue
            last = int(on.max())
            nxt = next((t for t in range(last + 1, sl.S) if obs[dd, t].sum() >= 2), None)
            if nxt is None:
                continue
            ends.append(last)
            at_once.append(int(q[dd, nxt].sum()) == 0)
            phase.append(last % 6)
        e = np.array(ends)
        print(f"   {split}, days {dl[first_day]}..{dl[-1]}: {len(e)} days with the queue after noon; last seen queued between "
              f"{e.min() // 12:02d}:{e.min() % 12 * 5:02d} and {e.max() // 12:02d}:{e.max() % 12 * 5:02d} (median {int(np.median(e)) // 12:02d}:{int(np.median(e)) % 12 * 5:02d}); "
              f"all observed links free at the next observed slot on {int(np.sum(at_once))} of {len(e)} days; "
              f"last queued slot on the half hour (:00 or :30) on {int(np.sum(np.array(phase) == 0))} days")


if __name__ == "__main__":
    main()
