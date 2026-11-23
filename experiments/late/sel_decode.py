"""1c. Window IoU of the production forecast and of selection-conditioned decode rules, on train, out of fold.

Reads ~/data/tfb/cache/late/sel_eval_cache.pkl (sel_eval.py --build). For every window:
  production   the expected-IoU set of the combined probability (pipeline_v1.predict_window; t2_public.WEIGHT, OFF_SCALE)
  state        the time-filled persistence state at T: per link, the latest reading stamped <= T is at or below
               0.6 free speed (masked layer at T, else the window history, else earlier masked-layer slots)
  rule A       flagged windows: steps 1-5 = state, step 6 = production; other windows = production
  k=...        flagged windows: steps 1..k = state, the rest production (k = 6 is full persistence)
  variants     of the state estimate, of step 6, and of where rule A is applied:
               "A, small state"       only when the state holds at most 8 queued links
               "A where production contradicts the flag (c <= x)": only when the production forecast, taken as
                    the truth, would have let the origin T-5 pass the cap: c = IoU of the state repeated over
                    T..T+25 with [state at T, production at steps 1-5]; the documented cap is 0.9

Scored as score_task2.score_window: IoU over the eligible target cells, truth = observed speed <= 0.6 free speed
(the released train truth). No rule here is fitted; every rule is evaluated on all its windows. What is
in-sample: the replica windows are chosen by a cap tested on the same observations that make their labels,
the official windows are not; the choice between the variants was made on these windows.

(The cache also holds the time-interpolated state at the horizon. On eligible cells it equals the observed
label, so it cannot serve as a noise-free second truth; it is not used.)

    python experiments/late/sel_decode.py [cache.pkl]
"""
import pickle
import sys

import numpy as np
import pandas as pd

import sel_lib as sl

CACHE_FILE = sl.LATE / "sel_eval_cache.pkl"
SMALL = 8


def state_last(r: dict, upto: int = 0) -> np.ndarray:
    """[L] queue state from the latest finite reading at or before T + upto*5 min (upto <= 0)."""
    v = r["v"] if upto == 0 else r["v"][:upto]
    return sl.last_reading(v) <= r["vc"]


def state_median(r: dict, n: int = 3) -> np.ndarray:
    """[L] queue state from the median of the last n finite readings at or before T (fewer if fewer exist)."""
    v = r["v"]
    out = np.zeros(v.shape[1], bool)
    for l in range(v.shape[1]):
        x = v[:, l][np.isfinite(v[:, l])][-n:]
        out[l] = len(x) > 0 and np.median(x) <= r["vc"][l]
    return out


def production(r: dict) -> np.ndarray:
    return sl.choose(sl.combine(r["ours"].astype(float), r["cand"], r["public"].astype(float)))


def rule_k(r: dict, k: int, state: np.ndarray | None = None, prod: np.ndarray | None = None) -> np.ndarray:
    """Steps 1..k from the persistence state, the rest from production."""
    pred = (production(r) if prod is None else prod).copy()
    s = state_last(r) if state is None else state
    pred[:k] = s[None, :]
    return pred


def rule_redecode(r: dict, p_in: float, p_out: float, k: int = 5) -> np.ndarray:
    """Steps 1..k get fixed probabilities from the state (p_in on it, p_out off it); the expected-IoU set is
    then chosen over the whole window, so step 6 is decided knowing that the first steps are near certain."""
    p = sl.combine(r["ours"].astype(float), r["cand"], r["public"].astype(float))
    s = state_last(r)
    p[:k] = np.where(s[None, :], p_in, p_out)
    return sl.choose(p)


def consistency(state: np.ndarray, pred: np.ndarray) -> float:
    """The cap IoU the origin T-5 would have had if `pred` came true: the state repeated over T..T+25 against
    [state at T, pred at steps 1-5]. A flagged window implies this is above 0.9 for the true future."""
    n = state.sum()
    inter, union = n + (pred[:5] & state[None]).sum(), n + (pred[:5] | state[None]).sum()
    return 1.0 if union == 0 else float(inter / union)


def rule_consistent(r: dict, threshold: float) -> np.ndarray:
    prod, s = production(r), state_last(r)
    return rule_k(r, 5, s, prod) if consistency(s, prod) <= threshold else prod


RULES = {
    "production": lambda r: production(r),
    "A (k=5)": lambda r: rule_k(r, 5),
    "k=1": lambda r: rule_k(r, 1),
    "k=2": lambda r: rule_k(r, 2),
    "k=3": lambda r: rule_k(r, 3),
    "k=4": lambda r: rule_k(r, 4),
    "k=6 (persistence)": lambda r: rule_k(r, 6),
    "A, state from T-5": lambda r: rule_k(r, 5, state_last(r, -1)),
    "A, median-3 state": lambda r: rule_k(r, 5, state_median(r, 3)),
    "A, step 6 empty": lambda r: np.concatenate([rule_k(r, 5)[:5], np.zeros((1, r["v"].shape[1]), bool)]),
    "A, step 6 = state & production": lambda r: np.concatenate([rule_k(r, 5)[:5], (state_last(r) & production(r)[5])[None]]),
    "A, step 6 = state | production": lambda r: np.concatenate([rule_k(r, 5)[:5], (state_last(r) | production(r)[5])[None]]),
    "redecode .97/.01": lambda r: rule_redecode(r, 0.97, 0.01),
    "redecode .95/.02": lambda r: rule_redecode(r, 0.95, 0.02),
    f"A, small state (<= {SMALL} links)": lambda r: rule_k(r, 5) if state_last(r).sum() <= SMALL else production(r),
    "A where production contradicts the flag (c <= 0.90)": lambda r: rule_consistent(r, 0.90),
    "A where production contradicts the flag (c <= 0.85)": lambda r: rule_consistent(r, 0.85),
    "A where production contradicts the flag (c <= 0.80)": lambda r: rule_consistent(r, 0.80),
}


def strata(n_state: int, add: int, rem: int) -> tuple[str, str]:
    size = f"small (<= {SMALL})" if n_state <= SMALL else "mid (9-40)" if n_state <= 40 else "large (> 40)"
    direction = "none" if add == 0 and rem == 0 else "adds" if rem == 0 else "removes" if add == 0 else "both"
    return size, direction


def score_all(recs: list) -> pd.DataFrame:
    rows = []
    for i, r in enumerate(recs):
        prod, s = production(r), state_last(r)
        S = np.broadcast_to(s, prod.shape)
        add, rem = int((S[:5] & ~prod[:5]).sum()), int((prod[:5] & ~S[:5]).sum())
        size, direction = strata(int(s.sum()), add, rem)
        row = dict(i=i, set=r["set"], panel=r["panel"], month=r["month"], kind=r["kind"], flag=bool(r["flag"]),
                   k_back=r["k_back"], n_state=int(s.sum()), n_true=int((r["truth"] & r["elig"]).sum()),
                   cells_added=add, cells_removed=rem, size=size, direction=direction, c_production=consistency(s, prod),
                   # the same regularity on the truth side: the state against the labels of steps 1-5
                   state_vs_truth_15=sl.iou(S[:5] & r["elig"][:5], r["truth"][:5] & r["elig"][:5]))
        for name, fn in RULES.items():
            pred = fn(r) if r["flag"] or name == "production" else prod
            row[name] = sl.window_iou(pred, r["truth"], r["elig"])
        for j, x in enumerate(sl.step_iou(prod, r["truth"], r["elig"])):
            row[f"prod_s{j + 1}"] = x
        for j, x in enumerate(sl.step_iou(S, r["truth"], r["elig"])):
            row[f"state_s{j + 1}"] = x
        rows.append(row)
    return pd.DataFrame(rows)


def report(df: pd.DataFrame, label: str) -> None:
    print(f"\n================ {label}: {len(df)} windows, {int(df.flag.sum())} flagged")
    out = []
    fl, un = df[df.flag], df[~df.flag]
    for name in RULES:
        d = fl[name] - fl["production"]
        out.append(dict(rule=name, all=df[name].mean(), flagged=fl[name].mean(), unflagged=un[name].mean(),
                        d_flagged=d.mean(), se=d.std(ddof=1) / np.sqrt(len(d)), wins=int((d > 1e-9).sum()), losses=int((d < -1e-9).sum()),
                        d_all=(df[name] - df["production"]).mean()))
    print(pd.DataFrame(out).round(4).to_string(index=False))
    print("  (all, flagged, unflagged: mean window IoU; d_flagged: mean change against production on flagged windows with its"
          " standard error; d_all: change of the mean over all windows of the set)")
    print("  per step, flagged windows:  production", [round(float(fl[f"prod_s{j}"].mean()), 3) for j in range(1, 7)],
          " state", [round(float(fl[f"state_s{j}"].mean()), 3) for j in range(1, 7)])
    print("  per step, unflagged windows: production", [round(float(un[f"prod_s{j}"].mean()), 3) for j in range(1, 7)],
          " state", [round(float(un[f"state_s{j}"].mean()), 3) for j in range(1, 7)])
    print("  production by what bound the selector (truth-based kind):",
          {k: f"{g['production'].mean():.4f} (n={len(g)})" for k, g in df.groupby("kind")})
    print("  rule A by kind:", {k: f"{g['A (k=5)'].mean():.4f}" for k, g in df.groupby("kind")})
    print(f"  truth side of the flag: IoU of the state with the labels of steps 1-5 in flagged windows: mean {fl.state_vs_truth_15.mean():.3f}, "
          f"minimum {fl.state_vs_truth_15.min():.3f}, above 0.9 in {(fl.state_vs_truth_15 > 0.9).mean():.0%}; "
          f"unflagged: mean {un.state_vs_truth_15.mean():.3f}, above 0.9 in {(un.state_vs_truth_15 > 0.9).mean():.0%}")
    print(f"  production forecast is (nearly) pure persistence of the state (c > 0.9) in {int((fl.c_production > 0.9).sum())} of {len(fl)} flagged windows")
    print("  rule A by the size of the state and by what it does to production's steps 1-5 (flagged windows):")
    g = fl.assign(dA=fl["A (k=5)"] - fl["production"]).groupby(["size", "direction"]).agg(
        n=("dA", "size"), dA=("dA", "mean"), se=("dA", "sem"), production=("production", "mean"),
        added=("cells_added", "mean"), removed=("cells_removed", "mean"))
    print("  " + g.round(4).to_string().replace("\n", "\n  "))
    ch = fl[(fl.cells_added + fl.cells_removed) > 0]
    print(f"  rule A changes {len(ch)} of {len(fl)} flagged windows; cells 0 -> 1: {int(fl.cells_added.sum())}, 1 -> 0: {int(fl.cells_removed.sum())}")


def main() -> None:
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    path = sys.argv[1] if len(sys.argv) > 1 else CACHE_FILE
    with open(path, "rb") as f:
        recs = pickle.load(f)
    df = score_all(recs)
    df.to_parquet(sl.LATE / "sel_decode_scores.parquet", index=False)
    off, rep = df[df.set == "official"], df[df.set == "replica"]
    print("panel-mean of production (bench convention): official %.4f, replica %.4f" %
          (off.groupby("panel")["production"].mean().mean(), rep.groupby("panel")["production"].mean().mean()))
    report(off, "40 official train windows (June 2030; models fitted without June)")
    if len(rep):
        report(rep, "per-month replica windows, 9 train months (models fitted without the window's month)")
        report(rep[rep.month != "2030-06"], "replica windows without June")
        print("\nrule A minus production per month, replica windows (mean over all windows of the month):")
        print(rep.assign(d=rep["A (k=5)"] - rep["production"]).groupby("month").d.agg(["mean", "count"]).round(4).T.to_string())
    print("\nflagged official windows one by one:")
    cols = ["panel", "kind", "k_back", "n_state", "n_true", "cells_added", "cells_removed", "c_production", "state_vs_truth_15",
            "production", "A (k=5)", "k=6 (persistence)", "A, step 6 empty"]
    print(off[off.flag][cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
