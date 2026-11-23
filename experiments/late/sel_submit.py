"""2. Late-submission candidates: v64's queue.csv with selection conditioning in the ongoing windows.

    late_sel_a   rule A: in flagged ongoing windows steps 1-5 (T+5..T+25) = the time-filled persistence state from
                 data stamped <= T; step 6 and every other window = v64.
    late_sel_b   rule B: every ongoing window is forecast as v64 does it, with one change: our LightGBM is the one
                 retrained on selector-faithful train windows with the selection flags as features (sel_model.py
                 --final, 5 seeds). The public all-link model, the two blend constants and the expected-IoU set
                 are v64's. (B_alone, not a named candidate: the new model without the public model on our
                 candidate cells; the train bench prefers it, as it did for the old model, where the leaderboard
                 disagreed. AC, not a named candidate either: rule A only in the flagged windows where v64's own
                 forecast contradicts the flag; it changes two private windows and no validation window.)

Inputs per scored ongoing window, all stamped <= T:
  - the masked layer of the window's split at slots <= T (tfb.dense arrays), whose slots T-60..T-5 are the
    released window history; the eligibility flags of those slots;
  - the origins of the earlier windows of the same panel and split (window_index.csv), for the 6 h spacing;
  - network files and models fitted on the train split;
  - rule A: v64's own prediction for the window; rule B: tfb.pipeline_v1.window_features and
    tfb.t2_public.window_input, the production inputs.
`decode_inputs` (rule A) and `b_predict` (rule B) are the single code paths; sel_causality.py reruns them with
every value after T blanked and on a release with every row stamped after T removed.

    python experiments/late/sel_submit.py --rule A --out late_sel_a
    python experiments/late/sel_submit.py --rule B --out late_sel_b
"""
import argparse
import json

import lightgbm as lgb
import numpy as np
import pandas as pd

import sel_lib as sl
import sel_model as sm
from tfb import pipeline_v1 as pv1
from tfb import t2_public
from tfb.dense import tod_index
from tfb.paths import RELEASE, SUBMISSIONS
from tfb.t2_onset import T2_PANELS
from tfb.task1_io import days, meta

BACK = 36
BASE = "v64"
B_SEEDS = 5


def window_table(index: pd.DataFrame, day_list: list) -> pd.DataFrame:
    """Ongoing windows of one panel and split with day index, slot and the gap (slots) to the previous origin."""
    w = index.copy()
    w["d"] = [day_list.index(x) for x in w.date]
    w["T"] = tod_index(w.forecast_origin)
    w["t"] = w.d * sl.S + w["T"]
    w = w.sort_values("t").reset_index(drop=True)
    w["gap"] = w.t.diff()
    return w[w.condition == "queue_ongoing"]


def decode_inputs(speed: np.ndarray, elig: np.ndarray, vc: np.ndarray, d: int, T: int, gap: float) -> dict:
    """Everything the rules read about the selection of one window, from slots <= T only: the flag and the persistence state."""
    v = sl.visible(speed, d, T, None, BACK)
    e = sl.visible(elig.astype(np.float32), d, T, None, BACK) > 0.5
    f = sl.flag_window(v, e, vc, gap, lag=1)
    f["state"] = sl.last_reading(v) <= vc
    return f


def rule_a(inp: dict, prod: np.ndarray) -> np.ndarray:
    pred = prod.copy()
    if inp["flag"]:
        pred[:5] = inp["state"][None, :]
    return pred


AC_THRESHOLD = 0.85


def rule_ac(inp: dict, prod: np.ndarray) -> np.ndarray:
    """Rule A only where v64's own forecast contradicts the flag: taken as the truth it would have let the origin
    T-5 pass the cap (sel_decode.consistency <= AC_THRESHOLD; the documented cap is 0.9, 0.85 leaves a margin
    for the noise of the state estimate)."""
    n = inp["state"].sum()
    inter, union = n + (prod[:5] & inp["state"][None]).sum(), n + (prod[:5] | inp["state"][None]).sum()
    c = 1.0 if union == 0 else inter / union
    return rule_a(inp, prod) if inp["flag"] and c <= AC_THRESHOLD else prod


def b_models() -> tuple[list, lgb.Booster]:
    new = [lgb.Booster(model_file=str(sm.model_path(None, s))) for s in range(B_SEEDS)]
    public = lgb.Booster(model_file=str(SUBMISSIONS / BASE / "t2_public.txt"))
    return new, public


def b_predict(ctx: dict, speed: np.ndarray, elig: np.ndarray, w, new: list, public: lgb.Booster, alone: bool = False) -> tuple[np.ndarray, dict]:
    """([6, L] prediction, selection inputs) of rule B for one ongoing window: pipeline_v1.predict_window with the
    selection-conditioned boosters in place of the production ones."""
    panel = ctx["panel"]
    T, x = pv1.window_features(ctx, w)
    inp = decode_inputs(speed, elig, sl.vcut(panel), int(w.d), T, w.gap)
    L = ctx["L"]
    p_new, cand = np.zeros((sl.Z, L)), np.zeros((sl.Z, L), bool)
    if len(x):
        x["panel_code"] = sl.CODES[panel]
        x["sel_flag"] = int(inp["flag"])
        x["sel_sb"] = int(w.gap == sl.SPACING)
        x["sel_flag_h"] = x.sel_flag * x.h
        hh, ll = x.h.to_numpy() - 1, x.link.to_numpy()
        p_new[hh, ll] = np.mean([b.predict(x[b.feature_name()], num_threads=sl.THREADS) for b in new], axis=0)
        cand[hh, ll] = True
    pub = t2_public.window_probs(public, ctx, w, T)
    p = sl.combine(p_new, cand, pub, weight=0.0 if alone else None)
    return sl.choose(p), inp


def predictions(rule: str, base: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """{window_id: [6, L] prediction} for the 80 scored ongoing windows, and a per-window summary."""
    by_window = base.groupby("window_id", sort=False).indices
    preds, summary = {}, []
    models = b_models() if rule.startswith("B") else None
    for split in ("validation", "private"):
        for panel in T2_PANELS:
            a = pv1.load(panel, split, "speed", "eligible")
            day_list = days(panel, split)["days"]
            vc = sl.vcut(panel)
            pos = {l: i for i, l in enumerate(meta(panel)["links"])}
            L = len(pos)
            ctx = pv1.panel_context(panel, split) if models else None
            index = pd.read_csv(RELEASE / "task2" / panel / split / "window_index.csv")
            for w in window_table(index, day_list).itertuples():
                rows = by_window[w.window_id]
                ts, links = base.timestamp.to_numpy()[rows], base.link_id.to_numpy()[rows]
                assert all(x[:10] == w.date for x in ts), w.window_id
                step = tod_index(pd.Series(ts)) - w.T - 1
                li = np.array([pos[x] for x in links])
                assert step.min() == 0 and step.max() == sl.Z - 1 and len(rows) == sl.Z * L, w.window_id
                prod = np.zeros((sl.Z, L), bool)
                prod[step, li] = base.queue_pred.to_numpy()[rows] == 1
                if models:
                    pred, inp = b_predict(ctx, a["speed"], a["eligible"], w, *models, alone=rule == "B_alone")
                else:
                    inp = decode_inputs(a["speed"], a["eligible"], vc, int(w.d), int(w.T), w.gap)
                    pred = rule_ac(inp, prod) if rule == "AC" else rule_a(inp, prod)
                preds[w.window_id] = (rows, pred[step, li].astype(int))
                summary.append(dict(split=split, panel=panel, window_id=w.window_id, flag=inp["flag"], spacing_bound=bool(w.gap == sl.SPACING),
                                    n_state=int(inp["state"].sum()), cells_changed=int((pred != prod).sum()),
                                    cells_0_to_1=int((pred & ~prod).sum()), cells_1_to_0=int((~pred & prod).sum()),
                                    step6_changed=int((pred[5] != prod[5]).sum()), queued_v64=int(prod.sum()), queued_new=int(pred.sum()),
                                    queued_v64_step6=int(prod[5].sum()), queued_new_step6=int(pred[5].sum())))
            print(f"  {panel}/{split}: done", flush=True)
    return preds, pd.DataFrame(summary)


def build(rule: str, out: str, path: str | None = None) -> None:
    base = pd.read_csv(SUBMISSIONS / BASE / "queue.csv", dtype={"link_id": str})
    preds, s = predictions(rule, base)
    new = base.copy()
    col = new.queue_pred.to_numpy().copy()
    for rows, values in preds.values():
        col[rows] = values
    new["queue_pred"] = col
    if path:                     # an extra file outside the submission folders
        target = sl.LATE / path
    else:
        work = SUBMISSIONS / out
        work.mkdir(parents=True, exist_ok=True)
        target = work / "queue.csv"
    new.to_csv(target, index=False)

    # checks by key join against v64
    again = pd.read_csv(target, dtype={"link_id": str})
    keys = ["window_id", "timestamp", "link_id"]
    m = base.merge(again, on=keys, how="outer", suffixes=("_v64", "_new"), validate="one_to_one", indicator=True)
    assert (m._merge == "both").all() and len(m) == len(base) == len(again), "row sets differ"
    assert list(again.columns) == list(base.columns) and again[keys].equals(base[keys]), "row order differs"
    assert set(again.queue_pred.unique()) <= {0, 1}
    cond = pd.concat([pd.read_csv(RELEASE / "task2" / p / sp / "window_index.csv")[["window_id", "condition", "split"]]
                      for sp in ("validation", "private") for p in T2_PANELS])
    m = m.merge(cond, on="window_id", validate="many_to_one")
    m["changed"] = m.queue_pred_v64 != m.queue_pred_new
    onset_changed = int(m[m.condition == "queue_onset"].changed.sum())
    assert onset_changed == 0
    if rule in ("A", "AC"):
        assert m[m.changed].window_id.isin(set(s[s.flag].window_id)).all(), "a row changed outside the flagged windows"
    sp = ("validation", "private")
    report = {"rule": rule, "base": BASE, "file": str(target),
              "ongoing_windows": {x: int((s.split == x).sum()) for x in sp},
              "flagged_windows": {x: int(s[s.split == x].flag.sum()) for x in sp},
              "spacing_bound_windows": {x: int(s[s.split == x].spacing_bound.sum()) for x in sp},
              "windows_changed": {x: int((s[s.split == x].cells_changed > 0).sum()) for x in sp},
              "flagged_windows_changed": {x: int((s[(s.split == x) & s.flag].cells_changed > 0).sum()) for x in sp},
              "rows_changed": {x: int(m[m.split == x].changed.sum()) for x in sp},
              "rows_changed_step6": {x: int(s[s.split == x].step6_changed.sum()) for x in sp},
              "rows_changed_total": int(m.changed.sum()), "rows_total": int(len(m)),
              "rows_changed_0_to_1": int(((m.queue_pred_v64 == 0) & (m.queue_pred_new == 1)).sum()),
              "rows_changed_1_to_0": int(((m.queue_pred_v64 == 1) & (m.queue_pred_new == 0)).sum()),
              "onset_rows_changed": onset_changed}
    if path:
        s.to_csv(sl.LATE / (path.replace(".csv", "") + "_windows.csv"), index=False)
    else:
        (target.parent / "sel_manifest.json").write_text(json.dumps(report, indent=2))
        s.to_csv(target.parent / "sel_windows.csv", index=False)
    print(json.dumps(report, indent=2))
    print("wrote", target)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rule", default="A", choices=["A", "AC", "B", "B_alone"])
    ap.add_argument("--out", default=None, help="submission folder (late_sel_a or late_sel_b)")
    ap.add_argument("--file", default=None, help="instead of a submission folder: a file name under ~/data/tfb/cache/late")
    args = ap.parse_args()
    assert (args.out is None) != (args.file is None), "give --out or --file"
    build(args.rule, args.out, args.file)


if __name__ == "__main__":
    main()
