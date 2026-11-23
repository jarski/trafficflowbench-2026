"""Causality tests of the two candidates for all 80 scored ongoing windows.

Rule A reads, besides v64's own prediction, the flag and the persistence state (sel_submit.decode_inputs).
Rule B recomputes the whole forecast (sel_submit.b_predict: production features, selection features, models).

  arrays   every value after T blanked in the dense arrays of the window's split (speed, flow, occupancy, ramps
           NaN, eligibility off; the rest of the day and all later days), with tfb.t2_bench.blanked_after:
           the result must be identical. Negative control: the same computation for an origin three slots later
           (it reads eligibility flags and ramp flows stamped after T, which the release does publish) must be
           caught by the same test.
  --files  the organizers' award check (thread 749601), as tfb.t2_bench.causality_files does it: for each window,
           every row stamped after its origin is removed from every file of the panel (states, ramps, window
           histories, window_index, templates), the dense arrays are rebuilt from those files, and the result
           recomputed from them, with the gap taken from the cut window_index: it must be identical.

For rule A the other ingredient, v64's own prediction, passed `tfb.t2_bench causality --files` at lock-in
(LOG 2026-10-31).

    python experiments/late/sel_causality.py [--rule B]
    python experiments/late/sel_causality.py --files [--rule B]
"""
import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import sel_lib as sl
import sel_submit as ss
from tfb import dense, t2_bench
from tfb import pipeline_v1 as pv1
from tfb.paths import DENSE, RELEASE
from tfb.t2_onset import T2_PANELS
from tfb.task1_io import days


def same(a: dict, b: dict) -> bool:
    return a.keys() == b.keys() and all(np.array_equal(a[k], b[k]) if isinstance(a[k], np.ndarray) else a[k] == b[k] for k in a)


def later(w, k: int = 3):
    """The window row moved k slots later on the same day (the negative control)."""
    ts = pd.Timestamp(w.forecast_origin) + pd.Timedelta(minutes=5 * k)
    return w._replace(forecast_origin=ts.strftime("%Y-%m-%dT%H:%M:%SZ"), T=w.T + k, t=w.t + k, gap=w.gap + k)


def context(rule: str, panel: str, split: str):
    """The production feature context from whatever the loaders currently return (rule B only)."""
    return pv1.panel_context(panel, split) if rule == "B" else None


def compute(rule: str, panel: str, split: str, w, models, ctx) -> dict:
    """The rule's result for one window from whatever the loaders currently return."""
    a = pv1.load(panel, split, "speed", "eligible")
    if rule == "A":
        return ss.decode_inputs(a["speed"], a["eligible"], sl.vcut(panel), int(w.d), int(w.T), w.gap)
    pred, inp = ss.b_predict(ctx, a["speed"], a["eligible"], w, *models)
    return {"pred": pred, **inp}


def arrays(rule: str) -> None:
    models = ss.b_models() if rule == "B" else None
    checked = caught = 0
    for split in ("validation", "private"):
        for panel in T2_PANELS:
            index = pd.read_csv(RELEASE / "task2" / panel / split / "window_index.csv")
            ctx = context(rule, panel, split)
            for w in ss.window_table(index, days(panel, split)["days"]).itertuples():
                d, T = int(w.d), int(w.T)
                full, full_late = compute(rule, panel, split, w, models, ctx), compute(rule, panel, split, later(w), models, ctx)
                with t2_bench.blanked_after(split, d, T):
                    ctx_blank = context(rule, panel, split)
                    blank, blank_late = compute(rule, panel, split, w, models, ctx_blank), compute(rule, panel, split, later(w), models, ctx_blank)
                assert same(full, blank), f"{w.window_id}: changed with data after T blanked"
                caught += not same(full_late, blank_late)
                checked += 1
            print(f"  {panel}/{split}: identical", flush=True)
    print(f"arrays, rule {rule}: {checked} ongoing windows identical with every value after T blanked")
    print(f"negative control: the same computation three slots late differs under the same blanking in {caught} of {checked} windows")
    assert caught == checked


def files(rule: str) -> None:
    models = ss.b_models() if rule == "B" else None
    checked = 0
    for split in ("validation", "private"):
        for panel in T2_PANELS:
            index = pd.read_csv(RELEASE / "task2" / panel / split / "window_index.csv")
            ctx = context(rule, panel, split)
            for w in ss.window_table(index, days(panel, split)["days"]).itertuples():
                full = compute(rule, panel, split, w, models, ctx)
                with tempfile.TemporaryDirectory() as tmp:
                    rel = t2_bench.truncated_release(Path(tmp), panel, w.forecast_origin)
                    root = Path(tmp) / "dense"
                    (root / panel).mkdir(parents=True)
                    m = dense.panel_meta(panel, rel)
                    (root / panel / "meta.json").write_text(json.dumps(m))
                    (root / panel / "train").symlink_to(DENSE / panel / "train")
                    dense.convert_split(panel, split, m, rel, root / panel / split)
                    cut_index = pd.read_csv(rel / "task2" / panel / split / "window_index.csv")
                    assert cut_index.forecast_origin.max() == w.forecast_origin
                    with t2_bench.dense_root(root):
                        day_list = days(panel, split)["days"]
                        wc = ss.window_table(cut_index, day_list)
                        wc = next(wc[wc.window_id == w.window_id].itertuples())
                        a = pv1.load(panel, split, "speed", "eligible")
                        assert day_list[-1] == w.date and not np.isfinite(a["speed"][wc.d, wc.T + 1:]).any() and not a["eligible"][wc.d, wc.T + 1:].any()
                        cut = compute(rule, panel, split, wc, models, context(rule, panel, split))
                assert same(full, cut), f"{w.window_id}: changed with rows after T removed"
                checked += 1
            print(f"  {panel}/{split}: unchanged", flush=True)
    print(f"files, rule {rule}: {checked} ongoing windows unchanged with every row after T removed from the release")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", action="store_true")
    ap.add_argument("--rule", default="A", choices=["A", "B"])
    args = ap.parse_args()
    files(args.rule) if args.files else arrays(args.rule)
