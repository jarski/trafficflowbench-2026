"""The window IoU used by the sel_ scripts against the organizers' own function, on the 40 official train windows.

For each official ongoing train window this builds the two frames organizer-repo/src/task2/score_task2.py works
on: the target (the released template rows with queue_true = observed speed <= 0.6 free speed and the released
is_score_eligible) and a submission (the same rows with queue_pred), and calls score_task2.score_window. The
result must equal sel_lib.window_iou on the arrays, for the production forecast and for candidate B's method
(both out of fold).

    python experiments/late/sel_official_scorer_check.py
"""
import pickle
import sys

import numpy as np
import pandas as pd

import sel_lib as sl
import sel_model as sm
from tfb.dense import tod_index
from tfb.paths import ORGANIZER_REPO, RELEASE
from tfb.task1_io import meta

sys.path.insert(0, str(ORGANIZER_REPO / "src"))
from task2.score_task2 import score_window  # noqa: E402


def main() -> None:
    with open(sl.LATE / "sel_eval_cache.pkl", "rb") as f:
        recs = pickle.load(f)
    with open(sm.PRED, "rb") as f:
        preds = pickle.load(f)
    worst, n, means = 0.0, 0, {"production": [], "B": []}
    templates = {}
    for r, new in zip(recs, preds):
        if r["set"] != "official":
            continue
        panel = r["panel"]
        if panel not in templates:
            t = pd.read_csv(RELEASE / "task2" / panel / "train" / "sample_submission_queue.csv", dtype={"link_id": str})
            templates[panel] = t
        t = templates[panel]
        rows = t[t.window_id == r["window_id"]].copy()
        pos = {l: i for i, l in enumerate(meta(panel)["links"])}
        step = tod_index(rows.timestamp) - r["T"] - 1
        li = rows.link_id.map(pos).to_numpy()
        assert len(rows) == r["truth"].size and step.min() == 0 and step.max() == sl.Z - 1
        rows["timestamp"] = pd.to_datetime(rows.timestamp, utc=True)
        target = rows[["window_id", "timestamp", "link_id"]].copy()
        target["queue_true"] = r["truth"][step, li]
        target["is_score_eligible"] = r["elig"][step, li]
        ours, cand, pub = r["ours"].astype(float), r["cand"], r["public"].astype(float)
        for name, pred in (("production", sl.choose(sl.combine(ours, cand, pub))), ("B", sl.choose(sl.combine(new.astype(float), cand, pub)))):
            sub = rows[["window_id", "timestamp", "link_id"]].copy()
            sub["queue_pred"] = pred[step, li].astype(int)
            official, missing, invalid = score_window(target, sub)
            mine = sl.window_iou(pred, r["truth"], r["elig"])
            assert missing == 0 and invalid == 0
            worst = max(worst, abs(official - mine))
            means[name].append(official)
            n += 1
    print(f"{n} window scores (40 official train windows x 2 forecasts): largest |score_task2.score_window - sel_lib.window_iou| = {worst:.2e}")
    print("mean IoU by the organizers' function: production %.4f, B %.4f" % (np.mean(means["production"]), np.mean(means["B"])))
    assert worst < 1e-12


if __name__ == "__main__":
    main()
