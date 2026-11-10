"""v0 submission: quick wins on every task.

Task 1  time interpolation of the masked layer, train weekday profile where a day is empty
Task 2  onset windows: time-of-day onset prior, top-k links at T+30 only (k per panel, chosen on train)
        ongoing windows: persistence of the latest queue state (T where observed, else T-5)
Task 4  the organizer NNLS solve (lambda 0.05) on each split's own counts and prior

    python -m tfb.pipeline_v0            # writes SUBMISSIONS/v0/submission.csv and .zip
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from tfb.dense import tod_index
from tfb.kaggle_submit import package
from tfb.paths import ORGANIZER_REPO, PANELS, RELEASE, SUBMISSIONS, TASK2_EXCLUDED
from tfb.t1_interp import interp_time, weekday, weekday_profile
from tfb.t2_rules import HORIZON_SLOTS, best_onset_k, onset_events, onset_prior, top_k, train_queue_flags, vcut
from tfb.task1_io import days, load, meta, state_frame

SCORED = ("validation", "private")


def task1(split: str) -> pd.DataFrame:
    frames = []
    for panel in PANELS:
        tr = load(panel, "train", "true_speed", "true_flow")
        wd_tr = weekday(days(panel, "train")["days"])
        a = load(panel, split, "speed", "flow")
        wd = weekday(days(panel, split)["days"])
        out = {}
        for ch in ("speed", "flow"):
            prof = weekday_profile(tr[f"true_{ch}"], wd_tr)[wd]
            x = interp_time(a[ch])
            out[ch] = np.where(np.isnan(x), prof, x)
        frames.append(state_frame(panel, split, out["speed"], out["flow"]))
    return pd.concat(frames, ignore_index=True)


def task2(split: str) -> pd.DataFrame:
    frames = []
    for panel in [p for p in PANELS if p not in TASK2_EXCLUDED]:
        q = train_queue_flags(panel)
        events = onset_events(q)
        months = np.array([d[:7] for d in days(panel, "train")["days"]])
        k, means = best_onset_k(q, events, months)
        print(f"  {panel}: onset k={k} (train LOMO IoU {means[k]:.3f}), {len(events)} train onsets", flush=True)
        vc = vcut(panel)
        pos = {l: i for i, l in enumerate(meta(panel)["links"])}
        dpos = {d: i for i, d in enumerate(days(panel, split)["days"])}
        speed = load(panel, split, "speed")["speed"]
        win = pd.read_csv(RELEASE / "task2" / panel / split / "window_index.csv")
        tmpl = pd.read_csv(RELEASE / "task2" / panel / split / "sample_submission_queue.csv", dtype=str)
        for w in win.itertuples():
            T = int(tod_index(pd.Series([w.forecast_origin]))[0])
            di = dpos[w.date]
            rows = tmpl[tmpl.window_id == w.window_id].copy()
            step = (tod_index(rows.timestamp) - T)  # 1..6
            li = rows.link_id.map(pos).to_numpy()
            if w.condition == "queue_onset":
                pred = top_k(onset_prior(q, events, T + HORIZON_SLOTS), k)
                rows["queue_pred"] = (pred[li] & (step == HORIZON_SLOTS)).astype(int)
            else:
                latest = np.where(np.isnan(speed[di, T]), speed[di, T - 1], speed[di, T])
                queued = np.nan_to_num(latest, nan=np.inf) <= vc
                rows["queue_pred"] = queued[li].astype(int)
            frames.append(rows[["window_id", "timestamp", "link_id", "queue_pred"]])
    return pd.concat(frames, ignore_index=True)


def task4(split: str, work: Path) -> pd.DataFrame:
    out = work / f"odme_{split}"
    subprocess.run([sys.executable, str(ORGANIZER_REPO / "src/task4/build_task4_odme_artifacts.py"),
                    "--release-root", str(RELEASE), "--split", split, "--output-root", str(out)],
                   check=True, stdout=subprocess.DEVNULL)
    return pd.read_csv(out / "public_reference.csv", dtype=str)  # the solve on this split's own counts


def validate(path: Path) -> None:
    sample = pd.read_csv(RELEASE / "sample_submission.csv", usecols=["submission_id", "task"])
    sub = pd.read_csv(path)
    assert list(sub.columns) == ["submission_id", "task", "speed_kmh", "flow_vph", "queue_pred", "path_flow"]
    assert len(sub) == len(sample) and (sub.submission_id.to_numpy() == sample.submission_id.to_numpy()).all()
    assert not sub.isna().any().any(), "blank cells"
    assert sub.queue_pred.isin([0, 1]).all(), "queue_pred not binary"
    assert np.isfinite(sub.path_flow).all() and (sub.path_flow >= 0).all(), "bad path_flow"
    st = sub[sub.task == "state"]
    assert (st.speed_kmh > 0).all() and (st.flow_vph > 0).all(), "non-positive state values"
    print("valid:", sub.task.value_counts().to_dict())


def main() -> None:
    work = SUBMISSIONS / "v0"
    work.mkdir(parents=True, exist_ok=True)
    print("Task 1 ...", flush=True)
    pd.concat([task1(s) for s in SCORED]).to_csv(work / "state.csv", index=False)
    print("Task 2 ...", flush=True)
    pd.concat([task2(s) for s in SCORED]).to_csv(work / "queue.csv", index=False)
    print("Task 4 ...", flush=True)
    pd.concat([task4(s, work) for s in SCORED]).to_csv(work / "odme.csv", index=False)
    print("Merging ...", flush=True)
    subprocess.run([sys.executable, str(ORGANIZER_REPO / "src/merge_submissions.py"),
                    "--state", str(work / "state.csv"), "--queue", str(work / "queue.csv"),
                    "--odme", str(work / "odme.csv"), "--key", str(RELEASE / "submission_key.csv"),
                    "--output", str(work / "submission.csv")], check=True)
    validate(work / "submission.csv")
    print("wrote", package(work / "submission.csv"))


if __name__ == "__main__":
    main()
