"""Measure our components against the strongest public pipeline, one task family at a time.

The public notebook "TrafficFlowBench: Trees + U-Net + CNN | LB 0.87" (yeashusemwal, run 2026-10-15,
public 0.87141) combines its own models with the Task 2 of the public anubisyy notebook. Its output
submission.csv is downloaded to ~/data/tfb/public_nb/unet087/.

Three probes, each v35 with exactly one part replaced, so each leaderboard delta isolates that part
(onset and ongoing windows are disjoint and the score is linear in them):
    v36  onset rows from the public notebook
    v37  ongoing rows from the public notebook
    v38  Task 1 speed and flow averaged 50/50 with the public notebook (Task 3 follows)
Also prints how much the two submissions differ, per task and family. Measurement only: nothing
from the public notebook enters our pipeline unless a probe shows a clear gain.

    python experiments/public_probe.py
"""
import numpy as np
import pandas as pd

from tfb.kaggle_submit import package
from tfb.paths import DATA_ROOT, RELEASE, SUBMISSIONS, PANELS
from tfb.pipeline_v0 import validate

BASE = SUBMISSIONS / "v35" / "submission.csv"
PUBLIC = DATA_ROOT / "public_nb" / "unet087" / "submission.csv"

ours, pub = pd.read_csv(BASE), pd.read_csv(PUBLIC)
assert (ours.submission_id.to_numpy() == pub.submission_id.to_numpy()).all() and (ours.task == pub.task).all()
key = pd.read_csv(RELEASE / "submission_key.csv", usecols=["submission_id", "task", "split", "panel", "window_id"], dtype=str)
assert (key.submission_id.astype(int).to_numpy() == ours.submission_id.to_numpy()).all()
cond = pd.concat([pd.read_csv(RELEASE / "task2" / p / s / "window_index.csv", usecols=["window_id", "condition"])
                  for p in PANELS for s in ("validation", "private")
                  if (RELEASE / "task2" / p / s / "window_index.csv").exists()])
family = key.window_id.map(dict(zip(cond.window_id, cond.condition))).fillna("").to_numpy()
split = key.split.to_numpy()

print("Task 2: cells predicted queued and agreement (Jaccard of the queued sets)")
for c in ("queue_onset", "queue_ongoing"):
    for s in ("validation", "private"):
        m = (family == c) & (split == s)
        a, b = ours.queue_pred.to_numpy()[m] == 1, pub.queue_pred.to_numpy()[m] == 1
        print(f"  {c:14s} {s:10s} ours {a.sum():5d}  public {b.sum():5d}  both {(a & b).sum():5d}  "
              f"Jaccard {(a & b).sum() / max((a | b).sum(), 1):.3f}")

st = (ours.task == "state").to_numpy()
print("Task 1: RMS difference ours vs public (speed km/h, flow veh/h)")
for s in ("validation", "private"):
    m = st & (split == s)
    for col in ("speed_kmh", "flow_vph"):
        d = ours[col].to_numpy()[m] - pub[col].to_numpy()[m]
        print(f"  {s:10s} {col:9s} RMS {np.sqrt(np.mean(d ** 2)):.3f}  mean {d.mean():+.3f}")
od = (ours.task == "odme").to_numpy()
d = np.abs(ours.path_flow.to_numpy()[od] - pub.path_flow.to_numpy()[od])
print(f"Task 4: max |diff| {d.max():.4g}, relative L1 {d.sum() / np.abs(ours.path_flow.to_numpy()[od]).sum():.3g}")


def write(version: str, sub: pd.DataFrame) -> None:
    work = SUBMISSIONS / version
    work.mkdir(parents=True, exist_ok=True)
    sub.to_csv(work / "submission.csv", index=False)
    validate(work / "submission.csv")
    print(version, "->", package(work / "submission.csv"))


for version, c in (("v36", "queue_onset"), ("v37", "queue_ongoing")):
    sub = ours.copy()
    m = family == c
    sub.loc[m, "queue_pred"] = pub.loc[m, "queue_pred"].astype(int).to_numpy()
    write(version, sub)
sub = ours.copy()
for col in ("speed_kmh", "flow_vph"):
    sub.loc[st, col] = 0.5 * ours.loc[st, col].to_numpy() + 0.5 * pub.loc[st, col].to_numpy()
write("v38", sub)
