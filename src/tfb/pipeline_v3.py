"""v3 submission: v1 with Task 1 replaced by the LightGBM correction of time interpolation.

Task 2 and Task 4 are copied from v1, so the leaderboard delta isolates Task 1
(0.35 * delta S_state + 0.15 * delta S_physics).

    python -m tfb.pipeline_v3
"""
from __future__ import annotations

import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

from tfb import t1_model
from tfb.kaggle_submit import package
from tfb.paths import ORGANIZER_REPO, PANELS, RELEASE, SUBMISSIONS
from tfb.pipeline_v0 import SCORED, validate
from tfb.task1_io import days, state_frame

N_PER_PANEL = 300_000


def main() -> None:
    work = SUBMISSIONS / "v3"
    work.mkdir(parents=True, exist_ok=True)
    for f in ("queue.csv", "odme.csv"):
        shutil.copy(SUBMISSIONS / "v1" / f, work / f)
    all_days = {p: np.ones(len(days(p, "train")["days"]), bool) for p in PANELS}
    df = t1_model.training_frame(PANELS, all_days, N_PER_PANEL)
    models = t1_model.fit(df)
    for name, (m, _) in models.items():
        m.save_model(str(work / f"t1_{name}.txt"))
    frames = []
    for i, panel in enumerate(PANELS):
        for split in SCORED:
            sp, fl, _ = t1_model.predict_panel(models, panel, split, i)
            frames.append(state_frame(panel, split, sp, fl))
        print(f"  {panel}: predicted", flush=True)
    pd.concat(frames, ignore_index=True).to_csv(work / "state.csv", index=False)
    subprocess.run([sys.executable, str(ORGANIZER_REPO / "src/merge_submissions.py"),
                    "--state", str(work / "state.csv"), "--queue", str(work / "queue.csv"),
                    "--odme", str(work / "odme.csv"), "--key", str(RELEASE / "submission_key.csv"),
                    "--output", str(work / "submission.csv")], check=True)
    validate(work / "submission.csv")
    print("wrote", package(work / "submission.csv"))


if __name__ == "__main__":
    main()
