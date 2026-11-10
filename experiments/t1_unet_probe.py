"""Leaderboard probe of the Task 1 U-Net (tfb.t1_unet): a base version's Task 1 values averaged with the U-Net.

At every target cell: (1 - w) * base + w * U-Net, then plausible_speed; Tasks 2 and 4 are copied from the base.
The base's values already passed plausible_speed once, so this differs from `pipeline_final` only where the
speed floor or density cap bound; the final run rebuilds everything with pipeline_final.

    TFB_UNET_DEVICE=mps python experiments/t1_unet_probe.py --out v47 --w 0.5 [--base v42] [--model cache/bench/t1_unet_all.pt]
"""
import argparse
import json
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

from tfb import t1_model, t1_unet
from tfb.kaggle_submit import package
from tfb.paths import CACHE, ORGANIZER_REPO, PANELS, RELEASE, SUBMISSIONS
from tfb.pipeline_v0 import SCORED, validate
from tfb.task1_io import state_frame, template

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--w", type=float, required=True)
ap.add_argument("--base", default="v42")
ap.add_argument("--model", default=str(CACHE / "bench" / "t1_unet_all.pt"), help="comma-separated: a seed average")
args = ap.parse_args()

work = SUBMISSIONS / args.out
work.mkdir(parents=True, exist_ok=True)
model = [t1_unet.load_model(m).to(t1_unet.DEVICE) for m in args.model.split(",")]
base = pd.read_csv(SUBMISSIONS / args.base / "state.csv")
frames = []
for panel in PANELS:
    for split in SCORED:
        t = template(panel, split)[["timestamp", "link_id", "di", "ti", "li"]]
        rows = base[(base.panel == panel) & base.timestamp.isin(set(t.timestamp))]
        m = t.merge(rows[["timestamp", "link_id", "speed_kmh", "flow_vph"]], on=["timestamp", "link_id"], how="left")
        assert len(m) == len(t) and m.speed_kmh.notna().all(), f"{panel}/{split}: base rows missing"
        us, uf = t1_unet.predict_split(model, panel, split)
        idx = (m.di.to_numpy(), m.ti.to_numpy(), m.li.to_numpy())
        sp, fl = np.full(us.shape, np.nan, np.float32), np.full(uf.shape, np.nan, np.float32)
        sp[idx] = (1 - args.w) * m.speed_kmh.to_numpy() + args.w * us[idx]
        fl[idx] = (1 - args.w) * m.flow_vph.to_numpy() + args.w * uf[idx]
        sp = t1_model.plausible_speed(panel, sp, fl)
        frames.append(state_frame(panel, split, sp, fl))
        print(f"  {panel}/{split}: {len(t):,} cells", flush=True)
pd.concat(frames, ignore_index=True).to_csv(work / "state.csv", index=False)
for name in ("queue.csv", "odme.csv"):
    shutil.copy(SUBMISSIONS / args.base / name, work / name)
subprocess.run([sys.executable, str(ORGANIZER_REPO / "src/merge_submissions.py"),
                "--state", str(work / "state.csv"), "--queue", str(work / "queue.csv"), "--odme", str(work / "odme.csv"),
                "--key", str(RELEASE / "submission_key.csv"), "--output", str(work / "submission.csv")], check=True)
validate(work / "submission.csv")
package(work / "submission.csv")
(work / "manifest.json").write_text(json.dumps({"probe": "t1_unet", "base": args.base, "t1_unet_weight": args.w, "model": args.model}, indent=2))
print("done ->", work / "submission.zip")
