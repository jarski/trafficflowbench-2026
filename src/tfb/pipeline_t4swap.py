"""Swap only Task 4 in an existing submission, so the leaderboard delta isolates it (0.20 * delta S_ODME).

    python -m tfb.pipeline_t4swap --base v1 --out v2a --start gravity_all
    python -m tfb.pipeline_t4swap --base v1 --out v2b --start prior
    python -m tfb.pipeline_t4swap --base v1 --out v2c --method nnls --start prior --scale sum
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

from tfb import t4
from tfb.kaggle_submit import package
from tfb.paths import ORGANIZER_REPO, PANELS, RELEASE, SUBMISSIONS
from tfb.pipeline_v0 import SCORED, validate


def task4(start: str, method: str = "ipf", scale: str = "none", lam: float = 0.05) -> pd.DataFrame:
    frames = []
    for panel in PANELS:
        for split in SCORED:
            data = t4.load(panel, split)
            f0 = t4.scale_to_counts(data["A"], data["c"], t4.estimators(panel, split, data)[start], scale)
            if method == "ipf":
                f = t4.ipf(data["A"], data["c"], f0, sweeps=5000, tol=1e-9)
            else:
                f = t4.nnls_l2(data["A"], data["c"], f0, lam)
            resid = np.abs(data["A"] @ f - data["c"]).max() / data["c"].min()
            assert np.isfinite(f).all() and (f >= 0).all() and resid < 1e-2, f"{panel}/{split}: fit residual {resid:.2e}"
            print(f"  {panel}/{split}: max rel count residual {resid:.1e}, zero share {(f < 1e-9).mean():.2f}", flush=True)
            x = data["paths"][["path_id", "origin_zone", "destination_zone"]].copy()
            x.insert(0, "departure_time", data["departure_time"])
            x.insert(0, "panel", panel)
            x["path_flow"] = f
            frames.append(x)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", default="v1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", choices=["prior", "gravity_own", "gravity_all", "geomean"], required=True)
    ap.add_argument("--method", choices=["ipf", "nnls"], default="ipf")
    ap.add_argument("--scale", choices=["none", "sum", "median"], default="none")
    ap.add_argument("--lam", type=float, default=0.05)
    args = ap.parse_args()
    work = SUBMISSIONS / args.out
    work.mkdir(parents=True, exist_ok=True)
    for f in ("state.csv", "queue.csv"):
        shutil.copy(SUBMISSIONS / args.base / f, work / f)
    task4(args.start, args.method, args.scale, args.lam).to_csv(work / "odme.csv", index=False)
    subprocess.run([sys.executable, str(ORGANIZER_REPO / "src/merge_submissions.py"),
                    "--state", str(work / "state.csv"), "--queue", str(work / "queue.csv"),
                    "--odme", str(work / "odme.csv"), "--key", str(RELEASE / "submission_key.csv"),
                    "--output", str(work / "submission.csv")], check=True)
    validate(work / "submission.csv")
    print("wrote", package(work / "submission.csv"))


if __name__ == "__main__":
    main()
