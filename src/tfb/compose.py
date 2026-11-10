"""Build a submission from per-task files of earlier versions, so one leaderboard delta isolates one task.

    python -m tfb.compose --out v5 --state v6 --queue v4 --odme v4
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys

from tfb.kaggle_submit import package
from tfb.paths import ORGANIZER_REPO, RELEASE, SUBMISSIONS
from tfb.pipeline_v0 import validate

FILES = {"state": "state.csv", "queue": "queue.csv", "odme": "odme.csv"}


def compose(out: str, sources: dict[str, str]) -> None:
    work = SUBMISSIONS / out
    work.mkdir(parents=True, exist_ok=True)
    for task, version in sources.items():
        if version != out:
            shutil.copy(SUBMISSIONS / version / FILES[task], work / FILES[task])
    subprocess.run([sys.executable, str(ORGANIZER_REPO / "src/merge_submissions.py"),
                    "--state", str(work / "state.csv"), "--queue", str(work / "queue.csv"),
                    "--odme", str(work / "odme.csv"), "--key", str(RELEASE / "submission_key.csv"),
                    "--output", str(work / "submission.csv")], check=True)
    validate(work / "submission.csv")
    package(work / "submission.csv")
    (work / "manifest.json").write_text(json.dumps({"composed_from": sources}, indent=2))
    print("wrote", work / "submission.zip")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    for task in FILES:
        ap.add_argument(f"--{task}", required=True, help=f"version folder to take {FILES[task]} from")
    args = ap.parse_args()
    compose(args.out, {t: getattr(args, t) for t in FILES})


if __name__ == "__main__":
    main()
