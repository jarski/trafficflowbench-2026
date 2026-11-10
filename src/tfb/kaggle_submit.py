"""Package, submit and track Kaggle submissions.

Kaggle rejects .gz uploads ("No .csv or .parquet found after extracting the
submission archive"), so the upload is a .zip holding submission.csv.

    python -m tfb.kaggle_submit ~/data/tfb/submissions/v1/submission.csv -m "v1: ..."
    python -m tfb.kaggle_submit --status          # latest submissions with scores and errors
"""
from __future__ import annotations

import argparse
import time
import zipfile
from pathlib import Path

COMPETITION = "2026-ieee-big-data-traffic-flow-bench"


def package(csv_path: Path) -> Path:
    out = csv_path.with_suffix(".zip")
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        z.write(csv_path, arcname="submission.csv")
    return out


def _api():
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    return api


def status(n: int = 5) -> None:
    for s in _api().competition_submissions(COMPETITION)[:n]:
        print(s.ref, s.date, str(s.status).split(".")[-1], "public:", s.public_score or "-",
              "|", s.description, "| error:", s.error_description or "-")


def wait_for_latest(timeout_s: int = 1800, every_s: int = 20):
    api = _api()
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        s = api.competition_submissions(COMPETITION)[0]
        if "PENDING" not in str(s.status):
            return s
        time.sleep(every_s)
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("csv", nargs="?", type=Path)
    ap.add_argument("-m", "--message", default="")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()
    if args.status or args.csv is None:
        status()
        return
    archive = package(args.csv.expanduser())
    _api().competition_submit(str(archive), args.message, COMPETITION, quiet=True)
    print("submitted", archive)
    s = wait_for_latest()
    print("result:", s.ref, str(s.status).split(".")[-1], "public:", s.public_score, "| error:", s.error_description)


if __name__ == "__main__":
    main()
