"""Wait until a UTC time, then submit queued files one after another and log the scores."""
import datetime as dt
import sys
import time
from pathlib import Path

from tfb.kaggle_submit import COMPETITION, _api, package, wait_for_latest

start = dt.datetime.fromisoformat(sys.argv[1]).replace(tzinfo=dt.timezone.utc)
queue = [(Path(sys.argv[i]).expanduser(), sys.argv[i + 1]) for i in range(2, len(sys.argv), 2)]
log = Path("~/data/tfb/submissions/queue.log").expanduser()
while dt.datetime.now(dt.timezone.utc) < start:
    time.sleep(30)
for csv, msg in queue:
    while not csv.with_suffix(".zip").exists():  # build may still be running
        time.sleep(30)
    time.sleep(60)
    _api().competition_submit(str(csv.with_suffix(".zip")), msg, COMPETITION, quiet=True)
    s = wait_for_latest()
    line = f"{dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M} {s.ref} {str(s.status).split('.')[-1]} public={s.public_score} error={s.error_description} | {msg}"
    with open(log, "a") as fh:
        fh.write(line + "\n")
    print(line, flush=True)
