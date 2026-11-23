"""Late diagnostic probes built from the final submission v64 (run once; 2026-11-23).

    late_p1: every Task 2 prediction zero            -> score = Tasks 1 + 3 + 4 of v64, exactly
    late_p2: speed and flow of every target times 2  -> S_state = 0 through its clip; q/v, and so S_LWR, unchanged;
                                                        S_FD changes by an amount fd_local.py computes
"""
import pandas as pd

from tfb.paths import SUBMISSIONS
from tfb.pipeline_v0 import validate

base = pd.read_csv(SUBMISSIONS / "v64" / "submission.csv")
for name in ("late_p1", "late_p2"):
    sub = base.copy()
    if name == "late_p1":
        sub.loc[sub.task == "queue", "queue_pred"] = 0.0
    else:
        st = sub.task == "state"
        sub.loc[st, ["speed_kmh", "flow_vph"]] = sub.loc[st, ["speed_kmh", "flow_vph"]] * 2.0
    out = SUBMISSIONS / name
    out.mkdir(exist_ok=True)
    sub.to_csv(out / "submission.csv", index=False)
    validate(out / "submission.csv")
    changed = (sub[["speed_kmh", "flow_vph", "queue_pred", "path_flow"]] != base[["speed_kmh", "flow_vph", "queue_pred", "path_flow"]]).any(axis=1)
    print(name, "rows changed by task:", sub.task[changed].value_counts().to_dict())
