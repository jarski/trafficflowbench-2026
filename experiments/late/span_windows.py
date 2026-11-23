"""Training windows for span-aware models: every window the corrected selector picks on a train month when its
forward scan starts on any day of that month (2026-11-23).

The scored months' windows are the scan started on day 1 (5 onset + 5 ongoing, origins 6 h apart). Starting the
same scan on each later day gives windows of the same kind on every day with a queue. The union over start days
is cached in cache/late/span_windows_<months>.json as {panel: [[day index, T, condition], ...]}.

    python experiments/late/span_windows.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import span_lib as L  # noqa: E402
import selector_check as SC  # noqa: E402


def union_windows(months) -> dict:
    path = L.LATE / f"span_windows_{months[0]}_{months[-1]}.json"
    if path.exists():
        return {p: [tuple(x) for x in v] for p, v in json.loads(path.read_text()).items()}
    out = {}
    for panel in L.T2_PANELS:
        q, elig, qf = L.queue_state(panel)
        mo = L.day_months(panel)
        found = set()
        for m in months:
            idx = np.flatnonzero(mo == m)
            for s in range(len(idx)):
                sub = idx[s:]
                found |= {(int(sub[d]), int(T), c) for d, T, c in SC.selector(q[sub], elig[sub], qf[sub])}
        out[panel] = sorted(found)
        print(f"  {panel}: {len(out[panel])} windows on {len({d for d, _, _ in found})} days; "
              f"onset {sum(1 for x in found if x[2] == 'queue_onset')}, ongoing {sum(1 for x in found if x[2] == 'queue_ongoing')}", flush=True)
    path.write_text(json.dumps(out))
    return out


def hourly_windows(months) -> dict:
    """More windows of the same kind, for augmentation: on every day and for every full hour o, the first origin
    T >= o that qualifies as an onset window and the first that qualifies as an ongoing window (the selector's
    conditions for one window; no spacing, no count limit). The real scan resumes 6 h after its previous window,
    so its resume times are a subset of these."""
    path = L.LATE / f"span_windows_hourly_{months[0]}_{months[-1]}.json"
    if path.exists():
        return {p: [tuple(x) for x in v] for p, v in json.loads(path.read_text()).items()}
    H, Z = SC.H, SC.Z
    out = {}
    for panel in L.T2_PANELS:
        q, elig, qf = L.queue_state(panel)
        mo = L.day_months(panel)
        found = set()
        for d in np.flatnonzero(np.isin(mo, months)):
            ok = {"queue_onset": [], "queue_ongoing": []}
            for T in range(H, L.S - Z):
                hor = q[d, T + 1:T + 1 + Z]
                if not hor.any() or elig[d, T - H:T].mean() < 0.7:
                    continue
                hist = q[d, T - H:T]
                if hist.sum() >= 2 and hist.sum(axis=0).max() >= 2:
                    horc = qf[d, T + 1:T + 1 + Z]
                    if SC.R.iou(np.broadcast_to(qf[d, T - 1], horc.shape), horc) <= 0.9:
                        ok["queue_ongoing"].append(T)
                elif not hist.any() and not q[d, T].any():
                    ok["queue_onset"].append(T)
            for c, ts in ok.items():
                ts = np.asarray(ts)
                for o in range(H, L.S - Z, 12):
                    later = ts[ts >= o]
                    if len(later):
                        found.add((int(d), int(later[0]), c))
        out[panel] = sorted(found)
        print(f"  {panel}: {len(out[panel])} hourly windows", flush=True)
    path.write_text(json.dumps(out))
    return out


def by_day(windows: list) -> dict:
    out = {}
    for d, T, c in windows:
        out.setdefault(d, []).append((T, c))
    return out


def pick(day_windows: list, rng: np.random.Generator, spacing: int = 72) -> list:
    """A random subset of a day's windows with origins at least `spacing` slots apart (the selector's 6 h rule)."""
    chosen = []
    for i in rng.permutation(len(day_windows)):
        T, c = day_windows[i]
        if all(abs(T - t) >= spacing for t, _ in chosen):
            chosen.append((T, c))
    return chosen


if __name__ == "__main__":
    for months in (L.TRAIN_MONTHS, L.TRAIN_MONTHS + L.HOLD_MONTHS):
        w = union_windows(months)
        print(months[0], "-", months[-1], ":", sum(len(v) for v in w.values()), "windows")
        w = hourly_windows(months)
        print(months[0], "-", months[-1], ":", sum(len(v) for v in w.values()), "hourly windows")
