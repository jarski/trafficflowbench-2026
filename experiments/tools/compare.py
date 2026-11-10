"""Compare two submission folders task by task (by key)."""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
a, b = (Path(x).expanduser() for x in sys.argv[1:3])
K = {"state.csv": (["panel", "timestamp", "station_id", "link_id", "mask_regime"], ["speed_kmh", "flow_vph"]),
     "queue.csv": (["window_id", "timestamp", "link_id"], ["queue_pred"]),
     "odme.csv": (["panel", "departure_time", "path_id"], ["path_flow"])}
for f, (keys, vals) in K.items():
    x = pd.read_csv(a / f, dtype=str); y = pd.read_csv(b / f, dtype=str)
    m = x.merge(y, on=keys, suffixes=("_a", "_b"), validate="one_to_one")
    out = [f"{f}: rows {len(x)}/{len(y)} matched {len(m)}"]
    for v in vals:
        d = np.abs(m[f"{v}_a"].astype(float) - m[f"{v}_b"].astype(float))
        out.append(f"{v}: max|diff| {d.max():.3g}, rel L1 {d.sum() / m[f'{v}_a'].astype(float).abs().sum():.3g}")
    print(" | ".join(out))
