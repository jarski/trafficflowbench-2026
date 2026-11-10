"""Task 1 submission I/O between dense arrays and the released template."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from tfb.dense import tod_index
from tfb.paths import DENSE, RELEASE


@lru_cache(maxsize=None)
def meta(panel: str) -> dict:
    return json.loads((DENSE / panel / "meta.json").read_text())


@lru_cache(maxsize=None)
def days(panel: str, split: str) -> dict:
    return json.loads((DENSE / panel / split / "days.json").read_text())


def load(panel: str, split: str, *names: str) -> dict[str, np.ndarray]:
    return {n: np.load(DENSE / panel / split / f"{n}.npy") for n in names}


def template(panel: str, split: str, release: Path = RELEASE) -> pd.DataFrame:
    """The per-panel Task 1 template with each row's (day, tod, link) index into the dense arrays."""
    t = pd.read_csv(release / "task1" / panel / split / "sample_submission_state.csv", dtype=str)
    day_pos = {d: i for i, d in enumerate(days(panel, split)["days"])}
    link_pos = {l: i for i, l in enumerate(meta(panel)["links"])}
    t["di"] = [day_pos[x[:10]] for x in t.timestamp]
    t["ti"] = tod_index(t.timestamp)
    t["li"] = t.link_id.map(link_pos).to_numpy()
    return t


def state_frame(panel: str, split: str, speed: np.ndarray, flow: np.ndarray) -> pd.DataFrame:
    """Fill the template's speed/flow from [day, tod, link] prediction arrays."""
    t = template(panel, split)
    idx = (t.di.to_numpy(), t.ti.to_numpy(), t.li.to_numpy())
    t["speed_kmh"] = speed[idx].astype(float)
    t["flow_vph"] = flow[idx].astype(float)
    if t[["speed_kmh", "flow_vph"]].isna().any().any():
        raise ValueError(f"{panel}/{split}: predictions missing at some target cells")
    return t[["panel", "timestamp", "station_id", "link_id", "mask_regime", "speed_kmh", "flow_vph"]]
