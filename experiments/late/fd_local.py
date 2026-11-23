"""S_FD of a state.csv on the scored splits, computed locally (it needs no hidden truth).

The official S_FD (score_task3.score_panel) runs over every eligible cell of a regime's days: released
observations plus the submitted target cells. Cells that are blank in the release and not targets (the rest
of the blank spans after Task 2 origins) are unknown to us and left out here; the organizers fill them with
truth. They are about 1.5% of the eligible cells, so this is an approximation at that level.

    python experiments/late/fd_local.py <state.csv> [--scale 2.0]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from tfb.metrics import REGIMES, _physics_static
from tfb.paths import DENSE
from tfb.task1_io import load, meta


def tod(ts: pd.Series) -> np.ndarray:
    return (ts.str.slice(11, 13).astype(int) * 12 + ts.str.slice(14, 16).astype(int) // 5).to_numpy()


def dense_state(df: pd.DataFrame, panel: str, split: str):
    info = json.loads((DENSE / panel / split / "days.json").read_text())
    dpos = {d: i for i, d in enumerate(info["days"])}
    pos = {l: i for i, l in enumerate(meta(panel)["links"])}
    a = load(panel, split, "speed", "flow", "target", "eligible")
    sp, fl = a["speed"].copy(), a["flow"].copy()
    di = df.timestamp.str[:10].map(dpos).to_numpy()
    tt, li = tod(df.timestamp), df.link_id.map(pos).to_numpy()
    assert a["target"][di, tt, li].all() and len(df) == int(a["target"].sum()), f"{panel}/{split}: rows != targets"
    sp[di, tt, li], fl[di, tt, li] = df.speed_kmh.to_numpy(), df.flow_vph.to_numpy()
    return sp, fl, a["eligible"], np.asarray(info["regime"])


def s_fd(panel: str, split: str, sp, fl, elig, regime) -> float:
    st = _physics_static(panel, split)
    lanes = np.clip(st["lanes"], 1.0, None)[None, None, :]
    out = []
    for r in REGIMES:
        sel = regime == r
        s, f, e = sp[sel], fl[sel], elig[sel] & np.isfinite(sp[sel]) & np.isfinite(fl[sel])
        k = f / np.clip(s, 1.0, None)
        kl, ql = k / lanes, f / lanes
        capl, kcl, kjl = (st["cap"] / st["lanes"])[None, None, :], (st["kcrit"] / st["lanes"])[None, None, :], (st["kjam"] / st["lanes"])[None, None, :]
        qfd = np.where(kl <= kcl, st["vf"][None, None, :] * kl, capl / np.maximum(kjl - kcl, 1e-9) * np.maximum(kjl - kl, 0))
        out.append(max(0.0, 1.0 - np.abs(ql - qfd)[e].sum() / (np.abs(ql)[e].sum() + 1e-9)))
    return float(np.mean(out))


def main() -> None:
    path = Path(sys.argv[1]).expanduser()
    scale = float(sys.argv[sys.argv.index("--scale") + 1]) if "--scale" in sys.argv else 1.0
    st = pd.read_csv(path, dtype={"link_id": str})
    st["split"] = np.where(st.timestamp.str[:7] == "2031-03", "validation", "private")
    for split in ("validation", "private"):
        vals, vals_s = [], []
        for panel, df in st[st.split == split].groupby("panel"):
            sp, fl, elig, regime = dense_state(df, panel, split)
            vals.append(s_fd(panel, split, sp, fl, elig, regime))
            if scale != 1.0:
                tgt = load(panel, split, "target")["target"]
                sp2, fl2 = np.where(tgt, sp * scale, sp), np.where(tgt, fl * scale, fl)
                vals_s.append(s_fd(panel, split, sp2, fl2, elig, regime))
        line = f"{split}: S_FD = {np.mean(vals):.5f} over {len(vals)} panels"
        if vals_s:
            line += f"; with targets x{scale:g}: {np.mean(vals_s):.5f}; difference {np.mean(vals) - np.mean(vals_s):.5f}"
        print(line)


if __name__ == "__main__":
    main()
