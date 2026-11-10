"""Local Task 3 (physics) scoring on the train split.

The leaderboard scores conservation against organizer boundary flows that are
never published. Per the organizer scorer (``load_boundary_flux`` in
organizer-repo/src/task3/score_task3.py) those flows are "projected onto the
released observations, so a participant who reproduces what was published
satisfies the conservation identity exactly". The right-hand side of every
scored transition is therefore the observed change in link accumulation
N = length_km * q / max(v, 1), with the eligible stations of a link averaged.

Train ships the unmasked observations, so this module writes that flux for
train, and the official scorer then runs unchanged:

    python -m tfb.task3_proxy --panel D12_I5_N
    python organizer-repo/src/task3/score_task3.py --split train --panel D12_I5_N \
        --release-root ~/data/tfb/kaggle_public \
        --state-submission state_submission.csv \
        --boundary-flux ~/data/tfb/cache/task3_flux/train_D12_I5_N.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from tfb.paths import CACHE, ORGANIZER_REPO, PANELS, RELEASE

sys.path.insert(0, str(ORGANIZER_REPO / "src"))
from task3.score_task3 import load_ramp_flows, network_parameters  # noqa: E402

DT_MAX_H = 5.1 / 60.0  # the scorer's transition window


def observed_accumulation(panel_dir: Path, split: str) -> pd.DataFrame:
    """Link-level N from the unmasked observations, built the way the scorer builds its state."""
    parts = []
    for path in sorted((panel_dir / split / "mainline_states").glob("**/*.parquet")):
        f = pd.read_parquet(path, columns=["timestamp", "link_id", "speed_kmh", "flow_vph", "is_score_eligible"])
        f = f[f.is_score_eligible.astype(bool)]
        parts.append(f[["timestamp", "link_id", "speed_kmh", "flow_vph"]])
    if not parts:
        raise FileNotFoundError(f"no unmasked {split} observations under {panel_dir / split}; only train ships them")
    state = pd.concat(parts, ignore_index=True)
    state["link_id"] = state.link_id.astype(str)
    state["timestamp"] = pd.to_datetime(state.timestamp, utc=True)
    state = state.groupby(["timestamp", "link_id"], as_index=False).mean(numeric_only=True)
    params = network_parameters(panel_dir)[["link_id", "length_km"]]
    state = state.merge(params, on="link_id", how="left")
    state["N"] = state.flow_vph / state.speed_kmh.clip(lower=1.0) * state.length_km
    return state[["timestamp", "link_id", "N"]]


def boundary_flux(panel: str, release: Path = RELEASE, split: str = "train") -> pd.DataFrame:
    """Inflow minus outflow such that the observed state satisfies conservation exactly."""
    panel_dir = release / "corridors" / panel
    x = observed_accumulation(panel_dir, split).sort_values(["link_id", "timestamp"])
    g = x.groupby("link_id")
    dn = g.N.shift(-1) - x.N
    dt_h = (g.timestamp.shift(-1) - x.timestamp).dt.total_seconds() / 3600.0
    ok = (dt_h > 0) & (dt_h <= DT_MAX_H)
    x["net_vph"] = np.where(ok, dn / dt_h, np.nan)
    ramps = load_ramp_flows(panel_dir, split)
    if not ramps.empty:
        x = x.merge(ramps[["timestamp", "link_id", "on_ramp_flow_vph", "off_ramp_flow_vph"]],
                    on=["timestamp", "link_id"], how="left")
    for c in ("on_ramp_flow_vph", "off_ramp_flow_vph"):
        if c not in x:
            x[c] = 0.0
        x[c] = x[c].fillna(0.0)
    # rhs = dt*(inflow + on - outflow - off) must equal dN, so inflow - outflow = dN/dt - on + off.
    x["inflow_vph"] = x.net_vph - x.on_ramp_flow_vph + x.off_ramp_flow_vph
    x["outflow_vph"] = 0.0
    x["panel"] = panel
    x = x.dropna(subset=["inflow_vph"])
    return x[["panel", "timestamp", "link_id", "inflow_vph", "outflow_vph"]]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--panel", action="append", help="default: all ten panels")
    ap.add_argument("--split", default="train")
    ap.add_argument("--release-root", type=Path, default=RELEASE)
    ap.add_argument("--output-dir", type=Path, default=CACHE / "task3_flux")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for panel in args.panel or PANELS:
        flux = boundary_flux(panel, args.release_root, args.split)
        out = args.output_dir / f"{args.split}_{panel}.parquet"
        flux.to_parquet(out, index=False)
        print(f"{panel}: {len(flux):,} flux rows -> {out}", flush=True)


if __name__ == "__main__":
    main()
