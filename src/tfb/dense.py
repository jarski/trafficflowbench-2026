"""Convert the release into dense per-panel arrays.

Layout: ``DENSE/<panel>/meta.json`` plus ``DENSE/<panel>/<split>/<name>.npy``,
each array shaped ``[day, 288, link]`` (ramps: ``[day, 288, ramp]``). Links are
stored in travel order: milepost order for N/E panels and reversed for S/W
panels, so index + 1 is always the downstream neighbour. There is exactly one
detector station per link in every panel, so station and link coincide.

Arrays per split:
  speed, flow, occ         the masked layer (the input every task sees), NaN where blank
  pct                      pct_observed (uint8), published even for blanked cells
  eligible                 is_score_eligible
  target                   Task 1 targets, read from the per-panel template
  true_speed, true_flow, true_occ   the unmasked layer (train only)
  ramp_flow, ramp_pct, ramp_eligible
and ``days.json`` holding the dates and each day's mask regime.

    python -m tfb.dense            # all panels, in parallel
    python -m tfb.dense --panel D12_I5_N
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from tfb.paths import DENSE, PANELS, RELEASE, SPLITS

N_TOD = 288
REVERSED_DIRECTIONS = ("S", "W")


def tod_index(ts: pd.Series) -> np.ndarray:
    """Slot 0..287 from the released text form YYYY-MM-DDTHH:MM:SSZ."""
    s = ts.astype(str)
    return (s.str.slice(11, 13).astype(int) * 12 + s.str.slice(14, 16).astype(int) // 5).to_numpy()


def panel_meta(panel: str, release: Path) -> dict:
    net = release / "corridors" / panel / "network"
    links = pd.read_csv(net / "links.csv", dtype={"link_id": str})
    fd = pd.read_csv(net / "fd_parameters.csv", dtype={"link_id": str})
    topo = pd.read_csv(net / "lwr_mainline_topology.csv", dtype=str)
    first = sorted((release / "corridors" / panel / "train" / "mainline_states").glob("**/*.parquet"))[0]
    st = pd.read_parquet(first, columns=["station_id", "link_id", "milepost"]).drop_duplicates("link_id")
    st = st.sort_values(["milepost", "link_id"])
    if panel.rsplit("_", 1)[1] in REVERSED_DIRECTIONS:
        st = st.iloc[::-1]
    order = st.link_id.astype(str).tolist()
    li = links.set_index("link_id").reindex(order)
    fdi = fd.set_index("link_id").reindex(order)
    ti = topo.set_index("link_id").reindex(order)
    ramps = pd.read_csv(net / "ramp_attachment_map.csv", dtype=str)
    pos = {l: i for i, l in enumerate(order)}
    return {
        "panel": panel,
        "links": order,
        "stations": st.station_id.astype(str).tolist(),
        "milepost": st.milepost.astype(float).tolist(),
        "length_km": li.length_km.astype(float).tolist(),
        "lanes": fdi.lanes.astype(float).tolist(),
        "free_speed_kmh": fdi.free_speed_kmh.astype(float).tolist(),
        "free_speed_links_csv": li.free_speed_kmh.astype(float).tolist(),
        "capacity_vph": fdi.capacity_vph.astype(float).tolist(),
        "critical_density": fdi.critical_density.astype(float).tolist(),
        "k_jam": fdi.k_jam.astype(float).tolist(),
        "on_ramps": ti.on_ramp_link_ids.fillna("").tolist(),
        "off_ramps": ti.off_ramp_link_ids.fillna("").tolist(),
        "ramps": ramps.ramp_link_id.tolist(),
        "ramp_type": ramps.ramp_type.tolist(),
        "ramp_link_index": [pos.get(l, -1) for l in ramps.nearest_mainline_link_id],
    }


def _fill(arrays: dict, frame: pd.DataFrame, day: int, col_index: np.ndarray, tod: np.ndarray, columns: dict) -> None:
    for name, col in columns.items():
        arrays[name][day, tod, col_index] = frame[col].to_numpy()


def convert_split(panel: str, split: str, meta: dict, release: Path, out: Path) -> dict:
    root = release / "corridors" / panel / split
    pos = {l: i for i, l in enumerate(meta["links"])}
    masked = {}
    for regime in ("R1", "R2", "R3"):
        for p in sorted((root / "mainline_states_masked" / f"mask_regime={regime}").glob("*.parquet")):
            masked[p.stem.replace("synthetic_mainline_", "").replace("_", "-")] = (regime, p)
    days = sorted(masked)
    day_pos = {d: i for i, d in enumerate(days)}
    n_days, n_links = len(days), len(meta["links"])
    shape = (n_days, N_TOD, n_links)
    a = {
        "speed": np.full(shape, np.nan, np.float32), "flow": np.full(shape, np.nan, np.float32),
        "occ": np.full(shape, np.nan, np.float32), "pct": np.zeros(shape, np.uint8),
        "eligible": np.zeros(shape, bool), "target": np.zeros(shape, bool),
    }
    filled = np.zeros(shape, np.uint8)
    cols = ["timestamp", "link_id", "speed_kmh", "flow_vph", "occupancy", "pct_observed", "is_score_eligible"]
    for d, (regime, p) in masked.items():
        f = pq.read_table(p, columns=cols).to_pandas()
        li, tod, di = f.link_id.map(pos).to_numpy(), tod_index(f.timestamp), day_pos[d]
        _fill(a, f, di, li, tod, {"speed": "speed_kmh", "flow": "flow_vph", "occ": "occupancy",
                                  "pct": "pct_observed", "eligible": "is_score_eligible"})
        np.add.at(filled, (di, tod, li), 1)
    assert (filled <= 1).all(), f"{panel}/{split}: duplicate rows in the masked layer"
    if not filled.all():   # a release with rows removed, e.g. the organizers' Task 2 check (tfb.t2_bench causality --files)
        print(f"  {panel}/{split}: {int((filled == 0).sum()):,} cells absent from the release, left blank", flush=True)

    truth_files = sorted((root / "mainline_states").glob("**/*.parquet"))
    if truth_files:
        for name in ("true_speed", "true_flow", "true_occ"):
            a[name] = np.full(shape, np.nan, np.float32)
        for p in truth_files:
            f = pq.read_table(p, columns=cols[:5]).to_pandas()
            d = p.stem.replace("synthetic_mainline_", "").replace("_", "-")
            li, tod = f.link_id.map(pos).to_numpy(), tod_index(f.timestamp)
            _fill(a, f, day_pos[d], li, tod, {"true_speed": "speed_kmh", "true_flow": "flow_vph", "true_occ": "occupancy"})
        seen = ~np.isnan(a["speed"])
        assert np.array_equal(a["speed"][seen], a["true_speed"][seen]), f"{panel}/{split}: masked and unmasked layers disagree"

    path = release / "task1" / panel / split / "sample_submission_state.csv"
    tmpl = (pd.read_csv(path, usecols=["timestamp", "link_id", "mask_regime"], dtype=str) if path.exists()
            else pd.DataFrame({c: pd.Series(dtype=str) for c in ("timestamp", "link_id", "mask_regime")}))
    tmpl = tmpl[tmpl.timestamp.str[:10].isin(days)]       # all rows unless days were removed from the release
    ti, tt, tl = np.array([day_pos[t[:10]] for t in tmpl.timestamp], int), tod_index(tmpl.timestamp), tmpl.link_id.map(pos).to_numpy()
    released = filled[ti, tt, tl] == 1                    # likewise for rows removed from a released day
    a["target"][ti[released], tt[released], tl[released]] = True
    regime_of_day = np.array([masked[d][0] for d in days])
    assert (tmpl.mask_regime.to_numpy() == regime_of_day[ti]).all(), f"{panel}/{split}: template regime != day regime"
    assert np.isnan(a["speed"][a["target"]]).all() and a["eligible"][a["target"]].all(), f"{panel}/{split}: odd target cells"

    ramp_pos = {r: i for i, r in enumerate(meta["ramps"])}
    rshape = (n_days, N_TOD, len(meta["ramps"]))
    r = {"ramp_flow": np.full(rshape, np.nan, np.float32), "ramp_pct": np.zeros(rshape, np.uint8),
         "ramp_eligible": np.zeros(rshape, bool)}
    for p in sorted((root / "ramp_states").glob("**/*.parquet")):
        f = pq.read_table(p, columns=["date", "timestamp", "ramp_link_id", "flow_vph", "pct_observed", "is_score_eligible"]).to_pandas()
        if f.empty or str(f.date.iloc[0]) not in day_pos:   # only in a release with rows removed
            continue
        ri, tod, di = f.ramp_link_id.map(ramp_pos).to_numpy(), tod_index(f.timestamp), day_pos[str(f.date.iloc[0])]
        _fill(r, f, di, ri, tod, {"ramp_flow": "flow_vph", "ramp_pct": "pct_observed", "ramp_eligible": "is_score_eligible"})
    a.update(r)

    out.mkdir(parents=True, exist_ok=True)
    for name, arr in a.items():
        np.save(out / f"{name}.npy", arr)
    (out / "days.json").write_text(json.dumps({"days": days, "regime": regime_of_day.tolist()}))
    return {"split": split, "days": n_days, "targets": int(a["target"].sum()), "template_rows": len(tmpl),
            "eligible_share": round(float(a["eligible"].mean()), 4)}


def convert_panel(panel: str, release: Path = RELEASE, dense: Path = DENSE) -> list[dict]:
    meta = panel_meta(panel, release)
    (dense / panel).mkdir(parents=True, exist_ok=True)
    (dense / panel / "meta.json").write_text(json.dumps(meta))
    return [{"panel": panel, **convert_split(panel, s, meta, release, dense / panel / s)} for s in SPLITS]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--panel", action="append")
    ap.add_argument("--workers", type=int, default=5)
    args = ap.parse_args()
    panels = args.panel or list(PANELS)
    with ProcessPoolExecutor(max_workers=min(args.workers, len(panels))) as ex:
        for rows in ex.map(convert_panel, panels):
            for row in rows:
                print(row, flush=True)


if __name__ == "__main__":
    main()
