"""Late-submission candidates: v64's state.csv with only the target cells inside the blank spans replaced.

    python experiments/late/span_candidates.py interp            # -> submissions/late_span_interp/state.csv
    python experiments/late/span_candidates.py fix --pred <dir>  # -> submissions/late_span_fix/state.csv
    python experiments/late/span_candidates.py verify <name>     # key join against v64, plausibility rules
    python experiments/late/span_candidates.py verify-submission <name>   # the composed submission.csv against v64's
<name> / --out is a folder name under ~/data/tfb/submissions or an absolute path (trial builds go to cache/late).

interp: every span target takes plain time interpolation of the split's masked layer (tfb.t1_interp.interp_time,
        speed and flow), then the production rule tfb.t1_model.plausible_speed (speed >= 5 km/h, q / v <= k_jam).
fix:    every span target takes the values stored in <dir>/<split>_<panel>.npz (speed, flow as [day, 288, link]),
        then the same rule.
The file is rewritten line by line: a line whose key is not a span target is copied byte for byte.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import span_lib as L  # noqa: E402

from tfb import t1_model as t1  # noqa: E402
from tfb.paths import PANELS, SUBMISSIONS  # noqa: E402
from tfb.t1_interp import interp_time  # noqa: E402
from tfb.task1_io import meta, template  # noqa: E402

SCORED = ("validation", "private")
BASE = SUBMISSIONS / "v64" / "state.csv"


def span_template(panel: str, split: str) -> pd.DataFrame:
    """The template rows of one panel and split with their array indices and whether they lie in a blank span."""
    t = template(panel, split)
    n_days = len(L.task1_io.days(panel, split)["days"])
    span, _ = L.slot_masks(n_days, L.official_origins(panel, split))
    t["in_span"] = span[t.di.to_numpy(), t.ti.to_numpy()]
    return t


def replacements(kind: str, pred_dir: Path | None) -> tuple[dict, dict]:
    """{(panel, timestamp, link_id): (speed text, flow text)} for the span targets, and counts."""
    out, stats = {}, {"span_targets": 0, "replaced": 0, "kept_no_value": 0}
    for panel in PANELS:
        for split in SCORED:
            t = span_template(panel, split)
            rows = t[t.in_span]
            stats["span_targets"] += len(rows)
            if rows.empty:
                continue
            a = L._ORIG_LOAD(panel, split, "speed", "flow", "target")
            idx = (rows.di.to_numpy(), rows.ti.to_numpy(), rows.li.to_numpy())
            assert a["target"][idx].all() and np.isnan(a["speed"][idx]).all()
            if kind == "interp":
                sp, fl = interp_time(a["speed"]), interp_time(a["flow"])
            else:
                z = np.load(pred_dir / f"{split}_{panel}.npz")
                sp, fl = z["speed"].astype(np.float32), z["flow"].astype(np.float32)
            ok = np.isfinite(sp[idx]) & np.isfinite(fl[idx])
            sp = t1.plausible_speed(panel, np.nan_to_num(sp, nan=50.0), np.nan_to_num(fl, nan=1000.0))
            sv, fv = sp[idx].astype(float), fl[idx].astype(float)
            for key, s, f, k in zip(zip(rows.panel, rows.timestamp, rows.link_id), sv, fv, ok):
                if k:
                    out[key] = (repr(float(s)), repr(float(f)))
            stats["replaced"] += int(ok.sum())
            stats["kept_no_value"] += int((~ok).sum())
    return out, stats


def folder(name: str) -> Path:
    return Path(name) if Path(name).is_absolute() else SUBMISSIONS / name


def write(name: str, repl: dict) -> int:
    work = folder(name)
    work.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(BASE, "r", newline="") as src, open(work / "state.csv", "w", newline="") as dst:
        dst.write(src.readline())
        for line in src:
            p = line.rstrip("\n").split(",")
            r = repl.get((p[0], p[1], p[3]))
            if r is None:
                dst.write(line)
            else:
                dst.write(",".join(p[:5] + [r[0], r[1]]) + "\n")
                n += 1
    return n


def verify(name: str) -> dict:
    """Key join against v64: rows that differ, whether all of them are span targets, and the plausibility rules."""
    base = pd.read_csv(BASE, dtype={"link_id": str, "station_id": str})
    new = pd.read_csv(folder(name) / "state.csv", dtype={"link_id": str, "station_id": str})
    keys = ["panel", "timestamp", "station_id", "link_id"]
    assert len(base) == len(new) and not new.duplicated(keys).any()
    m = base.merge(new, on=keys, how="outer", suffixes=("_v64", ""), validate="one_to_one", indicator=True)
    assert (m._merge == "both").all() and (m.mask_regime == m.mask_regime_v64).all()
    span = pd.concat([span_template(p, s)[["panel", "timestamp", "link_id", "in_span"]] for p in PANELS for s in SCORED])
    m = m.merge(span, on=["panel", "timestamp", "link_id"], how="left", validate="one_to_one")
    assert m.in_span.notna().all()
    m["in_span"] = m.in_span.astype(bool)
    diff = (m.speed_kmh != m.speed_kmh_v64) | (m.flow_vph != m.flow_vph_v64)
    kjam = pd.concat([pd.DataFrame({"panel": p, "link_id": meta(p)["links"], "kjam": meta(p)["k_jam"]}) for p in PANELS])
    m = m.merge(kjam, on=["panel", "link_id"], how="left", validate="many_to_one")
    month = m.timestamp.str[:7]
    res = {"rows": len(m), "rows_differ": int(diff.sum()), "differ_outside_spans": int((diff & ~m.in_span).sum()),
           "span_targets": int(m.in_span.sum()), "span_targets_unchanged": int((m.in_span & ~diff).sum()),
           "differ_validation": int((diff & (month == "2031-03")).sum()), "differ_private": int((diff & (month == "2031-04")).sum()),
           "min_speed": float(m.speed_kmh.min()), "speed_below_5": int((m.speed_kmh < 5.0).sum()),
           "density_above_kjam": int((m.flow_vph / m.speed_kmh > m.kjam * (1 + 1e-5)).sum()),
           "non_finite": int((~np.isfinite(m.speed_kmh) | ~np.isfinite(m.flow_vph)).sum()),
           "non_positive": int(((m.speed_kmh <= 0) | (m.flow_vph <= 0)).sum())}
    d = m[diff]
    res["rms_speed_change"] = float(np.sqrt(((d.speed_kmh - d.speed_kmh_v64) ** 2).mean())) if len(d) else 0.0
    res["rms_flow_change"] = float(np.sqrt(((d.flow_vph - d.flow_vph_v64) ** 2).mean())) if len(d) else 0.0
    for name_, sel in (("validation", month == "2031-03"), ("private", month == "2031-04")):
        dd = m[diff & sel]
        res[f"rms_speed_change_{name_}"] = float(np.sqrt(((dd.speed_kmh - dd.speed_kmh_v64) ** 2).mean())) if len(dd) else 0.0
        res[f"rms_flow_change_{name_}"] = float(np.sqrt(((dd.flow_vph - dd.flow_vph_v64) ** 2).mean())) if len(dd) else 0.0
    return res


def verify_submission(name: str) -> dict:
    """The composed submission.csv against v64's: which tasks' rows differ (only state rows may)."""
    cols = ["speed_kmh", "flow_vph", "queue_pred", "path_flow"]
    base = pd.read_csv(SUBMISSIONS / "v64" / "submission.csv")
    new = pd.read_csv(folder(name) / "submission.csv")
    assert len(base) == len(new) and (base.submission_id.to_numpy() == new.submission_id.to_numpy()).all()
    assert (base.task.to_numpy() == new.task.to_numpy()).all()
    changed = (base[cols].to_numpy() != new[cols].to_numpy()).any(axis=1)
    return {"rows": len(new), "rows_differ_by_task": new.task[changed].value_counts().to_dict()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=("interp", "fix", "verify", "verify-submission"))
    ap.add_argument("name", nargs="?")
    ap.add_argument("--pred", type=Path)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.kind == "verify":
        print(args.name, verify(args.name))
        return
    if args.kind == "verify-submission":
        print(args.name, verify_submission(args.name))
        return
    name = args.out or ("late_span_interp" if args.kind == "interp" else "late_span_fix")
    repl, stats = replacements(args.kind, args.pred)
    n = write(name, repl)
    print(name, stats, "lines rewritten:", n)
    print(name, verify(name))


if __name__ == "__main__":
    main()
