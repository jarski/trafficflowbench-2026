"""Facts about the all-link blank spans after Task 2 window origins in the scored months (2026-11-23).

For every Task 2 window (window_index.csv) of validation and private: which slots around the origin T are
blank on every link, how many Task 1 targets fall into T+1..T+18, how the target share inside the spans
compares with the rest of the day, and whether any fully blank slot exists outside those spans (and in train).

    python experiments/late/span_facts.py
"""
import json

import numpy as np
import pandas as pd

from tfb.paths import DENSE, PANELS, RELEASE
from tfb.task1_io import load

SPAN = 18


def tod(ts: str) -> int:
    return int(ts[11:13]) * 12 + int(ts[14:16]) // 5


def main() -> None:
    tot = {}
    for split in ("validation", "private", "train"):
        rows = []
        n_t = n_in = n_blank_slots = n_other_blank = 0
        for panel in PANELS:
            a = load(panel, split, "speed", "flow", "occ", "target", "eligible", "pct")
            info = json.loads((DENSE / panel / split / "days.json").read_text())
            dpos = {x: i for i, x in enumerate(info["days"])}
            D, S, L = a["speed"].shape
            blank = np.isnan(a["speed"]).all(axis=2)             # [day, slot]: no link has a speed
            blank_all = blank & np.isnan(a["flow"]).all(axis=2) & np.isnan(a["occ"]).all(axis=2)
            span = np.zeros((D, S), bool)
            wpath = RELEASE / "task2" / panel / split / "window_index.csv"
            if wpath.exists():
                wi = pd.read_csv(wpath)
                for r in wi.itertuples():
                    d, T = dpos[r.date], tod(r.forecast_origin)
                    flat = np.arange(d * S + T + 1, d * S + T + 1 + SPAN)
                    inside = flat < D * S
                    span.reshape(-1)[flat[inside]] = True
                    obs = ~np.isnan(a["speed"].reshape(D * S, L))
                    prof = {k: float(obs[d * S + T + k].mean()) if 0 <= d * S + T + k < D * S else np.nan for k in range(-13, 22)}
                    rows.append(dict(panel=panel, cond=r.condition, T=T, crosses_midnight=T + SPAN >= S,
                                     hist_obs=np.mean([prof[k] for k in range(-12, 0)]), obs_T=prof[0],
                                     span_obs=np.nanmean([prof[k] for k in range(1, 19)]), obs_T19=prof[19], obs_Tm13=prof[-13],
                                     tgt_in_span=int(a["target"].reshape(D * S, L)[flat[inside]].sum()),
                                     elig_in_span=int(a["eligible"].reshape(D * S, L)[flat[inside]].sum()),
                                     cells_in_span=int(inside.sum() * L), regime=info["regime"][d],
                                     pct_pos_in_span=float((a["pct"].reshape(D * S, L)[flat[inside]] > 0).mean())))
            tgt = a["target"]
            n_t += int(tgt.sum())
            n_in += int((tgt & span[:, :, None]).sum())
            n_blank_slots += int(blank_all.sum())
            n_other_blank += int((blank_all & ~span).sum())
            assert not (span & ~blank).any() or split == "train", f"{panel}/{split}: a span slot has an observation"
            if split != "train":
                # target share of eligible cells: inside spans against the same days outside
                day_has = span.any(axis=1)
                e_in, t_in = (a["eligible"] & span[:, :, None]).sum(), (tgt & span[:, :, None]).sum()
                e_out = (a["eligible"] & ~span[:, :, None])[day_has].sum()
                t_out = (tgt & ~span[:, :, None])[day_has].sum()
                print(f"  {split} {panel}: span slots {int(span.sum())}, targets in spans {int(t_in):,} of eligible {int(e_in):,} "
                      f"({t_in / max(e_in, 1):.3f}); same days outside {t_out / max(e_out, 1):.3f}; "
                      f"blank slots outside spans {int((blank_all & ~span).sum())}")
        print(f"{split}: targets {n_t:,}, in spans {n_in:,} ({n_in / n_t:.4%}); fully blank slots {n_blank_slots}, of them outside spans {n_other_blank}")
        tot[split] = (n_t, n_in)
        if rows:
            r = pd.DataFrame(rows)
            print(r.groupby("cond")[["hist_obs", "obs_Tm13", "obs_T", "span_obs", "obs_T19", "pct_pos_in_span"]].mean().round(4).to_string())
            print("  windows:", len(r), "crossing midnight:", int(r.crosses_midnight.sum()),
                  "| span target share of eligible by regime:",
                  (r.groupby("regime").tgt_in_span.sum() / r.groupby("regime").elig_in_span.sum()).round(3).to_dict(),
                  "| T range:", int(r["T"].min()), int(r["T"].max()))
            print("  origin slot histogram (hour):", (r["T"] // 12).value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
