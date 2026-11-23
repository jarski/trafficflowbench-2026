"""Does the selection-conditioned model's train gain depend on train-month recurrence? Three cuts of the same
out-of-fold results (sel_eval_cache.pkl, sel_model_pred*.pkl), all on the replica and official train windows.

  per month      the gain in each held-out month;
  atypical days  the 20% of each panel's days whose congestion map (queued share by link and slot, 05:00-22:00)
                 correlates least with the mean map of the other months: the project's stand-in for the shifted
                 scored months (experiments/t2_public_bench.py, LOG 2026-10-19);
  ablations      the same model without the selection features (tag faithnf) and without the features that can
                 carry recurrence (tag faithdyn), when their predictions exist.

    python experiments/late/sel_transfer.py
"""
import pickle

import numpy as np
import pandas as pd

import sel_lib as sl
import sel_model as sm
from tfb.t2_onset import T2_PANELS
from tfb.task1_io import days, load


def atypical_days(q: np.ndarray, month_of_day: np.ndarray, share: float = 0.2) -> set:
    """Days whose 05:00-22:00 congestion map correlates least with the other months' mean map (t2_public_bench.py)."""
    maps = q[:, 60:264].reshape(len(q), -1).astype(np.float32)
    corr = np.full(len(q), np.nan)
    for mo in np.unique(month_of_day):
        ref = maps[month_of_day != mo].mean(axis=0)
        for d in np.flatnonzero(month_of_day == mo):
            if maps[d].std() > 0:
                corr[d] = np.corrcoef(maps[d], ref)[0, 1]
    ok = np.isfinite(corr)
    return set(np.flatnonzero(ok & (corr <= np.nanquantile(corr, share))).tolist())


def main() -> None:
    pd.set_option("display.width", 250)
    with open(sl.LATE / "sel_eval_cache.pkl", "rb") as f:
        recs = pickle.load(f)
    preds = {}
    for tag in sm.TAG_DROP:
        if sm.pred_path(tag).exists():
            with open(sm.pred_path(tag), "rb") as f:
                preds[tag] = pickle.load(f)
    odd = {}
    for panel in T2_PANELS:
        a = load(panel, "train", "true_speed", "eligible")
        q = (a["true_speed"] <= sl.vcut(panel)) & a["eligible"]
        odd[panel] = atypical_days(q, np.array([x[:7] for x in days(panel, "train")["days"]]))
    rows = []
    for i, r in enumerate(recs):
        ours, cand, pub = r["ours"].astype(float), r["cand"], r["public"].astype(float)
        row = dict(set=r["set"], panel=r["panel"], month=r["month"], flag=bool(r["flag"]), atypical=int(r["d"]) in odd[r["panel"]],
                   production=sl.window_iou(sl.choose(sl.combine(ours, cand, pub)), r["truth"], r["elig"]),
                   ours_alone=sl.window_iou(sl.choose(sl.combine(ours, cand, pub, weight=0.0)), r["truth"], r["elig"]))
        for tag, pr in preds.items():
            new = pr[i].astype(float)
            row[f"{tag} blend"] = sl.window_iou(sl.choose(sl.combine(new, cand, pub)), r["truth"], r["elig"])
            row[f"{tag} alone"] = sl.window_iou(sl.choose(sl.combine(new, cand, pub, weight=0.0)), r["truth"], r["elig"])
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_parquet(sl.LATE / "sel_transfer_scores.parquet", index=False)

    def gain(part: pd.DataFrame, a: str, b: str) -> str:
        d = part[a] - part[b]
        return f"{d.mean():+.4f} ± {d.std(ddof=1) / np.sqrt(len(d)):.4f} (n={len(d)})"

    for wset in ("official", "replica"):
        part = df[df.set == wset]
        print(f"\n=== {wset}: {len(part)} windows; production {part.production.mean():.4f}, ours alone {part.ours_alone.mean():.4f}")
        for tag in preds:
            print(f"  [{tag}] blend {part[f'{tag} blend'].mean():.4f} ({gain(part, f'{tag} blend', 'production')} against production); "
                  f"alone {part[f'{tag} alone'].mean():.4f} ({gain(part, f'{tag} alone', 'ours_alone')} against ours alone)")
            for label, sub in (("flagged", part[part.flag]), ("unflagged", part[~part.flag])):
                print(f"      {label}: blend {gain(sub, f'{tag} blend', 'production')}; alone {gain(sub, f'{tag} alone', 'ours_alone')}")
            print(f"      without the public model on candidate cells, against with it: new model {gain(part, f'{tag} alone', f'{tag} blend')}; "
                  f"old model {gain(part, 'ours_alone', 'production')}")
    rep = df[df.set == "replica"]
    print("\nreplica windows, per held-out month: faith blend minus production (mean IoU over the month's windows)")
    print(rep.assign(d=rep["faith blend"] - rep.production).groupby("month").d.agg(["mean", "count"]).round(4).T.to_string())
    print("\nreplica windows on atypical and typical days:")
    for flag_label, sub in (("atypical days", rep[rep.atypical]), ("typical days", rep[~rep.atypical])):
        print(f"  {flag_label}: {len(sub)} windows, production {sub.production.mean():.4f}; "
              + "; ".join(f"[{tag}] blend {gain(sub, f'{tag} blend', 'production')}, alone {gain(sub, f'{tag} alone', 'ours_alone')}" for tag in preds))
    print("\nreplica windows per panel: faith blend minus production")
    print(rep.assign(d=rep["faith blend"] - rep.production).groupby("panel").d.agg(["mean", "sem", "count"]).round(4).to_string())


if __name__ == "__main__":
    main()
