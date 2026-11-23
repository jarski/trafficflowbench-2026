"""1c (iii, iv). Window IoU of rules that use the selection-conditioned model (sel_model.py), on train, out of fold.

Per window (sel_eval_cache.pkl and sel_model_pred.pkl, both fitted without the window's month):
  prod   = the production probability: on our candidate cells 0.65 ours + 0.35 public, off them 0.45 public
  new    = the selection-conditioned LightGBM on our candidate cells
Rules (the expected-IoU set is chosen from the resulting probabilities over the whole window):
  production                   prod everywhere (v64's method)
  new blend, all windows       0.65 new + 0.35 public on candidates, in every window
  new alone, all windows       new on candidates (public only off them), in every window
  new blend, flagged           the blend in flagged windows, production elsewhere
  new alone, flagged           new alone in flagged windows, production elsewhere
  step 6 new alone, flagged    flagged windows: prod at steps 1-5, new at step 6; production elsewhere
  step 6 new blend, flagged    the same with 0.65 new + 0.35 public at step 6
  mean(prod, new), flagged     flagged windows: the mean of prod and the new-alone probability on candidates
Nothing here is fitted on the evaluation windows; the choice between the rules is made on them, which is said
in the report. `--steps` adds per-step IoU.

    python experiments/late/sel_decode_b.py
"""
import pickle

import numpy as np
import pandas as pd

import sel_lib as sl
import sel_model as sm

W = 0.35   # tfb.t2_public.WEIGHT, read here only to label the rules


def probs(r: dict, new: np.ndarray) -> dict:
    ours, cand, pub = r["ours"].astype(float), r["cand"], r["public"].astype(float)
    new = new.astype(float)
    prod = sl.combine(ours, cand, pub)
    blend = sl.combine(new, cand, pub)
    alone = sl.combine(new, cand, pub, weight=0.0)
    fl = bool(r["flag"])
    s6a, s6b = prod.copy(), prod.copy()
    s6a[5], s6b[5] = alone[5], blend[5]
    return {
        "production": prod,
        "new blend, all windows": blend,
        "new alone, all windows": alone,
        "new blend, flagged": blend if fl else prod,
        "new alone, flagged": alone if fl else prod,
        "step 6 new alone, flagged": s6a if fl else prod,
        "step 6 new blend, flagged": s6b if fl else prod,
        "mean(prod, new), flagged": 0.5 * (prod + alone) if fl else prod,
        "new alone, unflagged": prod if fl else alone,
        "new blend, unflagged": prod if fl else blend,
        "ours alone (no public)": sl.combine(ours, cand, pub, weight=0.0),
    }


def score(recs: list, preds: list) -> pd.DataFrame:
    rows = []
    for r, new in zip(recs, preds):
        row = dict(set=r["set"], panel=r["panel"], month=r["month"], kind=r["kind"], flag=bool(r["flag"]), sb=r["gap"] == sl.SPACING)
        for name, p in probs(r, new).items():
            pred = sl.choose(p)
            row[name] = sl.window_iou(pred, r["truth"], r["elig"])
            for j, x in enumerate(sl.step_iou(pred, r["truth"], r["elig"])):
                row[f"{name}|s{j + 1}"] = x
        rows.append(row)
    return pd.DataFrame(rows)


def report(df: pd.DataFrame, label: str, names: list) -> None:
    print(f"\n================ {label}: {len(df)} windows, {int(df.flag.sum())} flagged")
    out = []
    for name in names:
        d = df[name] - df["production"]
        fl, un = df[df.flag], df[~df.flag]
        dfl, dun = fl[name] - fl["production"], un[name] - un["production"]
        out.append(dict(rule=name, all=df[name].mean(), d_all=d.mean(), se_all=d.std(ddof=1) / np.sqrt(len(d)),
                        wins=int((d > 1e-9).sum()), losses=int((d < -1e-9).sum()),
                        flagged=fl[name].mean(), d_flagged=dfl.mean(), se_fl=dfl.std(ddof=1) / np.sqrt(max(len(dfl), 1)),
                        unflagged=un[name].mean(), d_unfl=dun.mean(), se_un=dun.std(ddof=1) / np.sqrt(max(len(dun), 1))))
    print(pd.DataFrame(out).round(4).to_string(index=False))
    fl = df[df.flag]
    for name in ("production", "new alone, all windows", "new blend, all windows"):
        print(f"  per step, flagged, {name}:", [round(fl[f'{name}|s{j}'].mean(), 3) for j in range(1, 7)])
    un = df[~df.flag]
    for name in ("production", "new alone, all windows", "new blend, all windows"):
        print(f"  per step, unflagged, {name}:", [round(un[f'{name}|s{j}'].mean(), 3) for j in range(1, 7)])
    print("  by kind (production / new alone / new blend):",
          {k: f"{g['production'].mean():.3f} / {g['new alone, all windows'].mean():.3f} / {g['new blend, all windows'].mean():.3f} (n={len(g)})"
           for k, g in df.groupby("kind")})


def main() -> None:
    pd.set_option("display.width", 280)
    pd.set_option("display.max_rows", 500)
    with open(sl.LATE / "sel_eval_cache.pkl", "rb") as f:
        recs = pickle.load(f)
    with open(sm.PRED, "rb") as f:
        preds = pickle.load(f)
    df = score(recs, preds)
    df.to_parquet(sl.LATE / "sel_decode_b_scores.parquet", index=False)
    names = [c for c in df.columns if "|" not in c and c not in ("set", "panel", "month", "kind", "flag", "sb")]
    off, rep = df[df.set == "official"], df[df.set == "replica"]
    report(off, "40 official train windows (June 2030; every model fitted without June)", names)
    report(rep, "per-month replica windows, 9 train months (every model fitted without the window's month)", names)
    report(rep[rep.month != "2030-06"], "replica windows without June", names)
    print("\nper month, replica windows: 'new alone, flagged' minus production (mean over the month's windows), and its flagged count")
    print(rep.assign(d=rep["new alone, flagged"] - rep["production"]).groupby("month").agg(d=("d", "mean"), flagged=("flag", "sum"), n=("d", "size")).round(4).T.to_string())
    print("\nper panel, official windows: production / new alone, flagged / step 6 new alone, flagged")
    print(off.groupby("panel")[["production", "new alone, flagged", "step 6 new alone, flagged", "new alone, all windows"]].mean().round(4).to_string())
    print("\nper panel, replica windows:")
    print(rep.groupby("panel")[["production", "new alone, flagged", "step 6 new alone, flagged", "new alone, all windows"]].mean().round(4).to_string())


if __name__ == "__main__":
    main()
