"""Onset: can a dynamics-only model (no location priors) find non-recurrent onsets? LOMO CV on train."""
import numpy as np
import pandas as pd
import lightgbm as lgb
from tfb.paths import CACHE, PANELS, TASK2_EXCLUDED
from tfb.t2_onset import FEATURES_DROP, PARAMS, ROUNDS, choose_set
from tfb.t2_rules import iou
from tfb.task1_io import load, meta

df = pd.read_parquet(CACHE / "t2_onset_oof.parquet")
# time-of-day profiles per link (train truth), normalised like the features
for panel in [p for p in PANELS if p not in TASK2_EXCLUDED]:
    m = meta(panel); a = load(panel, "train", "true_speed", "true_flow", "true_occ")
    prof = {"speed": np.nanmean(a["true_speed"], axis=0) / np.array(m["free_speed_kmh"]),
            "flow": np.nanmean(a["true_flow"], axis=0) / np.array(m["capacity_vph"]),
            "occ": np.nanmean(a["true_occ"], axis=0) * 285.0 / np.array(m["critical_density"])}
    sel = df.panel == panel
    T = df.loc[sel, "tod"].to_numpy() - 6; L = df.loc[sel, "link"].to_numpy()
    for ch in ("speed", "flow", "occ"):
        df.loc[sel, f"{ch}_anom"] = df.loc[sel, f"{ch}_last"].to_numpy() - prof[ch][T, L]
        df.loc[sel, f"{ch}_anom_m15"] = df.loc[sel, f"{ch}_m15"].to_numpy() - prof[ch][T, L]
# neighbour anomalies (downstream density anomaly is the bottleneck signature)
df = df.sort_values(["key", "link"]).reset_index(drop=True)
for k in (1, 2, -1):
    df[f"occ_anom_n{k:+d}"] = df.groupby("key").occ_anom.shift(-k)
df["occ_anom_rank"] = df.groupby("key").occ_anom.rank(ascending=False, pct=True)
df["occ_anom_max_corr"] = df.groupby("key").occ_anom.transform("max")

prior_cols = [c for c in df.columns if c.startswith(("prior", "qfreq"))] + ["tod", "weekday", "panel_code"]
full_feats = [c for c in df.columns if c not in FEATURES_DROP | {"p"}]
dyn_feats = [c for c in full_feats if c not in prior_cols]
df["p_full2"], df["p_dyn"] = np.nan, np.nan
for month in sorted(df.month.unique()):
    tr, va = df[df.month != month], df.month == month
    for col, feats in (("p_full2", full_feats), ("p_dyn", dyn_feats)):
        df.loc[va, col] = lgb.train(PARAMS, lgb.Dataset(tr[feats], tr.y, weight=tr.w), ROUNDS).predict(df.loc[va, feats])
res = []
for key, g in df.groupby("key", sort=False):
    el = g.w.to_numpy() > 0; y = g.y.to_numpy().astype(bool) & el
    if not y.any(): continue
    pt = g.prior_tod.to_numpy()[y].max()
    row = {"panel": g.panel.iloc[0], "prior_true": pt}
    for col in ("p", "p_full2", "p_dyn"):
        row[col] = iou(choose_set(g[col].to_numpy()) & el, y)
    for w in (0.3, 0.5):
        row[f"blend{w}"] = iou(choose_set((1 - w) * g.p_full2.to_numpy() + w * g.p_dyn.to_numpy()) & el, y)
    res.append(row)
r = pd.DataFrame(res)
r["bin"] = pd.cut(r.prior_true, [-0.01, 0.05, 0.2, 0.5, 1.0])
cols = ["p", "p_full2", "p_dyn", "blend0.3", "blend0.5"]
print(r.groupby("bin", observed=True)[cols].mean().round(3).to_string())
print("overall (mean over panels):", r.groupby("panel")[cols].mean().mean().round(4).to_dict())
# what a validation-like mix (20% non-recurrent) would give
w = r.prior_true <= 0.2
mix = {c: 0.8 * r.loc[~w, c].mean() + 0.2 * r.loc[w, c].mean() for c in cols}
print("80/20 recurrent/non-recurrent mix:", {k: round(v, 4) for k, v in mix.items()})
