"""Onset: compare set-selection rules on leave-one-month-out probabilities."""
import numpy as np
import pandas as pd
import lightgbm as lgb
from tfb.paths import CACHE
from tfb.t2_onset import FEATURES_DROP, PARAMS, ROUNDS, choose_set
from tfb.t2_rules import iou

df = pd.read_parquet(CACHE / "t2_onset_train.parquet")
feats = [c for c in df.columns if c not in FEATURES_DROP] + ["panel_code"]
df["panel_code"] = df.panel.astype("category").cat.codes
df["p"] = np.nan
for month in sorted(df.month.unique()):
    tr, va = df[df.month != month], df.month == month
    df.loc[va, "p"] = lgb.train(PARAMS, lgb.Dataset(tr[feats], tr.y, weight=tr.w), ROUNDS).predict(df.loc[va, feats])
df.to_parquet(CACHE / "t2_onset_oof.parquet", index=False)

rng = np.random.default_rng(0)
def mc_set(p, kmax=8, n=400):
    order = np.argsort(-p)
    q = rng.random((n, len(p))) < p
    q = q[q.any(axis=1)]
    best, bk = -1, 1
    for k in range(1, kmax + 1):
        pred = np.zeros(len(p), bool); pred[order[:k]] = True
        v = np.mean([(pred & s).sum() / (pred | s).sum() for s in q])
        if v > best: best, bk = v, k
    pred = np.zeros(len(p), bool); pred[order[:bk]] = True
    return pred
rules = {
    "ratio_expect": choose_set,
    "thresh_0.5": lambda p: (p >= 0.5) | (np.arange(len(p)) == np.argmax(p)),
    "thresh_0.4": lambda p: (p >= 0.4) | (np.arange(len(p)) == np.argmax(p)),
    "thresh_0.3": lambda p: (p >= 0.3) | (np.arange(len(p)) == np.argmax(p)),
    "round_sum_p": lambda p: np.isin(np.arange(len(p)), np.argsort(-p)[:max(1, int(round(p.sum())))]),
    "mc_expected": mc_set,
}
res = []
for key, g in df.groupby("key", sort=False):
    el = g.w.to_numpy() > 0; y = g.y.to_numpy().astype(bool) & el; p = g.p.to_numpy()
    res.append({"panel": g.panel.iloc[0], **{k: iou(f(p) & el, y) for k, f in rules.items()}})
r = pd.DataFrame(res).groupby("panel").mean()
print(r.round(3).to_string()); print("mean:", r.mean().round(4).to_dict())
