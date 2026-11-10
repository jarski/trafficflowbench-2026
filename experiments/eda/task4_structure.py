"""Task 4: dimensions, count consistency, prior vs counts, priors across splits."""
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from tfb.paths import RELEASE, PANELS

def operator(panel):
    net = RELEASE / "corridors" / panel / "network"
    paths = pd.read_csv(net / "path_set.csv", dtype=str)
    inc = pd.read_csv(net / "path_link_incidence.csv", dtype=str)
    pid = {p: i for i, p in enumerate(paths.path_id)}
    return paths, inc, pid

rows = []
for panel in PANELS:
    paths, inc, pid = operator(panel)
    out = {"panel": panel, "paths": len(paths)}
    priors = {}
    for split in ("train", "validation", "private"):
        d = RELEASE / "task4" / panel / split
        c = pd.read_csv(d / "synthetic_link_counts.csv", dtype={"link_id": str})
        b = pd.read_csv(d / "synthetic_weak_prior.csv", dtype={"path_id": str}).set_index("path_id").path_flow.reindex(paths.path_id).to_numpy()
        lid = {l: i for i, l in enumerate(c.link_id)}
        A = np.zeros((len(c), len(paths)))
        sub = inc[inc.link_id.isin(lid)]
        A[sub.link_id.map(lid).to_numpy(), sub.path_id.map(pid).to_numpy()] = 1.0
        cv = c["count"].to_numpy()
        f, res = nnls(A, cv, maxiter=20000)
        priors[split] = b
        if split == "validation":
            out.update(counted=len(c), rank=np.linalg.matrix_rank(A), nnls_rel_resid=res / np.linalg.norm(cv),
                       Ab_over_c=np.median((A @ b) / cv), Ab_c_l1=np.abs(A @ b - cv).sum() / cv.sum(),
                       prior_zero_share=(b == 0).mean(), prior_cv=b.std() / b.mean(), departure=pd.read_csv(d / "synthetic_weak_prior.csv").departure_time.iloc[0])
    t, v, p = priors["train"], priors["validation"], priors["private"]
    out.update(corr_val_priv=np.corrcoef(v, p)[0, 1], corr_tr_val=np.corrcoef(t, v)[0, 1],
               ratio_val_priv_cv=np.std(v / p) / np.mean(v / p), mean_ratio_val_priv=np.mean(v) / np.mean(p))
    rows.append(out)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(pd.DataFrame(rows).round(4).to_string(index=False))
