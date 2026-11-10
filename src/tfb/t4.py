"""Task 4 (ODME) estimators and a held-out-block check against the released counts.

The counts are exactly reproducible by non-negative path flows, and each split's
prior looks like a gravity pattern times independent lognormal noise (sigma about
0.63), so the estimators fit the counts exactly and differ in the starting
structure: the raw prior, or a gravity model fitted to log prior flows.

    python -m tfb.t4 --cv
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.optimize import nnls

from tfb.paths import PANELS, RELEASE
from tfb.task1_io import meta

SPLITS = ("train", "validation", "private")


def load(panel: str, split: str) -> dict:
    net = RELEASE / "corridors" / panel / "network"
    paths = pd.read_csv(net / "path_set.csv", dtype=str)
    inc = pd.read_csv(net / "path_link_incidence.csv", dtype=str)
    pid = {p: i for i, p in enumerate(paths.path_id)}
    d = RELEASE / "task4" / panel / split
    c = pd.read_csv(d / "synthetic_link_counts.csv", dtype={"link_id": str})
    order = {l: i for i, l in enumerate(meta(panel)["links"])}
    c = c.assign(o=c.link_id.map(order)).sort_values("o").reset_index(drop=True)  # travel order
    lid = {l: i for i, l in enumerate(c.link_id)}
    A = np.zeros((len(c), len(paths)))
    sub = inc[inc.link_id.isin(lid)]
    A[sub.link_id.map(lid).to_numpy(), sub.path_id.map(pid).to_numpy()] = 1.0
    prior = pd.read_csv(d / "synthetic_weak_prior.csv", dtype={"path_id": str})
    b = prior.set_index("path_id").path_flow.reindex(paths.path_id).to_numpy()
    return {"paths": paths, "A": A, "c": c["count"].to_numpy(), "b": b,
            "departure_time": str(prior.departure_time.iloc[0])}


def gravity(paths: pd.DataFrame, logb: np.ndarray) -> np.ndarray:
    """Least-squares fit of log prior on origin + destination effects + distance decay."""
    gap = paths.destination_zone.str[1:].astype(int).to_numpy() - paths.origin_zone.str[1:].astype(int).to_numpy()
    X = pd.get_dummies(pd.DataFrame({"o": paths.origin_zone, "d": paths.destination_zone}), drop_first=True).astype(float)
    X["lgap"], X["gap"], X["c"] = np.log(np.abs(gap) + 1), gap, 1.0
    coef, *_ = np.linalg.lstsq(X.to_numpy(), logb, rcond=None)
    return np.exp(X.to_numpy() @ coef)


def ipf(A: np.ndarray, c: np.ndarray, f0: np.ndarray, sweeps: int = 2000, tol: float = 1e-7) -> np.ndarray:
    """KL projection of f0 onto {A f = c} by cyclic row scaling."""
    f = f0.astype(float).copy()
    rows = [np.flatnonzero(A[j]) for j in range(A.shape[0])]
    for _ in range(sweeps):
        worst = 0.0
        for j, idx in enumerate(rows):
            s = f[idx].sum()
            if s > 0:
                r = c[j] / s
                f[idx] *= r
                worst = max(worst, abs(r - 1))
        if worst < tol:
            break
    return f


def nnls_l2(A: np.ndarray, c: np.ndarray, b: np.ndarray, lam: float = 0.05) -> np.ndarray:
    """The organizer baseline: min ||A f - c||^2 + lam ||f - b||^2, f >= 0."""
    aa = np.vstack([A, np.sqrt(lam) * np.eye(A.shape[1])])
    f, _ = nnls(aa, np.concatenate([c, np.sqrt(lam) * b]), maxiter=50 * A.shape[1])
    return f


def estimators(panel: str, split: str, data: dict) -> dict[str, np.ndarray]:
    """Starting structures; each is then fitted to the counts by the caller."""
    logs = {s: np.log(load(panel, s)["b"]) for s in SPLITS}
    geo = np.exp(np.mean([logs[s] for s in SPLITS], axis=0))
    return {
        "prior": data["b"],
        "gravity_own": gravity(data["paths"], np.log(data["b"])),
        "gravity_all": gravity(data["paths"], np.mean([logs[s] for s in SPLITS], axis=0)),
        "geomean": geo * np.exp(np.log(data["b"]).mean() - np.log(geo).mean()),  # this split's overall level
    }


def scale_to_counts(A: np.ndarray, c: np.ndarray, f0: np.ndarray, how: str) -> np.ndarray:
    """Rescale a starting structure so its implied counts match the released counts in aggregate."""
    if how == "none":
        return f0
    pred = A @ f0
    alpha = c.sum() / pred.sum() if how == "sum" else np.median(c / pred)
    return alpha * f0


def block_cv(n_blocks: int = 4, frac: float = 0.15) -> pd.DataFrame:
    rows = []
    for panel in PANELS:
        for split in SPLITS:
            data = load(panel, split)
            A, c = data["A"], data["c"]
            starts = estimators(panel, split, data)
            n = len(c)
            w = max(2, int(round(frac * n)))
            for k in range(n_blocks):
                lo = int(k * (n - w) / max(n_blocks - 1, 1))
                hold = np.zeros(n, bool)
                hold[lo:lo + w] = True
                r = {"panel": panel, "split": split, "block": k}
                if split == "validation":  # the baseline solve is slow; one split is enough to compare
                    r["nnls_l2_prior"] = np.abs(A[hold] @ nnls_l2(A[~hold], c[~hold], data["b"]) - c[hold]).sum() / c[hold].sum()
                for name, f0 in starts.items():
                    f = ipf(A[~hold], c[~hold], f0)
                    r[f"ipf_{name}"] = np.abs(A[hold] @ f - c[hold]).sum() / c[hold].sum()
                rows.append(r)
        print(f"  {panel} done", flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cv", action="store_true")
    args = ap.parse_args()
    if args.cv:
        d = block_cv()
        cols = [c for c in d.columns if c.startswith(("nnls", "ipf"))]
        print(d.groupby("panel")[cols].mean().round(4).to_string())
        print("\nmean held-out relative L1 count error:", d[cols].mean().round(4).to_dict())


if __name__ == "__main__":
    main()
