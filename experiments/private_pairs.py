"""Public against private: every paired single-change submission re-read on the private month.

Reads the team's submission list from Kaggle (public and private score of every submission), pairs each
single-change version with its base, and writes experiments/private_pairs.csv for the report (Section 7.1
and the figure fig_public_vs_private.png).

    python experiments/private_pairs.py            # downloads the list with the Kaggle CLI
    python experiments/private_pairs.py subs.csv   # or reads a saved `kaggle competitions submissions --csv` file
"""
import io
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / "private_pairs.csv"
COMPETITION = "2026-ieee-big-data-traffic-flow-bench"

# (family, change, changed version, base version); each pair differs in one component (LOG entries by date)
PAIRS = [
    ("Task 1", "LightGBM residual model", "v3", "v1"),
    ("Task 1", "tuned LightGBM (600k cells, ±5 neighbours)", "v4", "v3"),
    ("Task 1", "leave-one-day-out training profiles", "v5", "v4"),
    ("Task 1", "profiles from the scored month", "v9", "v8"),
    ("Task 1", "day anomaly against the month profile", "v15", "v13"),
    ("Task 1", "without training-month profiles", "v16", "v15"),
    ("Task 1", "self-supervised rows from the scored months", "v20", "v16"),
    ("Task 1", "two seeds", "v21", "v16"),
    ("Task 1", "without time, position and panel features", "v24", "v16"),
    ("Task 1", "density adjustment", "v26", "v25"),
    ("Task 1", "larger self-supervised set", "v29", "v25"),
    ("Task 1", "three seeds instead of two", "v32", "v30"),
    ("Task 1", "extra congested training cells, 150k", "v34", "v31"),
    ("Task 1", "extra congested training cells, 300k", "v35", "v31"),
    ("Task 1", "50/50 average with the public notebook's Task 1", "v38", "v35"),
    ("Task 1", "public feature set at 0.25", "v42", "v40"),
    ("Task 1", "U-Net at 0.33", "v48", "v42"),
    ("Task 1", "U-Net at 0.5", "v47", "v42"),
    ("Task 1", "second U-Net seed", "v50", "v47"),
    ("Task 4", "KL projection of the raw prior", "v2b", "v1"),
    ("Task 4", "KL projection of the gravity fit", "v2a", "v1"),
    ("Task 4", "NNLS from the count-rescaled prior", "v2c", "v1"),
    ("ongoing", "dynamics features (A3b), 5 seeds", "v8", "v5"),
    ("ongoing", "recurrent far links + step shrink", "v11", "v10"),
    ("ongoing", "recurrent far links", "v12", "v10"),
    ("ongoing", "without training recurrence features", "v13", "v10"),
    ("ongoing", "without flow and arrival anomalies", "v17", "v13"),
    ("ongoing", "step shrink 0.2", "v22", "v16"),
    ("ongoing", "without time, weekday, position and panel", "v23", "v16"),
    ("ongoing", "10 seeds instead of 5", "v28", "v16"),
    ("ongoing", "causal estimate of links masked at T", "v31", "v30"),
    ("ongoing", "learned estimate of links masked at T", "v33", "v31"),
    ("ongoing", "the strongest public notebook's ongoing rows", "v37", "v35"),
    ("ongoing", "public all-link model averaged on candidate cells only", "v39", "v35"),
    ("ongoing", "public all-link model averaged over every link", "v40", "v35"),
    ("ongoing", "corridor CNN at 0.5", "v43", "v42"),
    ("ongoing", "corridor CNN at 0.33", "v44", "v42"),
    ("ongoing", "corridor CNN alone", "v45", "v42"),
    ("ongoing", "dynamics-only CNN at 0.33", "v46", "v42"),
    ("ongoing", "blend 0.35 / off-scale 0.45 (final) against 0.5 / 0.5", "v63", "v52"),
    ("ongoing", "blend 0.3 / off-scale 0.5 against 0.5 / 0.5", "v62", "v52"),
    ("onset", "5 seeds instead of 1", "v7", "v5"),
    ("onset", "5 seeds instead of 1 (replicate on a later base)", "v10", "v9"),
    ("onset", "anomaly features (v6 also has 5 ongoing seeds)", "v6", "v7"),
    ("onset", "sharper priors with month context", "v14", "v10"),
    ("onset", "without the recurrence prior", "v18", "v13"),
    ("onset", "priors adapted to the month's activity", "v19", "v13"),
    ("onset", "10 seeds instead of 5", "v27", "v16"),
    ("onset", "the strongest public notebook's onset rows", "v36", "v35"),
]
# the ongoing blend-weight probes, all against v52 (blend weight on candidate cells / public scale off them)
WEIGHTS = [("v54", "0.65 / 0.65"), ("v55", "0.35 / 0.35"), ("v56", "0.25 / 0.25"), ("v57", "0.5 / 0.35"),
           ("v58", "0.35 / 0.5"), ("v59", "0.2 / 0.5"), ("v60", "0.35 / 0.65"), ("v61", "0.2 / 0.65"),
           ("v62", "0.3 / 0.5"), ("v63", "0.35 / 0.45")]


def submissions(path: str | None) -> pd.DataFrame:
    if path:
        raw = Path(path).read_text()
    else:
        raw = subprocess.run(["kaggle", "competitions", "submissions", COMPETITION, "--csv", "--page-size", "200"],
                             capture_output=True, text=True, check=True).stdout
    raw = "\n".join(l for l in raw.splitlines() if not l.lower().startswith("warning"))
    d = pd.read_csv(io.StringIO(raw))
    d = d[d.status.astype(str).str.contains("COMPLETE")].copy()
    d["v"] = d.description.astype(str).str.extract(r"^(v\d+[a-z]?)\s*:", expand=False)
    return d.dropna(subset=["v"]).drop_duplicates("v").set_index("v")


def main() -> None:
    s = submissions(sys.argv[1] if len(sys.argv) > 1 else None)
    rows = []
    for fam, change, a, b in PAIRS:
        dp, dq = s.loc[a, "publicScore"] - s.loc[b, "publicScore"], s.loc[a, "privateScore"] - s.loc[b, "privateScore"]
        rows.append(dict(family=fam, change=change, pair=f"{a} vs {b}", public=round(dp, 5), private=round(dq, 5)))
    r = pd.DataFrame(rows)
    r["same_sign"] = np.where((r.public.abs() < 0.0002) | (r.private.abs() < 0.0002), "~0",
                              np.where(np.sign(r.public) == np.sign(r.private), "yes", "no"))
    r.to_csv(OUT, index=False)
    pd.set_option("display.width", 220)
    print(r.to_string(index=False))
    print()
    for fam, x in r.groupby("family", sort=False):
        clear = x[x.same_sign != "~0"]
        c = np.corrcoef(x.public, x.private)[0, 1] if len(x) > 2 else float("nan")
        print(f"{fam:8s} pairs {len(x):2d}  same sign {(clear.same_sign == 'yes').sum()}/{len(clear)}  correlation {c:+.2f}")
    print("\nblend-weight probes against v52 (public Δ, private Δ):")
    for v, w in WEIGHTS:
        print(f"  {w:12s} {v}: {s.loc[v, 'publicScore'] - s.loc['v52', 'publicScore']:+.5f}  {s.loc[v, 'privateScore'] - s.loc['v52', 'privateScore']:+.5f}")
    chain = {"Task 1": ["v3", "v4", "v5", "v9", "v15", "v16", "v25", "v26", "v29", "v35", "v42", "v47", "v50"],
             "Task 2": ["v7", "v8", "v13", "v28", "v31", "v40", "v63"]}
    base = {"v3": "v1", "v4": "v3", "v5": "v4", "v9": "v8", "v15": "v13", "v16": "v15", "v25": "v16", "v26": "v25", "v29": "v25",
            "v35": "v31", "v42": "v40", "v47": "v42", "v50": "v47", "v7": "v5", "v8": "v5", "v13": "v10", "v28": "v16",
            "v31": "v30", "v40": "v35", "v63": "v52"}
    print("\ngains along the chain of adopted changes (sum of the paired deltas):")
    for fam, vs in chain.items():
        dp = sum(s.loc[v, "publicScore"] - s.loc[base[v], "publicScore"] for v in vs)
        dq = sum(s.loc[v, "privateScore"] - s.loc[base[v], "privateScore"] for v in vs)
        print(f"  {fam}: public {dp:+.4f}, private {dq:+.4f}, transferred {dq / dp:.0%}")
    print(f"  v1 -> v64: public {s.loc['v64', 'publicScore'] - s.loc['v1', 'publicScore']:+.4f}, private {s.loc['v64', 'privateScore'] - s.loc['v1', 'privateScore']:+.4f}")
    best = s.privateScore.idxmax()
    print(f"\nbest private: {best} {s.loc[best, 'privateScore']:.5f}; selected v64 {s.loc['v64', 'privateScore']:.5f}; "
          f"public-private gap v1 {s.loc['v1', 'privateScore'] - s.loc['v1', 'publicScore']:+.4f}, v64 {s.loc['v64', 'privateScore'] - s.loc['v64', 'publicScore']:+.4f}")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
