"""Figures for the report (report/figures/*.png). Numbers come from released data, caches or LOG.md.

    python experiments/report_figures.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tfb.paths import CACHE, PROJECT
from tfb.t2_rules import vcut
from tfb.task1_io import load, meta

OUT = PROJECT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})


def fig_shift(panel: str = "D12_I5_N") -> None:
    """Queue share by link and time of day: train months vs the two scored months."""
    vc = vcut(panel)
    maps = {}
    for split, title in (("train", "train (9 months)"), ("validation", "validation (public)"), ("private", "private")):
        s = load(panel, split, "speed")["speed"]
        with np.errstate(all="ignore"):
            maps[title] = np.nanmean(s <= vc, axis=0)[60:264].T          # links x (05:00-22:00)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), sharey=True)
    for ax, (title, m) in zip(axes, maps.items()):
        im = ax.imshow(m, aspect="auto", origin="lower", cmap="magma_r", vmin=0, vmax=0.6,
                       extent=[5, 22, 0, m.shape[0]])
        ax.set_title(title)
        ax.set_xlabel("hour of day")
    axes[0].set_ylabel(f"{panel} link (travel order)")
    fig.colorbar(im, ax=axes, label="share of days queued", shrink=0.9)
    fig.savefig(OUT / "fig_shift_maps.png", bbox_inches="tight")
    plt.close(fig)


def fig_bench_vs_lb() -> None:
    """Local bench Δ (official-style selector windows) against the public-leaderboard Δ of the same change."""
    rows = [  # (label, bench delta on selector windows, leaderboard delta in total score); LOG 2026-10-06..11
        ("onset anomaly", 0.0059, -0.0060), ("onset sharper priors", 0.0050, -0.0078),
        ("onset adapted priors", -0.0029, -0.0045), ("ongoing far links", 0.0138, -0.0012),
        ("ongoing no recurrence", -0.0011, 0.0029), ("ongoing dynamics (A3b)", 0.0047, 0.0038),
        ("ongoing T estimate", 0.0006, 0.0010), ("ongoing learned T estimate", 0.0023, -0.0001),
        ("ongoing public-model average 0.5", -0.0122, 0.0027),   # LOG 2026-10-19: selector windows 0.8834 -> 0.8712
        ("ongoing CNN 0.5 (Dec-Feb holdout)", 0.0285, -0.0012),  # LOG 2026-10-20: +0.0285 per selector window; v43
    ]
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for label, b, l in rows:
        ax.scatter(b, l, color="tab:red" if b * l < 0 else "tab:blue", zorder=3)
        dy = -11 if "public-model" in label else 3          # keep this label clear of "no recurrence"
        ax.annotate(label, (b, l), textcoords="offset points", xytext=(4, dy), fontsize=7)
    ax.set_xlim(-0.016, 0.040)
    ax.axhline(0, color="grey", lw=0.8)
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_xlabel("local bench Δ IoU (selector windows, train months)")
    ax.set_ylabel("public leaderboard Δ (total score)")
    ax.set_title("Local validation vs the shifted month (red: signs disagree)")
    fig.savefig(OUT / "fig_bench_vs_lb.png", bbox_inches="tight")
    plt.close(fig)


def fig_progress() -> None:
    versions = [("baseline", 0.5534), ("v0", 0.8000), ("v1", 0.8404), ("v3", 0.8530), ("v4", 0.8549), ("v5", 0.8565),
                ("v8", 0.8603), ("v9", 0.8617), ("v10", 0.8630), ("v13", 0.8659), ("v16", 0.8676), ("v21", 0.8680),
                ("v26", 0.8686), ("v30", 0.8692), ("v31", 0.8702), ("v35", 0.8710), ("v40", 0.8736), ("v42", 0.8742),
                ("v47", 0.8763), ("v50", 0.8764), ("v58", 0.8771), ("v64", 0.8772)]
    fig, ax = plt.subplots(figsize=(8.5, 3))
    ax.plot(range(len(versions)), [v for _, v in versions], marker="o")
    ax.set_xticks(range(len(versions)), [n for n, _ in versions], rotation=45)
    ax.set_ylim(0.79, 0.885)
    ax.set_ylabel("public leaderboard")
    ax.set_title("Public score of each new best (organizer baseline 0.553 off scale)")
    fig.savefig(OUT / "fig_progress.png", bbox_inches="tight")
    plt.close(fig)


def fig_public_vs_private() -> None:
    """Public (March) against private (April) change of every paired single-change submission (experiments/private_pairs.csv)."""
    path = PROJECT / "experiments" / "private_pairs.csv"
    if not path.exists():
        return
    r = pd.read_csv(path)
    r = r[r.family != "Task 4"]          # the Task 4 probes (-0.03 to -0.07 on both) would set the scale
    colors = {"Task 1": "tab:blue", "ongoing": "tab:orange", "onset": "tab:red"}
    fig, ax = plt.subplots(figsize=(6.4, 5))
    lim = 0.0085
    ax.plot([-lim, lim], [-lim, lim], color="grey", lw=0.8, ls="--")
    ax.axhline(0, color="grey", lw=0.6)
    ax.axvline(0, color="grey", lw=0.6)
    for fam, x in r.groupby("family", sort=False):
        ax.scatter(x.public.clip(-lim, lim), x.private, color=colors[fam], label=f"{fam} ({len(x)} pairs)", zorder=3, s=28)
    notable = {"dynamics-only CNN at 0.33": (4, 4), "corridor CNN alone": (4, -9), "5 seeds instead of 1": (4, -9),
               "anomaly features (v6 also has 5 ongoing seeds)": (4, 4), "the strongest public notebook's ongoing rows": (4, -9),
               "public all-link model averaged over every link": (4, 4), "U-Net at 0.5": (4, 4),
               "LightGBM residual model": (-60, -10), "dynamics features (A3b), 5 seeds": (4, 4),
               "without training recurrence features": (4, -9), "sharper priors with month context": (4, 4)}
    short = {"dynamics-only CNN at 0.33": "dynamics-only CNN", "corridor CNN alone": "CNN alone",
             "5 seeds instead of 1": "onset 5 seeds", "anomaly features (v6 also has 5 ongoing seeds)": "onset anomaly features",
             "the strongest public notebook's ongoing rows": "notebook's ongoing rows",
             "public all-link model averaged over every link": "public all-link average", "U-Net at 0.5": "U-Net 0.5",
             "LightGBM residual model": "Task 1 LightGBM (+0.0126 / +0.0122)", "dynamics features (A3b), 5 seeds": "ongoing dynamics",
             "without training recurrence features": "no recurrence", "sharper priors with month context": "onset sharper priors"}
    for _, x in r.iterrows():
        if x.change in notable:
            ax.annotate(short[x.change], (min(max(x.public, -lim), lim), x.private), textcoords="offset points",
                        xytext=notable[x.change], fontsize=7)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-0.0045, 0.0045)
    ax.set_xlabel("public leaderboard Δ (March), total score")
    ax.set_ylabel("private leaderboard Δ (April), total score")
    ax.set_title("The same single change on the two scored months")
    ax.legend(loc="lower right", fontsize=8)
    fig.savefig(OUT / "fig_public_vs_private.png", bbox_inches="tight")
    plt.close(fig)


def fig_t1_errors() -> None:
    path = CACHE / "t1_holdout_errors.parquet"
    if not path.exists():
        return
    df = pd.read_parquet(path)
    g = df.groupby("state")
    rmse = pd.DataFrame({"interpolation": g.err_v_interp.apply(lambda e: np.sqrt(np.mean(e ** 2))),
                         "model": g.err_v_model.apply(lambda e: np.sqrt(np.mean(e ** 2)))}).loc[["free", "transition", "queued"]]
    share = g.size().loc[rmse.index] / len(df)
    fig, ax = plt.subplots(figsize=(5, 3))
    rmse.plot.bar(ax=ax, rot=0)
    for i, s in enumerate(share):
        ax.text(i, rmse.iloc[i].max() + 0.2, f"{s:.1%} of cells", ha="center", fontsize=7)
    ax.set_ylabel("speed RMSE (km/h)")
    ax.set_title("Task 1 holdout error by congestion state")
    fig.savefig(OUT / "fig_t1_errors.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_shift()
    fig_bench_vs_lb()
    fig_progress()
    fig_public_vs_private()
    fig_t1_errors()
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))
