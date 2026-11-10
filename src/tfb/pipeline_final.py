"""End-to-end pipeline: raw release -> dense arrays -> models -> validated upload file.

This is the reproducibility entry point. Every step is deterministic (fixed
seeds, deterministic LightGBM), and every model is saved beside the output.

    python -m tfb.pipeline_final --out final            # reuse caches where present
    python -m tfb.pipeline_final --out final --fresh    # rebuild everything from the release

    python -m tfb.pipeline_final --out v11 --tasks 2 --base v10   # rebuild Task 2 only, copy Tasks 1 and 4
    python -m tfb.pipeline_final --out check --queue-only         # Task 2 only, on any release (also one with rows removed)

Current components (see experiments/LOG.md for the evidence behind each):
  Task 1  LightGBM residual correction of time interpolation: leave-one-day-out train profiles plus
          month-adapted profiles of the scored month; speed floor and density cap; averaged at target
          cells with a public feature set retrained here (tfb.t1_public, CC0), weight T1_PUBLIC_WEIGHT
  Task 2  LightGBM onset (expected-IoU link set at T+30) and ongoing (link, step; bridged queue
          segments, dissipation and tail-dynamics features) models, each averaged over T2_SEEDS seeds;
          ongoing probabilities averaged with a retrained public all-link model (tfb.t2_public, CC0)
  Task 4  organizer NNLS L2 projection of each split's own prior (lambda 0.05)
"""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from tfb import dense, t1_model, t1_public, t1_unet, t2_cnn, t2_ongoing, t2_onset, t2_public
from tfb.kaggle_submit import package
from tfb.paths import CACHE, DENSE, ORGANIZER_REPO, PANELS, PROJECT, RELEASE, SUBMISSIONS
from tfb.pipeline_t4swap import task4
from tfb.pipeline_v0 import SCORED, validate
from tfb.pipeline_v1 import task2, train_models
from tfb.task1_io import days, state_frame

T1_CELLS_PER_PANEL = 600_000
# Seed-averaged Task 2 models. Onset: 5 (10 seeds lost -0.0011 public, v27); ongoing: 10 (+0.0005, v28).
T2_SEEDS = {"onset": 5, "ongoing": 10}
# D5: self-supervised Task 1 rows from the scored months (hide `rate` of their observed cells); None = off.
T1_PSEUDO: dict | None = {"rate": 0.15, "n": 300_000}  # +0.0003 public at rate 0.1 (v20); larger set +0.0001 (v29)
T4 = {"start": "prior", "method": "nnls", "scale": "none", "lam": 0.05}
# Task 1 average with the public feature set retrained here (tfb.t1_public), weight of that model at target
# cells; Dec-Feb holdout +0.0009 at 0.25, +0.0005 at 0.5, all 10 panels positive (LOG 2026-10-19). 0 = off.
T1_PUBLIC_WEIGHT = 0.25


def step(name: str):
    print(f"[{time.strftime('%H:%M:%S')}] {name}", flush=True)


def ensure_dense(fresh: bool) -> None:
    if fresh or not all((DENSE / p / "meta.json").exists() for p in PANELS):
        step("dense arrays")
        for p in PANELS:
            dense.convert_panel(p)


def ensure_t2_frames(fresh: bool) -> None:
    for name, build in (("t2_onset_train.parquet", t2_onset.train_frame), ("t2_ongoing_train.parquet", t2_ongoing.train_frame)):
        path = CACHE / name
        if fresh or not path.exists():
            step(f"Task 2 training frame {name}")
            CACHE.mkdir(parents=True, exist_ok=True)
            build().to_parquet(path, index=False)


def task1(work: Path, unet_weights: str | None = None) -> pd.DataFrame:
    step("Task 1 training")
    all_days = {p: np.ones(len(days(p, "train")["days"]), bool) for p in PANELS}
    df = t1_model.training_frame(PANELS, all_days, T1_CELLS_PER_PANEL)
    if T1_PSEUDO:
        rng = np.random.default_rng(2)
        df = pd.concat([df] + [t1_model.pseudo_frame(p, s, i, T1_PSEUDO["rate"], T1_PSEUDO["n"], rng)
                               for i, p in enumerate(PANELS) for s in SCORED], ignore_index=True)
    models = t1_model.fit(df)
    for name, (m, _) in models.items():
        m.save_model(str(work / f"t1_{name}.txt"))
    public = None
    if T1_PUBLIC_WEIGHT > 0:
        step("Task 1 public-feature model")
        public = t1_public.fit(tuple(sorted({d[:7] for d in days(PANELS[0], "train")["days"]})))
        for name, m in public.items():
            m.save_model(str(work / f"t1_public_{name}.txt"))
    unet = None   # a list of models (one per seed), averaged by t1_unet.predict_split
    if t1_unet.WEIGHT > 0:
        if unet_weights:   # the saved weights of an earlier run (comma-separated for a seed average): no retraining
            step(f"Task 1 U-Net from {unet_weights}")
            unet = [t1_unet.load_model(p).to(t1_unet.DEVICE) for p in unet_weights.split(",")]
        else:
            step(f"Task 1 U-Net training on {t1_unet.DEVICE.type}, seeds {t1_unet.SEEDS}")
            panels = {p: t1_unet.TrainPanel(p) for p in PANELS}
            unet_days = {p: np.arange(len(tp.months)) for p, tp in panels.items()}
            unet = [t1_unet.fit(panels, unet_days, seed=s) for s in t1_unet.SEEDS]
            del panels
    step("Task 1 prediction")
    frames = []
    for i, panel in enumerate(PANELS):
        for split in SCORED:
            sp, fl, pdata = t1_model.predict_panel(models, panel, split, i)
            blends = []
            if public is not None:
                blends.append((T1_PUBLIC_WEIGHT, *t1_public.predict(public, panel, split)))
            if unet is not None:
                blends.append((t1_unet.WEIGHT, *t1_unet.predict_split(unet, panel, split)))
            for w, ps, pf in blends:   # each family at its weight against the running average, at target cells
                at = pdata.target & np.isfinite(ps)
                sp = np.where(at, (1 - w) * sp + w * ps, sp)
                fl = np.where(at, (1 - w) * fl + w * pf, fl)
            if blends:
                sp = t1_model.plausible_speed(panel, sp, fl)
            frames.append(state_frame(panel, split, sp, fl))
    for i, m in enumerate(unet or []):   # t1_unet.pt, t1_unet_2.pt, ... in the order of t1_unet.SEEDS (or of --unet-weights)
        t1_unet.save(m, work / ("t1_unet.pt" if i == 0 else f"t1_unet_{i + 1}.pt"))
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="final")
    ap.add_argument("--fresh", action="store_true", help="rebuild dense arrays and training caches")
    ap.add_argument("--tasks", default="1,2,4", help="tasks to rebuild; the others are copied from --base")
    ap.add_argument("--base", default=None, help="version folder to copy the tasks that are not rebuilt from")
    ap.add_argument("--queue-only", action="store_true",
                    help="Task 2 only: write queue.csv and stop. Runs on a release with every row after some time removed")
    ap.add_argument("--unet-weights", default=None,
                    help="load these saved Task 1 U-Net weights instead of training (comma-separated for a seed average)")
    args = ap.parse_args()
    tasks = {"2"} if args.queue_only else set(args.tasks.split(","))
    if tasks != {"1", "2", "4"} and not args.base and not args.queue_only:
        ap.error("--base is required when not all tasks are rebuilt")
    t0 = time.time()
    work = SUBMISSIONS / args.out
    work.mkdir(parents=True, exist_ok=True)
    ensure_dense(args.fresh)
    for task, name in (("1", "state.csv"), ("2", "queue.csv"), ("4", "odme.csv")):
        if task not in tasks and not args.queue_only:
            shutil.copy(SUBMISSIONS / args.base / name, work / name)
    if "1" in tasks:
        task1(work, args.unet_weights).to_csv(work / "state.csv", index=False)
    if "2" in tasks:
        ensure_t2_frames(args.fresh)
        step("Task 2 training and prediction")
        codes, m_on, m_og = train_models(T2_SEEDS)
        for name, model in (("onset", m_on[0]), ("ongoing", m_og[0])):
            for i, b in enumerate(getattr(model, "boosters", [model])):
                b.save_model(str(work / f"t2_{name}_s{i}.txt"))
        if m_og[2] is not None:
            m_og[2].save_model(str(work / "t2_public.txt"))
        if m_og[3] is not None:
            t2_cnn.save(m_og[3], work / "t2_cnn.pt")
        pd.concat([task2(s, codes, m_on, m_og) for s in SCORED]).to_csv(work / "queue.csv", index=False)
    if args.queue_only:
        step(f"done in {round((time.time() - t0) / 60, 1)} min -> {work / 'queue.csv'}")
        return
    if "4" in tasks:
        step("Task 4")
        task4(**T4).to_csv(work / "odme.csv", index=False)
    step("merge and validate")
    subprocess.run([sys.executable, str(ORGANIZER_REPO / "src/merge_submissions.py"),
                    "--state", str(work / "state.csv"), "--queue", str(work / "queue.csv"),
                    "--odme", str(work / "odme.csv"), "--key", str(RELEASE / "submission_key.csv"),
                    "--output", str(work / "submission.csv")], check=True)
    validate(work / "submission.csv")
    package(work / "submission.csv")
    git = subprocess.run(["git", "-C", str(PROJECT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    manifest = {"git": git or "uncommitted", "python": platform.python_version(), "platform": platform.platform(),
                "rebuilt_tasks": sorted(tasks), "copied_from": args.base if tasks != {"1", "2", "4"} else None,
                "task4": T4, "t1_cells_per_panel": T1_CELLS_PER_PANEL, "t1_pseudo": T1_PSEUDO, "t2_seeds": T2_SEEDS,
                "t2_public_weight": t2_public.WEIGHT, "t2_cnn_weight": t2_cnn.WEIGHT, "t1_public_weight": T1_PUBLIC_WEIGHT,
                "t1_unet_weight": t1_unet.WEIGHT, "t1_unet_device": t1_unet.DEVICE.type if t1_unet.WEIGHT > 0 else None,
                "t1_unet_seeds": list(t1_unet.SEEDS) if t1_unet.WEIGHT > 0 and not args.unet_weights else None,
                "t1_unet_loaded_from": args.unet_weights,
                "runtime_min": round((time.time() - t0) / 60, 1),
                "packages": {m: __import__(m).__version__ for m in ("numpy", "pandas", "pyarrow", "scipy", "lightgbm", "torch")}}
    (work / "manifest.json").write_text(json.dumps(manifest, indent=2))
    step(f"done in {manifest['runtime_min']} min -> {work / 'submission.zip'}")


if __name__ == "__main__":
    main()
