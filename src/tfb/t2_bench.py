"""Task 2 test bench: leave-one-month-out evaluation on three window sets, plus a causality check.

  official  the official train windows (onset 39 of 40: the 40th has no observed queue at T+30)
  selector  the replicated organizer selector run on every train month (<= 5 + 5 per panel and month)
  all       every replicated training window (the cached training frames)

Official-style windows are what the leaderboard scores; "all" is what the models are trained on and
is optimistic (ongoing 0.887 there against 0.839 on official windows, LOG 2026-10-06). For each set
the report shows the mean IoU over panels and the self-assessed IoU (the ratio-of-expectations value
the set rule maximizes). Out-of-fold predictions are cached in cache/bench/<condition>_<tag>.parquet.

    python -m tfb.t2_bench onset --tag base [--seeds 3] [--frame file.parquet] [--drop f1,f2]
    python -m tfb.t2_bench ongoing --tag base
    python -m tfb.t2_bench report onset --tags base,seeds3
    python -m tfb.t2_bench causality     # every scored window's features are unchanged when data after T is blanked
    python -m tfb.t2_bench causality --files --submission v35   # the organizers' award check, on the files
"""
from __future__ import annotations

import argparse
import contextlib
import json
import tempfile
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

from tfb import dense
from tfb import pipeline_v1 as pv1
from tfb import t2_cnn
from tfb import t2_ongoing as og
from tfb import t2_onset as on
from tfb import t2_public
from tfb import task1_io
from tfb.dense import tod_index
from tfb.paths import CACHE, DENSE, RELEASE, SUBMISSIONS
from tfb.t2_rules import iou, selector_windows, vcut
from tfb.task1_io import days, load, meta

BENCH = CACHE / "bench"
CODES = {p: i for i, p in enumerate(sorted(on.T2_PANELS))}
FRAMES = {"onset": CACHE / "t2_onset_train.parquet", "ongoing": CACHE / "t2_ongoing_train.parquet"}
MODULES = {"onset": on, "ongoing": og}
SETS = ("official", "selector", "all")


# ------------------------------------------------------------------ window sets on train
def official_windows(panel: str) -> list[tuple]:
    dl = days(panel, "train")["days"]
    w = pd.read_csv(RELEASE / "task2" / panel / "train" / "window_index.csv")
    return [(dl.index(r.date), int(tod_index(pd.Series([r.forecast_origin]))[0]), r.condition) for r in w.itertuples()]


def month_selector_windows(panel: str, q: np.ndarray, elig: np.ndarray) -> list[tuple]:
    """The selector replica run on each train month separately, as if every month were a scored split."""
    months = np.array([d[:7] for d in days(panel, "train")["days"]])
    out = []
    for mo in np.unique(months):
        idx = np.flatnonzero(months == mo)
        out += [(int(idx[d]), T, c) for d, T, c in selector_windows(q[idx], elig[idx])]
    return out


def window_sets() -> dict:
    """{panel: {"official": [...], "selector": [...]}} of (day, T, condition) on train."""
    out = {}
    for panel in on.T2_PANELS:
        a = load(panel, "train", "true_speed", "eligible")
        q = (a["true_speed"] <= vcut(panel)) & a["eligible"]
        out[panel] = {"official": official_windows(panel), "selector": month_selector_windows(panel, q, a["eligible"])}
    return out


# ------------------------------------------------------------------ training
def fold_models(df: pd.DataFrame, feats: list, params: dict, rounds: int, seeds: int) -> dict:
    """{held-out month: Ensemble of seed models trained on the other months}."""
    models = {}
    for month in sorted(df.month.unique()):
        tr = df[df.month != month]
        # free_dataset: a trained booster otherwise keeps its binned training data alive
        models[month] = pv1.Ensemble([lgb.train({**params, "seed": params["seed"] + i},
                                                lgb.Dataset(tr[feats], tr.y, weight=tr.w), rounds).free_dataset()
                                      for i in range(seeds)])
        print(f"  fold {month}: {seeds} model(s)", flush=True)
    return models


def oof(cond: str, frame: str, seeds: int, drop: set, params: dict | None = None, rounds: int | None = None) -> pd.DataFrame:
    """Out-of-fold predictions on the training frame plus the official/selector windows missing from it."""
    mod = MODULES[cond]
    df = pd.read_parquet(frame)
    df["panel_code"] = df.panel.map(CODES)
    feats = [c for c in df.columns if c not in mod.FEATURES_DROP and c not in drop]
    models = fold_models(df, feats, params or mod.PARAMS, rounds or mod.ROUNDS, seeds)
    df["p"] = np.nan
    for month, m in models.items():
        sel = df.month == month
        df.loc[sel, "p"] = m.predict(df.loc[sel, feats])
    keep = ["key", "panel", "month", "link", "y", "w", "p"] + (["h", "outside_true"] if cond == "ongoing" else [])
    out = df[keep].assign(set="all")
    sets = window_sets()
    extra = []
    for panel, s in sets.items():
        tp = og.TrainPanel(panel) if cond == "ongoing" else None
        for name in ("official", "selector"):
            wins = [(d, T) for d, T, c in s[name] if c == f"queue_{cond}"]
            if cond == "onset":
                keys = {f"{panel}|{d}|{T + 6}" for d, T in wins}
                extra.append(out[out.key.isin(keys)].assign(set=name))
                missing = len(keys - set(out.key))
                if missing:
                    print(f"  {panel} {name}: {missing} onset window(s) not among the replicated events", flush=True)
            else:
                for d, T in wins:
                    r = tp.window(d, T)
                    if r.empty:
                        continue
                    r["panel_code"] = CODES[panel]
                    r["p"] = models[r.month.iloc[0]].predict(r[feats])
                    extra.append(r[keep].assign(set=name))
    return pd.concat([out] + extra, ignore_index=True)


# ------------------------------------------------------------------ scoring
def renormalize(p: np.ndarray, target: float) -> np.ndarray:
    """Onset windows always contain a queue at T+30: scale p up when the expected count sum(p) is below target."""
    s = p.sum()
    return np.clip(p * (target / s), 0, 1) if 0 < s < target else p


def window_scores(df: pd.DataFrame, cond: str, transform=None) -> pd.DataFrame:
    rows = []
    for (wset, key), g in df.groupby(["set", "key"], sort=False):
        p = g.p.to_numpy()
        if transform is not None:
            p = transform(p)
        el = g.w.to_numpy() > 0
        y = g.y.to_numpy().astype(bool) & el
        if cond == "onset":
            sel = on.choose_set(p)
            score = iou(sel & el, y)
        else:
            sel = og.choose_window(p, g.h.to_numpy())
            union = ((sel & el) | y).sum() + g.outside_true.iloc[0]
            score = ((sel & el) & y).sum() / union if union else 1.0
        inside = p[sel].sum()
        rows.append((wset, key, g.panel.iloc[0], score, inside / (sel.sum() + p.sum() - inside)))
    return pd.DataFrame(rows, columns=["set", "key", "panel", "iou", "self"])


def summary(scores: pd.DataFrame) -> pd.DataFrame:
    """Per window set: mean over panels of the mean window IoU (the official family weighting for 8 panels)."""
    per_panel = scores.groupby(["set", "panel"])[["iou", "self"]].mean().groupby("set").mean()
    per_panel["windows"] = scores.groupby("set").size()
    return per_panel.reindex([s for s in SETS if s in per_panel.index])


def cache_path(cond: str, tag: str):
    return BENCH / f"{cond}_{tag}.parquet"


def report(cond: str, tags: list[str], renorm: tuple = ()) -> None:
    for tag in tags:
        df = pd.read_parquet(cache_path(cond, tag))
        print(f"\n{cond} [{tag}]")
        print(summary(window_scores(df, cond)).round(4).to_string())
        for target in renorm:
            print(f"  renormalized to sum(p) >= {target}:")
            print(summary(window_scores(df, cond, lambda p, t=target: renormalize(p, t))).round(4).to_string())


# ------------------------------------------------------------------ causality
@contextlib.contextmanager
def blanked_after(split: str, d: int, T: int):
    """Patch the loaders so the scored split has nothing after slot T of day d (later days blank too)."""
    real = load

    def truncated(panel, sp, *names):
        arrs = real(panel, sp, *names)
        if sp != split:
            return arrs
        out = {}
        for k, v in arrs.items():
            v = np.array(v, copy=True)
            fill = np.nan if v.dtype.kind == "f" else 0
            v[d, T + 1:] = fill
            v[d + 1:] = fill
            out[k] = v
        return out

    saved = (pv1.load, on.load)
    pv1.load = on.load = truncated
    try:
        yield
    finally:
        pv1.load, on.load = saved


def causality() -> None:
    """Every released window's features must be identical when all data after its T is blanked."""
    checked = 0
    for split in ("validation", "private"):
        for panel in on.T2_PANELS:
            ctx = pv1.panel_context(panel, split)
            win = pd.read_csv(RELEASE / "task2" / panel / split / "window_index.csv")
            for w in win.itertuples():
                T, x = pv1.window_features(ctx, w)
                d = ctx["days"].index(w.date)
                with blanked_after(split, d, T):
                    ctx_blank = pv1.panel_context(panel, split)
                    _, x_blank = pv1.window_features(ctx_blank, w)
                pd.testing.assert_frame_equal(x, x_blank, check_exact=True)
                if w.condition == "queue_ongoing":   # the all-link models' inputs too
                    np.testing.assert_array_equal(t2_public.window_input(ctx, w, T), t2_public.window_input(ctx_blank, w, T))
                    np.testing.assert_array_equal(t2_cnn.window_input(ctx, w, T), t2_cnn.window_input(ctx_blank, w, T))
                checked += 1
        print(f"  {split}: all windows identical", flush=True)
    print(f"causality OK: {checked} windows, features unchanged with every value after T blanked")


# ------------------------------------------------------------------ causality on the files (the organizers' check)
def _cut(src: Path, dst: Path, column: str, origin: str) -> None:
    """Copy src to dst without the rows whose `column` is after `origin` (text timestamps compare in time order).

    Links the file when no row is cut and leaves it out when no row is left.
    """
    if src.suffix == ".parquet":
        t = pq.read_table(src)
        keep = pc.less_equal(t[column], origin)
        n = pc.sum(keep).as_py() or 0
    else:
        t = pd.read_csv(src, dtype=str)
        keep = (t[column] <= origin).to_numpy()
        n = int(keep.sum())
    if n == 0:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if n == len(t):
        dst.symlink_to(src)
    elif src.suffix == ".parquet":
        pq.write_table(t.filter(keep), dst)
    else:
        t[keep].to_csv(dst, index=False)


def truncated_release(root: Path, panel: str, origin: str) -> Path:
    """The panel's part of the release with every row stamped after `origin` removed, in every file:
    mainline and ramp states, window histories, window_index (by origin) and both templates."""
    rel = root / "release"
    (rel / "corridors" / panel).mkdir(parents=True)
    (rel / "config").symlink_to(RELEASE / "config")
    for name in ("network", "train"):   # the train split is stamped long before every scored origin
        (rel / "corridors" / panel / name).symlink_to(RELEASE / "corridors" / panel / name)
    for split in ("validation", "private"):
        src = RELEASE / "corridors" / panel / split
        for f in sorted(src.rglob("*.parquet")):
            _cut(f, rel / "corridors" / panel / split / f.relative_to(src), "timestamp", origin)
        t2 = RELEASE / "task2" / panel / split
        if t2.exists():
            _cut(t2 / "window_index.csv", rel / "task2" / panel / split / "window_index.csv", "forecast_origin", origin)
            for name in ("window_history.parquet", "sample_submission_queue.csv"):
                _cut(t2 / name, rel / "task2" / panel / split / name, "timestamp", origin)
        name = Path("task1") / panel / split / "sample_submission_state.csv"
        _cut(RELEASE / name, rel / name, "timestamp", origin)
    return rel


@contextlib.contextmanager
def dense_root(path: Path):
    """Point the array loaders (tfb.task1_io) at another dense root."""
    saved = task1_io.DENSE
    task1_io.DENSE = path
    task1_io.meta.cache_clear()
    task1_io.days.cache_clear()
    try:
        yield
    finally:
        task1_io.DENSE = saved
        task1_io.meta.cache_clear()
        task1_io.days.cache_clear()


def causality_files(submission: str | None) -> None:
    """The award check of thread 749601: rerun each scored window on a release with every row stamped after
    its origin removed, rebuilding the dense arrays from those files; the prediction must not change.
    With `submission`, the predictions on the full release must also equal that version's queue.csv."""
    from tfb.pipeline_final import T2_SEEDS
    codes, m_on, m_og = pv1.train_models(T2_SEEDS)
    sub = None
    if submission:
        q = pd.read_csv(SUBMISSIONS / submission / "queue.csv", dtype={"link_id": str})
        sub = dict(tuple(q.groupby("window_id")))
    checked = 0
    for split in ("validation", "private"):
        for panel in on.T2_PANELS:
            ctx = pv1.panel_context(panel, split)
            pos = {l: i for i, l in enumerate(meta(panel)["links"])}
            win = pd.read_csv(RELEASE / "task2" / panel / split / "window_index.csv")
            for w in win.itertuples():
                T, pred = pv1.predict_window(ctx, w, codes[panel], m_on, m_og)
                if sub is not None:
                    rows = sub[w.window_id]
                    got = pred[tod_index(rows.timestamp) - T - 1, rows.link_id.map(pos).to_numpy()]
                    assert np.array_equal(got, rows.queue_pred.to_numpy() == 1), f"{w.window_id}: differs from {submission}"
                with tempfile.TemporaryDirectory() as tmp:
                    rel = truncated_release(Path(tmp), panel, w.forecast_origin)
                    cut_index = pd.read_csv(rel / "task2" / panel / split / "window_index.csv")
                    w_cut = next(cut_index[cut_index.window_id == w.window_id].itertuples())
                    root = Path(tmp) / "dense"
                    (root / panel).mkdir(parents=True)
                    m = dense.panel_meta(panel, rel)
                    (root / panel / "meta.json").write_text(json.dumps(m))
                    (root / panel / "train").symlink_to(DENSE / panel / "train")
                    dense.convert_split(panel, split, m, rel, root / panel / split)
                    with dense_root(root):
                        T_cut, pred_cut = pv1.predict_window(pv1.panel_context(panel, split), w_cut, codes[panel], m_on, m_og)
                assert T_cut == T and np.array_equal(pred, pred_cut), f"{w.window_id}: prediction changed with rows after T removed"
                checked += 1
            print(f"  {panel}/{split}: {len(win)} windows unchanged", flush=True)
    print(f"causality (files) OK: {checked} windows; predictions unchanged with every row after T removed from the release"
          + (f", and equal to {submission}" if submission else ""))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["onset", "ongoing", "report", "causality"])
    ap.add_argument("cond", nargs="?", choices=["onset", "ongoing"], help="for report")
    ap.add_argument("--tag", default="base")
    ap.add_argument("--tags", default="base")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--frame", default=None)
    ap.add_argument("--drop", default="")
    ap.add_argument("--renorm", default="", help="report: comma-separated sum(p) targets for onset renormalization")
    ap.add_argument("--files", action="store_true", help="causality: remove rows from the release files and compare predictions")
    ap.add_argument("--submission", default=None, help="causality --files: also require equality with this version's queue.csv")
    args = ap.parse_args()
    if args.cmd == "causality":
        causality_files(args.submission) if args.files else causality()
    elif args.cmd == "report":
        report(args.cond, args.tags.split(","), tuple(float(x) for x in args.renorm.split(",") if x))
    else:
        t0 = time.time()
        BENCH.mkdir(parents=True, exist_ok=True)
        drop = {c for c in args.drop.split(",") if c}
        df = oof(args.cmd, args.frame or FRAMES[args.cmd], args.seeds, drop)
        df.to_parquet(cache_path(args.cmd, args.tag), index=False)
        print(f"{args.cmd} [{args.tag}] seeds={args.seeds} in {(time.time() - t0) / 60:.1f} min")
        print(summary(window_scores(df, args.cmd)).round(4).to_string())


if __name__ == "__main__":
    main()
