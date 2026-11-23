"""Span-aware fine-tuning of the Task 1 U-Net (tfb.t1_unet), 2026-11-23.

The production U-Net was trained with cell-wise Bernoulli masks only. Here its weights are trained further on days
that also carry all-link blank spans as the scored months do: after a window origin T (span_windows.py: the
selector's windows on the training months) slots T+1..T+18 are blanked on every link and T-12..T-1 are shown
unmasked. The loss is the production loss (scorer-weighted MSE) on the blank-span cells, plus the ordinary hidden
cells at weight --w-other.

    python experiments/late/span_unet_ft.py --init <weights.pt> --out <name> [--months holdout|all] [--epochs 8] [--exclude panels]
Writes cache/late/<name>.pt. `--months holdout` trains on Jun-Nov only and prints the span loss on the emulated
Dec-Feb spans after every epoch; `--months all` trains on Jun-Feb for the scored months.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import span_lib as L  # noqa: E402
import span_windows as W  # noqa: E402

from tfb import t1_unet as U  # noqa: E402


class SpanPanel(U.TrainPanel):
    """Train states of one panel; a sample hides a Bernoulli mask and blanks spans after window origins."""

    def __init__(self, panel: str, windows: dict, w_other: float, fixed: bool = False):
        super().__init__(panel)
        self.windows, self.w_other, self.fixed = windows, w_other, fixed

    def sample(self, d: int, rng: np.random.Generator):
        tv, tq, to = self.a["true_speed"][d], self.a["true_flow"][d], self.a["true_occ"][d]
        hide = (rng.random(tv.shape) < U.RATES[rng.integers(3)]) & self.ok[d]
        blank, hist = np.zeros(L.S, bool), np.zeros(L.S, bool)
        wins = self.windows.get(d, [])
        for T, _ in (wins if self.fixed else W.pick(wins, rng)):
            blank[T + 1:T + 1 + L.SPAN] = True
            hist[max(T - L.HIST, 0):T] = True
        hist &= ~blank
        hide &= ~hist[:, None]
        gone = hide | blank[:, None]
        x, iv, iq = U.day_input(np.where(gone, np.nan, tv), np.where(gone, np.nan, tq), np.where(gone, np.nan, to), self.st)
        y = np.nan_to_num(np.stack([tv / U.V_UNIT - iv, tq / self.st.q_unit - iq]), nan=0.0).astype(np.float32)
        w = np.where(blank[:, None] & self.ok[d], 1.0, np.where(hide, self.w_other, 0.0)).astype(np.float32)
        return x, y, w


def span_loss(model, panels: dict, day_sets: dict, seed: int = 123) -> float:
    """Production loss on the blank-span cells only, fixed masks."""
    model.eval()
    rng = np.random.default_rng(seed)
    tot = n = 0.0
    with torch.no_grad():
        for p, idx in day_sets.items():
            for i in range(0, len(idx), U.BATCH):
                x, y, w = U._stack(panels[p], np.asarray(idx[i:i + U.BATCH]), rng)
                w = (w == 1.0).astype(np.float32)
                tot, n = tot + U._loss(model, x, y, w).item() * w.sum(), n + w.sum()
    model.train()
    return tot / max(n, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True, help="weights to start from (a production or holdout U-Net)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--months", choices=("holdout", "all"), default="holdout")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--w-other", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--exclude", default="", help="comma-separated panels left out of the training (a leave-family-out test)")
    args = ap.parse_args()
    U.THREADS = 1
    U._setup(args.seed)
    months = L.TRAIN_MONTHS if args.months == "holdout" else L.TRAIN_MONTHS + L.HOLD_MONTHS
    windows = W.union_windows(months)
    panels = {p: SpanPanel(p, W.by_day(windows[p]), args.w_other) for p in L.T2_PANELS if p not in args.exclude.split(",")}
    day_sets = {p: np.array(sorted(tp.windows)) for p, tp in panels.items()}
    valid = vsets = None
    if args.months == "holdout":   # the emulated Dec-Feb spans: reported, not used for any choice
        origins = L.emulated_origins()
        valid = {p: SpanPanel(p, W.by_day(origins[p]), 0.0, fixed=True) for p in L.T2_PANELS}
        vsets = {p: np.array(sorted(tp.windows)) for p, tp in valid.items()}
    model = U.load_model(args.init).to(U.DEVICE)
    model.train()
    rng = np.random.default_rng(args.seed)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    n_batches = sum(-(-len(v) // U.BATCH) for v in day_sets.values())
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * n_batches, pct_start=0.15)
    t0 = time.time()
    print(f"{sum(len(v) for v in day_sets.values())} training days with windows, {n_batches} batches per epoch, device {U.DEVICE.type}", flush=True)
    if valid:
        print(f"  start: span loss on the emulated Dec-Feb spans {span_loss(model, valid, vsets):.5f}", flush=True)
    for ep in range(args.epochs):
        tot, n = 0.0, 0
        for p, idx in U._batches(panels, day_sets, rng):
            loss = U._loss(model, *U._stack(panels[p], idx, rng))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot, n = tot + loss.item(), n + 1
        msg = f"  epoch {ep + 1}/{args.epochs}: train {tot / n:.5f}"
        if valid:
            msg += f"  Dec-Feb span loss {span_loss(model, valid, vsets):.5f}"
        print(msg + f"  ({(time.time() - t0) / 60:.1f} min)", flush=True)
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, str(L.LATE / f"{args.out}.pt"))
    print("done ->", L.LATE / f"{args.out}.pt")


if __name__ == "__main__":
    main()
