"""Adapting the span-aware U-Net to a scored month on its own released observations (2026-11-23).

The scored months are shifted against train. Their released observations can teach a span model their own
patterns without any hidden truth: blank a pseudo-span (span_pseudo.py: an 18-slot all-link span at a
selector-like origin found on the masked layer, clear of the real spans) and train on the observations that were
blanked. Batches of such self-supervised days from March and April are mixed with the supervised train days of
span_unet_x.py, starting from a span-aware model.

    python experiments/late/span_adapt.py --init x.pt --out <name> [--months holdout|all] [--days even|all] [--epochs 5]
                                          [--periods March,April | Dec-Feb]
--months is the supervised part (holdout = Jun-Nov), --days the days of the adapted months used (even: a test of
the idea, scored on the odd days with `span_pseudo.py --days odd`; all: the model for the candidate).
--periods Dec-Feb adapts to the emulated holdout months instead (their masked layer only, no truth), so the
adapted model can be scored on the emulated spans with truth.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import span_lib as L  # noqa: E402
import span_pseudo as P  # noqa: E402
import span_unet_x as X  # noqa: E402
import span_windows as W  # noqa: E402

from tfb import t1_unet as U  # noqa: E402


class PseudoPanel:
    """A scored month of one panel: a sample blanks pseudo-spans in a day's masked layer; the loss is on their observed cells."""

    def __init__(self, panel: str, period: str, extras: tuple, parity: str):
        (self.split, self.sp, self.fl, self.oc, self.elig, _, _, _), cand = P.pseudo_windows(panel, period, parity)
        self.windows = W.by_day(cand)
        self.st = U.Static(panel)
        self.xin = X.Extras(panel, self.split, extras)

    def sample(self, d: int, rng: np.random.Generator):
        sp, fl, oc = self.sp[d], self.fl[d], self.oc[d]
        blank, cond = np.zeros(L.S, bool), np.zeros(L.S, np.int8)
        for T, c in W.pick(self.windows[d], rng):
            blank[T + 1:T + 1 + L.SPAN] = True
            cond[T + 1:T + 1 + L.SPAN] = 1 if c == "queue_onset" else 2
        b = blank[:, None]
        vm, qm = np.where(b, np.nan, sp), np.where(b, np.nan, fl)
        x, iv, iq = U.day_input(vm, qm, np.where(b, np.nan, oc), self.st)
        x = np.concatenate([x, self.xin.channels(d, vm, qm, iv, iq, cond)])
        y = np.nan_to_num(np.stack([sp / U.V_UNIT - iv, fl / self.st.q_unit - iq]), nan=0.0).astype(np.float32)
        w = (b & np.isfinite(sp) & np.isfinite(fl) & self.elig[d]).astype(np.float32)
        return x, y, w


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--months", choices=("holdout", "all"), default="holdout")
    ap.add_argument("--days", choices=("even", "all"), default="even")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--repeat", type=int, default=2, help="times a scored-month day is drawn per epoch")
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--periods", default="March,April")
    args = ap.parse_args()
    U.THREADS = 1
    U._setup(args.seed)
    blob = torch.load(args.init, map_location="cpu")
    extras = tuple(blob["extras"])
    model = U.UNet(blob["state"]["e1.net.0.weight"].shape[1], blob["state"]["e1.net.0.weight"].shape[0])
    model.load_state_dict(blob["state"])
    model = model.to(U.DEVICE).train()
    months = L.TRAIN_MONTHS if args.months == "holdout" else L.TRAIN_MONTHS + L.HOLD_MONTHS
    windows, aug = W.union_windows(months), W.hourly_windows(months)
    pools, day_sets = {}, {}
    for p in L.T2_PANELS:
        pools[("train", p)] = X.SpanPanelX(p, W.by_day(windows[p]), 0.1, extras, aug=W.by_day(aug[p]), p_aug=0.5)
        day_sets[("train", p)] = np.array(sorted(set(pools[("train", p)].windows) | set(pools[("train", p)].aug)))
        for period in args.periods.split(","):
            pp = PseudoPanel(p, period, extras, args.days)
            pools[(period, p)] = pp
            day_sets[(period, p)] = np.repeat(np.array(sorted(pp.windows)), args.repeat)
    n_sup = sum(len(v) for k, v in day_sets.items() if k[0] == "train")
    n_self = sum(len(v) for k, v in day_sets.items() if k[0] != "train")
    valid = vsets = None
    if args.months == "holdout":
        origins = L.emulated_origins()
        valid = {p: X.SpanPanelX(p, W.by_day(origins[p]), 0.0, extras, fixed=True) for p in L.T2_PANELS}
        vsets = {p: np.array(sorted(tp.windows)) for p, tp in valid.items()}
    rng = np.random.default_rng(args.seed)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    n_batches = sum(-(-len(v) // U.BATCH) for v in day_sets.values())
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * n_batches, pct_start=0.15)
    t0 = time.time()
    print(f"extras {extras}; per epoch {n_sup} supervised train days and {n_self} self-supervised day draws of {args.periods} "
          f"({args.days} days), {n_batches} batches, device {U.DEVICE.type}", flush=True)
    if valid:
        print(f"  start: span loss on the emulated Dec-Feb spans {X.span_loss(model, valid, vsets):.5f}", flush=True)
    for ep in range(args.epochs):
        tot = {"train": [0.0, 0], "self": [0.0, 0]}
        for key, idx in U._batches(pools, day_sets, rng):
            loss = U._loss(model, *U._stack(pools[key], idx, rng))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            t = tot["train" if key[0] == "train" else "self"]
            t[0], t[1] = t[0] + loss.item(), t[1] + 1
        msg = f"  epoch {ep + 1}/{args.epochs}: supervised {tot['train'][0] / max(tot['train'][1], 1):.5f}  self-supervised {tot['self'][0] / max(tot['self'][1], 1):.5f}"
        if valid:
            msg += f"  Dec-Feb span loss {X.span_loss(model, valid, vsets):.5f}"
        print(msg + f"  ({(time.time() - t0) / 60:.1f} min)", flush=True)
        torch.save({"state": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "extras": list(extras)}, str(L.LATE / f"{args.out}.pt"))
    print("done ->", L.LATE / f"{args.out}.pt")


if __name__ == "__main__":
    main()
