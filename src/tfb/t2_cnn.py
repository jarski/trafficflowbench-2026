"""Task 2 ongoing: a convolutional model over the corridor, a second model family for the ongoing average.

One sample is a window. Per link it holds the 12-slot history T-60..T-5 and the slot at T (the masked
layer, with blank links filled causally as in the production features) of speed, flow and density, plus
missing and queue flags, static link attributes and the time of day. A 1-D residual convolution along
the corridor (receptive field about +-36 links) outputs queue logits for every link and horizon step, so it
scores far links like the public all-link model and unlike our candidate-cell LightGBM. It is trained on
the train split only, on the CPU with fixed seeds, so it reproduces exactly.

    python -m tfb.t2_cnn --holdout        # train Jun-Nov, report Dec-Feb loss per epoch, save the model
    python -m tfb.t2_cnn --oof            # leave-one-month-out probabilities for the bench windows (experiments/t2_cnn_bench.py)
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from tfb import t2_ongoing as og
from tfb.paths import CACHE
from tfb.t1_interp import weekday
from tfb.t2_features import K_OCC, at_origin, impute_origin, static_features
from tfb.t2_onset import T2_PANELS
from tfb.t2_rules import vcut
from tfb.task1_io import days, load, meta

HISTORY, STEPS = 12, og.STEPS
SLOTS = HISTORY + 1                      # T-60..T-5 plus T
STRIDE, CAP = 3, 12                      # training windows: every 3rd valid origin, at most 12 per day (22,500 windows)
WIDTH, DILATIONS = 64, (1, 2, 4, 8, 1, 2)
EPOCHS, LR, BATCH, SEED = 12, 2e-3, 16, 3
THREADS = 10
# Production trains on the CPU (deterministic). TFB_CNN_DEVICE=mps speeds up experiments but is not bit-reproducible.
DEVICE = torch.device(os.environ.get("TFB_CNN_DEVICE", "cpu"))
WEIGHT = 0.0                             # weight of this model in the ongoing average; 0 = off (neutral on the leaderboard, v46)
# Static link attributes and the time of day let the model memorise which links queue when in the train months:
# with them the average lost -0.0006 public at weight 0.33 (v44), without them +0.0001 (v46); LOG 2026-10-20.
STATIC_CHANNELS = os.environ.get("TFB_CNN_STATIC", "0") == "1"
N_STATIC, N_TIME = 8, 3


def c_in() -> int:
    return 5 * SLOTS + (N_STATIC + N_TIME if STATIC_CHANNELS else 0)


C_IN = c_in()


# ------------------------------------------------------------------ inputs
def link_static(panel: str) -> np.ndarray:
    """[N_STATIC, L] static link attributes, roughly unit scale."""
    st = static_features(panel)
    return np.stack([st["lanes"] / 6, st["cap_lane"] / 2500, st["cap_dn_ratio"] - 1, st["min_cap_dn5_ratio"] - 1,
                     st["on_ramps"] > 0, st["off_ramps"] > 0, st["pos"], st["length"]]).astype(np.float32)


def window_tensor(panel: str, hist: dict, at_t: dict, T: int, wd: int, static: np.ndarray) -> np.ndarray:
    """[C_IN, L] input of one window. hist[ch]: [12, L] complete history; at_t[ch]: [L] at T, NaN where blank."""
    m = meta(panel)
    vf, cap, kc = (np.asarray(m[k], np.float32) for k in ("free_speed_kmh", "capacity_vph", "critical_density"))
    chans = []
    miss = None
    for ch, scale in (("speed", vf), ("flow", cap), ("occ", kc / K_OCC)):
        x = np.vstack([hist[ch], at_t[ch][None]]).astype(np.float32) / scale        # [13, L]
        if ch == "speed":
            miss = np.isnan(x)
            x_last = np.where(miss[-1], hist["speed"][-1] / vf, x[-1])              # blank at T: fall back to T-5
            q = np.nan_to_num(np.vstack([x[:-1], x_last[None]]), nan=np.inf) <= (vcut(panel) / vf)
        chans.append(np.nan_to_num(x, nan=0.0))
    chans += [miss.astype(np.float32), q.astype(np.float32)]
    if STATIC_CHANNELS:
        L = static.shape[1]
        tod = np.full((N_TIME, L), 0, np.float32)
        tod[0], tod[1], tod[2] = np.sin(2 * np.pi * T / 288), np.cos(2 * np.pi * T / 288), float(wd >= 5)
        chans += [static, tod]
    return np.vstack(chans).astype(np.float32)


class TrainPanel:
    """Train arrays of one panel; windows as (input, label, weight) tensors."""

    def __init__(self, panel: str):
        self.panel = panel
        self.a = load(panel, "train", "true_speed", "true_flow", "true_occ", "speed", "flow", "occ", "eligible")
        self.q = (self.a["true_speed"] <= vcut(panel)) & self.a["eligible"]
        dl = days(panel, "train")["days"]
        self.months = np.array([d[:7] for d in dl])
        self.wd = weekday(dl)
        self.static = link_static(panel)

    def window(self, d: int, T: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        a = self.a
        hist = {ch: a[f"true_{ch}"][d, T - HISTORY:T] for ch in ("speed", "flow", "occ")}
        at_t = impute_origin(at_origin(a, d, T), hist, "ongoing", self.panel, T, self.months[d])
        x = window_tensor(self.panel, hist, at_t, T, self.wd[d], self.static)
        return x, self.q[d, T + 1:T + 1 + STEPS], a["eligible"][d, T + 1:T + 1 + STEPS]

    def samples(self, exclude: tuple = (), stride: int | None = None, cap: int | None = None) -> dict:
        """Stacked tensors of this panel's training windows outside the excluded months."""
        cands = [(d, T) for d, T in og.candidates(self.q, self.a["eligible"], stride or STRIDE, cap or CAP)
                 if self.months[d] not in exclude]
        xs, ys, ws = zip(*[self.window(d, T) for d, T in cands]) if cands else ((), (), ())
        return {"x": np.stack(xs).astype(np.float16), "y": np.stack(ys), "w": np.stack(ws),
                "month": np.array([self.months[d] for d, T in cands]), "keys": [f"{self.panel}|{d}|{T}" for d, T in cands]}


# ------------------------------------------------------------------ model
class Block(nn.Module):
    def __init__(self, width: int, dilation: int):
        super().__init__()
        self.norm = nn.GroupNorm(8, width)
        self.conv1 = nn.Conv1d(width, width, 5, padding=2 * dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(width, width, 1)
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        return x + self.drop(self.conv2(F.gelu(self.conv1(self.norm(x)))))


class Net(nn.Module):
    def __init__(self, c_in: int = C_IN, width: int = WIDTH):
        super().__init__()
        self.inp = nn.Conv1d(c_in, width, 5, padding=2)
        self.blocks = nn.ModuleList([Block(width, d) for d in DILATIONS])
        self.out = nn.Conv1d(width, STEPS, 1)

    def forward(self, x):                                   # [B, C, L] -> [B, STEPS, L] logits
        h = F.gelu(self.inp(x))
        for b in self.blocks:
            h = b(h)
        return self.out(h)


def _setup(seed: int) -> None:
    torch.manual_seed(seed)
    torch.set_num_threads(THREADS)
    torch.use_deterministic_algorithms(DEVICE.type == "cpu")


def _batches(data: dict, rng: np.random.Generator, batch: int) -> list:
    out = []
    for panel, s in data.items():
        idx = rng.permutation(len(s["x"]))
        out += [(panel, idx[i:i + batch]) for i in range(0, len(idx), batch)]
    rng.shuffle(out)
    return out


def _loss(model: Net, s: dict, idx: np.ndarray) -> torch.Tensor:
    x = torch.from_numpy(s["x"][idx].astype(np.float32)).to(DEVICE)
    y = torch.from_numpy(s["y"][idx].astype(np.float32)).to(DEVICE)
    w = torch.from_numpy(s["w"][idx].astype(np.float32)).to(DEVICE)
    return F.binary_cross_entropy_with_logits(model(x), y, weight=w, reduction="sum") / w.sum().clamp(min=1)


def fit(data: dict, epochs: int = EPOCHS, seed: int = SEED, valid: dict | None = None, log: bool = True) -> Net:
    """Train on {panel: samples}; `valid` (same shape) only reports its loss per epoch."""
    _setup(seed)
    rng = np.random.default_rng(seed)
    model = Net(c_in(), WIDTH).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    n_batches = sum(-(-len(s["x"]) // BATCH) for s in data.values())
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=epochs * n_batches, pct_start=0.15)
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        tot, n = 0.0, 0
        for panel, idx in _batches(data, rng, BATCH):
            loss = _loss(model, data[panel], idx)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot, n = tot + loss.item(), n + 1
        msg = f"  epoch {ep + 1}/{epochs}: train {tot / n:.4f}"
        if valid is not None:
            msg += f"  valid {evaluate(model, valid):.4f}"
        if log:
            print(msg + f"  ({(time.time() - t0) / 60:.1f} min)", flush=True)
    model.eval()
    return model


def evaluate(model: Net, data: dict) -> float:
    model.eval()
    tot, n = 0.0, 0.0
    with torch.no_grad():
        for s in data.values():
            for i in range(0, len(s["x"]), 64):
                idx = np.arange(i, min(i + 64, len(s["x"])))
                w = s["w"][idx].sum()
                tot, n = tot + _loss(model, s, idx).item() * w, n + w
    model.train()
    return tot / max(n, 1)


def predict(model: Net, x: np.ndarray) -> np.ndarray:
    """[B, STEPS, L] probabilities from [B, C, L] inputs."""
    model.eval()
    dev = next(model.parameters()).device
    with torch.no_grad():
        return torch.sigmoid(model(torch.from_numpy(np.asarray(x, np.float32)).to(dev))).cpu().numpy()


def save(model: Net, path) -> None:
    torch.save(model.cpu().state_dict(), str(path))


def load_model(path) -> Net:
    state = torch.load(str(path), map_location="cpu")
    model = Net(state["inp.weight"].shape[1], state["inp.weight"].shape[0])
    model.load_state_dict(state)
    return model.eval()


# ------------------------------------------------------------------ production
def window_input(ctx: dict, w, T: int) -> np.ndarray:
    """[C_IN, L] input of a released window from the split arrays: history T-60..T-5 and the masked slot at T."""
    a, d = ctx["a"], ctx["days"].index(w.date)
    hist = {ch: a[ch][d, T - HISTORY:T] for ch in ("speed", "flow", "occ")}
    at_t = impute_origin(at_origin(a, d, T), hist, "ongoing", ctx["panel"], T, None)
    return window_tensor(ctx["panel"], hist, at_t, T, ctx["wd"][d], ctx["cnn_static"])


def window_probs(model: Net, ctx: dict, w, T: int) -> np.ndarray:
    """[step, link] queue probabilities."""
    return predict(model, window_input(ctx, w, T)[None])[0]


def train_data(exclude: tuple = ()) -> dict:
    data = {}
    for panel in T2_PANELS:
        data[panel] = TrainPanel(panel).samples(exclude)
        print(f"  {panel}: {len(data[panel]['x'])} windows", flush=True)
    return data


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--holdout", action="store_true", help="train on Jun-Nov, report the Dec-Feb loss per epoch")
    ap.add_argument("--oof", action="store_true", help="leave-one-month-out models, saved under cache/bench/t2_cnn_without_<month>.pt")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    args = ap.parse_args()
    t0 = time.time()
    data = train_data()
    months = sorted({m for s in data.values() for m in np.unique(s["month"])})
    if args.holdout:
        hold = tuple(m for m in months if m >= "2030-12")
        split = lambda keep: {p: {k: (v[np.isin(s["month"], hold) == keep] if k != "keys" else v) for k, v in s.items()} for p, s in data.items()}
        model = fit(split(False), args.epochs, valid=split(True))
        save(model, CACHE / "bench" / "t2_cnn_holdout.pt")
    if args.oof:
        for m in months:
            path = CACHE / "bench" / f"t2_cnn_without_{m}.pt"
            if path.exists():
                continue
            print(f"fold without {m}", flush=True)
            fold = {p: {k: (v[s["month"] != m] if k != "keys" else v) for k, v in s.items()} for p, s in data.items()}
            save(fit(fold, args.epochs), path)
    print(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
