"""Task 1: a 2-D U-Net over each day's [time, link] grid, a third model family for the Task 1 average.

Input per day: the masked view (speed, flow, density) with an observed-cell flag, plus each link's linear
time interpolation of speed and flow. Output: corrections to the interpolation at every cell in the scorer's
units (speed / 25 km/h, flow per lane / 600 vph). Trained on the train split's unmasked states with fresh
Bernoulli masks every epoch at the organizer's regime rates (20 / 30 / 50% of eligible cells). Training uses
the GPU when there is one (TFB_UNET_DEVICE=cpu forces the deterministic CPU path, about 4.5 hours); the
trained weights are saved beside the submission and can be loaded instead (pipeline_final --unet-weights).

    python -m tfb.t1_unet --holdout     # train Jun-Nov (Dec-Feb loss per epoch), save cache/bench/t1_unet_holdout.pt
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

from tfb.paths import CACHE, PANELS
from tfb.t1_model import K_OCC
from tfb.task1_io import days, load, meta

V_UNIT, Q_UNIT = 25.0, 600.0           # the scorer's speed scale and per-lane flow scale
RATES = (0.2, 0.3, 0.5)                 # masking regimes R1-R3, share of eligible cells hidden
WIDTH, EPOCHS, LR, BATCH, SEED = 32, 20, 1e-3, 4, 5
C_IN, C_OUT = 6, 2
THREADS = 10


def _default_device() -> str:
    """The GPU when there is one (about 25 min of training; two runs on an Apple GPU gave bit-identical weights, which
    PyTorch does not guarantee in general), else the CPU (deterministic, ~4.5 h)."""
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


DEVICE = torch.device(os.environ.get("TFB_UNET_DEVICE") or _default_device())
WEIGHT = 0.5                            # weight of this model at the Task 1 target cells; 0 = off. Public +0.0021 (v47), 0.33: +0.0018 (v48)
SEEDS = (5, 6)                          # production trains one model per seed and averages their corrections. Second seed: public +0.0002 (v50), holdout +0.0004


# ------------------------------------------------------------------ inputs
def interp_time(x: np.ndarray) -> np.ndarray:
    """[288, L]: linear interpolation along time per link over NaN cells, ends held; all-NaN links stay NaN."""
    return pd.DataFrame(x).interpolate(axis=0, limit_direction="both").to_numpy(np.float32)


class Static:
    def __init__(self, panel: str):
        m = meta(panel)
        self.lanes = np.clip(np.asarray(m["lanes"], np.float32), 1.0, None)
        self.q_unit = Q_UNIT * self.lanes
        self.kcrit = np.asarray(m["critical_density"], np.float32)


def day_input(v: np.ndarray, q: np.ndarray, o: np.ndarray, st: Static) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """([C_IN, 288, L] input, interpolated speed / V_UNIT, interpolated flow / q_unit) from a day's masked view."""
    obs = np.isfinite(v) & np.isfinite(q)
    iv = interp_time(np.where(obs, v, np.nan)) / V_UNIT
    iq = interp_time(np.where(obs, q, np.nan)) / st.q_unit
    for arr, fallback in ((iv, 100.0 / V_UNIT), (iq, 0.5)):   # links without any observation that day
        bad = np.isnan(arr)
        if bad.any():
            arr[bad] = np.nanmean(arr) if (~bad).any() else fallback
    x = np.stack([np.where(obs, v / V_UNIT, 0.0), np.where(obs, q / st.q_unit, 0.0),
                  np.where(np.isfinite(o), o * K_OCC / st.kcrit, 0.0), obs, iv, iq]).astype(np.float32)
    return x, iv, iq


class TrainPanel:
    """Unmasked train states of one panel; samples hide a fresh Bernoulli mask at a regime rate."""

    def __init__(self, panel: str):
        self.panel = panel
        self.a = load(panel, "train", "true_speed", "true_flow", "true_occ", "eligible")
        self.months = np.array([d[:7] for d in days(panel, "train")["days"]])
        self.st = Static(panel)
        self.ok = self.a["eligible"] & np.isfinite(self.a["true_speed"]) & np.isfinite(self.a["true_flow"])

    def sample(self, d: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        tv, tq, to = self.a["true_speed"][d], self.a["true_flow"][d], self.a["true_occ"][d]
        hide = (rng.random(tv.shape) < RATES[rng.integers(3)]) & self.ok[d]
        x, iv, iq = day_input(np.where(hide, np.nan, tv), np.where(hide, np.nan, tq), np.where(hide, np.nan, to), self.st)
        y = np.nan_to_num(np.stack([tv / V_UNIT - iv, tq / self.st.q_unit - iq]), nan=0.0).astype(np.float32)
        return x, y, hide


# ------------------------------------------------------------------ model
class Block(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(c_in, c_out, 3, padding=1), nn.GroupNorm(8, c_out), nn.GELU(),
                                 nn.Conv2d(c_out, c_out, 3, padding=1), nn.GroupNorm(8, c_out), nn.GELU())

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.reduce = nn.Conv2d(c_in, c_out, 1)
        self.block = Block(2 * c_out, c_out)

    def forward(self, x, skip):
        x = self.reduce(F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False))
        return self.block(torch.cat([x, skip], dim=1))


class UNet(nn.Module):
    def __init__(self, c_in: int = C_IN, width: int = WIDTH, c_out: int = C_OUT):
        super().__init__()
        w = width
        self.e1, self.e2, self.e3, self.b = Block(c_in, w), Block(w, 2 * w), Block(2 * w, 4 * w), Block(4 * w, 8 * w)
        self.u3, self.u2, self.u1 = Up(8 * w, 4 * w), Up(4 * w, 2 * w), Up(2 * w, w)
        self.out = nn.Conv2d(w, c_out, 1)

    def forward(self, x):                                   # [B, C, 288, L] -> [B, C_OUT, 288, L]
        L = x.shape[-1]
        pad = (-L) % 8
        if pad:
            x = F.pad(x, (0, pad))
        e1 = self.e1(x)
        e2 = self.e2(F.max_pool2d(e1, 2))
        e3 = self.e3(F.max_pool2d(e2, 2))
        b = self.b(F.max_pool2d(e3, 2))
        y = self.out(self.u1(self.u2(self.u3(b, e3), e2), e1))
        return y[..., :L] if pad else y


def _setup(seed: int) -> None:
    torch.manual_seed(seed)
    torch.set_num_threads(THREADS)
    torch.use_deterministic_algorithms(DEVICE.type == "cpu")


def _loss(model: UNet, x: np.ndarray, y: np.ndarray, w: np.ndarray) -> torch.Tensor:
    """Scorer-weighted MSE on the hidden cells: 0.54 speed + 0.46 flow, in the scorer's units."""
    xt = torch.from_numpy(x).to(DEVICE)
    yt = torch.from_numpy(y).to(DEVICE)
    wt = torch.from_numpy(w.astype(np.float32)).to(DEVICE)[:, None]
    err = ((model(xt) - yt) ** 2) * wt
    n = wt.sum().clamp(min=1)
    return (0.54 * err[:, 0].sum() + 0.46 * err[:, 1].sum()) / n


def _batches(panels: dict, day_sets: dict, rng: np.random.Generator) -> list:
    out = []
    for p, idx in day_sets.items():
        idx = rng.permutation(idx)
        out += [(p, idx[i:i + BATCH]) for i in range(0, len(idx), BATCH)]
    rng.shuffle(out)
    return out


def _stack(tp: TrainPanel, idx: np.ndarray, rng: np.random.Generator):
    xs, ys, ws = zip(*[tp.sample(int(d), rng) for d in idx])
    return np.stack(xs), np.stack(ys), np.stack(ws)


def fit(panels: dict, day_sets: dict, epochs: int = EPOCHS, seed: int = SEED, valid: dict | None = None,
        log: bool = True, checkpoint=None) -> UNet:
    """panels: {name: TrainPanel}; day_sets: {name: training day indices}; valid: {name: day indices} for the loss.
    checkpoint: a path that receives the weights after every epoch, so a killed run keeps its last epoch."""
    _setup(seed)
    rng = np.random.default_rng(seed)
    model = UNet(C_IN, WIDTH).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    n_batches = sum(-(-len(v) // BATCH) for v in day_sets.values())
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=epochs * n_batches, pct_start=0.15)
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        tot, n = 0.0, 0
        for p, idx in _batches(panels, day_sets, rng):
            loss = _loss(model, *_stack(panels[p], idx, rng))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot, n = tot + loss.item(), n + 1
        msg = f"  epoch {ep + 1}/{epochs}: train {tot / n:.5f}"
        if valid is not None:
            msg += f"  valid {evaluate(model, panels, valid):.5f}"
        if log:
            print(msg + f"  ({(time.time() - t0) / 60:.1f} min)", flush=True)
        if checkpoint:
            torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, str(checkpoint))
    model.eval()
    return model


def evaluate(model: UNet, panels: dict, day_sets: dict, seed: int = 123) -> float:
    """Loss on fixed synthetic masks of the given days."""
    model.eval()
    rng = np.random.default_rng(seed)
    tot, n = 0.0, 0.0
    with torch.no_grad():
        for p, idx in day_sets.items():
            for i in range(0, len(idx), BATCH):
                x, y, w = _stack(panels[p], np.asarray(idx[i:i + BATCH]), rng)
                tot, n = tot + _loss(model, x, y, w).item() * w.sum(), n + w.sum()
    model.train()
    return tot / max(n, 1)


def save(model: UNet, path) -> None:
    torch.save(model.cpu().state_dict(), str(path))


def load_model(path) -> UNet:
    state = torch.load(str(path), map_location="cpu")
    model = UNet(C_IN, state["e1.net.0.weight"].shape[0])
    model.load_state_dict(state)
    return model.eval()


# ------------------------------------------------------------------ prediction
def predict_split(model: UNet | list | tuple, panel: str, split: str, day_sel: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """[day, 288, link] speed and flow for the selected days (NaN on other days), from the split's masked view.

    `model` may be a list of models (a seed average): their corrections are averaged before the clipping."""
    models = list(model) if isinstance(model, (list, tuple)) else [model]
    a = load(panel, split, "speed", "flow", "occ")
    st = Static(panel)
    n = a["speed"].shape[0]
    sel = np.ones(n, bool) if day_sel is None else day_sel
    speed = np.full(a["speed"].shape, np.nan, np.float32)
    flow = np.full(a["flow"].shape, np.nan, np.float32)
    devs = []
    for m in models:
        m.eval()
        devs.append(next(m.parameters()).device)
    with torch.no_grad():
        for d in np.flatnonzero(sel):
            x, iv, iq = day_input(a["speed"][d], a["flow"][d], a["occ"][d], st)
            xt = torch.from_numpy(x[None])
            y = np.mean([m(xt.to(dev))[0].cpu().numpy() for m, dev in zip(models, devs)], axis=0)
            speed[d] = np.clip((iv + y[0]) * V_UNIT, 1.0, 180.0)
            flow[d] = np.clip((iq + y[1]) * st.q_unit, 0.0, None)
    return speed, flow


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--holdout", action="store_true", help="train on Jun-Nov and report the Dec-Feb loss per epoch (default: all months)")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", default=None, help="default cache/bench/t1_unet_{holdout|all}[_s<seed>].pt")
    args = ap.parse_args()
    out = args.out or str(CACHE / "bench" / f"t1_unet_{'holdout' if args.holdout else 'all'}{'' if args.seed == SEED else f'_s{args.seed}'}.pt")
    t0 = time.time()
    panels = {p: TrainPanel(p) for p in PANELS}
    print(f"loaded {len(panels)} panels in {(time.time() - t0) / 60:.1f} min", flush=True)
    if args.holdout:
        train = {p: np.flatnonzero(tp.months < "2030-12") for p, tp in panels.items()}
        valid = {p: np.flatnonzero(tp.months >= "2030-12")[::4] for p, tp in panels.items()}   # every 4th day: a quick loss
    else:
        train = {p: np.arange(len(tp.months)) for p, tp in panels.items()}
        valid = None
    model = fit(panels, train, args.epochs, seed=args.seed, valid=valid, checkpoint=out)   # every epoch lands in `out`
    save(model, out)
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {out}")


if __name__ == "__main__":
    main()
