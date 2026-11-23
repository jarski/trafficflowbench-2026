"""Span-aware U-Net with extra input channels: the month's own profile, ramp flows, the window condition (2026-11-23).

Same fine-tuning as span_unet_ft.py (the production U-Net trained further on days with all-link blank spans at
selector-chosen origins), with input channels the production model does not have. Their first-layer weights start
at zero, so training starts from the production model exactly.

    prof   2 channels: the month's time-of-day profile of the link (same day type, the day left out, observed cells
           of the masked layer) shifted to the day's values around each gap, minus the plain interpolation; speed
           and flow. The scored month's own recurrence, which the grid of one day cannot show.
    ramp   4 channels: on-ramp and off-ramp flow attached to the link (valid readings: eligible, pct >= 75) and a
           flag for each. Ramp flows are not blanked inside the spans of the scored months.
    cond   2 channels: the cell lies in the span of an onset / an ongoing window (window_index.csv gives it).

    python experiments/late/span_unet_x.py train --init <w.pt> --out <name> --extra prof,ramp,cond [--months holdout|all] [--epochs 8]
    python experiments/late/span_unet_x.py holdout --models a.pt,b.pt --name <family>      # world B -> cache/late/pred
    python experiments/late/span_unet_x.py scored --models a.pt,b.pt --out <dir>           # validation, private
The extras of a model are stored in its file.
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
from tfb.t1_interp import group_profile_by_day, weekday  # noqa: E402
from tfb.task1_io import days, meta  # noqa: E402

N_EXTRA = {"prof": 2, "ramp": 4, "cond": 2}
R_UNIT = 100.0


class Extras:
    """Extra input channels of one panel and split, built from a masked layer."""

    def __init__(self, panel: str, split: str, extras: tuple, speed: np.ndarray | None = None, flow: np.ndarray | None = None):
        self.extras, self.st = extras, U.Static(panel)
        m = meta(panel)
        self.L = len(m["links"])
        if "prof" in extras:
            info = days(panel, split)
            wd = weekday(info["days"])
            groups = np.array([f"{dd[:7]}|{w >= 5}" for dd, w in zip(info["days"], wd)])
            if speed is None:
                a = L._ORIG_LOAD(panel, split, "speed", "flow")
                speed, flow = a["speed"], a["flow"]
            self.ms, self.mf = group_profile_by_day(speed, groups), group_profile_by_day(flow, groups)
        if "ramp" in extras:
            r = L._ORIG_LOAD(panel, split, "ramp_flow", "ramp_pct", "ramp_eligible")
            self.rvalid = r["ramp_eligible"] & (r["ramp_pct"] >= 75) & np.isfinite(r["ramp_flow"])
            self.rflow = np.where(self.rvalid, r["ramp_flow"], 0.0).astype(np.float32)
            li, typ = np.asarray(m["ramp_link_index"]), np.asarray([t.upper() for t in m["ramp_type"]])
            self.m_on = np.zeros((len(li), self.L), np.float32)
            self.m_off = np.zeros((len(li), self.L), np.float32)
            for j in range(len(li)):
                if li[j] >= 0:
                    (self.m_on if typ[j] in ("OR", "ON") else self.m_off)[j, li[j]] = 1.0

    def channels(self, d: int, v: np.ndarray, q: np.ndarray, iv: np.ndarray, iq: np.ndarray, cond: np.ndarray | None) -> np.ndarray:
        """[n, 288, L] for day d: v, q are the day's masked view, iv / iq its interpolation in model units,
        cond [288] the window condition of each slot (0 none, 1 onset span, 2 ongoing span)."""
        out = []
        if "prof" in self.extras:
            for x, prof, base, unit in ((v, self.ms[d], iv, U.V_UNIT), (q, self.mf[d], iq, self.st.q_unit)):
                obs = np.isfinite(v) & np.isfinite(q)
                shift = prof + U.interp_time(np.where(obs, x, np.nan) - prof)
                out.append(np.where(np.isfinite(shift), shift / unit - base, 0.0))
        if "ramp" in self.extras:
            val = self.rvalid[d].astype(np.float32)
            out += [self.rflow[d] @ self.m_on / R_UNIT, self.rflow[d] @ self.m_off / R_UNIT,
                    (val @ self.m_on > 0).astype(np.float32), (val @ self.m_off > 0).astype(np.float32)]
        if "cond" in self.extras:
            c = np.zeros(L.S, np.int8) if cond is None else cond
            out += [np.broadcast_to((c == 1)[:, None], (L.S, self.L)), np.broadcast_to((c == 2)[:, None], (L.S, self.L))]
        if not out:
            return np.zeros((0, L.S, self.L), np.float32)
        return np.stack(out).astype(np.float32)


def cond_of_day(n_days: int, origins) -> np.ndarray:
    """[day, 288]: 1 in the span of an onset window, 2 in the span of an ongoing window, 0 elsewhere."""
    c = np.zeros(n_days * L.S, np.int8)
    for d, T, cnd in origins:
        t = d * L.S + T
        c[t + 1:t + 1 + L.SPAN] = 1 if cnd == "queue_onset" else 2
    return c.reshape(n_days, L.S)


class SpanPanelX(U.TrainPanel):
    """Train states of one panel; a sample hides a Bernoulli mask, blanks spans after window origins and adds the extras."""

    def __init__(self, panel: str, windows: dict, w_other: float, extras: tuple, fixed: bool = False,
                 aug: dict | None = None, p_aug: float = 0.0):
        super().__init__(panel)
        self.windows, self.w_other, self.fixed = windows, w_other, fixed
        self.aug, self.p_aug = aug or {}, p_aug          # hourly windows used instead with probability p_aug
        self.xin = Extras(panel, "train", extras)

    def sample(self, d: int, rng: np.random.Generator):
        tv, tq, to = self.a["true_speed"][d], self.a["true_flow"][d], self.a["true_occ"][d]
        hide = (rng.random(tv.shape) < U.RATES[rng.integers(3)]) & self.ok[d]
        blank, hist, cond = np.zeros(L.S, bool), np.zeros(L.S, bool), np.zeros(L.S, np.int8)
        wins = self.windows.get(d, [])
        if self.p_aug > 0 and self.aug.get(d) and (not wins or rng.random() < self.p_aug):
            wins = self.aug[d]
        for T, c in (wins if self.fixed else W.pick(wins, rng)):
            blank[T + 1:T + 1 + L.SPAN] = True
            cond[T + 1:T + 1 + L.SPAN] = 1 if c == "queue_onset" else 2
            hist[max(T - L.HIST, 0):T] = True
        hist &= ~blank
        hide &= ~hist[:, None]
        gone = hide | blank[:, None]
        vm, qm = np.where(gone, np.nan, tv), np.where(gone, np.nan, tq)
        x, iv, iq = U.day_input(vm, qm, np.where(gone, np.nan, to), self.st)
        x = np.concatenate([x, self.xin.channels(d, vm, qm, iv, iq, cond)])
        y = np.nan_to_num(np.stack([tv / U.V_UNIT - iv, tq / self.st.q_unit - iq]), nan=0.0).astype(np.float32)
        w = np.where(blank[:, None] & self.ok[d], 1.0, np.where(hide, self.w_other, 0.0)).astype(np.float32)
        return x, y, w


def span_loss(model, panels: dict, day_sets: dict, seed: int = 123) -> float:
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


def widen(state: dict, n_extra: int) -> dict:
    """Production weights with zero weights for the extra input channels."""
    w = state["e1.net.0.weight"]
    if w.shape[1] == U.C_IN and n_extra:
        state = dict(state)
        state["e1.net.0.weight"] = torch.cat([w, torch.zeros(w.shape[0], n_extra, *w.shape[2:])], dim=1)
    return state


def load_x(path) -> tuple[U.UNet, tuple]:
    blob = torch.load(str(path), map_location="cpu")
    state, extras = (blob["state"], tuple(blob["extras"])) if "state" in blob else (blob, ())
    model = U.UNet(state["e1.net.0.weight"].shape[1], state["e1.net.0.weight"].shape[0])
    model.load_state_dict(state)
    return model.eval(), extras


def train(args) -> None:
    extras = tuple(e for e in args.extra.split(",") if e)
    U.THREADS = 1
    U._setup(args.seed)
    months = L.TRAIN_MONTHS if args.months == "holdout" else L.TRAIN_MONTHS + L.HOLD_MONTHS
    windows = W.union_windows(months)
    excl = args.exclude.split(",")
    aug = W.hourly_windows(months) if args.aug > 0 else {p: [] for p in L.T2_PANELS}
    panels = {p: SpanPanelX(p, W.by_day(windows[p]), args.w_other, extras, aug=W.by_day(aug[p]), p_aug=args.aug)
              for p in L.T2_PANELS if p not in excl}
    day_sets = {p: np.array(sorted(set(tp.windows) | (set(tp.aug) if args.aug > 0 else set()))) for p, tp in panels.items()}
    valid = vsets = None
    if args.months == "holdout":   # the emulated Dec-Feb spans: reported, not used for any choice
        origins = L.emulated_origins()
        valid = {p: SpanPanelX(p, W.by_day(origins[p]), 0.0, extras, fixed=True) for p in L.T2_PANELS}
        vsets = {p: np.array(sorted(tp.windows)) for p, tp in valid.items()}
    n_extra = sum(N_EXTRA[e] for e in extras)
    blob = torch.load(args.init, map_location="cpu")
    state = widen(blob["state"] if "state" in blob else blob, n_extra)
    model = U.UNet(U.C_IN + n_extra, state["e1.net.0.weight"].shape[0])
    model.load_state_dict(state)
    model = model.to(U.DEVICE).train()
    rng = np.random.default_rng(args.seed)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    n_batches = sum(-(-len(v) // U.BATCH) for v in day_sets.values())
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * n_batches, pct_start=0.15)
    t0 = time.time()
    print(f"extras {extras}; {sum(len(v) for v in day_sets.values())} training days with windows, {n_batches} batches per epoch, device {U.DEVICE.type}", flush=True)
    if valid:
        print(f"  start: span loss on the emulated Dec-Feb spans {span_loss(model, valid, vsets):.5f}", flush=True)
        if excl != [""]:
            ex = {p: vsets[p] for p in excl}
            print(f"  start: the same on the left-out panels {span_loss(model, valid, ex):.5f}", flush=True)
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
            if excl != [""]:
                msg += f"  left-out panels {span_loss(model, valid, {p: vsets[p] for p in excl}):.5f}"
        print(msg + f"  ({(time.time() - t0) / 60:.1f} min)", flush=True)
        torch.save({"state": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "extras": list(extras)}, str(L.LATE / f"{args.out}.pt"))
    print("done ->", L.LATE / f"{args.out}.pt")


def predict(models: list, extras: tuple, panel: str, split: str, speed, flow, occ, origins, day_sel=None):
    """[day, 288, link] speed and flow from a masked layer; the models' corrections are averaged before the clip."""
    st = U.Static(panel)
    xin = Extras(panel, split, extras, speed, flow)
    cond = cond_of_day(speed.shape[0], origins)
    sel = np.ones(speed.shape[0], bool) if day_sel is None else day_sel
    sp, fl = np.full(speed.shape, np.nan, np.float32), np.full(speed.shape, np.nan, np.float32)
    with torch.no_grad():
        for d in np.flatnonzero(sel):
            x, iv, iq = U.day_input(speed[d], flow[d], occ[d], st)
            if extras:
                x = np.concatenate([x, xin.channels(d, speed[d], flow[d], iv, iq, cond[d])])
            xt = torch.from_numpy(x[None]).to(U.DEVICE)
            y = np.mean([m(xt)[0].cpu().numpy() for m in models], axis=0)
            sp[d] = np.clip((iv + y[0]) * U.V_UNIT, 1.0, 180.0)
            fl[d] = np.clip((iq + y[1]) * st.q_unit, 0.0, None)
    return sp, fl


def load_models(paths: str):
    loaded = [load_x(p) for p in paths.split(",")]
    extras = loaded[0][1]
    assert all(e == extras for _, e in loaded), "models with different extras"
    return [m.to(U.DEVICE) for m, _ in loaded], extras


def holdout(args) -> None:
    models, extras = load_models(args.models)
    origins = L.emulated_origins()
    for p in L.T2_PANELS:
        lay = L.emulated_layer(p, origins[p])
        hold = np.isin(L.day_months(p), L.HOLD_MONTHS)
        sp, fl = predict(models, extras, p, "train", lay["speed"], lay["flow"], lay["occ"], origins[p], hold)
        np.savez_compressed(L.LATE / "pred" / f"B_{args.name}_{p}.npz", speed=sp[hold], flow=fl[hold])
        print(p, "done", flush=True)
    for p in set(L.PANELS) - set(L.T2_PANELS):       # no spans there: NaN arrays keep the evaluation code uniform
        hold = np.isin(L.day_months(p), L.HOLD_MONTHS)
        shape = (int(hold.sum()), L.S, len(meta(p)["links"]))
        np.savez_compressed(L.LATE / "pred" / f"B_{args.name}_{p}.npz", speed=np.full(shape, np.nan, np.float32), flow=np.full(shape, np.nan, np.float32))


def scored(args) -> None:
    models, extras = load_models(args.models)
    out = L.LATE / args.out
    out.mkdir(exist_ok=True)
    for split in ("validation", "private"):
        for p in L.T2_PANELS:
            a = L._ORIG_LOAD(p, split, "speed", "flow", "occ")
            sp, fl = predict(models, extras, p, split, a["speed"], a["flow"], a["occ"], L.official_origins(p, split))
            np.savez_compressed(out / f"{split}_{p}.npz", speed=sp, flow=np.maximum(fl, 1.0))
            print(split, p, "done", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--init", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--extra", default="prof,ramp,cond")
    t.add_argument("--months", choices=("holdout", "all"), default="holdout")
    t.add_argument("--epochs", type=int, default=8)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--w-other", type=float, default=0.1)
    t.add_argument("--seed", type=int, default=5)
    t.add_argument("--exclude", default="")
    t.add_argument("--aug", type=float, default=0.0, help="probability that a sample day uses the hourly windows (span_windows.hourly_windows)")
    h = sub.add_parser("holdout")
    h.add_argument("--models", required=True)
    h.add_argument("--name", required=True)
    s = sub.add_parser("scored")
    s.add_argument("--models", required=True)
    s.add_argument("--out", required=True)
    args = ap.parse_args()
    U.THREADS = 1
    torch.set_num_threads(1)
    {"train": train, "holdout": holdout, "scored": scored}[args.cmd](args)


if __name__ == "__main__":
    main()
