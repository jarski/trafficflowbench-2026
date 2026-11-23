"""Scores of stored holdout predictions inside and outside the emulated blank spans (Dec-Feb), and the cost of the
spans in total-score units.

A method is a stored family (interp, ours, public, unet, ...), `blend` (the production recipe: ours -> public 0.25
-> U-Net 0.5 -> plausible_speed, as tfb.pipeline_final.task1), or `base>mix`: the base everywhere and, at the span
targets only, a weighted mix such as `blend>unetft` or `blend>0.5*unetft+0.5*spec`, then plausible_speed again
(what a late-submission candidate does to v64).

The two LightGBM families are predicted on the days that carry a window only (span_predict.py --days window). On
the other days `pair` and `blend` take the project's cached holdout predictions of the same pair on the released
layer (cache/t1_holdout_base: ours with the full training configuration and 1 seed, public at 0.25); the two worlds
do not differ there apart from a small change of the month profiles. `pairc` / `blendc` use the cached pair on
every day (world A only: the project's own holdout numbers). `ours` and `public` alone are scored on the window
days only: RMSE, no scorer replica.

For each method: speed RMSE and per-lane flow RMSE pooled over the span targets and over the other targets, then
the exact scorer replicas (tfb.metrics) per panel and month in world B (the emulated month):

    B        as predicted
    B-excl   the span targets left out of S_state
    B-truth  truth at the span targets (S_physics: what the span cells cost)
    A        (--world-a, plain methods only) the same models on the released layer without spans, same target set

Totals are 0.35 S_state + 0.15 S_physics, panels -> family -> overall as on the leaderboard; each holdout month is
scored on its own (a scored split is one month) and the three months are averaged.

    python experiments/late/span_eval.py --methods interp,ours,public,unet,blend --world-a
    python experiments/late/span_eval.py --methods "blend,blend>interp,blend>unetft" --ref blend
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import span_lib as L  # noqa: E402

from tfb import t1_model as t1  # noqa: E402
from tfb.paths import CACHE, PANELS  # noqa: E402
from tfb.task1_io import days, meta  # noqa: E402

PRED = L.LATE / "pred"
W_PUBLIC, W_UNET = 0.25, 0.5


def stored(world: str, family: str, panel: str):
    z = np.load(PRED / f"{world}_{family}_{panel}.npz")
    return z["speed"], z["flow"]


def production_blend(panel: str, target: np.ndarray, ours, public, unet, w_public=W_PUBLIC, w_unet=W_UNET):
    """pipeline_final.task1: each family at its weight against the running average at target cells, then plausible_speed."""
    sp, fl = ours[0].copy(), ours[1].copy()
    for w, (ps, pf) in ((w_public, public), (w_unet, unet)):
        if w == 0:
            continue
        at = target & np.isfinite(ps)
        sp = np.where(at, (1 - w) * sp + w * ps, sp)
        fl = np.where(at, (1 - w) * fl + w * pf, fl)
    return t1.plausible_speed(panel, sp, fl), fl


class Holdout:
    """Truth, masks and day bookkeeping of one panel's Dec-Feb days in world B."""

    def __init__(self, panel: str, origins):
        self.panel = panel
        mo = L.day_months(panel)
        self.hold = np.isin(mo, L.HOLD_MONTHS)
        self.months = mo[self.hold]
        self.n_days = len(mo)
        a = L._ORIG_LOAD(panel, "train", "true_speed", "true_flow", "target", "eligible")
        self.lay = L.emulated_layer(panel, origins)
        self.ts, self.tf = a["true_speed"][self.hold], a["true_flow"][self.hold]
        self.target_full = self.lay["target"]                      # [all days]: B's target set
        self.target = self.lay["target"][self.hold]
        self.target_release = a["target"][self.hold]               # the release's (world A's) target set
        self.window_day = np.broadcast_to(self.lay["span"][self.hold].any(axis=1)[:, None, None], self.target.shape)
        self.span = np.broadcast_to(self.lay["span"][self.hold][:, :, None], self.target.shape)
        self.lanes = np.clip(np.asarray(meta(panel)["lanes"], float), 1.0, None)
        self.regime = np.asarray(days(panel, "train")["regime"])[self.hold]
        self.cond = np.zeros(self.lay["span"].shape, np.int8)       # per span slot: 1 onset, 2 ongoing
        self.pos = np.zeros(self.lay["span"].shape, np.int8)        # and the position 1..18
        self.win = np.full(self.lay["span"].shape, -1, np.int16)    # and the window number
        fc, fp, fw = self.cond.reshape(-1), self.pos.reshape(-1), self.win.reshape(-1)
        for i, (d, T, c) in enumerate(origins):
            t = d * L.S + T
            n = len(fc[t + 1:t + 1 + L.SPAN])
            fc[t + 1:t + 1 + L.SPAN] = 1 if c == "queue_onset" else 2
            fp[t + 1:t + 1 + L.SPAN] = np.arange(1, n + 1)
            fw[t + 1:t + 1 + L.SPAN] = i
        self.cond, self.pos, self.win = self.cond[self.hold], self.pos[self.hold], self.win[self.hold]

    def full(self, x: np.ndarray) -> np.ndarray:
        """[holdout day, ...] -> [all train days, ...] with NaN elsewhere (what tfb.metrics expects)."""
        out = np.full((self.n_days,) + x.shape[1:], np.nan, np.float32)
        out[self.hold] = x
        return out

    def sse(self, sp: np.ndarray, fl: np.ndarray, mask: np.ndarray) -> tuple[int, float, float]:
        ds = (sp[mask] - self.ts[mask]).astype(float)
        dq = ((fl - self.tf) / self.lanes)[mask].astype(float)
        return int(mask.sum()), float((ds ** 2).sum()), float((dq ** 2).sum())

    def scores(self, sp: np.ndarray, fl: np.ndarray, target_full: np.ndarray | None = None, physics: bool = True) -> list[dict]:
        """One row per holdout month: the exact replicas on that month's days."""
        spf, flf = self.full(sp), self.full(fl)
        rows = []
        for m in L.HOLD_MONTHS:
            sel = L.day_months(self.panel) == m
            rows.append({"panel": self.panel, "month": m,
                         **L.score(self.panel, spf, flf, sel, self.target_full if target_full is None else target_full, physics)})
        return rows


def cached_pair(panel: str):
    z = np.load(CACHE / "t1_holdout_base" / f"{panel}.npz")
    return z["speed"], z["flow"]


def plain(world: str, name: str, panel: str, h: Holdout):
    if name in ("pair", "blend", "pairc", "blendc"):
        sp, fl = cached_pair(panel)                      # the released layer, every holdout day
        target = h.target_release                        # the release's targets: also the history cells in world A
        if name in ("pair", "blend") and (PRED / f"{world}_ours_{panel}.npz").exists():
            os_, of_ = stored(world, "ours", panel)
            win = np.isfinite(os_).any(axis=(1, 2))[:, None, None]      # the days predicted with the holdout models of this run
            ps, pf = production_blend(panel, target, (os_, of_), stored(world, "public", panel), (None, None), w_unet=0)
            sp, fl = np.where(win, ps, sp), np.where(win, pf, fl)
        if name in ("pair", "pairc"):
            return sp, fl
        us, uf = stored(world if name == "blend" else "A", "unet", panel)
        at = target & np.isfinite(us)
        sp, fl = np.where(at, (1 - W_UNET) * sp + W_UNET * us, sp), np.where(at, (1 - W_UNET) * fl + W_UNET * uf, fl)
        return t1.plausible_speed(panel, sp, fl), fl
    return stored(world, name, panel)


def method_arrays(world: str, method: str, panel: str, h: Holdout):
    """Speed and flow [holdout day, 288, link] of a method; see the module docstring for the grammar."""
    base, _, mix = method.partition(">")
    sp, fl = plain(world, base, panel, h)
    if not mix:
        return sp, fl
    at = h.target & h.span
    ms, mf, wsum = np.zeros(sp.shape, np.float64), np.zeros(sp.shape, np.float64), 0.0
    for term in mix.split("+"):
        w, _, name = term.rpartition("*")
        w = float(w) if w else 1.0
        try:
            s2, f2 = plain(world, name, panel, h)
        except FileNotFoundError:        # a LightGBM family on a panel without windows: no span target there, nothing to replace
            assert not (h.target & h.span).any()
            return sp, fl
        at = at & np.isfinite(s2) & np.isfinite(f2)
        ms, mf, wsum = ms + w * np.nan_to_num(s2), mf + w * np.nan_to_num(f2), wsum + w
    assert abs(wsum - 1.0) < 1e-9, f"weights of {mix} sum to {wsum}"
    sp, fl = np.where(at, ms, sp).astype(np.float32), np.where(at, np.maximum(mf, 1.0), fl).astype(np.float32)
    return t1.plausible_speed(panel, sp, fl), fl


def evaluate(methods, physics: bool = True, world_a: bool = False, panels=PANELS) -> tuple[pd.DataFrame, pd.DataFrame]:
    origins = L.emulated_origins()
    L.install(None)
    cell_rows, score_rows = [], []
    for p in panels:
        h = Holdout(p, origins[p])
        span_full = np.broadcast_to(h.lay["span"][:, :, None], h.target_full.shape)
        for meth in methods:
            try:
                sp, fl = method_arrays("B", meth, p, h)
            except FileNotFoundError:                         # a LightGBM family on a panel without windows: nothing stored
                continue
            partial = bool(np.isnan(sp[h.target]).any())      # a family predicted on the window days only
            have = np.isfinite(sp) & np.isfinite(fl)
            sp, fl = np.nan_to_num(sp, nan=0.0), np.nan_to_num(fl, nan=0.0)   # the scorer's rule; only target cells are read
            n, sv, sq = h.sse(sp, fl, h.target & ~h.span & h.window_day & have)
            cell_rows.append(dict(panel=p, method=meth, where="out_windays", regime="all", month="all", n=n, sse_v=sv, sse_q=sq))
            if partial:
                n, sv, sq = h.sse(sp, fl, h.target & h.span & have)
                cell_rows.append(dict(panel=p, method=meth, where="in", regime="R0", month="all", n=n, sse_v=sv, sse_q=sq))
                for c, cname in ((1, "onset"), (2, "ongoing")):
                    n, sv, sq = h.sse(sp, fl, h.target & h.span & have & (h.cond == c)[:, :, None])
                    cell_rows.append(dict(panel=p, method=meth, where="in_" + cname, regime="all", month="all", n=n, sse_v=sv, sse_q=sq))
                continue
            for name, mask in (("in", h.target & h.span), ("out", h.target & ~h.span)):
                for r in ("R1", "R2", "R3"):
                    for mo in L.HOLD_MONTHS:
                        n, sv, sq = h.sse(sp, fl, mask & ((h.regime == r) & (h.months == mo))[:, None, None])
                        cell_rows.append(dict(panel=p, method=meth, where=name, regime=r, month=mo, n=n, sse_v=sv, sse_q=sq))
            for c, cname in ((1, "onset"), (2, "ongoing")):
                n, sv, sq = h.sse(sp, fl, h.target & h.span & (h.cond == c)[:, :, None])
                cell_rows.append(dict(panel=p, method=meth, where="in_" + cname, regime="all", month="all", n=n, sse_v=sv, sse_q=sq))
            for k in range(1, L.SPAN + 1):
                n, sv, sq = h.sse(sp, fl, h.target & h.span & (h.pos == k)[:, :, None])
                cell_rows.append(dict(panel=p, method=meth, where=f"k{k:02d}", regime="all", month="all", n=n, sse_v=sv, sse_q=sq))
            kinds = [("B", None, sp, fl), ("B-excl", h.target_full & ~span_full, sp, fl)]
            if physics:
                kinds.append(("B-truth", None, np.where(h.target & h.span, h.ts, sp), np.where(h.target & h.span, h.tf, fl)))
            for kind, target, s2, f2 in kinds:
                score_rows += [dict(method=meth, kind=kind, **r) for r in h.scores(s2, f2, target, physics and kind != "B-excl")]
            if world_a and ">" not in meth:
                sa, fa = method_arrays("A", meth, p, h)
                score_rows += [dict(method=meth, kind="A", **r) for r in h.scores(np.nan_to_num(sa, nan=0.0), np.nan_to_num(fa, nan=0.0), None, physics)]
        print(f"  {p} scored", flush=True)
    return pd.DataFrame(cell_rows), pd.DataFrame(score_rows)


def rmse_table(cells: pd.DataFrame, methods, where) -> pd.DataFrame:
    g = cells[cells["where"].isin(where)].groupby(["method", "where"])[["n", "sse_v", "sse_q"]].sum()
    g["rmse_v"], g["rmse_q_lane"] = np.sqrt(g.sse_v / g.n), np.sqrt(g.sse_q / g.n)
    return g[["n", "rmse_v", "rmse_q_lane"]].unstack("where").reindex(methods)


def monthly(sc: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ("S_state", "S_physics", "S_FD", "S_LWR") if c in sc]
    rows = [dict(method=meth, kind=kind, month=month, **L.macro(df, cols).to_dict()) for (meth, kind, month), df in sc.groupby(["method", "kind", "month"])]
    m = pd.DataFrame(rows)
    if "S_physics" in m:
        m["total"] = 0.35 * m.S_state + 0.15 * m.S_physics
    return m


def report(cells: pd.DataFrame, sc: pd.DataFrame, methods, ref: str | None = None) -> None:
    pd.set_option("display.width", 250)
    print("\nPooled RMSE over the Dec-Feb targets of the 10 panels (speed km/h, flow veh/h per lane);")
    print("in = span targets, out = all other targets, out_windays = the other targets of the days that carry a window:")
    print(rmse_table(cells, methods, ["in", "out", "out_windays"]).round(3).to_string())
    scored_methods = [m for m in methods if m in set(sc.method)]
    print("\nInside the spans, by window condition:")
    print(rmse_table(cells, methods, ["in_onset", "in_ongoing"]).round(3).to_string())
    kk = cells[cells["where"].str.startswith("k")].groupby(["method", "where"])[["n", "sse_v"]].sum()
    print("\nSpeed RMSE by position in the span (slot T+k):")
    print(np.sqrt(kk.sse_v / kk.n).unstack("where").reindex(scored_methods).round(2).to_string())
    methods = scored_methods
    m = monthly(sc)
    cols = [c for c in ("S_state", "S_physics", "S_FD", "S_LWR", "total") if c in m]
    mean = m.groupby(["method", "kind"])[cols].mean()
    print("\nScorer replicas, mean of the three holdout months (each scored as one split):")
    print(mean.unstack("kind").reindex(methods).round(5).to_string())
    out = []
    for meth in methods:
        x = mean.loc[meth]
        row = dict(method=meth, S_state_B=x.loc["B", "S_state"], dS_state_excl=x.loc["B", "S_state"] - x.loc["B-excl", "S_state"])
        if "A" in x.index:
            row["dS_state_vsA"] = x.loc["B", "S_state"] - x.loc["A", "S_state"]
        if "S_physics" in x and "B-truth" in x.index:
            row.update(S_physics_B=x.loc["B", "S_physics"], dS_physics_truth=x.loc["B", "S_physics"] - x.loc["B-truth", "S_physics"])
            row["total_cost_substitution"] = 0.35 * row["dS_state_excl"] + 0.15 * row["dS_physics_truth"]
            if "A" in x.index:
                row["dS_physics_vsA"] = x.loc["B", "S_physics"] - x.loc["A", "S_physics"]
                row["total_cost_vsA"] = 0.35 * row["dS_state_vsA"] + 0.15 * row["dS_physics_vsA"]
        out.append(row)
    print("\nCost of the spans (negative = the spans lower the score):")
    print(pd.DataFrame(out).set_index("method").round(5).to_string())
    b = m[m.kind == "B"].set_index(["method", "month"])[cols]
    if ref is not None:
        d = (b - b.loc[ref].reindex(b.index, level="month").to_numpy()).reset_index()
        print(f"\nWorld B, change against `{ref}` per month and their mean (total = 0.35 S_state + 0.15 S_physics):")
        t = d.pivot(index="method", columns="month", values=[c for c in ("S_state", "S_physics", "total") if c in d]).reindex(methods)
        for c in ("S_state", "S_physics", "total"):
            if c in d:
                t[(c, "mean")] = d.groupby("method")[c].mean().reindex(methods)
        print(t.sort_index(axis=1).round(5).to_string())
    else:
        print("\nWorld B per month:")
        print(b.unstack("month").reindex(methods).round(5).to_string())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="interp,ours,public,unet,blend")
    ap.add_argument("--no-physics", action="store_true")
    ap.add_argument("--world-a", action="store_true")
    ap.add_argument("--ref", default=None)
    ap.add_argument("--out", default="span_eval")
    args = ap.parse_args()
    methods = args.methods.split(",")
    cells, sc = evaluate(methods, not args.no_physics, args.world_a)
    cells.to_csv(L.LATE / f"{args.out}_cells.csv", index=False)
    sc.to_csv(L.LATE / f"{args.out}_scores.csv", index=False)
    report(cells, sc, methods, args.ref)


if __name__ == "__main__":
    main()
