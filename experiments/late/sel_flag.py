"""1b. The cap-bound flag from data stamped <= T, on every split, and its precision and recall on train.

For an ongoing window with origin T the flag says: the origin five minutes earlier met every condition of the
selector that can be checked at T (spacing to the previous window, history coverage, an established queue in
its history), so the selector can only have passed it over because persistence still scored above the cap.

Inputs, all stamped <= T (sel_lib.visible, sel_lib.flag_window):
  - the masked layer up to T, whose slots T-60..T-5 are the complete window history;
  - the eligibility flags of those slots;
  - the origins of the earlier windows of the same panel and split in window_index.csv.
On train the same view is emulated for windows that have no released history: slots T-60..T-5 from the unmasked
layer, everything else from the masked layer.

Truth on train (not available at test time), from sel_replica.py:
  cap_m5 > 0.9        the cap fails at T-5 on the interpolated state, origin state repeated (the chosen replica)
  cap_m5_ffill > 0.9  the cap fails at T-5 on the forward-filled state, T-10 state repeated (selector_check.py)
  kind == "cap"       the above and T-5 otherwise qualified

Writes ~/data/tfb/cache/late/sel_flags.parquet (train official, train replica, validation, private).

    python experiments/late/sel_flag.py
"""
import numpy as np
import pandas as pd

import sel_lib as sl
from tfb.paths import RELEASE
from tfb.t2_onset import T2_PANELS
from tfb.task1_io import days, load, meta

BACK = 36


def window_flags(masked, elig, vc, d, T, gap, history=None) -> dict:
    v = sl.visible(masked, d, T, history, BACK)
    e = sl.visible(elig.astype(np.float32), d, T, None, BACK) > 0.5
    f = sl.flag_window(v, e, vc, gap, lag=1)
    k = 0                                   # consecutive earlier origins that qualified (T-5, T-10, ...)
    while k < 12 and sl.flag_window(v, e, vc, gap, lag=k + 1)["flag"]:
        k += 1
    state = sl.last_reading(v) <= vc        # the time-filled persistence state at T
    f.update(k_back=k, n_state=int(state.sum()), n_obs_T=int(np.isfinite(v[-1]).sum()),
             q_seen_T=int(((v[-1] <= vc) & e[-1]).sum()))
    return f


def check_history_is_complete(panel: str, split: str, a: dict) -> int:
    """The masked layer in T-60..T-5 of every released window equals window_history.parquet (so nothing else is needed)."""
    h = pd.read_parquet(RELEASE / "task2" / panel / split / "window_history.parquet",
                        columns=["window_id", "timestamp", "link_id", "speed_kmh", "is_score_eligible"])
    pos = {l: i for i, l in enumerate(meta(panel)["links"])}
    dl = {x: i for i, x in enumerate(days(panel, split)["days"])}
    di = h.timestamp.str[:10].map(dl).to_numpy()
    ti = sl.tod_index(h.timestamp)
    li = h.link_id.map(pos).to_numpy()
    got = a["speed"][di, ti, li]
    want = h.speed_kmh.to_numpy(np.float32)
    assert np.array_equal(np.isnan(got), np.isnan(want)) and np.allclose(got[~np.isnan(got)], want[~np.isnan(want)], atol=1e-4), (panel, split)
    assert np.array_equal(a["eligible"][di, ti, li], h.is_score_eligible.to_numpy().astype(bool)), (panel, split)
    return len(h)


def main() -> None:
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    wins = pd.read_parquet(sl.LATE / "sel_windows.parquet")
    rows, checked = [], 0
    for panel in T2_PANELS:
        vc = sl.vcut(panel)
        a = load(panel, "train", "speed", "true_speed", "eligible")
        for r in wins[wins.panel == panel].itertuples():
            f = window_flags(a["speed"], a["eligible"], vc, r.d, r.T, r.gap, a["true_speed"][r.d, r.T - sl.H:r.T])
            if r.set == "official":          # the released train windows need no emulation: check that it changes nothing
                f0 = window_flags(a["speed"], a["eligible"], vc, r.d, r.T, r.gap)
                assert f0 == f, (panel, r.d, r.T)
            rows.append(dict(split="train", set=r.set, panel=panel, window_id=r.window_id, d=r.d, T=r.T, month=r.month, gap=r.gap,
                             kind=r.kind, cap_m5=r.cap_m5, cap_0=r.cap_0, cap_m5_ffill=r.cap_m5_ffill, **f))
        for split in ("validation", "private"):
            a = load(panel, split, "speed", "eligible")
            checked += check_history_is_complete(panel, split, a)
            off = sl.official(panel, split)
            for r in off[off.condition == "queue_ongoing"].itertuples():
                f = window_flags(a["speed"], a["eligible"], vc, r.d, r.T, r.gap)
                rows.append(dict(split=split, set="official", panel=panel, window_id=r.window_id, d=r.d, T=r.T,
                                 month=r.date[:7], gap=r.gap, kind=None, **f))
    print(f"masked layer == window_history on the scored splits: {checked:,} history cells checked (speed and eligibility)")
    df = pd.DataFrame(rows)
    df.to_parquet(sl.LATE / "sel_flags.parquet", index=False)

    tr = df[df.split == "train"]
    for name, part in (("40 official train windows", tr[tr.set == "official"]), ("per-month replica windows, 9 train months", tr[tr.set == "replica"])):
        print(f"\n=== {name}: {len(part)} windows, flagged {int(part.flag.sum())}")
        print("flag by what bound the selector at T-5 (truth):")
        print(pd.crosstab(part.kind, part.flag).to_string())
        for label, truth in (("cap fails at T-5, interpolated state (chosen replica)", part.cap_m5 > sl.CAP),
                             ("cap fails at T-5, forward-filled state, T-10 repeated (selector_check)", part.cap_m5_ffill > sl.CAP),
                             ("cap-bound: the above (interpolated) and T-5 otherwise qualified", part.kind == "cap")):
            tp, fp, fn = int((part.flag & truth).sum()), int((part.flag & ~truth).sum()), int((~part.flag & truth).sum())
            print(f"  truth = {label}: {int(truth.sum())} windows; precision {tp}/{tp + fp} = {tp / max(tp + fp, 1):.3f}, "
                  f"recall {tp}/{tp + fn} = {tp / max(tp + fn, 1):.3f}")
        if part.set.iloc[0] == "replica":
            print("  by month (flagged / cap-bound / windows):",
                  {m: f"{int(g.flag.sum())}/{int((g.kind == 'cap').sum())}/{len(g)}" for m, g in part.groupby("month")})
        fl = part[part.flag]
        print("  flagged windows: components of misses and false alarms")
        print("   false alarms (flag, cap passes at T-5):")
        print(fl[fl.cap_m5 <= sl.CAP][["panel", "d", "T", "gap", "kind", "cap_m5", "cap_0", "cap_m5_ffill", "k_back", "n_state"]].round(3).head(30).to_string(index=False))
        ms = part[~part.flag & (part.kind == "cap")]
        print("   misses (no flag, cap-bound):")
        print(ms[["panel", "d", "T", "gap", "kind", "cap_m5", "spacing_ok", "coverage", "established"]].round(3).head(30).to_string(index=False))

    print("\n=== scored splits: ongoing windows flagged")
    for split in ("validation", "private"):
        part = df[df.split == split]
        print(f"{split}: {int(part.flag.sum())} of {len(part)} flagged; spacing-bound (gap = 360 min) {int((part.gap == sl.SPACING).sum())}; "
              f"not flagged for want of an established queue at T-5: {int((part.spacing_ok & ~part.established).sum())}; "
              f"coverage below 0.7 at T-5: {int((part.spacing_ok & (part.coverage < sl.COVERAGE)).sum())}")
        print("  per panel:", part.groupby("panel").flag.sum().to_dict())
    print(df[df.split != "train"][["split", "window_id", "gap", "spacing_ok", "coverage", "established", "flag", "k_back", "n_state", "n_obs_T", "q_seen_T"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
