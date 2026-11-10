"""Task 1 building blocks: time interpolation and historical profiles on [day, tod, link] arrays."""
from __future__ import annotations

import numpy as np


def interp_time(x: np.ndarray) -> np.ndarray:
    """Linear interpolation along time within each day; edges take the nearest value; all-NaN stays NaN."""
    D, T, L = x.shape
    valid = ~np.isnan(x)
    idx = np.broadcast_to(np.arange(T)[None, :, None], x.shape)
    last = np.maximum.accumulate(np.where(valid, idx, -1), axis=1)
    nxt = np.minimum.accumulate(np.where(valid, idx, T)[:, ::-1], axis=1)[:, ::-1]
    xl = np.take_along_axis(x, np.clip(last, 0, T - 1), axis=1)
    xn = np.take_along_axis(x, np.clip(nxt, 0, T - 1), axis=1)
    w = (idx - last) / np.maximum(nxt - last, 1)
    out = np.where(last < 0, xn, np.where(nxt >= T, xl, xl + w * (xn - xl)))
    return np.where(valid, x, out).astype(np.float32)


def weekday(dates: list[str]) -> np.ndarray:
    """Monday = 0."""
    return (np.array(dates, dtype="datetime64[D]").astype(int) + 3) % 7


def weekday_profile(x: np.ndarray, weekdays: np.ndarray) -> np.ndarray:
    """[7, tod, link] mean over the given days, NaN-aware."""
    prof = np.full((7,) + x.shape[1:], np.nan, np.float32)
    for wd in range(7):
        sel = weekdays == wd
        if sel.any():
            prof[wd] = np.nanmean(x[sel], axis=0)
    return prof


def weekday_profile_by_day(x: np.ndarray, weekdays: np.ndarray, keep: np.ndarray,
                           target_weekdays: np.ndarray, leave_own_day_out: bool) -> np.ndarray:
    """[target day, tod, link] weekday mean of x over the kept days.

    With leave_own_day_out (target days are the days of x), each day's own values are
    removed from its profile, so training rows see the same kind of profile as test
    rows. Cells without data for that weekday fall back to the all-days mean.
    """
    valid = ~np.isnan(x) & keep[:, None, None]
    xv = np.where(valid, x, 0.0)
    s = np.zeros((7,) + x.shape[1:])
    c = np.zeros((7,) + x.shape[1:])
    for wd in range(7):
        sel = weekdays == wd
        s[wd], c[wd] = xv[sel].sum(axis=0), valid[sel].sum(axis=0)
    s_day, c_day = s[target_weekdays], c[target_weekdays]
    s_all, c_all = s.sum(axis=0)[None], c.sum(axis=0)[None]
    if leave_own_day_out:
        s_day, c_day = s_day - xv, c_day - valid
        s_all, c_all = s_all - xv, c_all - valid
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(c_day > 0, s_day / c_day, np.where(c_all > 0, s_all / c_all, np.nan))
    return out.astype(np.float32)


def group_profile_by_day(x: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """[day, tod, link] mean of x over the other days of the same group (each day left out of its own)."""
    valid = ~np.isnan(x)
    xv = np.where(valid, x, 0.0)
    out = np.full(x.shape, np.nan, np.float32)
    for g in np.unique(groups):
        sel = groups == g
        s_other = xv[sel].sum(axis=0)[None] - xv[sel]
        c_other = valid[sel].sum(axis=0)[None] - valid[sel]
        with np.errstate(invalid="ignore", divide="ignore"):
            out[sel] = np.where(c_other > 0, s_other / c_other, np.nan)
    return out
