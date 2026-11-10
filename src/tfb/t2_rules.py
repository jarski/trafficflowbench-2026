"""Task 2 building blocks: queue flags, replicated onset events, the time-of-day onset prior."""
from __future__ import annotations

import numpy as np

from tfb.task1_io import load, meta

HISTORY_SLOTS, HORIZON_SLOTS = 12, 6
TOD_WINDOW = 24  # onset prior uses onsets within +-2 h of the target time of day


def vcut(panel: str) -> np.ndarray:
    """Queue threshold per link: 0.6 x free speed (links.csv and fd_parameters agree in this release)."""
    return 0.6 * np.asarray(meta(panel)["free_speed_links_csv"])


def train_queue_flags(panel: str) -> np.ndarray:
    """[day, tod, link] observed-speed queue flags on train (a noisy stand-in for the hidden labels)."""
    a = load(panel, "train", "true_speed", "eligible")
    return (a["true_speed"] <= vcut(panel)) & a["eligible"]


def onset_events(q: np.ndarray) -> np.ndarray:
    """Earliest onset per day: first slot with a queue after 17 queue-free slots (history + horizon).

    Returns rows (day, t0) where t0 is the first queued slot, i.e. T + 30 min.
    """
    events = []
    for di in range(q.shape[0]):
        any_q = q[di].any(axis=1)
        for t0 in range(HISTORY_SLOTS + HORIZON_SLOTS - 1, q.shape[1]):
            if any_q[t0] and not any_q[t0 - (HISTORY_SLOTS + HORIZON_SLOTS - 1):t0].any():
                events.append((di, t0))
                break
    return np.array(events, dtype=int).reshape(-1, 2)


def onset_prior(q: np.ndarray, events: np.ndarray, t0: int) -> np.ndarray:
    """Share of onsets (within +-2 h of t0's time of day) in which each link was queued."""
    near = events[np.abs(events[:, 1] - t0) <= TOD_WINDOW]
    if len(near) < 5:
        near = events
    return q[near[:, 0], near[:, 1]].mean(axis=0)


def top_k(score: np.ndarray, k: int) -> np.ndarray:
    pred = np.zeros(score.shape, bool)
    pred[np.argsort(-np.nan_to_num(score, nan=-np.inf))[:k]] = True
    return pred


def selector_windows(q: np.ndarray, elig: np.ndarray, per_condition: int = 5, spacing: int = 72) -> list[tuple]:
    """Replica of the organizer window selector on one split or month: [(day, T, condition)].

    Forward scan in time. A window needs a queue in the horizon and history coverage >= 0.7.
    It is ongoing if the history has an established queue (>= 2 queued cells, >= 2 on one link)
    and persistence from T-5 scores IoU <= 0.9, and onset if nothing is queued in the history or at T.
    Origins of both conditions are >= 6 h apart; at most 5 per condition. On train this reproduces
    39/40 official onset windows but only 14/40 ongoing ones (LOG 2026-10-06).
    """
    S = q.shape[1]
    chosen, count = [], {"queue_onset": 0, "queue_ongoing": 0}
    for d in range(q.shape[0]):
        for T in range(HISTORY_SLOTS, S - HORIZON_SLOTS):
            if min(count.values()) >= per_condition:
                return chosen
            hor = q[d, T + 1:T + 1 + HORIZON_SLOTS]
            if not hor.any() or elig[d, T - HISTORY_SLOTS:T].mean() < 0.7:
                continue
            hist = q[d, T - HISTORY_SLOTS:T]
            if hist.sum() >= 2 and hist.sum(axis=0).max() >= 2:
                if iou(np.broadcast_to(q[d, T - 1], hor.shape), hor) > 0.9:
                    continue
                c = "queue_ongoing"
            elif not hist.any() and not q[d, T].any():
                c = "queue_onset"
            else:
                continue
            t = d * S + T
            if count[c] >= per_condition or any(abs(t - (x[0] * S + x[1])) < spacing for x in chosen):
                continue
            chosen.append((d, T, c))
            count[c] += 1
    return chosen


def iou(pred: np.ndarray, truth: np.ndarray) -> float:
    union = (pred | truth).sum()
    return 1.0 if union == 0 else float((pred & truth).sum() / union)


def best_onset_k(q: np.ndarray, events: np.ndarray, months: np.ndarray, ks=(1, 2, 3)) -> tuple[int, dict]:
    """Choose how many prior-ranked links to predict, by leave-one-month-out IoU on train onsets."""
    scores = {k: [] for k in ks}
    for di, t0 in events:
        other = events[months[events[:, 0]] != months[di]]
        prior = onset_prior(q, other, t0)
        for k in ks:
            scores[k].append(iou(top_k(prior, k), q[di, t0]))
    means = {k: float(np.mean(v)) for k, v in scores.items()}
    return max(means, key=means.get), means
