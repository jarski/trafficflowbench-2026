"""Task 1: extra importance-weighted congested training cells (t1_model.CONGESTED_EXTRA), Dec-Feb holdout.

Congested target cells are ~4% of cells but ~21% of the squared speed error. Baseline and variant use
the same settings (1 seed for speed; density adjustment on), so the difference isolates the extra cells.

    python experiments/t1_congested.py
"""
from tfb import t1_model as t1

t1.SEEDS = 1
for extra in (0, 150_000):
    t1.CONGESTED_EXTRA = extra
    print(f"\n===== CONGESTED_EXTRA = {extra}", flush=True)
    t1.cv(600_000)
