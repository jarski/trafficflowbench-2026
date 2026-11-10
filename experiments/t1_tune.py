"""Task 1 tuning: same holdout protocol as `python -m tfb.t1_model --cv`, different settings."""
import sys
from tfb import t1_model

n, max_off, leaves, rounds = (int(x) for x in sys.argv[1:5])
t1_model.OFFSETS = tuple(k for k in range(-max_off, max_off + 1) if k != 0)
t1_model.PARAMS = {**t1_model.PARAMS, "num_leaves": leaves}
t1_model.ROUNDS = rounds
print(f"n={n} offsets=±{max_off} leaves={leaves} rounds={rounds}", flush=True)
t1_model.cv(n)
