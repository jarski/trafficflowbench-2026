"""Holdout sweep of tfb.t2_cnn settings: train Jun-Nov, save cache/bench/t2_cnn_holdout_<name>.pt, then score
each with experiments/t2_cnn_bench.py --mode holdout --model _<name>.

    TFB_CNN_DEVICE=mps python experiments/t2_cnn_sweep.py
"""
import subprocess
import sys
import time

import numpy as np

from tfb import t2_cnn
from tfb.paths import CACHE

VARIANTS = {
    "e20": dict(EPOCHS=20),
    "w96_e16": dict(WIDTH=96, EPOCHS=16),
    "s2_e12": dict(STRIDE=2, CAP=18, EPOCHS=12),
    "s2_w96_e20": dict(STRIDE=2, CAP=18, WIDTH=96, EPOCHS=20),
}
defaults = {k: getattr(t2_cnn, k) for v in VARIANTS.values() for k in v}
data_cache = {}
for name, cfg in VARIANTS.items():
    path = CACHE / "bench" / f"t2_cnn_holdout_{name}.pt"
    for k, v in defaults.items():
        setattr(t2_cnn, k, cfg.get(k, v))
    if not path.exists():
        t0 = time.time()
        key = (t2_cnn.STRIDE, t2_cnn.CAP)
        if key not in data_cache:
            data_cache[key] = t2_cnn.train_data()
        data = data_cache[key]
        hold = tuple(m for m in sorted({m for s in data.values() for m in np.unique(s["month"])}) if m >= "2030-12")
        split = lambda keep: {p: {k: (v[np.isin(s["month"], hold) == keep] if k != "keys" else v) for k, v in s.items()} for p, s in data.items()}
        print(f"== {name}: {cfg}", flush=True)
        t2_cnn.save(t2_cnn.fit(split(False), t2_cnn.EPOCHS, valid=split(True)), path)
        print(f"   trained in {(time.time() - t0) / 60:.1f} min", flush=True)
    print(f"== bench {name}", flush=True)
    out = subprocess.run([sys.executable, "experiments/t2_cnn_bench.py", "--mode", "holdout", "--model", f"_{name}"],
                         capture_output=True, text=True).stdout
    print("\n".join(l for l in out.splitlines() if "Mean ongoing" in l or l.startswith(("set", "selector", "all", "official"))), flush=True)
