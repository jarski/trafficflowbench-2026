"""Span loss per panel of saved span-aware U-Nets on the emulated Dec-Feb spans (fixed masks, as the training logs).

For the leave-family-out test: a model fine-tuned without a corridor family against the model fine-tuned on all
panels, on the left-out panels. Loss = the production loss (0.54 speed + 0.46 flow, scorer units) on the blank-span
cells; `rmse_v` is the speed RMSE over the same cells in km/h.

    python experiments/late/span_unet_panel_loss.py a.pt b.pt ...
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import span_lib as L  # noqa: E402
import span_unet_x as X  # noqa: E402
import span_windows as W  # noqa: E402

from tfb import t1_unet as U  # noqa: E402

U.THREADS = 1
torch.set_num_threads(1)
origins = L.emulated_origins()
rows = []
for path in sys.argv[1:]:
    model, extras = X.load_x(path)
    model = model.to(U.DEVICE)
    for p in L.T2_PANELS:
        panel = X.SpanPanelX(p, W.by_day(origins[p]), 0.0, extras, fixed=True)
        idx = np.array(sorted(panel.windows))
        rng = np.random.default_rng(123)
        tot = n = sv = sq = 0.0
        with torch.no_grad():
            for i in range(0, len(idx), U.BATCH):
                x, y, w = U._stack(panel, idx[i:i + U.BATCH], rng)
                w = (w == 1.0).astype(np.float32)
                out = model(torch.from_numpy(x).to(U.DEVICE)).cpu().numpy()
                e = (out - y) ** 2 * w[:, None]
                sv, sq, n = sv + e[:, 0].sum(), sq + e[:, 1].sum(), n + w.sum()
        rows.append(dict(model=Path(path).stem, panel=p, n=int(n), loss=(0.54 * sv + 0.46 * sq) / n, rmse_v=np.sqrt(sv / n) * U.V_UNIT,
                         rmse_q_lane=np.sqrt(sq / n) * U.Q_UNIT, sv=sv, sq=sq))
r = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print(r.pivot(index="panel", columns="model", values="loss").round(5).to_string())
print(r.pivot(index="panel", columns="model", values="rmse_v").round(3).to_string())
g = r.groupby("model")[["n", "sv", "sq"]].sum()
print(((0.54 * g.sv + 0.46 * g.sq) / g.n).round(5).to_string())
