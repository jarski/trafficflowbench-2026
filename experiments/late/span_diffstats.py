"""How far a span replacement moves the blend, on the emulated holdout (to compare with a candidate's change of v64).

RMS difference at the span targets between the production blend and each given method, speed (km/h) and total flow
(veh/h), pooled over the emulated Dec-Feb spans. The same statistic of a candidate against v64 on the scored
months comes from `span_candidates.py verify` (rms_speed_change, rms_flow_change).

    python experiments/late/span_diffstats.py interp x5 ...
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import span_eval as E  # noqa: E402
import span_lib as L  # noqa: E402

origins = L.emulated_origins()
L.install(None)
acc = {m: [0, 0.0, 0.0] for m in sys.argv[1:]}
for p in L.T2_PANELS:
    h = E.Holdout(p, origins[p])
    bs, bf = E.method_arrays("B", "blend", p, h)
    at = h.target & h.span
    for m in acc:
        ms, mf = E.method_arrays("B", f"blend>{m}", p, h)
        acc[m][0] += int(at.sum())
        acc[m][1] += float(((ms[at] - bs[at]).astype(float) ** 2).sum())
        acc[m][2] += float(((mf[at] - bf[at]).astype(float) ** 2).sum())
for m, (n, sv, sq) in acc.items():
    print(f"blend -> {m}: {n} span targets, RMS change speed {np.sqrt(sv / n):.2f} km/h, flow {np.sqrt(sq / n):.1f} veh/h")
