"""Task 1 second pass, upper bound: neighbour features read the TRUE value at masked neighbour cells.

If even perfect neighbour estimates gain little, a real second pass (neighbours filled with first-pass
predictions) is not worth building. Dec-Feb holdout, 1 seed, density adjustment off (both arms equal).

    python experiments/t1_second_pass_oracle.py
"""
import numpy as np
import pandas as pd

from tfb import t1_model as t1
from tfb.metrics import s_physics, s_state
from tfb.paths import PANELS
from tfb.task1_io import days, load

t1.SEEDS, t1.DENSITY = 1, None
hold, train_days = {}, {}
for p in PANELS:
    months = np.array([x[:7] for x in days(p, "train")["days"]])
    hold[p] = np.isin(months, ["2030-12", "2031-01", "2031-02"])
    train_days[p] = ~hold[p]


def truth_fill(pdata):
    tr = load(pdata.panel, "train", "true_speed", "true_flow")
    pdata.set_neighbour_fill(tr["true_speed"], tr["true_flow"])


rows = []
for name, fill in (("base", None), ("oracle neighbours", truth_fill)):
    models = t1.fit(t1.training_frame(PANELS, train_days, 600_000, fill=fill))
    for i, p in enumerate(PANELS):
        sp, fl, _ = t1.predict_panel(models, p, "train", i, hold[p], profile_days=train_days[p], fill=fill)
        st, ph = s_state(p, "train", sp, fl, hold[p]), s_physics(p, "train", sp, fl, hold[p])
        rows.append(dict(arm=name, panel=p, S_state=st["S_state"], S_physics=ph["S_physics"],
                         R3_rmse_v=st["R3_rmse_speed"], R1_rmse_v=st["R1_rmse_speed"]))
    print(f"  {name}: done", flush=True)
r = pd.DataFrame(rows)
r["family"] = r.panel.map(t1.FAMILY)
out = r.groupby(["arm", "family"])[["S_state", "S_physics", "R1_rmse_v", "R3_rmse_v"]].mean().groupby("arm").mean()
out["composite"] = 0.35 * out.S_state + 0.15 * out.S_physics
print(out.round(5).to_string())
