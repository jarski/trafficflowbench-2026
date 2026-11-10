"""Write truth / smoothed-truth Task 1 submissions for one panel's train split."""
import sys
import numpy as np
from tfb.paths import CACHE
from tfb.task1_io import load, state_frame

panel = sys.argv[1]
out = CACHE / "t3check"
a = load(panel, "train", "true_speed", "true_flow")
state_frame(panel, "train", a["true_speed"], a["true_flow"]).to_csv(out / f"{panel}_truth.csv", index=False)

def smooth(x):  # 3-point temporal mean within each day, a stand-in for the noise-free state
    p = np.pad(x, ((0, 0), (1, 1), (0, 0)), mode="edge")
    return np.nanmean(np.stack([p[:, :-2], p[:, 1:-1], p[:, 2:]]), axis=0)
state_frame(panel, "train", smooth(a["true_speed"]), smooth(a["true_flow"])).to_csv(out / f"{panel}_smooth3.csv", index=False)
print("written")
