# Late diagnostics (after the deadline)

Code for the late submissions of 2026-11-23 and 2026-11-24: report Section 7.2, and `experiments/LOG.md`, entries of 2026-11-23 and 2026-11-24. It is diagnostic code. It is not part of the pipeline that reproduces the selected submission, and it writes its data to `~/data/tfb/cache/late/` and to `~/data/tfb/submissions/late_*`.

| files | question |
|---|---|
| `make_probes.py`, `fd_local.py` | The two probes: v64 with every Task 2 prediction zero, and v64 with speed and flow of every target doubled. `fd_local.py` computes the S_FD term that the second probe needs. |
| `selector_check.py`, `sel_*.py` | The selection regularity in ongoing windows: the corrected selector replica, the flag computed from data at or before T, decode rules, the LightGBM retrained on selector-faithful windows, and the time-rule checks. Start with `sel_lib.py`. |
| `span_*.py` | The blank spans of the scored months: their emulation on the holdout, the span-aware U-Net, and the two candidates. Start with `span_lib.py`. |

The onset variant without training-month features was built with `experiments/tools/t2_variant.py --drop ...`; the command is in the log.

The fix for the blank spans was trained on a shared GPU, so a retrain may not be bit-identical.
