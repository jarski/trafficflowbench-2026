# Report outline (draft): becomes the PDF due 2026-11-10

Scoring rubric: technical soundness 40, analysis and insight 40, clarity and documentation 20. Every number below must come from `experiments/LOG.md`.

**Central story (analysis and insight).** The scored months are shifted from the training months, and local validation cannot see it. Seeing that, and turning it into a decision rule, is what moved the leaderboard after week 1. Sections 2.4 and 9 carry it; each task section shows how it changed the method.

## 1. Summary
- One paragraph per task: approach, local validation score, leaderboard evidence.
- The final private leaderboard score.
- Reproducibility statement: runtime, hardware, single entry point.

## 2. Data and evaluation methodology
### 2.1 Representation and noise
- **Dense representation:** `[day, 288, link]`; one station per link; travel order, with S/W panels reversed, verified from the topology.
- **Measurement noise from physics:** adjacent ramp-free links carry the same 5-minute flow, so their difference isolates the noise. Flow is about 2.7% and speed about 2 km/h, close to independent in time. This sets the ceilings for Tasks 1 and 3.
- Ineligible readings are not noisier: the second-difference RMS is flat across pct_observed buckets.

### 2.2 Local scoring we can trust
- Task 1 and Task 3 scorers re-implemented on dense arrays, checked to 6 decimals against the official scorers.
- Task 3 made scorable on train, contrary to the documentation: the hidden boundary flux is "projected onto the released observations", so it equals the observed change in accumulation.
- **Task 2 test bench (`tfb.t2_bench`):**
  - leave-one-month-out on three window sets: the official train windows, the selector replica on every month, and all replicated windows;
  - self-assessed IoU to show calibration;
  - the selector replica reproduces 39/40 onset windows but only 14/40 ongoing ones.

### 2.3 Causality of Task 2 inputs
- An automated test rebuilds every scored window's features with all data after T blanked. All 160 windows are identical.
- Negative controls (mainline data after the buffer, ramps inside the horizon, the next day) are caught.

### 2.4 The scored months are shifted (key finding)
- **Congestion-pattern correlation** with the train months: train months 0.955 among themselves, validation 0.843, private 0.835. Private against validation: 0.801.
- **How they differ:** the same bottlenecks (94–98% of congested links), no time-of-day shift, but about 50 active hot links instead of 63.
- **Consequence:** leave-one-month-out CV cannot see the shift. Paired leaderboard tests show it:
  - changes built on train recurrence or train profiles won on the bench and lost on the leaderboard, five times;
  - removing such features gained, twice.
- **Decision rule adopted:** judge shift-sensitive changes on the public leaderboard, which is shifted like private; use the bench for shift-free mechanics only.
- Figure: the paired bench Δ against leaderboard Δ for every tested change.

### 2.5 Leaderboard discipline
- One change per submission against a fixed base.
- Task 2 family scores add up exactly, because onset and ongoing windows are disjoint and the aggregation is linear. v10 was predicted at 0.86296 and scored 0.86297.
- Leaderboard probing for labels is avoided; zeroing a task only measures our own component scores.

## 3. Task 1: state reconstruction
- Time interpolation is already strong (0.909). The remaining error sits in congestion transitions.
- **LightGBM residual model.** The key feature is each neighbour's deviation from its own straight-line interpolation over the target's gap (wave propagation). Held-out months: S_state 0.9093 → 0.9416, S_physics 0.678 → 0.725.
- **Profile leak fixed:** leave-one-day-out profiles for training rows. Holdout +0.0013, leaderboard +0.0016.
- **Month adaptation:**
  - profiles from the scored month's own observed cells: leaderboard +0.0014, 7× the holdout estimate;
  - the day's anomaly against the month profile: +0.0007;
  - dropping the train-profile features: +0.0011;
  - self-supervised rows from the scored months (D5) and seeds (D3): pending.
- Safety limits: speed ≥ 5 km/h, q/v ≤ k_jam.
- Error analysis to add: by regime, by congestion state, by gap length.

## 4. Task 3: physics consistency
- **Derivation:** E_LWR = Σ|ΔN_sub − ΔN_obs| / Σ|ΔN_obs|. Only transitions touching targets carry error, and the measurement noise caps the achievable S_LWR (analysis by regime).
- Gains come through Task 1 accuracy on density. The profile leak fix alone added +0.0064 S_physics on the holdout; occupancy-based density was tested and rejected.

## 5. Task 2: queue forecasting
### Onset
- The queue first appears at T+30, so the question is which link breaks first. The slowest link at T scores 0.03 IoU, because free flow carries no speed signal.
- A time-of-day prior gives 0.78; LightGBM gives 0.907 (0.948 on the official train windows).
- **Non-recurrent onsets are unpredictable from data ≤ T** (prior ≤ 0.05: IoU 0.13). They do not recur within the month either.
- **Every feature change except seed averaging lost on the shifted month:**
  - anomaly features: −0.006;
  - sharper priors: −0.0078;
  - no congestion map: −0.0039;
  - month-adapted priors: −0.0045.

  Interpretation: the changed windows are the low-confidence ones, where the train prior's top link is still the best bet.

### Ongoing
- A (link, step) model on queue geometry and dynamics.
- **Error anatomy:** the largest error is kept-too-long queues (48% of error cells on official windows), then queues beyond 12 links (20%).
- **What transferred:** bridging links without a reading inside queues, plus dissipation and tail-dynamics features (+0.0038), and removing the train recurrence features (+0.0029).
- **What did not:** recurrent far links: +0.025 on the bench, −0.0012 on the leaderboard.
- Decision rule: expected-IoU set selection; alternatives within ±0.007.

## 6. Task 4: path-flow estimation
- **Structure:**
  - counts are exactly feasible;
  - rank ≈ number of zones;
  - the prior is about 2× too high;
  - the prior is a gravity pattern (R² 0.74) times lognormal noise (σ ≈ 0.63).
- The hidden truth behaves like the organizer NNLS projection of the month's own prior: predicted leaderboard deltas within 0.007. S_ODME ≈ 0.993.
- **Negative results, which are instructive:** multiplicative (KL) projections and gravity smoothing lose badly on the leaderboard (−0.15 and −0.37 S_ODME).

## 7. Reproducibility
- `python -m tfb.pipeline_final [--fresh]`: deterministic LightGBM with fixed seeds, about 18 minutes; partial rebuilds with `--tasks`. Every submitted version is reproduced exactly.
- Environment: Python 3.12, pinned packages. No external data.

## 8. Limitations and future work
- Non-recurrent onsets.
- Only one labelled shifted month (the public leaderboard) to judge shift-sensitive choices, with 40 windows per Task 2 family.
- A generator-level model (cell transmission) was not attempted.

## 9. Lessons
- Validate on what the scorer sees: local CV over near-identical months over-rewards memorizing the training scenario.
- Paired single-change submissions turn a noisy leaderboard into a measuring instrument.
