# Experiment log

One entry per experiment or submission: date, what changed, local scores, leaderboard score, conclusion. This log becomes the source for the final PDF report.

## 2026-09-30 — setup
- Conda env `tfb` (Python 3.12, conda-forge): numpy 2.5.3, pandas 2.3.3 (pinned below 3 for the organizer scripts), pyarrow 25.0.0, scipy 1.18.1, scikit-learn 1.9.1, lightgbm 4.7.0, polars 1.44.2; Kaggle CLI 2.2.4.
- Data downloaded to `~/data/tfb` (outside Dropbox). Release files are dated 2026-09-11, the rebuilt dataset.

## 2026-09-30 — data layout, noise, local Task 3 scorer
- Dense arrays built (`python -m tfb.dense`, 31 s, 4 GB in `~/data/tfb/dense`). Every panel has exactly one station per link; links are 0.1–0.5 km. Links are stored in travel order (S/W panels reversed), verified from `outgoing_link_ids`.
- Measurement noise (ramp-free adjacent links, train): flow about 2.7% relative (100–170 veh/h per link), speed about 1.3–2.8 km/h in free flow; both close to independent in time (lag-1 autocorrelation mostly < 0.15). The noise-free ceilings are therefore roughly S_state ≈ 0.92–0.95 and S_LWR well below 1.
- Local Task 3 scorer validated. Using the reconstructed flux (`tfb.task3_proxy`), the official `score_task3.py` gives S_LWR = 1.000 for the true observations. `tfb.metrics` reproduces the official scorers to 6 decimals in 0.2 s per panel. D12_I405_S train:

| submission | S_state | S_FD | S_LWR R1/R2/R3 | S_physics |
|---|---|---|---|---|
| truth | 1.000 | 0.991 | 1.00/1.00/1.00 | 0.997 |
| 3-point smoothed truth (includes the target's own value) | 0.949 | 0.991 | 0.79/0.69/0.49 | 0.767 |
| organizer historical mean | 0.783 | 0.987 | 0.33/0.02/0.00 | 0.408 |

## 2026-09-30 — Task 1 v0, Task 2 exploration, first submission file
- Task 1, train, family mean (`experiments/t1_v0_interp.py`). The historical-mean profile is built leaving the evaluated month out:

| method | S_state | S_physics | S_LWR R1/R2/R3 (typical) |
|---|---|---|---|
| historical mean | 0.720 | 0.368 | 0.0–0.5 / 0–0.3 / 0–0.1 |
| time interpolation | 0.909 | 0.678 | 0.70 / 0.56 / 0.32 |
| 0.6 time + 0.4 raw neighbours | 0.892 | 0.555 | worse: neighbours differ in level |

  Time interpolation is already close to the noise-limited S_LWR ceiling. Remaining Task 1 errors are concentrated in congestion (R3 speed RMSE up to 3.8 km/h on D7_I405_N).
- Task 2, official train windows (`experiments/eda/task2_windows.py`). Observed-speed labels reproduce the rule: in onset windows the first queue appears at T+30 in 39/40, and the 40th has no observed queued cell. Ongoing persistence IoU is 0.51. The selector takes the earliest events of each month, at least 6 h apart. History covers T−60…T−5; the masked layer also has the observation at T for about 68% of links, which the rules allow ("at or before T").
- Onset heuristics on 1,959 replicated train onsets (`experiments/t2_onset_heuristics*.py`), IoU at T+30:
  - slowest link at T: 0.03 (before onset everything runs near free speed);
  - single-snapshot saturation or density: < 0.1;
  - onset-location prior: 0.36;
  - ±2 h time-of-day prior with k chosen per panel: 0.52, leave-one-month-out.
- v0 submission built with `python -m tfb.pipeline_v0` (8 min): 6,980,503 rows, validated, `~/data/tfb/submissions/v0/submission.csv.gz` (106 MB). Not yet submitted. Expected about 0.74: S_state 0.91, S_physics 0.68, S_queue about 0.52, S_ODME about 0.84.

## 2026-10-01 — first leaderboard score
- **Submission 56757531, v0: public LB 0.80004**, against the organizer baseline of 0.553. That is above the local estimate of about 0.74, so the validation month probably scores higher on Task 2 or Task 4 than our train-based estimates. Single-task swaps will show which.
- Submission 56757487 (the same content as `.gz`) failed with "No .csv or .parquet found after extracting the submission archive". **Upload a `.zip` containing `submission.csv`.** `tfb.kaggle_submit` does this, and reads the error text, which the CLI does not show.
- Task 2 window selector (`experiments/eda/task2_selector.py`): a forward scan through time with 6 h spacing across both conditions, at most 5 per condition, persistence IoU ≤ 0.9 and history coverage ≥ 0.7. It reproduces 53/80 official train windows exactly, including every onset window. The replicated ongoing windows come out slightly earlier than the official ones, probably because the official labels are noise-free.

## 2026-10-01 — Task 2 learned models (v1)
- **Onset** (`tfb.t2_onset`): LightGBM over (event, link) rows built from 2,327 replicated train onsets. The predicted link set at T+30 maximizes the ratio-of-expectations IoU. Leave-one-month-out IoU, eligible cells only as in the official scorer:
  - all events: **0.907** (time-of-day prior 0.783);
  - earliest onset of each day: 0.901;
  - 39 of the 40 official train onset windows: **0.948** (prior 0.830).

  Top features: onset priors, the recurrent-congestion map, time of day, downstream density.
- **Ongoing** (`tfb.t2_ongoing`): LightGBM over (window, link, step) rows within 12 links of the queue, from 11,291 replicated windows (persistence IoU ≤ 0.9). Leave-one-month-out IoU **0.883** (persistence 0.755). On the 40 official train ongoing windows: **0.837** (persistence 0.724).
- **Submission 56757764, v1 = v0 with only Task 2 replaced: public LB 0.84035** (v0 0.80004). The +0.0403 implies S_queue +0.134 on validation, consistent with the local estimates.

## 2026-10-01 — Task 4 structure and leaderboard probes (Task 4 only swapped vs v1)
- Structure (`experiments/eda/task4_structure.py`, `task4_prior_info.py`):
  - Counts sit on mainline links only and are exactly reproducible by non-negative path flows (zero NNLS residual).
  - rank(A) ≈ number of zones (35–100), against 630–5,050 paths.
  - The prior's implied counts are 1.6–2.5× the actual counts.
  - In log terms, a gravity model (origin + destination + distance decay) explains 74% of the variance, equal to the correlation between months' priors.
  - On mainline counts, each month's own prior is no better than another month's prior (0.246 vs 0.248 relative L1).
- Leaderboard probes:

| submission | Task 4 method | public LB | implied ΔS_ODME vs v1 |
|---|---|---|---|
| v1 (56757764) | organizer NNLS L2 projection, λ=0.05, own prior (zeroes 59% of paths) | 0.84035 | — |
| v2b (56757871) | IPF (KL projection) from the own raw prior, all paths positive | 0.81010 | −0.151 |
| v2a (56757846) | IPF from the gravity fit of all three priors | 0.76712 | −0.366 |

- **Lesson:** the path-level variation in each month's prior is strongly informative about that month's truth, even though aggregate counts cannot show it. The additive L2 projection beats the multiplicative one, which suggests the truth is sparse or additively related to the prior. **Count-based proxies cannot rank Task 4 methods; only leaderboard swaps can.** Keep v1's Task 4 and probe variants around the L2 projection.

## 2026-10-01 — Task 1 LightGBM correction (`tfb.t1_model`)
- Residual models for speed and flow per lane on top of time interpolation. Features:
  - neighbours' deviation from their own straight-line interpolation across the target's gap, at ±1..3 links, for speed, flow and occupancy;
  - anchor values, gap lengths, profile anomalies at the anchors;
  - occupancy density, link attributes, regime and pct_observed.
- Trained on target cells of Jun–Nov train (250k per panel), scored on Dec–Feb with `tfb.metrics` (exact scorer replicas). Profiles use training months only.
- Family mean: S_state 0.9093 → **0.9391**, S_physics 0.6781 → **0.7179**; every panel improves. R3 speed RMSE 1.2–2.3 km/h, close to the noise floor. Expected total gain about +0.016.
- Density from interpolated occupancy (285·occ) is worse than interpolated flow/speed (relative MAE 0.07 vs 0.04), so it is not used for Task 3 post-processing.
- Task 4 block CV (held-out contiguous count blocks): NNLS 0.194 is best, consistent with the leaderboard. But it ranks gravity-IPF above prior-IPF, the reverse of the leaderboard. Count CV is not reliable for Task 4.

## 2026-10-01 — v3 build, safety limits, reproducibility
- v3 = v1 with Task 1 replaced by the LightGBM correction (300k cells per panel, all train months). Queued for submission right after the UTC reset.
- Task 1 safety limits (`t1_model.plausible_speed`): speed ≥ 5 km/h (train minimum 4.9) and density q/v ≤ k_jam. 123 v3 values changed. Near-zero speeds imply absurd accumulations in the Task 3 conservation check.
- Task 2 ongoing fix: candidate links now also cover queues seen anywhere in the history. Previously one validation/private window had no queue at T or T−30 and got an empty forecast, a sure IoU of 0. Leave-one-month-out IoU 0.882 on 11,321 windows (was 0.883 on 11,291, which excluded those windows). Cached as `t2_ongoing_train_v2.parquet`; goes into v4.
- **Reproducibility:** `python -m tfb.pipeline_final` (Task 1 model + Task 2 models + NNLS Task 4) reproduces v3 exactly in 9.0 min on the M4 Pro: Task 1 max |diff| 5e-7 km/h, Task 2 identical, Task 4 max |diff| 2e-13. `tfb.t4.nnls_l2` equals the organizer builder's solve.
- Queued after 00:00 UTC: v3 (Task 1), v2c (Task 4 NNLS from count-rescaled prior), v2e (Task 4 NNLS from rescaled gravity of all priors).

## 2026-10-01 — Task 1 tuning and v4
- `experiments/t1_tune.py`, same holdout protocol (train Jun–Nov, score Dec–Feb):
  - 600k cells per panel, neighbours ±5, 255 leaves, 1,000 rounds: S_state **0.9416**, S_physics **0.7246**;
  - previous model (250k, ±3, 127 leaves, 600 rounds): 0.9391 / 0.7179.

  Adopted as the defaults in `t1_model` and `pipeline_final` (600k cells per panel). Expected total gain about +0.002.
- The Task 2 ongoing cache was replaced by the version with history-wide neighbourhoods. The old one is kept as `t2_ongoing_train_v1.parquet`.
- v4 = `pipeline_final --out v4`: tuned Task 1, Task 2 with the ongoing fix, NNLS Task 4.

## 2026-10-02 — leaderboard results, Task 4 solved, Task 2 diagnosis
- **v3 (56761350): public LB 0.85295** (v1 0.84035). The Task 1 model adds +0.0126; the local holdout estimate was +0.0165.
- v2c (56761510): 0.78575. Task 4 NNLS from the prior rescaled to the counts: ΔS_ODME −0.273.
- **Task 4 is effectively solved by v1** (`experiments/t4_hypothesis.py`). Assume the hidden truth = the organizer NNLS (λ=0.05) projection of the released prior, and score the probed variants against v1's solution. Predicted vs observed ΔS_ODME:

| variant | predicted | observed |
|---|---|---|
| v2a | −0.373 | −0.366 |
| v2b | −0.158 | −0.151 |
| v2c | −0.280 | −0.273 |

  The constant offset of about 0.007 implies S_ODME(v1) ≈ 0.993. The remaining Task 4 headroom is about 0.0014 of total, so the v2e/v2f probes were cancelled.
- **Component isolation on validation:**
  - v3 with all Task 2 predictions zero (56761729 → 0.62714) gives **S_queue(v3) = 0.753**.
  - v3 with onset windows zeroed (0.73832) gives **onset IoU 0.764, ongoing IoU 0.741**.
  - Local estimates were 0.90–0.95 for onset and 0.84–0.88 for ongoing. **Task 2 generalizes much worse to the validation month than LOMO CV suggests.**
  - Tasks 1+3: 0.35·S_state + 0.15·S_physics ≈ 0.4285 on validation (train-holdout estimate 0.436).
- **Why the Task 2 gap? Checks so far:**
  - Not a capacity shift: the 99.9th-percentile flow/capacity is the same in all splits.
  - Not label noise: queued links at onset are typically 22–52 km/h below v_cut; only D7_I10_W is marginal.
  - Congestion levels differ by panel and month: D12_I5_N validation has 0.022 queued share against a train range of 0.036–0.070; D7_I405_S has more.

  Most likely cause: month-specific demand and incidents, with only 80 windows, concentrated in the first days of the month.
- **Task 2 gap explained (onset):** onset IoU on train by how recurrent the true onset location is, measured by the leave-one-month-out time-of-day prior:

| prior at the true onset location | share of train onsets | model IoU | prior-only IoU |
|---|---|---|---|
| ≤ 0.05 | 3.8% | 0.13 | 0.00 |
| 0.05–0.2 | 1.7% | 0.57 | 0.00 |
| 0.2–0.5 | 23% | 0.91 | 0.48 |
| > 0.5 | 71% | 0.97 | 0.95 |

  About 20% non-recurrent onsets would put the average near 0.77, matching validation's 0.764. Train months with the official-like selection range 0.854–0.950 (sd 0.041), so validation is an outlier month. Noise levels and free-flow speeds are identical across splits, so the simulator is unchanged; demand and incidents differ.
- **Non-recurrent onsets cannot be predicted from data ≤ T** (`experiments/t2_onset_dynamics.py`):

| model | overall | rare locations (≤ 0.05) |
|---|---|---|
| current | 0.9071 | 0.131 |
| + anomaly features vs time-of-day profile | 0.9101 | 0.132 |
| dynamics-only | 0.8715 | 0.090 |
| blends | ≤ 0.909 | — |

  Adopt the anomaly features (small gain). The rest of the gap is irreducible for any team.
- **v4 (56763069): public LB 0.85494**, new best (v3 0.85295, +0.0020). Built with `pipeline_final`: tuned Task 1 (600k, ±5 neighbours, 255 leaves, 1,000 rounds), Task 2 ongoing neighbourhood fix, NNLS Task 4. The gain matches the tuned-Task 1 holdout estimate (about +0.002).
- Submissions on 2026-10-02: v3, v2c, v3_q0, v3_onset0, v4 (5/5).

## 2026-10-06 — diagnostics for the improvement plan
- **Leaderboard context (Oct 5):**
  - top 0.924;
  - #10 0.893;
  - #20 0.883;
  - a public notebook at 0.86517 was #48 of 320.
  - Our 0.85494 is about #60.

  Tasks 1, 3 and 4 are near their ceilings (Task 4 ≈ 0.993; Tasks 1+3 ≈ 0.43 weighted, noise-limited). A 0.92 therefore needs S_queue ≈ 0.93 on validation; we have 0.753.
- **The Task 2 model knows which months are hard** (`experiments/eda/t2_confidence.py`). Self-assessed IoU is the ratio-of-expectations value the set selector maximizes, with v4 models and inputs at or before T only.

| windows | onset: self-assessed | onset: actual | ongoing: self-assessed | ongoing: actual |
|---|---|---|---|---|
| train out-of-fold, replicated | 0.894 | 0.921 | 0.882 | 0.887 |
| train out-of-fold, official 80 | 0.899 | 0.949 | 0.886 | 0.839 |
| validation (public) | 0.741 | 0.764 | 0.797 | 0.741 |
| private | 0.863 | ? | 0.842 | ? |

  - **Onset is calibrated.** Validation's low score is visible in advance and concentrated in D7_I10_W (self-assessed 0.27) and D12_I5_N/S (0.55/0.58). It is not a weekend effect.
  - **Ongoing is calibrated on replicated windows but overconfident on official windows** (−0.05 on train, −0.056 on validation).
  - **The private month looks more train-like.** It has 7/40 weekend onset windows against 16/40 in validation, and it starts on a Tuesday, where validation starts on a Saturday. Expect private S_queue around 0.83–0.84, so a private total around 0.88 if nothing else changes.
- **Non-recurrent onsets are one-offs.** For train onsets whose location prior (from other months) is ≤ 0.05:
  - the same location was queued on an earlier day of the same month (±2 h) in only 7% of cases, against 67–73% for recurrent onsets;
  - it was queued earlier the same day in 0% of cases.

  An online same-month prior cannot rescue them. It does help the small 0.05–0.2 band (1.7% of events; same-month frequency 0.59).
- **Ongoing errors on the 40 official train windows** (`experiments/eda/t2_ongoing_errors.py`, model trained without the window's month). IoU 0.839, persistence 0.723. Error cells:
  - 38% are queued-at-T cells that clear while we keep them (dissipation timing);
  - 33% are missed new queue cells more than 3 links from the queue at T (secondary bottlenecks; links beyond 12 are never predicted);
  - tail and head movement together make up about 18%.

  Errors grow with the step: step 6 has 2.6× the errors of step 1.
- **The ongoing selector cannot be reproduced exactly.** We tried:
  - condition rules: established, queued at T−5, queued at T, any queue in history;
  - persistence from T−5 or T;
  - global or per-condition spacing;
  - median-denoised labels.

  None matches more than 14/40 official ongoing windows; onset stays 38–39/40. The official ongoing windows are bimodal in queue age (20% under 30 min, 72% over 4 h of continuous congestion). Reweighting the replicated out-of-fold results to that mix still gives 0.88, so queue age does not explain the gap to 0.84.
- **Task 1:**
  - Ineligible readings (pct_observed < 75) are not noisier: the second-difference RMS is flat across pct buckets. Quality-aware anchors are not worth adding.
  - **Training-time profile leak:** `t1_model.PanelData` builds the weekday profile from the true values of all training days, including the target's own day. Training rows therefore see a slightly better profile than test rows. Fix: leave-one-day-out profiles. A public notebook reports +0.004 on Task 1 for the same fix.

## 2026-10-06 — week 1: Task 2 bench, Task 1 profile leak, onset anomaly features, seed ensembles
- **Task 2 test bench (`python -m tfb.t2_bench`).** Leave-one-month-out IoU and self-assessed IoU on three window sets, with out-of-fold predictions cached in `cache/bench/`:
  - **official:** the 80 official train windows (39 onset).
  - **selector:** the selector replica (now `t2_rules.selector_windows`) run on every train month. That gives 357 onset and 359 ongoing windows.
  - **all:** every replicated training window.

  The replica's ongoing windows score like "all" (0.879), not like the official ones (0.839), because it reproduces only 14/40 official ongoing windows. **For ongoing, only the 40 official windows show official difficulty**, and they are noisy.
- **Causality test (`python -m tfb.t2_bench causality`).** The production Task 2 feature path is refactored into `pipeline_v1.panel_context` and `window_features`, and still reproduces v4's queue.csv exactly (174,000 rows). The test rebuilds all 160 scored windows with every value after T blanked (rest of the day, later days, ramps) and requires identical features: **all 160 identical**. Negative controls are caught: a feature reading mainline data after the buffer, ramp flows inside the horizon, or the next day. (Reading the horizon itself is NaN in the release, so it cannot leak.)
- **Task 1 leave-one-day-out profiles** (`t1_interp.weekday_profile_by_day`). The training profile no longer contains the target day. Dec–Feb holdout, tuned settings (600k cells per panel, ±5 neighbours, 255 leaves, 1,000 rounds):
  - S_state 0.9416 → **0.9427**;
  - S_physics 0.7246 → **0.7310**;
  - every one of the 10 panels improves on both.

  Expected about +0.0013 total.
- **Onset** (bench, IoU on official / selector / all):

| variant | official | selector | all |
|---|---|---|---|
| v4 model (1 seed) | 0.9481 | 0.9053 | 0.9071 |
| 3 seeds | 0.9431 | 0.9046 | 0.9090 |
| 5 seeds | 0.9481 | 0.9070 | 0.9104 |
| + anomaly features, 1 seed | 0.9508 | 0.9116 | 0.9099 |
| **+ anomaly features, 5 seeds** | 0.9456 | **0.9129** | **0.9115** |

  - The anomaly features are now in `t2_onset` (`train_profiles`, `anomaly_features`; the profile leaves out the event's month). The cache is rebuilt as `t2_onset_train.parquet`; the old one is kept as `t2_onset_train_v1.parquet`.
  - The 39 official windows move by ±0.005 between seeds: noise.
  - **Renormalization (B3) dropped.** It helped the v4 model (selector +0.0045 at a sum(p) target of 2), but on the anomaly 5-seed model it gives +0.0000 at 1.5 and −0.0005 at 2.
- **Ongoing baseline on the bench** (1 seed): official 0.8392 (self-assessed 0.8865), selector 0.8792, all 0.8819.
- **Ongoing A3 features** (`t2_ongoing.window_rows`, frame `t2_ongoing_train_a3.parquet`, old columns unchanged):
  - recovery: speed change over 15 and 30 minutes; queued slots in the last 15 minutes;
  - demand: flow against the train profile (month left out), and arriving flow at each queue tail with its 30-minute trend;
  - distances in km;
  - change in the recurrent queue frequency from T to T+h;
  - per queue segment: shockwave tail speed w = (q_arr − q_q)/(k_arr − k_q), the signed km from the tail predicted for each step, and discharge flow and speed below the head;
  - **gap bridging:** links with no reading (masked at T and ineligible at T−5) between queued links count as queued for the segment features. In one official window this turned 23 fragments into 3 queues, and the bridged links are queued 76% of the time at T+30.

  Bench, 1 seed, IoU on official / selector / all:
  - base: 0.8392 / 0.8792 / 0.8819;
  - **A3: 0.8488 / 0.8846 / 0.8856**, better on every set.

  The bridged flag takes 64% of the gain importance. Recent persistence, distance upstream in km, the frequency trend, the 15-minute speed slope and the flow anomaly follow. The physics features (tail speed, predicted-tail gap, discharge) are minor at ≤ 1.4% each.
- **Bench memory:** a trained LightGBM booster keeps its binned dataset, so 9 folds × 5 seeds held several GB and the machine swapped. `free_dataset()` is now called after training, in the bench and in `pipeline_v1.train_models`.
- **New tools:**
  - `python -m tfb.compose --out vX --state vA --queue vB --odme vC` builds a submission from per-task files of earlier versions.
  - `pipeline_final --tasks 2 --base v6` rebuilds only the listed tasks and copies the others.
- **v6 build:** `pipeline_final` took 17.7 min with 5 seeds per Task 2 model. v5 = v6's state.csv with v4's queue and odme files. Compare tool: v4→v5 changes only Task 1, v5→v6 only Task 2 (1.4% of queue cells flip).
- **Leaderboard, 2026-10-06.** Each submission changes one thing against a fixed base, so its delta isolates it:

| submission | change | public | Δ | implied |
|---|---|---|---|---|
| v5 (56889484) | v4 + Task 1 leave-one-day-out profiles | 0.85650 | +0.0016 vs v4 | holdout predicted +0.0013 |
| v6 (56889514) | v5 + onset anomaly features + 5 seeds, ongoing 5 seeds | 0.85150 | −0.0050 vs v5 | — |
| v7 (56889665) | v5 + onset 5 seeds, v4 onset features (onset only) | 0.85774 | +0.0012 vs v5 | onset IoU +0.008 |
| **v8 (56889687)** | v5 + ongoing A3b + 5 seeds (ongoing only) | **0.86033** | **+0.0038 vs v5** | ongoing IoU +0.026 |

  - **The onset anomaly features hurt on validation** (about −0.006 total, onset IoU about −0.04), although they helped on every train window set. They changed the picks in 7 low-confidence validation windows (D7_I10_W, D12_I5_N/S, D7_I210_W), where the model's top probability is about 0.1. Validation anomaly levels are within about 1 sd of train, so this is not a month-wide demand shift; those windows are close to coin flips, and this time they went against us. Decision: **drop the anomaly features** and keep v7's onset (old features, 5 seeds).
  - **The ongoing A3b gain transfers.** Bench +0.013 / +0.005 / +0.006 (official / selector / all); validation ongoing IoU +0.026.
  - **Task 2 scores add up exactly across families.** Onset and ongoing windows are disjoint and the aggregation is linear, so v7's onset plus v8's ongoing would score exactly 0.85650 + 0.00124 + 0.00383 = 0.86157. No submission is needed to know that.
- **Task 1 D2, month-adapted profiles** (`t1_interp.group_profile_by_day`). The profile comes from the scored month's own observed cells, by day type (weekday or weekend), with the target day left out. Dec–Feb holdout: S_state 0.9427 → 0.9430, S_physics 0.7310 → 0.7317. The total gain is positive on all 10 panels (+0.0001 to +0.0004 each). Adopted; v9 = v8 with Task 1 rebuilt, to check it on the leaderboard.
- **v9 (56889898): public 0.86172**, +0.0014 over v8. This is v8 with Task 1 rebuilt with the month-adapted profiles (D2). On validation the gain is about 7× the train-holdout estimate (+0.0002). That fits each scored month having its own demand draw: train months resemble each other, March does not.
- **v10:** `pipeline_final --out v10 --tasks 2 --base v9`, 3.1 min. Onset uses v4 features with 5 seeds; ongoing uses A3b with 5 seeds. It reproduces v7's onset rows and v8's ongoing rows exactly (0 of 87,000 + 87,000 differ), so it is the full current configuration. Predicted public score: 0.86172 + 0.00124 = **0.86296**. Not yet submitted (5/5 used on Oct 6).
- **B2, sharper onset priors with same-day and same-month context** (bench, 5 seeds, official / selector / all):
  - v4 features: 0.9481 / 0.9070 / 0.9104;
  - + B2: 0.9547 / 0.9120 / 0.9075;
  - + B2 + anomaly: 0.9597 / 0.9094 / 0.9113.

  Mixed, so not adopted. After v6, onset changes ship only with a clear, consistent gain. The optional onset groups now sit behind `t2_onset.EXTRA_FEATURES` ("anomaly", "context"), off by default; a fresh onset frame equals the production cache.
- **v10 (56894164): public 0.86297, new best**, against a predicted 0.86296: family scores add up exactly as expected. Submitted by `submit_queue.py` at 00:05 UTC on Oct 7. v10 is the full current `pipeline_final` configuration.

## 2026-10-07 — the scored months are shifted; leaderboard tests of shift sensitivity
- **v10 (56894164): 0.86297.**
- **Ongoing errors left after A3b** (`experiments/eda/t2_ongoing_errors.py --tag a3b`), shares of error cells, official / selector windows:
  - queued-at-T cells kept after they clear: 48% / 27%; these grow with the step (8 cells at T+5, 103 at T+30 on official windows);
  - queue cells more than 12 links away, never modelled: 20% / 18%;
  - misses 1–12 links up- or downstream: about 25% / 30%.
- **Recurrent far links** (`FAR_QFREQ = 0.02`): also model links beyond 12 where the train queue frequency in the horizon is ≥ 0.02. That covers 96% of the far true cells and adds 46% rows.
  - Bench, 1 seed: 0.8520 → **0.8771** / 0.8839 → 0.8977 / 0.8876 → 0.8940. Positive on all 9 months on "all", 6 of 9 on selector.
  - Calibration on official windows: self-assessed 0.883 against actual 0.877 (before: 0.905 against 0.852).
- **Decision rule on the far model.** Bench, official / selector / all:
  - expected-IoU: 0.8771 / 0.8977 / 0.8940;
  - on p^1.25: 0.8787 / 0.9016 / 0.8951;
  - p shrunk linearly by up to 20% at the last step: 0.8843 / 0.9006 / 0.8949;
  - threshold 0.5: 0.8711 / 0.9023 / 0.8941.
- **Leaderboard:**

| submission | change vs v10 | public | Δ | implied family IoU |
|---|---|---|---|---|
| v11 (56904934) | ongoing: far links + 20% step shrink | 0.86195 | −0.0010 | ongoing −0.007 |
| v12 (56905064) | ongoing: far links only | 0.86178 | −0.0012 | ongoing −0.008 (so the shrink is +0.0002) |
| **v13 (56905412)** | ongoing **without** train-recurrence features qfreq_h, qfreq_T, qfreq_trend | **0.86586** | **+0.0029** | ongoing +0.019 |
| v14 (56905447) | onset: B2 sharper train priors plus month and day context (no anomaly) | 0.85517 | −0.0078 | onset −0.052 |

- **Why (the key finding).** Congestion-pattern correlation (queue share by time of day and link, 06–21 h, masked layers) of each month against the train months:
  - each train month against the other train months: **0.955** (0.84–0.98);
  - validation: **0.843**;
  - private: **0.835**, with D12_I5_N at 0.64 and D7_I405_S at 0.77, below every train month.

  Flow profiles differ by 3.2% in both scored months, against 2.2% between train months. The train months are near copies of one scenario, while validation and private were drawn independently and are both shifted, private even more on some panels. Consequences:
  - **Leave-one-month-out CV cannot see this shift.** Anything built on train recurrence or train profiles looks good on the bench and fails on the scored months: far links (train queue frequency), sharper onset priors, onset anomaly features (train profile), recurrence features in ongoing.
  - **What transfers** is current dynamics (A3b) and adaptation to the month itself (Task 1 month profiles, +0.0014 public, 7× the holdout estimate).
  - **The public leaderboard is the better proxy for private for any change that depends on train patterns**, because both scored months are shifted. The bench stays the check for shift-free mechanics.
  - The private self-assessed onset IoU of 0.863 (LOG 2026-10-06) is not trustworthy: it relies on train priors.
- **Main line** = v13: ongoing `FEATURES_DROP` now includes qfreq_*, `FAR_QFREQ = None`, `STEP_SHRINK = 0`. A fresh `pipeline_final --tasks 2` reproduces v13 exactly.
- **Task 1 D2b, day anomaly** (`t1_model._window_mean`): the target link's mean deviation from the month profile over ±1 h of observed cells, for speed and flow. Dec–Feb holdout:
  - S_state 0.9430 → 0.9435;
  - S_physics 0.7317 → 0.7346;
  - positive on all 10 panels (+0.0003 to +0.0010 total).

  To be tested on the leaderboard as v15.
- **Private does not resemble validation either.** Congestion-pattern correlation, mean over the 8 Task 2 panels:
  - private vs validation: 0.801;
  - private vs train: 0.835;
  - validation vs train: 0.843.

  Panels differ (D12_I5_N: private against validation 0.80, against train 0.64; D12_I5_S: 0.58 against 0.82). Each scored month is its own draw, so March data is no general template for April, even though it is legal input for April Task 2 windows. Robustness has to come from adapting within each month and relying less on fixed train patterns.
- **v13 is the main line**, and a fresh `pipeline_final --tasks 2` reproduces it exactly. Queued for Oct 8 (`submit_queue.py`):
  - v15 = v13 + Task 1 D2b;
  - v16 = v15 with Task 1 trained without train-profile features;
  - v17 = v13 with ongoing also without flow_anom/arr_anom;
  - v18 = v13 with onset without qfreq_t0.
- **Task 1 D5, self-supervised rows from the scored months** (`t1_model.pseudo_frame`). Hide 10% of the observed eligible non-target cells (all channels), rebuild the features on that layer, and train on the hidden observations. Holdout check: pseudo rows from Dec–Feb plus the usual Jun–Nov rows, scored on the real Dec–Feb targets (result below).
- **D5 holdout result.** 1.48M self-supervised rows from Dec–Feb (148k per panel) added to the Jun–Nov training rows:
  - S_state 0.9435 → **0.9438**;
  - S_physics 0.7346 → **0.7357**;
  - positive on all 10 panels (+0.00003 to +0.0007 total, mean +0.0003).

  Small on the near-identical train months; it should matter more on the shifted scored months. It is `pipeline_final.T1_PSEUDO` (off until the leaderboard confirms it), to be tested on top of the better of v15 and v16 on Oct 9.
- **How the scored months differ** (congested link = queue share ≥ 0.03, 06–21 h):
  - 94% (validation) and 98% (private) of their congested links are train bottlenecks, and the time-of-day peaks are not shifted (lag 0–1 min);
  - but only about 50 links are congested, against 63 in train months: **the scored months activate a subset of the train bottlenecks**;
  - validation's new bottlenecks sit almost entirely in D7_I10_E (11%) and D7_I10_W (17%), the panels where onset scored worst.
- **Onset "adapt"** (`t2_onset.month_activity_ratio`, `EXTRA_FEATURES = ("adapt",)`):
  - the train priors (prior_tod, prior_all, qfreq_t0 and the neighbouring priors) are rescaled by each link's queue activity on the month's earlier days within ±2 h, shrunk toward train with 2 days of pseudo-evidence;
  - the raw priors are replaced, so the model cannot fall back on them;
  - day 1 of a month is unchanged.

  Bench, 5 seeds, official / selector / all: 0.9613 / 0.9041 / 0.9043 (v4 features: 0.9481 / 0.9070 / 0.9104). That is a small cost on train months, which have no shift to correct. It changes 9 validation and 5 private onset windows. Queued as v19 for Oct 8.

## 2026-10-08 — leaderboard: Task 1 sheds train patterns; onset changes keep losing
| submission | change | public | Δ vs base |
|---|---|---|---|
| v15 (56925181) | v13 + Task 1 day anomaly (D2b) | 0.86652 | +0.0007 vs v13 (holdout +0.0006) |
| **v16 (56925367)** | v15 + Task 1 **without train-profile features** (month profiles only) | **0.86763** | **+0.0011 vs v15** |
| v17 (56925489) | v13 + ongoing without flow_anom/arr_anom | 0.86547 | −0.0004 (ongoing IoU −0.003): not adopted |
| v18 (56925591) | v13 + onset without qfreq_t0 | 0.86199 | −0.0039 (onset IoU −0.026): not adopted |
| v19 (56925946) | v13 + onset priors rescaled by the month's own bottleneck activity | 0.86141 | −0.0045 (onset IoU −0.030): not adopted |

- **Task 1.** Dropping the train-profile features (prof_s, prof_f_lane, anom_s_prev/next) helped on the shifted month, as predicted. They are now in `t1_model.FEATURES_DROP` by default, so the main line is v16.
- **Onset: every change except seed averaging has lost 0.004–0.008 on validation:**
  - anomaly features (v6);
  - sharper priors and context (v14);
  - no qfreq_t0 (v18);
  - month-adapted priors (v19).

  That is too consistent to be noise. The changed windows are the low-confidence ones, where the train prior's top link is still usually right: the scored months use the same bottlenecks, only fewer of them. Every feature change pulls picks away from it. **Onset is frozen at the v4 features with 5 seeds.**
- **Task 1 D3, 2 seeds** (Dec–Feb holdout, D2b features): S_state 0.9435 → 0.9437, S_physics 0.7346 → 0.7364.
- **Queued for Oct 9** (built and checked: each changes only its own part), all against v16:
  - v20: Task 1 D5 (self-supervised rows, rate 0.1, 150k per panel per scored month);
  - v21: Task 1 with 2 seeds;
  - v22: ongoing step shrink 0.2;
  - v23: ongoing without tod/weekday/pos/panel_code;
  - v24: Task 1 without tod/pos/panel_code.
- **A4, twice the ongoing training windows** (stride 3, cap 12 per day: 22,535 windows, 8.6M rows). Bench, 1 seed, official / selector: 0.8421 / 0.8783 against 0.8426 / 0.8828 for the main configuration. **No gain; dropped.**
- **Main ongoing configuration on the bench** (A3b without qfreq, 1 seed): 0.8426 / 0.8828 / 0.8816, against A3b with qfreq at 0.8520 / 0.8839 / 0.8876. Removing the recurrence features costs −0.009 on official bench windows but gained +0.019 on validation: the shift effect again.
- **Task 1 D4, density adjustment for Task 3** (`experiments/t1_density.py`; Dec–Feb holdout, current Task 1 configuration). A third LightGBM (L1 loss) predicts log density against the interpolation, k = q/v; each prediction's q/v is then moved β of the way (in logs) toward it, with share α of the move put on flow. Composite 0.35·S_state + 0.15·S_physics:

| β | α (flow share) | S_state | S_physics | Δ composite |
|---|---|---|---|---|
| 0 | — | 0.94320 | 0.73375 | 0 |
| 0.25 | 1.0 | 0.94304 | 0.73729 | +0.00047 |
| **0.5** | **1.0** | **0.94241** | **0.73945** | **+0.00058** |
| 0.5 | 0.5 | 0.94227 | 0.73945 | +0.00053 |
| 0.75 | 1.0 | 0.94135 | 0.74007 | +0.00030 |
| 1.0 | 1.0 | 0.93987 | 0.73888 | −0.00040 |

  Moving flow only, halfway, is best: S_physics +0.0057 for S_state −0.0008. It is in `t1_model.DENSITY = {"beta": 0.5}` (off by default) and goes on the leaderboard once the Oct 9 results settle the Task 1 base.

## 2026-10-09 — Task 1 seeds and self-supervision help a little; rank #72
| submission | change vs v16 (0.86763) | public | Δ |
|---|---|---|---|
| v20 (56979741) | Task 1 self-supervised scored-month rows (D5) | 0.86792 | +0.0003 |
| **v21 (56979952)** | Task 1 2-seed average (D3) | **0.86796** | **+0.0003** |
| v22 (56980114) | ongoing step shrink 0.2 | 0.86767 | +0.0000 |
| v23 (56980243) | ongoing without tod/weekday/pos/panel_code | 0.86721 | −0.0004 |
| v24 (56980381) | Task 1 without tod/pos/panel_code | 0.86743 | −0.0002 |

- D5 and D3 both help; D4 (density) is still to be measured. Time of day, position and panel ID stay in both models; the step shrink stays off.
- **Leaderboard on Oct 9:**
  - we are **#72 of 374** at 0.86796;
  - #50 0.8726, #30 0.8815, #20 0.8878, #10 0.8945, top 0.924;
  - a cluster of teams sits at 0.887–0.897.

  By our component arithmetic (Task 1+3 ≈ 0.436, Task 4 ≈ 0.199), a 0.89 needs S_queue ≈ 0.84 on validation against our ≈ 0.78. The gap is in Task 2.
- **Public notebooks:** the new "0.90+ target" notebook repackages the 0.865 notebook (CatBoost Task 1; LightGBM + CatBoost onset; threshold tweaks). It does not explain the cluster.
- **Open question (thread 745700, Oct 3):** may Task 2 use the observation at T itself (≤ T), or only < T? No organizer answer yet.
  - We use ≤ T, as rules.md and the organizer reply in thread 742068 state.
  - `t2_features.USE_T_SLOT` now switches every read of the T slot (mainline at T, ramp flows at T) in one place.
  - With it on, a fresh Task 2 build reproduces v16 exactly and the causality check passes.
- **Queue persistence is not shifted.** P(still queued 30 min later | queued now), from observed readings, mean over the 8 Task 2 panels:
  - train months: 0.862 (panel range 0.64–0.95);
  - validation: 0.868;
  - private: 0.853.

  At 15 min: 0.929 / 0.932 / 0.925. Per panel, the scored months stay within or near the train range. **The shift is in where and how often queues form, not in how a queue evolves.** That is consistent with dynamics features (A3b) transferring while recurrence features (qfreq, far links, sharper priors) failed.
- **What option B would cost** (no observation at T: `USE_T_SLOT = False`, frames rebuilt, bench official / selector / all):
  - onset, 5 seeds: 0.9418 / 0.8979 / 0.9069, against 0.9481 / 0.9070 / 0.9104 with T;
  - ongoing, 1 seed: 0.8381 / 0.8569 / 0.8633, against 0.8426 / 0.8828 / 0.8816.

  About −0.012 S_queue in total, roughly −0.004 overall. Keep ≤ T, as the rules allow; switch with one flag if the organizers rule otherwise.
- **Built and checked for Oct 10**, all against v16:
  - v25: Task 1 D5 + 2 seeds;
  - v26: v25 + D4 density, β 0.5 uncapped;
  - v27: onset 10 seeds, which changes 10 onset cells;
  - v28: ongoing 10 seeds, which changes 44 ongoing cells;
  - v29: Task 1 2 seeds + D5 at rate 0.15 with 300k rows.
- **Size of the density adjustment** (v26 against v25, all 6.7M Task 1 cells):
  - the flow ratio is 0.996–1.004 for 80% of cells and 0.943–1.067 for 99.8%;
  - only about 135 cells (2e-5) move by more than 30%, the largest 0.46× and 1.68×, e.g. 5,392 → 9,051 veh/h.

  Under RMSE a few large errors can cancel the small gains, so the factor can now be capped (`DENSITY["cap"]`). The holdout is being re-run with caps 1.1, 1.25 and 1.5.
- **Density cap on the holdout** (β 0.5, flow only). Every variant gives composite +0.00058:
  - cap 1.1: S_state 0.94271, S_physics 0.73877;
  - cap 1.25: 0.94247, 0.73933;
  - cap 1.5 or none: 0.94241–0.94242, 0.73945.

  The extremes are too rare in train months to matter. **Use cap 1.25** as protection on the shifted months at no holdout cost; v26 is rebuilt with it.
- **v26 rebuilt with cap 1.25** (flow ratio to v25 within 0.8–1.25; only Task 1 differs from v16). Queued for Oct 10.
- **Hypothesis rejected: no selection regularity in ongoing windows.**
  - The idea: the selector picks the earliest valid origin, so a change could enter only at T+30, as in onset.
  - Persistence IoU one step before T (origin T−5) averages 0.52 for the 40 official windows, the same as random valid windows. None exceed 0.9, with raw, median5 or 3×3-mean denoised labels alike. The persistence condition holds almost everywhere, so it does not set T.
  - Step by step, about 6 links appear and 6 clear at every horizon step in official windows (about 4 in random windows), with no last-step burst. The difference comes from bigger queues: 27 queued links at T−5 against 18.
- **Label noise** (training labels are observed speeds; official labels are the noise-free state) touches only about 2% of horizon cells near the threshold. Denoised training labels could gain at most a few thousandths of IoU, and only the leaderboard could measure it.

## 2026-10-10 — Task 1 options add up; onset 10 seeds lost again
| submission | change | public | Δ |
|---|---|---|---|
| v25 (57024489) | v16 + Task 1 D5 (rate 0.1, 150k) + 2 seeds | 0.86829 | +0.0007 vs v16 (≈ D5 + D3) |
| **v26 (57024644)** | v25 + Task 1 density adjustment (β 0.5, cap 1.25) | **0.86860** | **+0.0003 vs v25** (holdout +0.0006) |
| v27 (57024761) | v16 + onset 10 seeds | 0.86657 | −0.0011 (onset IoU −0.007) |
| v28 (57024863) | v16 + ongoing 10 seeds | 0.86812 | +0.0005 (ongoing IoU +0.003) |
| v29 (57024954) | v16 + Task 1 2 seeds + D5 at rate 0.15, 300k | 0.86840 | +0.0001 vs v25 |

- **New defaults** (`pipeline_final`):
  - Task 1: 2 seeds, D5 at rate 0.15 with 300k rows, density β 0.5 with cap 1.25;
  - Task 2: onset 5 seeds, ongoing 10 seeds (`T2_SEEDS` per family).

  Predicted public ≈ 0.86860 + 0.0001 (larger D5) + 0.00049 (ongoing 10 seeds) ≈ **0.8692**. v30 is built end to end with these defaults.
- Onset again: more seeds reshuffle the low-confidence validation windows (−0.0011). **Onset stays at 5 seeds.**
- **Can atypical train days stand in for the shifted months?** Days whose congestion map correlates least with the other months' average (bottom 20% per panel), with cached bench predictions re-scored on them. Δ IoU against the base (all days → atypical days):

| test | all days | atypical days | leaderboard verdict |
|---|---|---|---|
| ongoing far links | +0.0064 | +0.0065 | −0.008 |
| ongoing without recurrence | −0.0060 | −0.0029 | +0.019 |
| onset anomaly | +0.0011 | −0.0005 | −0.04 |
| onset sharper priors + context | +0.0009 | −0.0022 | −0.05 |
| onset adapted priors | −0.0061 | −0.0061 | −0.03 |

  Onset directions match on atypical days (3 of 3); ongoing directions do not (0 of 2). This is not a reliable proxy, so the leaderboard stays the judge for shift-sensitive changes.
- **v30 = the new defaults, end to end** (`pipeline_final --out v30`, 36.4 min):
  - onset rows equal v16 (5 seeds) and ongoing rows equal v28 (10 seeds), so Task 2 is deterministic;
  - Task 4 equals v16;
  - Task 1 equals v29's models plus the capped density adjustment (flow ratio to v29 within 0.8–1.25).

  Queued for Oct 11 (predicted ≈ 0.8692).
- **Causal estimate of links masked at T** (`t2_features.impute_origin`, per family via `IMPUTE_T_SLOT`): own T−5 value plus the mean T−5 → T change of observed links within ±2. It touches 30% of ongoing rows and flips the queued-at-T flag in 0.3%. Bench, official / selector / all:
  - ongoing, 1 seed: 0.8452 / 0.8834 / 0.8844 against 0.8426 / 0.8828 / 0.8816, **+0.003 / +0.001 / +0.003**;
  - onset, 5 seeds: 0.9399 / 0.9054 / 0.9088 against 0.9481 / 0.9070 / 0.9104, worse.

  Causality check passes with it on. Tested on the leaderboard for ongoing only as v31 (v30 + imputation, 10 seeds); v32 = v30 with Task 1 on 3 seeds.
- **Task 1 error analysis** (`experiments/t1_error_analysis.py`; current configuration trained on Jun–Nov, Dec–Feb holdout targets; speed RMSE km/h, flow RMSE veh/h/lane, interp → model):

| group | share of cells | speed RMSE | flow RMSE | share of model's squared speed error |
|---|---|---|---|---|
| free flow (v > 0.85 vf) | 95.4% | 2.40 → 1.46 | 44.0 → 33.0 | 77% |
| queued (v ≤ 0.6 vf) | 4.4% | 7.31 → 3.52 | 67.7 → 53.5 | 21% |
| transition | 0.2% | 7.57 → 6.16 | 178.5 → 105.1 | 3% |
| R1 / R2 / R3 | 18 / 30 / 52% | 2.65 / 2.69 / 2.95 → 1.48 / 1.54 / 1.72 | 43.5 / 44.4 / 47.7 → 33.3 / 33.6 / 35.3 | 15 / 27 / 58% |
| gap 1 / 2–3 / 4+ slots | 80 / 19 / 1% | 2.60 / 3.48 / 4.77 → 1.56 / 1.84 / 2.15 | — | 74 / 24 / 2% |

  - Free-flow errors are at the measurement-noise floor (σ 1.3–2.8 km/h, about 2.7% of flow). The only real Task 1 headroom is in queued cells: 4.4% of cells but 21% of the squared speed error, where the model already halves interpolation error.
  - R3 days carry 58% of the squared error.
  - Saved as `cache/t1_holdout_errors.parquet` for report figures.

## 2026-10-11 — v30 as predicted; the masked-link estimate at T transfers
| submission | change | public | Δ |
|---|---|---|---|
| v30 (57055736) | full new defaults, end to end | 0.86920 | predicted ≈ 0.8692 |
| **v31 (57056176)** | v30 + ongoing causal estimate of links masked at T | **0.87019** | **+0.0010** (ongoing IoU +0.0066; bench +0.001 to +0.003) |
| v32 (57056275) | v30 + Task 1 3 seeds | 0.86926 | +0.0001: not worth 50% more Task 1 runtime |

- **New defaults:** `t2_features.IMPUTE_T_SLOT = {"ongoing"}`; the production ongoing cache is the imputed frame (the previous one kept as `t2_ongoing_train_a3b.parquet`). A fresh `pipeline_final --tasks 2` reproduces v31's queue exactly, and the causality check passes.
- **Best public: v31 0.87019.**

## 2026-10-12 — learned estimate of masked links at T: more accurate, no leaderboard gain
- **`tfb.t2_impute`**, a learned causal estimate for links masked at the origin T.
  - Training: 1.09M train cells that are Task 1 targets at their slot, within 12 links of a queue five minutes earlier.
  - Features: own last 30 minutes, neighbours ±3 at T and T−5, the rule estimate, link data, time of day. LightGBM per channel, fitted month-held-out.
  - Held-out RMSE at T (normalized), T−5 value → rule → model:
    - speed 0.0620 → 0.0484 → **0.0293**;
    - flow 0.0382 → 0.0381 → **0.0244**;
    - occupancy 0.1358 → 0.1050 → **0.0634**.

    That is about 40% better than the rule.
  - Ongoing bench, 1 seed, against the rule: 0.8473 / 0.8857 / 0.8870 vs 0.8452 / 0.8834 / 0.8844 (+0.002 to +0.003 on every set).
  - **v33 (57110024)**, v31 with the learned estimate for ongoing at 10 seeds: **0.87010, −0.0001** (ongoing IoU −0.0006). No transfer.
- **Decision:** keep the simple rule (`IMPUTE_MODEL = False`), which is leaderboard-validated (+0.0010, v31) and simpler. The module stays as an option.
- **Task 1: extra congested training cells** (`t1_model.CONGESTED_EXTRA`, `experiments/t1_congested.py`).
  - Add up to N extra target cells per panel where the interpolated speed is below 0.85 × free speed.
  - Importance weights (uniform rate / inclusion probability) keep the loss unbiased.
  - Dec–Feb holdout, 1 seed, density on:
    - S_state 0.9425 → **0.9438**;
    - S_physics 0.7390 → **0.7420**;
    - positive on all 10 panels, mean total **+0.0009**, largest on the congested D12_I5_S/N, D7_I405_N and D7_I210_W.
  - To the leaderboard as v34 (150k per panel) and v35 (300k), each v31 with Task 1 rebuilt.
- **Leaderboard, extra congested cells** (Task 1 only, against v31 0.87019):
  - v34 (57119388), 150k per panel: 0.87092 (+0.0007);
  - **v35 (57120901), 300k per panel: 0.87097 (+0.0008), new best.**

  The holdout gain (+0.0009) transferred almost fully. **Default is now `CONGESTED_EXTRA = 300_000`.**

## 2026-10-19 — organizer rulings, award-check robustness, probes against the strongest public pipeline
- **Standing:** v35 (0.87097) is #109 of 511 on the public leaderboard; on Oct 9 we were #72 of 374. The top 10 start at 0.8999 and #50 is at 0.8828. A public notebook at 0.8714 (Oct 15) lifted many teams. Registration e-mail sent (Oct 19).
- **Organizer rulings, Oct 15–18** (`docs/discussions/745700-748789-749601-task2-origin-refit-compliance.md`):
  - the observation at T may be used (745700), so `USE_T_SLOT` stays and the −0.004 risk is gone;
  - a Task 2 model may be refitted on data before T, and April windows may use all March files (748789). Little use to us: the scored windows sit in the first days of each month (validation onset days 1–7, median day 3; ongoing median 3.5; private medians 3–4), and March is no template for April (LOG 2026-10-07);
  - the award check (749601) reruns the code with every row stamped after each window's origin removed; the Task 2 predictions must not change. The report should explain design choices with evidence, for example train-split results.
- **The award check would have crashed our pipeline.** `tfb.dense` asserted that every released day is a complete grid, so a release with rows removed stopped at the first step.
  - Now absent rows, days and files are left blank. The arrays from the complete release are byte-identical (checked on 4 panel-splits).
  - New `t2_bench causality --files`: the organizers' procedure for every scored window. It removes every row after the window's origin from every file, rebuilds the dense arrays from those files, predicts, and compares with the full-release prediction and the submitted `queue.csv`.
  - New `pipeline_final --queue-only`: Task 2 alone, on any release.
- **Probes against the 0.8714 public notebook** (its output `submission.csv`; each probe is v35 with one part replaced, `experiments/public_probe.py`):

| submission | change against v35 (0.87097) | public | Δ |
|---|---|---|---|
| v36 (57389210) | onset rows from the public notebook | 0.87150 | +0.0005 (onset IoU +0.004) |
| v37 (57389239) | ongoing rows from the public notebook | **0.87346** | **+0.0025** (ongoing IoU +0.017) |
| v38 (57389282) | Task 1 averaged 50/50 with the public notebook | 0.87202 | +0.0011 |

  - How different they are: queued-cell Jaccard onset 0.87 / 0.83 (validation / private), ongoing 0.92 / 0.90; Task 1 speed differs by 1.35 km/h RMS. Task 4 is the same solution (relative L1 5e-6).
  - **Onset**: +0.0005 is about one window; onset stays frozen.
  - **Ongoing is where we lag.** The notebook's own log credits most of its ongoing strength to averaging the probabilities of two different ongoing models (+0.0023) and to a 0.45 threshold on the average (+0.0011). One of the two is the public all-link model of the anubisyy notebook (CC0 assets).
  - **Task 1**: averaging with a different model family adds +0.001, although ours is the stronger one on its own.
- **Averaging our ongoing probabilities with the public all-link model** (`experiments/t2_public_blend.py`).
  - The public model is the 36-feature LightGBM of the anubisyy assets (CC0). It scores every link from the window history T−60..T−5.
  - p = 0.5 p_ours + 0.5 p_public, then our expected-IoU set (`choose_window`). Weight 0 reproduces v35 exactly (checked).

| submission | change against v35 (0.87097) | public | Δ |
|---|---|---|---|
| v39 (57389595) | average on our candidate cells (links within 12 of a history queue) | 0.87231 | +0.0013 (ongoing IoU +0.009) |
| **v40 (57389666)** | average over every link (p_ours = 0 off our candidates) | **0.87364** | **+0.0027** (ongoing IoU +0.018) |

  - **v40 is the new best public.** It beats the public notebook's own ongoing rows (v37, +0.0025) and changes 5% of our queued cells (Jaccard 0.947).
  - Far links help here, unlike the train-recurrence far-link model (−0.0012, v15). These probabilities come from current dynamics, not from train patterns.
- **File-level award check passes:** `causality --files --submission v35` gives all 160 windows unchanged with every row after T removed from the release, and equal to the submitted v35 rows.
- **Ported and retrained** the public model (`tfb.t2_public`, CC0): the same features and training (two origins per day and condition, Feb as the early-stopping month), deterministic LightGBM.
  - It reproduces the original checkpoint exactly: correlation 1.0000, mean |Δp| < 0.0001 on the 87,000 scored ongoing cells. Training takes 48 s.
  - In `pipeline_v1.predict_window`: p = 0.5 p_public + 0.5 p_ours over every link, then `choose_window`.
  - `pipeline_final --tasks 2 --base v35` → **v41, identical to v40 in every queue cell** (public 0.87364, no submission needed).
- **Train-split bench of the average** (`experiments/t2_public_bench.py`): leave-one-month-out for both models (ours from the production-feature bench, 1 seed), every link scored. Mean IoU over panels:

| window set | ours | public alone | avg 0.25 | avg 0.5 | avg 0.75 | avg 0.5 on our candidates |
|---|---|---|---|---|---|---|
| official (40) | 0.8452 | 0.7926 | 0.8359 | 0.8239 | 0.8110 | 0.8233 |
| selector (359) | 0.8834 | 0.8371 | 0.8825 | 0.8712 | 0.8525 | 0.8684 |
| all (11,321) | 0.8844 | 0.8307 | 0.8788 | 0.8636 | 0.8444 | 0.8616 |
| atypical days, all (2,066) | 0.8149 | 0.7518 | 0.8062 | 0.7862 | 0.7641 | 0.7848 |

  - **On train months the average hurts.** Per window, avg 0.5 − ours is −0.021 ± 0.0075 (SE) on the 40 official windows and −0.012 ± 0.0034 on the selector windows. Atypical train days (the 20% least like the other months) show the same.
  - **On the shifted March month it gains +0.018 IoU** over the 40 validation windows, about +2.3 SE at the train per-window spread.
  - The two estimates disagree well beyond noise. This is the month shift again (LOG 2026-10-07): our model is strong on train-like months, and averaging with a simpler current-dynamics model pays off only where the month differs.
  - **Consequence:**
    - Only the leaderboard supports this change; train-split evidence opposes it.
    - The report must say so, because the organizers ask for evidence behind design choices (749601).
    - It is the natural candidate for the hedge in the final pair: one submission with it and one without.
- **Clean-install reproduction test passed** (`~/data/tfb_clean/clean_test.sh`):
  - setup: a fresh conda environment from `environment.yml`, a fresh clone of commit e46296c, the organizer repository cloned from GitHub at c88cddf, and an empty data root that only links the raw release;
  - run: `pipeline_final --fresh`, 65.9 min while other jobs shared the CPU;
  - result: `submission.csv` is **byte-identical to v35**.
  - Repeat it for the final configuration at lock-in.
- **Both causality checks pass for v41** (with the public model):
  - `causality`: features and the public model's input are unchanged with every value after T blanked;
  - `causality --files --submission v41`: all 160 windows are unchanged with every row after T removed from the files, and equal to v41.
- **Second Task 1 model family** (`tfb.t1_public`, `experiments/t1_public_holdout.py`).
  - What it is: the 94-feature set of the same public notebook (CC0), retrained here with LightGBM and our Task 1 parameters instead of its GPU CatBoost.
  - How it trains: on synthetic masks of the unmasked train states (20/30/50% by a hashed day regime, two realizations per month, 750 cells per day).
  - Dec–Feb holdout: ours in the production configuration with 1 seed, both models trained on Jun–Nov. Each column is family-averaged.

| public weight | S_state | S_physics | R3 speed RMSE | R3 flow RMSE | composite | Δ |
|---|---|---|---|---|---|---|
| 0 (ours) | 0.94399 | 0.74258 | 1.580 | 33.93 | 0.44179 | — |
| **0.25** | **0.94491** | **0.74628** | 1.580 | 32.81 | **0.44266** | **+0.0009** |
| 0.5 | 0.94442 | 0.74463 | 1.628 | 32.40 | 0.44224 | +0.0005 |
| 1 (public alone) | 0.93948 | 0.72831 | 1.853 | 33.79 | 0.43807 | −0.0037 |

  - Weight 0.25 is positive on all 10 panels (+0.0006 to +0.0011). Alone the public feature set is clearly weaker, but its errors differ from ours. Unlike the ongoing average, this one has train-split evidence.
  - **New default** `pipeline_final.T1_PUBLIC_WEIGHT = 0.25`. v42 = v41 with Task 1 rebuilt; it is queued for the leaderboard.

## 2026-10-20 — the Task 1 average transfers
| submission | change | public | Δ |
|---|---|---|---|
| **v42 (57413788)** | v41 + Task 1 averaged with the public feature set (`t1_public`, weight 0.25) | **0.87420** | **+0.0006** (holdout +0.0009 with 1 seed) |

- **New best public: v42, 0.87420.** About two thirds of the holdout gain transferred, in line with earlier Task 1 changes.
- **Physics-based dissipation forecast for ongoing: not viable on this generator** (`experiments/eda/t2_dissipation.py`). The plan was kinematic-wave features: the queue tail's shockwave speed from the arriving and queued states, the bottleneck discharge, and a time-to-clear per link. On the train split:
  - the fundamental diagram holds (queued cells sit on the congested branch of the triangular FD, free cells on the free branch), but **the queue tail does not move like a shockwave**: over 10,809 training windows every kinematic estimate (production `w_tail`, FD density, whole-queue state, 15-min averaged states, demand-trend adjusted) correlates 0.03–0.10 with the actual tail motion over T+5..T+30 and has a larger error than persistence (MAE 3.2–4.0 km against 1.3 km at step 6). The tail's own motion over the previous 30 min is *negatively* correlated with its next motion (−0.6 at step 1);
  - **72% of the queued cells that clear are interior cells** (median 4 links from the tail and 4 from the head), and they jump straight to free speed (speed / threshold 1.63–1.69 for 75% of them; only 7.6% borderline);
  - space-time diagrams show why: congestion is a patchwork of per-link episodes (constant congested speed for hours, free links interleaved), with abrupt free ⇄ congested switches and no wave propagation. The clearing hazard is **memoryless**: P(still queued after 30 min) is 0.89 at every queue age (1 to 13+ slots) and depends only on depth (0.92 at v ≤ 0.3 v_f, 0.80 at 0.45–0.6 v_f).
  - What does predict clearing per cell (train Jun–Nov, test Dec–Feb, 1.1 M queued cells): speed trends (AUC 0.85 → 0.93 at step 3 on top of speed, age, time of day), density trends +0.004, neighbours +0.007. The production model already ranks clearing at **AUC 0.98–0.99** out of fold (official, selector and all windows), so dissipation is modelled about as well as the stochastic switching allows.
  - **Oracle bounds** (truth substituted for one cell class before the expected-IoU choice, OOF `imp` predictions): official windows 0.8452 → in-queue +0.058, near-new (≤ 3 links) +0.025, far-new (4–12) +0.018, all modelled cells +0.104; "keep every queued cell" −0.032. The headroom is spread over the classes and none of it is physics. Dropped; the remaining lever for ongoing is model diversity (a different family in the average), which is what paid on validation.
- **Convolutional ongoing model (`tfb.t2_cnn`), the second family for the ongoing average.** One sample is a window: per link, the 12-slot history T−60..T−5 and the slot at T (masked layer, blank links filled causally as in production) of speed, flow and density, with missing and queue flags, 8 static link attributes and the time of day (76 channels). A 1-D residual convolution along the corridor (6 blocks, dilations 1–8, receptive field about ±36 links, 174k parameters) outputs queue logits for every link and step. Weighted BCE on eligible cells, AdamW with a one-cycle schedule, 12 epochs, batch 16, seed 3; training windows are every 3rd valid origin, at most 12 per day (22,500 windows). PyTorch 2.14.1 (conda-forge, CPU build, added to `environment.yml`); the CPU run is deterministic (checked), the Apple GPU (`TFB_CNN_DEVICE=mps`, 10 s per epoch) is used for experiments only.
  - **Holdout** (train Jun–Nov, Dec–Feb bench windows, every link scored; `experiments/t2_cnn_bench.py --mode holdout`; "ours" is the leave-one-month-out production model, which has seen more months):

| windows | ours | public | cnn | prod (0.5/0.5) | prod + cnn 0.25 | prod + cnn 0.33 | prod + cnn 0.5 | ours + cnn 0.5 |
|---|---|---|---|---|---|---|---|---|
| selector (119) | 0.8911 | 0.8410 | **0.8987** | 0.8764 | 0.8929 | 0.8973 | **0.9048** | 0.9004 |
| all (1,511) | 0.8834 | 0.8287 | **0.9110** | 0.8622 | 0.8832 | 0.8904 | 0.9042 | **0.9104** |
| atypical days, all (314) | 0.8198 | 0.7589 | 0.8448 | 0.7878 | 0.8169 | 0.8246 | 0.8446 | **0.8566** |

  - Alone it beats our LightGBM on every set, and the average with the production pair at weight 0.5 gains +0.0285 ± 0.0074 (SE) per selector window and +0.042 ± 0.002 on all windows. The Dec–Feb loss was still falling at epoch 12 (0.0189).
  - The caveat is the month shift: a convolution with position and static channels can memorise link-specific train patterns, which the bench cannot penalise. Leaderboard probes decide: v43 = v42 with p = 0.25 ours + 0.25 public + 0.5 cnn, v44 at 0.335 / 0.335 / 0.33, v45 cnn alone (`experiments/t2_cnn_probe.py`, GPU-trained models).
  - **Leaderboard probes** (each is v42 with ongoing rebuilt; Tasks 1 and 4 and the onset rows are identical, checked):

| submission | ongoing probabilities | public | Δ vs v42 (0.87420) |
|---|---|---|---|
| v43 (57434502) | 0.25 ours + 0.25 public + 0.5 cnn | 0.87303 | −0.0012 (ongoing IoU −0.008) |
| v44 (57434593) | 0.335 ours + 0.335 public + 0.33 cnn | 0.87365 | −0.0006 (−0.004) |
| v45 (57434778) | cnn alone | 0.86847 | −0.0057 (−0.038) |

  - **The reversal is 0.03 IoU.** Alone, the CNN is about 0.02 IoU below our LightGBM on validation (v45 against the ours-alone level v42 − 0.0027) and 0.008 above it on the train holdout. The loss grows linearly with its weight, so no weight helps. Same lesson as the train-recurrence features (LOG 2026-10-07): the bench rewards what the train months share, and a convolution over the corridor learns the train months' congestion structure very well.
  - Holdout sweep (`experiments/t2_cnn_sweep.py`; cnn alone, selector / all): 20 epochs 0.9089 / 0.9156, width 96 0.9025 / 0.9163, every 2nd origin 0.9067 / 0.9184, against 0.8987 / 0.9110 at the defaults. All within the per-window noise, and the bench cannot rank this family anyway.
  - **Dynamics-only variant** (`STATIC_CHANNELS = False`: no position, static or time-of-day channels, 65 channels): holdout cnn alone 0.9043 / 0.9022, so the static channels add nothing on train; whether they are what fails on the shifted months is probe v46 (0.335 / 0.335 / 0.33, dynamics-only).
  - **v46 (57435112): 0.87427, +0.0001** (dynamics-only CNN at 0.335 / 0.335 / 0.33). Removing the static and time channels recovers the whole loss of v44 (−0.0006 → +0.0001), so they were what failed to transfer; what is left is neutral within noise. The CNN family stays **off** (`t2_cnn.WEIGHT = 0`, `STATIC_CHANNELS = False` by default). Not worth the extra dependency for +0.0001.
  - **Lesson for the report:** on this benchmark a model's train-split strength is not evidence of validation strength for the ongoing family. The three transfers so far: the public all-link LightGBM (36 local history features, early-stopped) gained +0.018 IoU on validation while losing 0.02 on the train bench; the CNN lost 0.02–0.04 on validation while gaining 0.01–0.03 on the bench; our own recurrence features lost. Simple, local, current-dynamics models transfer; models that learn the corridor's congestion structure do not.
- Submissions today: v42, v43, v44, v45, v46 (5/5).

## 2026-10-20 (evening) — Task 1 U-Net, a third model family
- **`tfb.t1_unet`:** a 2-D U-Net over each day's [288 slots × links] grid. Inputs (6 channels): the masked view's speed / 25, flow per lane / 600 and density / k_crit (0 where blank), an observed flag, and each link's linear time interpolation of speed and flow. Output: corrections to the interpolation at every cell in the scorer's units. Width 32 (1.8 M parameters), 3 pooling levels, GroupNorm + GELU; loss 0.54 MSE_speed + 0.46 MSE_flow on the hidden cells (the scorer's weights and units). Training on the unmasked train states with a fresh Bernoulli mask per day and epoch at the organizer's regime rates (20 / 30 / 50% of eligible cells; the released masks are i.i.d. per cell at those rates, run lengths 1–3 slots); AdamW, one-cycle, 20 epochs, batch 4 days, seed 5. One epoch is 1 min on the Apple GPU and 13 min on the CPU (0.6–2.7 s per batch of 4 days at L = 107–352), so `t1_unet.DEVICE` defaults to the GPU when there is one (`TFB_UNET_DEVICE` overrides; the CPU path is deterministic, the GPU path is not bit-reproducible) and `pipeline_final --unet-weights` loads the saved weights of a run for an exact reproduction.
- **Holdout** (train Jun–Nov, Dec–Feb with the real masks and the exact scorer replicas). Loss on fixed synthetic masks: interpolation only 0.00999, U-Net 0.00362 after 20 epochs (still falling slowly).

| model (family mean) | S_state | S_physics | R3 RMSE speed | R3 RMSE flow/lane | composite |
|---|---|---|---|---|---|
| ours alone (LOG 2026-10-19) | 0.94399 | 0.74258 | 1.580 | 33.93 | 0.44179 |
| public feature set alone | 0.93948 | 0.72831 | 1.853 | 33.79 | 0.43807 |
| production pair (0.75 / 0.25) | 0.94491 | 0.74628 | 1.580 | 32.81 | 0.44266 |
| **U-Net alone** | **0.9449** | 0.7419 | 1.554 | 33.24 | 0.4420 |

  - Alone, the U-Net matches the production pair on S_state and beats our LightGBM alone, with a completely different mechanism (2-D pattern completion against hand-crafted neighbour features). The average with the pair is scored next (`experiments/t1_unet_holdout.py`).
- **Average with the production pair** (`experiments/t1_unet_holdout.py`; base = the cached Dec–Feb predictions of ours (1 seed) 0.75 / public 0.25; at the target cells (1 − w) base + w U-Net, then `plausible_speed`):

| U-Net weight | S_state | S_physics | R3 RMSE speed | R3 RMSE flow/lane | composite | Δ |
|---|---|---|---|---|---|---|
| 0 (pair) | 0.94491 | 0.74628 | 1.580 | 32.81 | 0.44266 | — |
| 0.2 | 0.94642 | 0.75094 | 1.519 | 32.05 | 0.44389 | +0.0012 |
| 0.33 | 0.94701 | 0.75255 | 1.494 | 31.78 | 0.44433 | +0.0017 |
| **0.5** | **0.94729** | **0.75284** | **1.478** | **31.70** | **0.44448** | **+0.0018** |
| 1 (U-Net alone) | 0.94489 | 0.74190 | 1.554 | 33.24 | 0.44200 | −0.0007 |

  - Positive on all 10 panels at 0.5 (+0.0011 to +0.0033). The largest Task 1 holdout gain since the tuned model; Task 1 gains have transferred at about two thirds, so expect +0.001 on the leaderboard.
  - Probes queued for the UTC reset (`experiments/t1_unet_probe.py`: v42's Task 1 values averaged with the all-months U-Net, GPU-trained, 20 epochs, train loss 0.00297): **v47 at weight 0.5, v48 at 0.33.** Tasks 2 and 4 identical to v42 (checked).

## 2026-10-21 — U-Net adopted (public 0.87628)
- **v47 (57457356): 0.87628, +0.0021** over v42 (Task 1 averaged with the U-Net at weight 0.5). **v48 (57457601): 0.87599, +0.0018** (weight 0.33).
  - The holdout predicted +0.0018 / +0.0017, so this gain transferred in full (ratio 1.2), unlike the earlier Task 1 changes (about two thirds). The leaderboard also prefers the heavier U-Net weight more than the holdout does (0.33 → 0.5: +0.0003 public, +0.0001 holdout).
  - Largest single gain since the A3 ongoing features (v8).
- **Heavier weights on the holdout** (`t1_unet_holdout.py --weights 0,0.5,0.6,0.67,0.75`; composite Δ vs the pair): 0.5 **+0.0018**, 0.6 +0.0017, 0.67 +0.0015, 0.75 +0.0011; 0.5 is the peak on every panel. Even with the leaderboard's stronger preference for the U-Net (3× the holdout's 0.33 → 0.5 step), 0.67 would be within ±0.0001 of 0.5, so no further weight probe.
- **Decision:** `t1_unet.WEIGHT = 0.5`. The real candidate is a full `pipeline_final` run with the U-Net trained inside it (v49), compared with the probe v47 below.
- **v49** = full `pipeline_final` run, all tasks rebuilt, 89.6 min: Task 1 LightGBM 38 min, public feature set 6 min, U-Net 24 min on MPS, Task 1 prediction 12 min, Task 2 5 min, Task 4 4 min, merge 1 min. (Earlier Task-1-only rebuilds took 44–54 min, so the U-Net adds about 35 min.)
  - The U-Net trained inside the pipeline is **bit-identical** to the one trained standalone for the probe (64 of 64 tensors equal): MPS training was reproducible across two separate runs on this machine, with the same seed, data and software.
  - Against v47: every flow identical; 15 of 6.7 M speed cells differ (max 0.84 km/h, rel L1 7e-9) where the density cap binds between the two blends (the probe blended v42's already-capped values; the pipeline caps once after all blends). Tasks 2 and 4 byte-identical. Submitted as the real candidate (3/5 today).
  - **v49 (57460522): 0.87627** (v47: 0.87628; the 15 cells are worth 0.00001). **New current best configuration**, the full `pipeline_final` product at `t1_unet.WEIGHT = 0.5`, git ee9737b plus the weight change.
- Submissions today: v47, v48, v49 (3/5).

## 2026-10-26 — U-Net seed average
- A second U-Net seed (6) for the average, trained with the same recipe (`python -m tfb.t1_unet --holdout --seed 6`; the CLI now takes `--seed`, trains on all months without `--holdout`, and checkpoints every epoch after two runs were killed by Claude Code's memory guard while a 30-hour job of another project held most of the RAM). `t1_unet.predict_split` averages the corrections of a list of models; `t1_unet.SEEDS` lists the production seeds and `pipeline_final --unet-weights a.pt,b.pt` loads several.
- Holdout losses at epoch 20: seed 5 0.00362, seed 6 0.00356. Blend with the production pair at the target cells (`t1_unet_holdout.py --model holdout.pt,holdout_s6.pt`), composite Δ vs the pair:

| U-Net | weight 0.5 | weight 0.6 | alone (1.0) |
|---|---|---|---|
| seed 5 | +0.0018 | +0.0017 | −0.0007 |
| seed 6 | +0.0020 | — | −0.0003 |
| **seeds 5 + 6 averaged** | **+0.0022** (S_state 0.94768, S_physics 0.75435) | +0.0022 | +0.0006 |

  - The seed average adds +0.0004 over one seed at weight 0.5 and keeps 0.5 as the weight (0.6 equal). Alone, the 2-seed U-Net now beats the production pair (+0.0006), which one seed did not.
  - Probe: v50 = v42's Task 1 values averaged at 0.5 with the mean of the two all-months models (`t1_unet_probe.py --model all.pt,all_s6.pt`), to compare with v47 (0.87628). The all-months seed-6 training took 42.6 min on the GPU (25 min when the machine is idle).
- **v50 (57679479): 0.87644, +0.0002 over v47**, new best public (corrected 2026-10-31: first logged as +0.0016, a misplaced decimal). About half the holdout's +0.0004, the usual Task 1 transfer ratio. Against v47 the Task 1 values differ by rel L1 0.0007 (speed) and 0.0016 (flow); Tasks 2 and 4 identical.
- **Decision:** `t1_unet.SEEDS = (5, 6)`: the production U-Net is the average of two seeds' corrections. Real candidate v51 = Task 1 rebuilt with `pipeline_final --tasks 1 --base v49 --unet-weights v49/t1_unet.pt,cache/bench/t1_unet_all_s6.pt` (loading equals training: the seed-5 weights trained inside the pipeline were bit-identical to the CLI's, and the CLI trains seed 6 with the same call); the lock-in run trains both seeds from scratch.
- **v51 built** (Task 1 rebuilt, Tasks 2 and 4 copied from v49): 176 min because the machine was shared with the other project's job (LightGBM training 130 min against 38 min on 2026-10-21; loading the U-Nets takes seconds, prediction 28 min). Against v50: every flow identical, speeds rel L1 7e-9 (the same 15 density-cap cells as v49 vs v47); Tasks 2 and 4 identical. Saved as `t1_unet.pt` (seed 5) and `t1_unet_2.pt` (seed 6).
  - **v51 (57684017): 0.87643** (v50: 0.87644). **New current best configuration**, the `pipeline_final` product at `t1_unet.SEEDS = (5, 6)`, `WEIGHT = 0.5`; git ee9737b plus the uncommitted seed-average changes.
- Submissions: 2026-10-26 v50 (1/5); 2026-10-27 v51 (1/5).

## 2026-10-28 — Lock-in run v52 from commit c22e05d
- `python -m tfb.pipeline_final --out v52 --fresh` on an idle machine, from the clean commit: 116 min (dense arrays 2, Task 1 LightGBM 37, public feature set 6, U-Net seeds 5 and 6 on the GPU 48, Task 1 prediction 12, Task 2 frames 1, Task 2 training and prediction 6, Task 4 4, merge 1).
- **Byte-identical to v51** in all three files, and both U-Net seeds are bit-identical to the models trained separately (the seed-5 one for the third time). The documented command reproduces the submission exactly, GPU training included.
- **v52 (57714111): 0.87643**, as v51. **Lock-in candidate:** commit c22e05d, `pipeline_final --out v52 --fresh`, manifest in `submissions/v52`.
- **Award check passed:** `t2_bench causality --files --submission v52`, all 160 windows: with every row after each window's origin removed from every file of the release, the Task 2 predictions are unchanged and equal to v52's `queue.csv` (a re-confirmation: the check last ran on v41, and Task 2 is unchanged since). 35 min.
- Submissions today: v52 (1/5).

## 2026-10-29 — Clean-install test and the ongoing public-model weight
- **Clean install:** `conda env create -f environment.yml -p ~/data/tfb_clean/env2` + `pip install -e .` (10 min). Same versions as the working env, newer conda-forge build numbers of pytorch/libtorch, pyarrow and scipy, llvm-openmp 23.1.4 vs 23.1.2; PyTorch finds the GPU. The full pipeline runs in it as v53 (`--fresh`), to be compared with v52.
  - **v53 (clean environment) is byte-identical to v52** in all three files, and both U-Net seeds are bit-identical, despite the different build numbers. 168 min (the machine was also building the probes below). The install instructions in REPRODUCE.md therefore reproduce the submission exactly. Not submitted (identical content).
- **Hedge probes = the first tuning of the ongoing public-model weight** (`t2_public.WEIGHT`, 0.5 since v40 with no other value tried; `t2_cnn_probe.py --pub w --cnn 0 --base v52`, Task 2 only):

| version | public weight | public | Δ vs v52 | queue cells changed |
|---|---|---|---|---|
| v54 (57782592) | 0.65 | 0.87385 | **−0.0026** | 4.3% |
| v52 | 0.5 | 0.87643 | — | — |
| **v55 (57782829)** | **0.35** | **0.87656** | **+0.0001** | 2.7% |

  - Strongly asymmetric. The weight does two things in `pipeline_v1.predict_window`: it blends the two models on our candidate cells (links within 12 of a history queue) and it scales the public model's probability alone off them (ours is 0 there), so a larger weight both trusts the public model more and lets more far cells into the expected-IoU set.
  - **v56 (57783051), weight 0.25: 0.87568, −0.0008** vs v52 and −0.0009 vs v55 (3.7% of queue cells changed). Along the coupled axis: 0.65 −0.0026, 0.5 0, 0.35 +0.0001, 0.25 −0.0008. (Corrected 2026-10-31: the v55 and v56-vs-v55 deltas were first logged ten times too large.)
  - **Decoupling** (`t2_public.OFF_SCALE`, the public model's scale off our candidate cells, `None` = the blend weight; probe builder `--off`): v57 = blend 0.5 on the candidate cells with off-scale 0.35; v58 = blend 0.35 with off-scale 0.5. Their deltas against v52 split v55's +0.0001 between the two effects.

| version | blend on candidate cells (public weight) | public scale off them | public | Δ vs v52 | queue cells changed |
|---|---|---|---|---|---|
| v52 | 0.5 | 0.5 | 0.87643 | — | — |
| v55 | 0.35 | 0.35 | 0.87656 | +0.0001 | 2.7% |
| v57 (57783233) | 0.5 | 0.35 | 0.87559 | −0.0008 | 0.35% |
| **v58 (57783362)** | **0.35** | **0.5** | **0.87712** | **+0.0007** | 2.4% |

  - The two effects pull in opposite directions: on our candidate cells the public model should weigh *less* than 0.5 (0.35: +0.0007 with the off-scale kept), while off them its scale should stay near 0.5 (lowering it to 0.35 costs 0.0008). v55 moved both and the two cancelled. A modest gain, comparable to the causal estimate at T (v31, +0.0010); S_queue rises by about 0.0023. (Corrected 2026-10-31: first logged as +0.0069, a misplaced decimal.)
  - Reading: `choose_window` maximizes the expected IoU from the probabilities, so their calibration decides the set. Ours is trained on the candidate cells and calibrated there, so mixing in more of the public model hurts; off the candidates only the public model exists, and its probabilities need the 0.5 scale to balance against the candidate cells' values.
  - Next (queued for the UTC reset): v59 blend 0.2 / off 0.5, v60 blend 0.35 / off 0.65, v61 blend 0.2 / off 0.65; then a refinement and the lock-in candidate at the chosen setting. v52 (untuned weights, 0.87643) becomes the hedge pick.
- Submissions today: v54, v55, v56, v57, v58 (5/5).

## 2026-10-30 — Around the peak

| version | blend on candidate cells | public scale off them | public | Δ vs v58 (0.87712) |
|---|---|---|---|---|
| v59 (57785941) | 0.2 | 0.5 | 0.87681 | −0.0003 |
| v60 (57786176) | 0.35 | 0.65 | 0.87640 | −0.0007 |
| v61 (57786335) | 0.2 | 0.65 | 0.87595 | −0.0012 |

  - Both axes fall away from v58. Blend at off-scale 0.5: 0.2 +0.0004, 0.35 +0.0007, 0.5 0 (vs v52); a quadratic through the three points peaks at 0.32, within 0.0001 of 0.35. Off-scale at blend 0.35: 0.5 best, 0.65 −0.0007; at blend 0.5, 0.35 cost 0.0008, so the off-scale peak is between 0.4 and 0.55. (Corrected 2026-10-31: these deltas were first logged ten times too large.)
  - Refinement probes: v62 blend 0.3 / off 0.5, v63 blend 0.35 / off 0.45. The best of v58/v62/v63 becomes the production setting.

| version | blend on candidate cells | public scale off them | public | Δ vs v58 | queue cells changed vs v58 |
|---|---|---|---|---|---|
| v62 (57787121) | 0.3 | 0.5 | 0.87703 | −0.0001 | 0.5% |
| **v63 (57787778)** | **0.35** | **0.45** | **0.87721** | **+0.0001** | 0.1% |

  - A plateau: blend 0.3–0.35 with off-scale 0.45–0.5 are all within 0.0002. **Production: `t2_public.WEIGHT = 0.35`, `t2_public.OFF_SCALE = 0.45`** (v63, the measured best, inside the plateau on both axes). Total from the weight tuning: +0.0008 over v52 (S_queue about +0.0026), small but consistent across the plateau (v58, v62, v63: +0.0006 to +0.0008).
  - For the report: this is a leaderboard-tuned choice (40 validation ongoing windows; the train bench cannot rank it because the public model loses there, LOG 2026-10-20). The hedge pick v52 keeps the untuned 0.5 / 0.5.
- Submissions today: v59, v60, v61, v62, v63 (5/5). The lock-in run at the new setting (v64, fresh from the next commit) and its award check run tonight; v64 is submitted after the UTC reset.

## 2026-10-31 — Lock-in run v64 from commit e89d9c4
- `pipeline_final --out v64 --fresh` from the clean commit e89d9c4: 247 min (the machine was shared again; 116 min idle). **Byte-identical to the probe v63** in all three files, and to v52 in Tasks 1 and 4 (Task 2 differs in 2.5% of the queue cells, the tuned weights). Both U-Net seeds therefore trained to the same weights for the fourth time.
- **Award check passed** on v64: all 160 windows unchanged with every row after T removed from the files, and equal to v64's `queue.csv`.
- **v64 (57820324): 0.87721**, equal to v63. **Final candidate:** commit e89d9c4, `submissions/v64` (manifest, models, zip). Final pair to select on Kaggle: **v64** (primary) and **v52** (hedge: the untuned 0.5 / 0.5 ongoing average, 0.87643, commit c22e05d).
- Submissions today: v64 (1/5).
- **Corrections (2026-10-31):** while writing the report every leaderboard delta was recomputed from the raw scores. Six deltas logged on 2026-10-26, 29 and 30 had a misplaced decimal (ten times too large): v50 +0.0002 (not +0.0016), v55 +0.0001, v58 +0.0007 (not +0.0069), v59 −0.0003, v60 −0.0007, v61 −0.0012, and the weight tuning as a whole +0.0008 (not +0.0078). The entries above are corrected in place; the decisions they supported are unchanged (the tuned weights are a small, consistent gain; v52 remains the hedge pick).
- **Report:** `report/REPORT.md` rewritten for the final configuration (every number from this log), figures regenerated (`experiments/report_figures.py`), PDF built with `experiments/report_build.py` (python-markdown + headless Chrome; 13 pages). The private score is left to fill in.

## 2026-11-06 — Final submissions selected
- On Kaggle, the user selected **v64** (57820324, public 0.87721) and **v52** (57714111, public 0.87643) as the two final submissions.

## 2026-11-06 — Disk cleanup
- The disk had 1.6 GiB free. Nothing in the final build reads older versions; only partial rebuilds, the old pipelines' own entry points and experiment tools do. Deleted:
  - `state.csv` and `submission.csv` of every version except v52 and v64 (137 files, 66.1 GiB). Each version keeps its uploaded zip, `queue.csv`, `odme.csv`, models and manifest;
  - the downloaded competition zip (7.8 GiB), after checking that all 9,688 of its files match the unpacked release in size and CRC;
  - the clean-install test's conda environments and outputs in `~/data/tfb_clean` (7.0 GiB); its script, log and repository clone are kept.
- 81 GiB free afterwards. v52 and v64 are complete, including v64's two U-Net weight files.

## 2026-11-07 — Private leaderboard
| submission | public | private | private − public |
|---|---|---|---|
| **v64** (57820324), tuned ongoing weights 0.35 / 0.45 | 0.87721 | **0.86769** | −0.0095 |
| v52 (57714111), untuned 0.5 / 0.5 | 0.87643 | 0.86647 | −0.0100 |

- **Final private score: 0.86769** (v64, the better of the two picks).
- v64 and v52 differ only in the two ongoing blend weights, so the private month tests that leaderboard-only choice: **+0.0012 on private** against +0.0008 on public. The tuning transferred.
- Both picks lost about 0.01 from public to private. Component probes exist only for the public split, so the gap cannot be split by task; the private month was the less train-like of the two scored months (LOG 2026-10-07).
- Report updated on 2026-11-08 (page 1, summary, Sections 5.3, 7 and 9) and the PDF rebuilt.

## 2026-11-08 — Final package
- `README.md` is now the package's overview and quick start: the selected submissions, the entry script, the runtime and hardware, reproduction steps, the award check, contents and declarations. The earlier competition notes moved to `docs/competition-notes.md`.
- `models/` ships the selected run's two U-Net models (`t1_unet.pt` for seed 5, `t1_unet_2.pt` for seed 6, byte-identical to `submissions/v64`) and `manifest_v64.json`. REPRODUCE.md's exact-reproduction command now loads them from there.
- The e-mail package is built into `dist/`, which git ignores: the code as `git archive` of this commit, the report PDF and the e-mail text.
- **Package test passed.** The zip (commit 6d8fbe7) was unpacked into `~/data/tfb_pkgtest` and run as its README says, with the organizer repository cloned from GitHub at c88cddf: `pipeline_final --out pkgtest --fresh --unet-weights models/t1_unet.pt,models/t1_unet_2.pt`, in 67.1 min. All three task files and `submission.csv` are **byte-identical to v64**.
- E-mail ready in `dist/`: `trafficflowbench-jarski-code.zip` (14.4 MB), `trafficflowbench-jarski-report.pdf` (0.7 MB) and `email.txt`.

## 2026-11-08 — What held up on the private month
- Kaggle now shows the private score of every submission, so each paired single-change test that was judged on the public month (March) can be re-read on the private month (April). Δ = total score, the changed version against its base:

| family | change | pair | public Δ | private Δ | same sign |
|---|---|---|---|---|---|
| Task 1 | LightGBM residual model | v3 vs v1 | +0.0126 | +0.0122 | yes |
| Task 1 | leave-one-day-out training profiles | v5 vs v4 | +0.0016 | +0.0024 | yes |
| Task 1 | profiles from the scored month | v9 vs v8 | +0.0014 | +0.0015 | yes |
| Task 1 | day anomaly against the month profile | v15 vs v13 | +0.0007 | +0.0005 | yes |
| Task 1 | without training-month profiles | v16 vs v15 | +0.0011 | +0.0012 | yes |
| Task 1 | self-supervised rows and two seeds | v25 vs v16 | +0.0007 | +0.0007 | yes |
| Task 1 | density adjustment | v26 vs v25 | +0.0003 | +0.0003 | yes |
| Task 1 | extra congested training cells | v35 vs v31 | +0.0008 | +0.0008 | yes |
| Task 1 | public feature set at 0.25 | v42 vs v40 | +0.0006 | +0.0005 | yes |
| Task 1 | U-Net at 0.5 | v47 vs v42 | +0.0021 | +0.0021 | yes |
| Task 1 | second U-Net seed | v50 vs v47 | +0.0002 | +0.0001 | yes |
| Task 4 | KL projection of the raw prior | v2b vs v1 | -0.0302 | -0.0300 | yes |
| Task 4 | KL projection of the gravity fit | v2a vs v1 | -0.0732 | -0.0735 | yes |
| ongoing | dynamics features (A3b) | v8 vs v5 | +0.0038 | +0.0017 | yes |
| ongoing | without training recurrence features | v13 vs v10 | +0.0029 | +0.0010 | yes |
| ongoing | recurrent far links | v12 vs v10 | -0.0012 | -0.0010 | yes |
| ongoing | causal estimate of links masked at T | v31 vs v30 | +0.0010 | +0.0015 | yes |
| ongoing | learned estimate of links masked at T | v33 vs v31 | -0.0001 | +0.0002 | NO |
| ongoing | without time, weekday, position and panel | v23 vs v16 | -0.0004 | +0.0012 | NO |
| ongoing | 10 seeds instead of 5 | v28 vs v16 | +0.0005 | -0.0003 | NO |
| ongoing | ongoing rows of the strongest public notebook | v37 vs v35 | +0.0025 | -0.0010 | NO |
| ongoing | average with the public all-link model | v40 vs v35 | +0.0027 | +0.0003 | yes |
| ongoing | corridor CNN at 0.5 | v43 vs v42 | -0.0012 | +0.0014 | NO |
| ongoing | corridor CNN alone | v45 vs v42 | -0.0057 | +0.0018 | NO |
| ongoing | dynamics-only CNN at 0.33 | v46 vs v42 | +0.0001 | +0.0031 | yes |
| ongoing | blend weights set separately | v63 vs v52 | +0.0008 | +0.0012 | yes |
| onset | 5 seeds instead of 1 | v7 vs v5 | +0.0012 | -0.0027 | NO |
| onset | sharper priors with month context | v14 vs v10 | -0.0078 | -0.0006 | yes |
| onset | without the recurrence prior | v18 vs v13 | -0.0039 | +0.0012 | NO |
| onset | priors adapted to the month's activity | v19 vs v13 | -0.0044 | +0.0013 | NO |
| onset | 10 seeds instead of 5 | v27 vs v16 | -0.0011 | +0.0006 | NO |

- **Task 1 and Task 4 transferred exactly:** 11 of 11 Task 1 changes have the same sign and nearly the same size on both months (correlation +1.00), and the Task 4 probes match to 0.0003.
- **Ongoing: large, mechanistic changes held** (dynamics features, removing recurrence, the masked-at-T estimate, no far links, the separately set blend weights); **model-family choices did not:** the public notebook's ongoing rows (+0.0025 → −0.0010), the average with the public all-link model (+0.0027 → +0.0003) and the corridor CNN (−0.0012 → +0.0014 at 0.5; −0.0057 → +0.0018 alone). The dynamics-only CNN at 0.33, dropped as neutral on public (+0.0001), was the largest Task 2 gain on private (+0.0031). Removing time, weekday, position and panel features from ongoing lost on public and gained on private (+0.0012).
- **Onset: public verdicts were noise.** 1 of 5 signs agree (correlation −0.44): the changes that lost on public (adapted priors, no recurrence prior, 10 seeds) gained on private, and 5 seeds, kept for its public gain, lost 0.0027 on private. With about 40 onset windows per month, onset effects below about 0.005 cannot be read from one month.
- Both write-ups posted after the deadline report the same pattern (Task 1 transfers; less train-month climatology in Task 2 helps on April; the public leaderboard is a weak guide for Task 2).
- The best private score among all 64 versions is v62 (0.86840); the selected v64 is 0.0007 below it.
- **Scores above 0.90 on the private leaderboard** are submissions made before the 11 September rebuild, when horizon information leaked; the organizers confirmed on 2026-11-08 (thread 756746) that they are void and excluded from the final standings. Compliant teams reported private scores around 0.860–0.873.

## 2026-11-09 — Report: the private-month evidence
- `experiments/private_pairs.py` recomputes every paired public/private delta from the Kaggle submission list and writes `experiments/private_pairs.csv` (49 pairs); `report_figures.py` draws them as `fig_public_vs_private.png`.
- Report changes: Section 7.1 "What held up on the private month" (full table, figure, per-family summary, the chain decomposition: Task 1 +0.0240 public → +0.0237 private, Task 2 +0.0129 → +0.0027), the private column in the blend-weight table (5.3), the onset verdicts reversed on April (5.1), the CNN section retitled "lost on March, won on April" (5.4), the decision-rule caveat (2.4), the summary paragraph, and Lessons 1–4. The report, README, REPRODUCE.md and the e-mail are in the first person; a "Use of AI" declaration was added. Date on the report: 2026-11-09.

## 2026-11-10 — Final package sent
- The user e-mailed the final package to trafficflowbench@gmail.com: `dist/trafficflowbench-jarski-code.zip` (the repository at commit 1243da7, including `models/` with the selected run's two U-Net models) and `dist/trafficflowbench-jarski-report.pdf` (the report of 2026-11-09). Subject "Traffic Flow Bench Final Package – jarski". Deadline was 2026-11-10 23:59 AoE.
- Still to come from the organizers: the final standings with the void pre-rebuild submissions removed, the reproducibility run (section 8 of the report and REPRODUCE.md), and the award review. Winners are to be announced on 2026-11-18. Until prizes are settled, keep `~/data/tfb` (release, dense arrays, caches, `submissions/v52` and `submissions/v64`) and the `tfb` environment unchanged.
## 2026-11-18 — Correction: compliant scores reach 0.899
- The 2026-11-08 entry above said compliant teams reported private scores around 0.860–0.873. Write-ups published since then show compliant private scores up to 0.899: 8th place 0.89874 (development stopped 2026-10-20, so post-rebuild), 24th 0.88992, 83rd 0.87590, against 0.86748 (130th), 0.86044 (212th) and this team's 0.86769.
- The 8th-place write-up's March component levels (S_state 0.947, S_physics 0.754, onset IoU 0.832, ongoing IoU 0.883) put Tasks 1 and 3 about level with this solution and Task 2 well above it. The stronger solutions trained the ongoing model on selector-faithful windows (the 83rd-place replica matched 36 of 40 official ongoing windows by evaluating the persistence cap on a time-filled state; +0.0047 public for them, +0.0034 private for 24th place) and adapted Task 2 to the scored month per date (8th place).
- Report revised (dated 2026-11-18): Section 7.1 "Reading across teams" reworded and a bullet on where the stronger compliant solutions gained added; Section 7 "the gap to the top of the leaderboard" now reads "most of the gap to the best compliant solutions is in Task 2". The PDF e-mailed on 2026-11-10 and the one attached to the Kaggle write-up keep the old sentence.

## 2026-11-19 — Final results
- The organizers announced the winners (thread 760311): Gold Ashkan Zare, Silver jkatdare, Bronze takaito, K-12 award CoReRazed.
- Field: 1,802 entrants, 801 teams, 18,124 submissions; 126 teams registered for awards and 73 sent code and reports.
- **Final ranking (thread 760505):** 69 teams with a registration and a final package, ranked by private score. Gold 0.92823, silver 0.92560, bronze 0.90196, 4th–11th 0.891–0.900; **jarski 33rd, 0.86769**. The organizers state that the winners' code and reports were reviewed against the Task 2 time rule, so the 0.92+ scores are compliant. The reading in the 2026-11-08 entry, that the 0.90+ scores are void leak entries, was wrong for these teams.

## 2026-11-23 — Late submissions: four open questions tested
After the final ranking, four open questions were tested with late submissions, which Kaggle still scores. Code is in `experiments/late/`. Every forecast below was written before its submission.

**Probes (no modelling).**
- `late_p1`: v64 with every Task 2 prediction zero. Its score is Tasks 1 + 3 + 4 of v64, exactly, on both months.
  - Forecast: 0.6365–0.6385 public and private (from the v3 probes and the chain; the range is the unknown Task 1 share of v4).
- `late_p2`: v64 with speed and flow of every target times 2. S_state becomes 0 through its clip (normalizers 25 km/h and 600 veh/h per lane). Density q/v, and with it S_LWR, does not change. S_FD falls by 0.2378 on both months (`experiments/late/fd_local.py`: 0.9880 → 0.7502 validation, 0.9879 → 0.7501 private; the blank non-target cells are left out, about 1.5% of the eligible cells). So S_state = (v64 − late_p2 − 0.05 × 0.2378) / 0.35, and Tasks 2 and 4 cancel.
  - Forecast for S_state 0.948 (the holdout's level): 0.5335 public, 0.5240 private. Band for S_state 0.935–0.955: 0.5310–0.5380 public, 0.5215–0.5285 private.
  - This rests on one assumption: the organizers' conservation right-hand side does not depend on the submitted flows (LOG 2026-09-30, the flux "projected onto the observations").
- **Results of the two probes (2026-11-23 03:48 and 03:49 UTC; both inside their bands).**
  - `late_p1` (58756888): 0.63735 public, 0.63694 private. So Tasks 1 + 3 + 4 of v64 are level on the two months, and Task 2 of v64 is 0.23986 public (0.7995) and 0.23075 private (0.7692). With onset at 0.11588 and 0.10431, ongoing is 0.12398 public (IoU 0.8266) and 0.12644 private (IoU 0.8430).
  - `late_p2` (58756955): 0.53732 public, 0.52743 private. So **S_state of v64 is 0.9371 on the public month and 0.9382 on the private month**, against 0.948 on the Dec–Feb holdout. With Task 4 at 0.1986, S_physics is 0.738 public and 0.733 private (±0.002 for the Task 4 estimate), against 0.754 on the holdout; S_LWR is then about 0.61.
  - The gap between holdout and leaderboard in Tasks 1 + 3 (0.4448 against 0.4388 and 0.4383) is therefore about 0.0038 from state and 0.0024–0.0031 from physics.
  - A check of the blank-span hypothesis: 1.9% of the targets at a speed error of 9.6 km/h (plain interpolation across the span) instead of 1.6 km/h lower S_state by about 0.011. That is the measured difference. It does not prove the cause; the span experiment below tests it.

**Onset without training-month memory.**
- Train bench, leave one month out, 5 seeds (official / selector / all windows):
  - production (v4 features): 0.9481 / 0.9070 / 0.9104;
  - `late_nomem_s5`, without the priors, the clock and the position (`prior_tod`, `prior_all`, `qfreq_t0`, `prior_tod_up1`, `prior_tod_dn1`, `tod`, `weekday`, `pos`, `panel_code`): 0.8687 / 0.8485 / 0.8296;
  - `late_dyn_s5`, also without the static link attributes: 0.7800 / 0.7684 / 0.7638.
- For comparison, the production onset model scores 0.773 on the public month and 0.695 on the private month, far below its bench level. The bench cannot say what the no-memory model scores there.
- Candidate `late_onset_nomem`: v64 with the onset model retrained without those nine features (5 seeds, `experiments/tools/t2_variant.py`). A control build with no feature dropped must reproduce v64's onset rows.
  - Forecast: −0.005 public (band −0.013 to +0.003), +0.002 private (band −0.008 to +0.012). Reasons: on train-like months the change costs 0.06 IoU, which would be −0.009; the one comparable submission (v18, without `qfreq_t0` only) read −0.0039 public and +0.0012 private.
  - What would count as a finding: the same change differs between the two months by 0.0058 (root mean square over our onset pairs), so only a private change beyond ±0.006, or both months moving the same way by more than 0.004, says something.
  - The control build reproduced v64's queue rows exactly (0 of 174,000 differ). The variant changes 123 rows in 36 onset windows (18 per month); the other 22 onset windows per month are identical.
  - **Result (58758560, 2026-11-23 04:27 UTC): 0.87049 public, 0.87093 private. Against v64: −0.0067 public, +0.0032 private.** Both inside the forecast bands.
  - Onset IoU without the memory features: 0.728 public and 0.717 private, against 0.773 and 0.695 with them. So the memory features are worth +0.045 IoU in March and −0.022 in April, and without them the two months are level.
  - Reading, by the criterion written above: the private gain is below 0.006 and the two months move in opposite directions, so this is not evidence that a model without memory is better. Over both months it is worse by 0.0035. The same pattern as v18 and v19, now with all memory removed.

**Selection regularity in ongoing windows.** Scripts: `experiments/late/sel_*.py`.
- Selector replica: with the cap tested on the time-interpolated state and the state at the origin repeated, it matches 37 of 40 official ongoing train windows exactly and 39 within five minutes (forward-filled as in `selector_check.py`: 34 and 37; logged state: 14 and 16). Onset stays 39 of 40. What kept the selector from taking T−5 in the 40 official windows: the cap in 22, the 6 h spacing in 9, no established queue yet in 7, borderline 2.
- A flag computed from data ≤ T (T−5 was at least 6 h after the previous origin, had history coverage ≥ 0.7 and an established queue) finds all 22 cap-bound official train windows and flags 24; on 359 per-month replica windows precision is 180/182 and recall 180/180. Scored months: 23 of 40 validation and 20 of 40 private ongoing windows are flagged.
- Window IoU out of fold (the window's month left out), 40 official / 359 replica windows:
  - production: 0.8346 / 0.8378 (flagged windows 0.8819 / 0.8799);
  - rule A, flagged windows with steps 1–5 set to the persistence state from data ≤ T: 0.8252 / 0.8390. Production already forecasts near-pure persistence in 22 of the 24 flagged official windows, and where it departs on large queues it is right. No decode-only rule beats production on the official windows;
  - **candidate B, our LightGBM retrained on 11,678 selector-faithful train windows with the flag, a spacing flag and flag × step as features, inside the unchanged v64 blend: 0.8633 / 0.8646**, +0.0287 ± 0.0080 and +0.0268 ± 0.0035. Positive in all nine held-out months. Most of the gain is step 6 of flagged windows (0.64 → 0.82 for the new model alone). Without the three selection features the gain is +0.0160 / +0.0212, so the training windows carry most of it.
  - Negative control: on 480 ordinary origins that the selector did not choose, the new model is worse than production (−0.0243 ± 0.0028). Production scores 0.889 on ordinary origins and 0.835–0.838 on selector-chosen ones.
  - Controls of the evaluation: it reproduces LOG 2026-10-19 exactly (0.8452 / 0.7926 / 0.8239), the organizers' `score_window` agrees in 80 of 80 cases, and v64's probabilities rebuilt from its saved models reproduce v64's `queue.csv` in all 87,000 ongoing cells.
- Causality: for all 80 scored ongoing windows both candidates are identical with every value after T blanked in the arrays, and with every row stamped after T removed from the release files (the award check). A three-slot shift is caught in 80 of 80 windows.
- Disclosure: one ad-hoc look during this work printed, for five validation windows of D7_I210_E, the first readings after their blank spans (data stamped after T). They are used in no candidate and no forecast.
- Warning signs for B: 44% of the replica gain comes from one panel (D7_I210_E), the gain is nil on the 38 replica windows of the least typical days, and the train bench has reversed sign for ongoing model changes before.
- **Forecasts, written before submission** (change in total score against v64):
  - `late_sel_a` (rule A; 481 rows in 10 + 11 windows): public −0.001 (−0.003 to +0.001), private +0.002 (−0.002 to +0.004).
  - `late_sel_b` (candidate B; 314 rows in 30 + 30 windows): public +0.003 (−0.001 to +0.006), private +0.004 (0.000 to +0.008). At train rates it would be +0.004 on both.
- **Result of `late_sel_b` (58760776, 2026-11-23 05:22 UTC): 0.87833 public, 0.87131 private. Against v64: +0.0011 public, +0.0036 private.** Both inside the forecast bands. Ongoing IoU rises from 0.8266 to 0.8341 on the public month and from 0.8430 to 0.8671 on the private month. The private score is above every one of the 64 versions submitted before the deadline (best: v62, 0.86840).

**Blank spans in Task 1.** Scripts: `experiments/late/span_*.py`.
- Emulation on the Dec–Feb holdout (models trained on Jun–Nov only): 239 windows from the corrected selector, spans T+5 to T+90 blank on all links, histories unmasked and not targets; 201,255 span targets, 1.92% of the targets.
  - Speed RMSE inside the spans / outside: interpolation 9.85 / 2.79 km/h; our LightGBM 9.05; public-feature LightGBM 9.30; U-Net 9.45; **production blend 9.14 / 1.47**. The same cells when the spans are not blank: 1.90.
  - Production blend: S_state 0.9478 without spans and **0.9371 with them** (per month 0.9372 / 0.9373 / 0.9368). The leaderboard has 0.9371 public and 0.9382 private. S_physics 0.7578 → 0.7445; the leaderboard has about 0.738 and 0.733, so the spans explain about half of the physics difference.
  - **Cost of the spans: −0.0057 of total score** (S_state −0.0106, S_physics −0.0133).
- Fixes on the emulated holdout, change of total against the blend: span cells from interpolation −0.0008; from the U-Net alone −0.0004; best existing blend +0.0002; month profile shifted to the day −0.0007; the production U-Net fine-tuned on days with synthetic all-link spans +0.0032 (1 seed); with extra inputs (the month's profile, ramp flows, the window condition), 2 seeds, +0.0036; the same adapted to the month by self-supervision **+0.0037** (speed RMSE inside the spans 9.14 → 4.94 km/h; positive in all 24 panel-months).
- Transfer check on the scored months' own released observations with pseudo-spans: squared speed error against interpolation falls 73% on Dec–Feb, 69% on March and 63% on April.
- **Forecasts, written before submission** (change in total score against v64; each candidate changes exactly the 128,465 span targets, 64,720 in validation and 63,745 in private):
  - `late_span_interp` (span cells from plain interpolation; a diagnostic of what v64's models are worth there): −0.0008 on both months (−0.0012 to −0.0004).
  - `late_span_fix` (span cells from the span-aware U-Net): +0.0035 public (+0.0022 to +0.0043), +0.0032 private (+0.0020 to +0.0041).
- Not verified: the physics part of the gain on the leaderboard (its level is only half explained), and bit-identical retraining of the fix (trained on a shared GPU).
- **Result of `late_span_fix` (58765522, 2026-11-23 07:06 UTC): 0.87949 public, 0.86993 private. Against v64: +0.0023 public, +0.0022 private.** Both inside the forecast bands, at their lower edge: the gain is about two thirds of the holdout's +0.0037, which fits the state part alone transferring and the physics part only partly.

**Summary of the five late submissions of 2026-11-23** (all five landed inside the bands written before submission):

| submission | what it tests | public | private | against v64 |
|---|---|---|---|---|
| `late_p1` | Tasks 1 + 3 + 4 of v64 | 0.63735 | 0.63694 | Task 2 of v64 = 0.2399 / 0.2308 |
| `late_p2` | state against physics | 0.53732 | 0.52743 | S_state = 0.9371 / 0.9382 |
| `late_onset_nomem` | onset without training-month memory | 0.87049 | 0.87093 | −0.0067 / +0.0032 |
| `late_sel_b` | ongoing trained on selector-faithful windows | 0.87833 | 0.87131 | +0.0011 / +0.0036 |
| `late_span_fix` | span-aware model for the blank spans | 0.87949 | 0.86993 | +0.0023 / +0.0022 |

- The two changes that gained on both months touch disjoint rows, so together they give 0.8806 public and 0.8736 private (+0.0034 and +0.0059 over v64).
- Not submitted yet: `late_span_interp` and `late_sel_a`, two diagnostics with forecasts above.
