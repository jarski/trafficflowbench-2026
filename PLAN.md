# Solution plan: TrafficFlowBench (2026 IEEE Big Data Cup)

Written 2026-09-29, before the data was downloaded. Everything here comes from the organizers' scorer code and docs. Claims marked **(verify)** are inferences that Phase 1 checks against the data before we rely on them.

## 0. Improvement plan (2026-10-06): supersedes the phase plans in §5–§7

**Where we stand.** Public 0.85494 (v4), about #60 of 320. The top is 0.924, #10 is 0.893 and #20 is 0.883. Validation components:
- Task 4 ≈ 0.993: done.
- Tasks 1+3 ≈ 0.43 weighted: near the noise ceiling.
- Task 2 0.753: onset 0.764, ongoing 0.741.

The evidence is in LOG 2026-10-06.

**What this means:**
- **Task 2 is the only large lever.** Inside it, ongoing is the most fixable.
- **Onset failures on validation are mostly unpredictable from data ≤ T.** The model is calibrated (self-assessed 0.741 against 0.764 actual) and already rated those windows as hard.
- **Ongoing has two concrete error sources.** On the official train windows it scores 0.839 against a self-assessed 0.886:
  - queues we keep after they clear (38% of error cells);
  - new queues more than 3 links away that we miss (33%).
- **Our own model rates the private month easier than validation** (onset 0.86 against 0.74 self-assessed). So we keep only gains that hold on train-like months, not ones that fit validation's few odd windows.
- **A 0.92 needs S_queue ≈ 0.93 on validation.** We see no way to reach that with data ≤ T. Some top scores may use data after T, which the rules forbid and the organizers check before awards. We can't tell which, and we won't do it.
- **Realistic target:**
  - public +0.006–0.015 (about 0.862–0.870);
  - private around 0.88–0.89;
  - a strong report and clean eligibility.

### Ground rules for every change
1. **Decide on train, not on the leaderboard.**
   - Task 2: decide on the official-like CV (A1).
   - Task 1: decide on the Dec–Feb holdout with both `s_state` and `s_physics`.
   - The leaderboard only confirms single-task swaps. Never tune to validation's unusual windows (D7_I10_W, D12_I5_N).
2. One task per submission, against the current best. Log every result in `experiments/LOG.md`.
3. **Task 2 inputs stay at or before T.** A1 adds an automated causality test: blank every array after T and require identical features.
4. Delete the matching Task 2 cache whenever its features change. Keep `pipeline_final` deterministic and under about 30 minutes.

### A. Task 2 ongoing (0.15 of the total; official train windows 0.839, validation 0.741)
- **A1. Evaluation harness (first; everything else depends on it).**
  - Report the leave-one-month-out IoU on three window sets: the 80 official train windows, the replicated selector applied to all 9 months, and all replicated windows.
  - Add self-assessed IoU, so calibration is visible.
  - Add the causality test.
  - Done when one command prints these numbers for both the onset and ongoing models.
- **A2. Whole-corridor coverage.**
  - Score every link, not only those within 12 links of the queue.
  - Give each link the onset-style features (dynamic features, onset priors, upstream density), so the model can see secondary bottlenecks.
  - Targets the 33% "far" misses.
- **A3. Dissipation features.** Targets the 38% kept-too-long cells. Candidates:
  - time since congestion began at the head;
  - the usual end of congestion for this head and time of day (from train queue frequency);
  - upstream demand trend against profile;
  - discharge flow just downstream of the head against capacity (an incident or capacity-drop signature);
  - how recurrent the head location is.
- **A4. Training data and weighting.**
  - Use every valid origin, not stride 6 with a cap of 6 per day.
  - Up-weight windows that look like the official ones (the earliest valid origin of an episode).
  - Keep only what improves the official-like CV.
- **A5. Decision rule.** Compare the expected-IoU set with a fixed threshold, and with shrinking toward persistence, on official-like windows only.
- **Expected:** +0.02–0.04 official-window IoU, which is +0.003–0.006 total.

### B. Task 2 onset (0.15 of the total; validation 0.764, private self-assessed 0.863)
- **B1.** Adopt the anomaly features (out-of-fold 0.9071 → 0.9101, LOG 2026-10-02).
- **B2. Sharper priors:**
  - onset-time windows of ±30 or ±60 minutes instead of ±2 h;
  - day type (weekday / Saturday / Sunday);
  - "first onset of the day" conditioning;
  - a same-month, earlier-days prior as a feature. This is small: it helps only the 0.05–0.2 prior band.
- **B3. Renormalize when the probabilities are too low.** Every onset window has a queue at T+30, so rescale p when Σp is far below 1. Test it on the cached out-of-fold predictions; no rebuild needed.
- **Expected:** +0.01–0.02 onset IoU, which is +0.0015–0.003 total.

### C. Both Task 2 families
- **C1.** Seed or bagging ensembles (5 deterministic seeds) for onset and ongoing. A public notebook reports +0.008 public LB from bagging its Task 2 members.
- **C2. Stretch, only if A and B plateau by Oct 25.** Feasibility check for using a cell-transmission-model forward simulation as features. Go only if it reproduces a train day's queue head and tail within one link from boundary flows.

### D. Tasks 1 and 3 (0.50 of the total; small headroom)
- **D1. Leave-one-day-out profiles for training rows.** This fixes the training-time leak (LOG 2026-10-06). A public notebook gained +0.004 on the leaderboard from the same fix; their leak may have been larger.
- **D2. Month-adapted profiles.** Add a profile built from the scored month's own observed cells, leaving out the target day. Task 1 is offline, so this is allowed. It targets the 0.0075 gap between the holdout estimate and validation: each month has its own demand draw.
- **D3.** Two or three seeds, and all target cells instead of a 600k sample, if the runtime allows.
- **D4. Density-aware adjustment for Task 3.** Add a density (q/v) model and nudge (q, v) toward it. Tune on 0.35·S_state + 0.15·S_physics.
- **Expected:** +0.001–0.003 total.

### E. Task 4
No further work: ≈ 0.993, with about 0.0014 of total headroom.

### Schedule

| Dates | Work | Submissions |
|---|---|---|
| Oct 6–11 | A1 harness, D1, B1, B3, C1 | Task 1 swap (D1); Task 2 ensemble swap |
| Oct 12–18 | A2, A3, A4 (ongoing v2) | ongoing v2 swap |
| Oct 19–25 | B2, A5, D2–D4; **registration e-mail by Oct 25** | onset and Task 1 swaps |
| Oct 26–Nov 1 | C2 if worthwhile; report figures and error analysis | best combinations |
| Nov 2–6 | Lock in: `pipeline_final --fresh` rerun, reproducibility check, choose the final 2; last upload Nov 6 (cutoff Nov 7 06:55 UTC) | final |
| Nov 7–10 | Report PDF, code package, e-mail | — |

### Final-package checklist (due Nov 10)
- [x] `REPRODUCE.md`: hardware, runtime, data setup, organizer commit, single command, seeds, declarations.
- [x] `environment.yml` with exact versions.
- [x] License for our code: MIT (`LICENSE`).
- [ ] Clean-install test: create the environment from `environment.yml` in a fresh prefix and run `pipeline_final --fresh`; the result must match the selected submission. Passed for v35 on Oct 19 (byte-identical, `~/data/tfb_clean/clean_test.sh`); repeat for the final configuration.
- [ ] Check on Kaggle's Submissions page how many final submissions can be selected (probably 2). Then pick a hedged pair.
- [ ] Report PDF, from `experiments/REPORT_OUTLINE.md` and LOG.
  - First draft is `report/REPORT.md` (2026-10-12), with figures from `experiments/report_figures.py` in `report/figures/`.
  - Final numbers and the private score go in after selection; then convert to PDF.
- [x] Registration e-mail by **Oct 25** (template in §10): sent Oct 19.
- [ ] Award check (thread 749601): `python -m tfb.t2_bench causality --files --submission <final>` passes, and REPRODUCE.md documents it with `pipeline_final --queue-only`.

### Status 2026-10-19 (details in LOG 2026-10-19)
- **Best public: v40 = v41, 0.87364** (+0.0027 over v35). Ongoing probabilities are averaged 50/50 with a ported public all-link model (`tfb.t2_public`, CC0). Before it, v35 was #109 of 511; #50 is 0.8828 and #10 0.8999.
- **Rulings settled:**
  - observations at T are allowed;
  - the award check reruns Task 2 with every row after each origin removed. Our pipeline now survives that (`tfb.dense` accepts missing rows), and `causality --files` passes on all 160 windows;
  - the report must give the evidence behind design choices, preferably from the train split.
- **The averaging gain is leaderboard-only:** the train bench says −0.012 to −0.021 IoU, atypical days too. Candidate hedge for the final pair: with and without it.
- **Probes against the 0.8714 public notebook:** onset +0.0005 (noise level, onset stays frozen), ongoing +0.0025 (now captured), Task 1 averaging +0.0011 (open).
- **Next:**
  1. a second Task 1 model family, judged on the Dec–Feb holdout first;
  2. Task 1 second pass, if time allows;
  3. a clean-install test of the final configuration;
  4. the report, reframed around train-split evidence plus the leaderboard.

### Status 2026-10-09 (details in LOG 2026-10-09)
- **Best public: v21 0.86796** (#72 of 374). #50 is 0.8726, #20 0.8878; the gap is in Task 2 (≈ 0.78 against ≈ 0.84 for the 0.89 cluster).
- Task 1 self-supervised rows +0.0003 and 2 seeds +0.0003. Ongoing step shrink neutral; dropping tod/pos/panel features hurts both tasks.
- **Shift is in where and how often queues form**, not in their dynamics: queue persistence is equal across months.
- **Risk:** the T-slot question (thread 745700) is unanswered. Option B would cost about −0.004; the `USE_T_SLOT` switch is ready.
- **Queued for Oct 10:**
  - v25: Task 1 self-supervision + 2 seeds;
  - v26: v25 + density adjustment;
  - v27: onset 10 seeds;
  - v28: ongoing 10 seeds;
  - v29: v25 with twice the self-supervised data.

### Status 2026-10-08 (details in LOG 2026-10-08)
- **Best: v16 0.86763**. Task 1 now leaves out the train-profile features (+0.0011); the day anomaly gained +0.0007.
- Onset changes lost again: no qfreq_t0 −0.0039, month-adapted priors −0.0045. **Onset is frozen** at the v4 features with 5 seeds. Ongoing without the flow anomalies: −0.0004, not adopted.
- **Queued for Oct 9**, each against v16:
  - v20: Task 1 self-supervised rows (D5);
  - v21: Task 1 with 2 seeds (D3);
  - v22: ongoing step shrink;
  - v23: ongoing without tod/weekday/pos/panel_code;
  - v24: Task 1 without tod/pos/panel_code.

### Status 2026-10-07: strategy change (details in LOG 2026-10-07)

**Finding.** The train months are near copies of one scenario (congestion-pattern correlation 0.955 between them). Validation (0.843) and private (0.835) were drawn independently, and both are shifted.

**What follows:**
- Leave-one-month-out CV cannot see this shift. Every change built on train recurrence or train profiles helped the bench and lost on the leaderboard:
  - onset anomaly features: −0.006;
  - far links: −0.0012;
  - sharper onset priors: −0.0078.
- Removing recurrence features from ongoing gained **+0.0029** (v13, best 0.86586). Month-adapted Task 1 profiles gained 7× their holdout estimate.

**New rules:**
1. A change that relies on train-month patterns is judged on the public leaderboard, because both scored months are shifted. The bench judges only shift-free mechanics.
2. Prefer features about current dynamics and adaptation to the scored month. Reduce dependence on train recurrence, train profiles and train priors.
3. Final selection (probably 2 submissions; check the Submissions page): hedge across the uncertain shift-sensitive choices.

**Queued for Oct 8** (`submit_queue.py`; the Mac must be awake around 03:00 Helsinki), one change each:
- v15 = v13 + Task 1 day anomaly (D2b);
- v16 = v15 with Task 1 trained without train-profile features;
- v17 = v13 with ongoing also without the train-profile flow anomalies;
- v18 = v13 with onset without the train congestion map (qfreq_t0);
- v19 = v13 with onset priors rescaled by each bottleneck's activity this month.

**Ready for Oct 9:**
- D5: Task 1 self-supervised rows from the scored months (`T1_PSEUDO`). It improved the holdout on all 10 panels; it goes on top of the better of v15 and v16.
- D3: Task 1 seed averaging (`t1_model.SEEDS`), holdout running.

**Next:**
- onset priors made robust to the shift: wider and smoothed, or blended with the scored month's own earlier days;
- Task 1 trained on re-masked cells of the scored months (D5);
- registration e-mail by Oct 25.

### Status 2026-10-06 (end of day 1 of week 1; details in LOG 2026-10-06)

| item | result | decision |
|---|---|---|
| A1 bench and causality test | done; all 160 scored windows pass, and negative controls are caught | use for every Task 2 change |
| D1 leave-one-day-out profiles | v5 +0.0016 public | adopted |
| D2 month-adapted profiles (pulled forward) | v9 +0.0014 public, 7× the holdout estimate | adopted |
| B1 onset anomaly features | v6 cost about −0.006 public despite bench gains | **dropped** |
| B3 renormalization | no gain on the better model | dropped |
| C1 seeds | onset 5 seeds +0.0012 public (v7); ongoing 5 seeds inside v8 | adopted |
| A3 ongoing dissipation, tail dynamics and gap bridging (pulled forward) | bench gains on every set; v8 +0.0038 public | adopted |
| B2 sharper onset priors and context | bench mixed (official +0.007, selector +0.005, all −0.003) | not adopted |

- **Best: v10 0.86297 public** (Oct 7), the full current configuration. Predicted 0.86296: family scores add up exactly.
- **Lesson:** onset changes mostly flip validation's low-confidence windows, so the leaderboard cannot confirm small onset gains. Ongoing and Task 1 changes transfer predictably.
- **Next:**
  - more ongoing work: the remaining kept-too-long errors; tail growth more than 3 links upstream; the decision rule (A5);
  - Task 1 seeds and the density adjustment (D3, D4);
  - registration e-mail by Oct 25.

## 1. Where the points are

`S_total = 0.35·S_state + 0.30·S_queue + 0.15·S_physics + 0.20·S_ODME`

| Task | Weight | Organizer baseline (validation) | Our v0 (train, local) | Target | Gain vs baseline |
|---|---:|---:|---:|---:|---:|
| 1 State reconstruction | 0.35 | 0.693 (historical mean) | 0.909 (time interpolation) | 0.92–0.93 (noise ceiling ≈ 0.93–0.95) | +0.08 |
| 2 Queue forecast | 0.30 | 0.302 (persistence) | ≈ 0.52 | 0.75–0.85 | +0.14 |
| 3 Physics (scored on Task 1 file) | 0.15 | 0.355 | 0.678 | 0.69–0.72 (noise-limited) | +0.05 |
| 4 ODME | 0.20 | 0.836 | ≈ 0.84 | 0.92–0.96 | +0.02 |
| **Total** | | **0.553** | **≈ 0.74** | **0.83–0.88** | |

### Status 2026-09-30 (what the data showed; details in experiments/LOG.md)
- **Measurement noise:** independent from one 5-minute step to the next, about 2.7% on flow and 1.3–2.8 km/h on speed. This caps Task 1 near 0.93–0.95. It caps Task 3's S_LWR far below 1: time interpolation already reaches 0.70 / 0.56 / 0.32 across R1 / R2 / R3, and a perfect state estimate would reach only about 0.73 / 0.61 / 0.40.
- **Priorities change.** Tasks 1 and 3 are close to their ceiling with plain interpolation. The remaining Task 1 work is congestion handling with a learned model: modest gains, but still worth doing. **Task 2 (onset link choice and ongoing dynamics) and Task 4 are now where the leaderboard is decided.**
- **Local scoring we can trust:** `tfb.metrics` reproduces the official Task 1 and Task 3 scorers exactly (Task 3 via the reconstructed flux, S_LWR = 1.000 for the truth). Task 2 labels and window selection reproduce on the official train windows.
- **Leaderboard 2026-10-01:**
  - v0 0.80004 (quick wins);
  - **v1 0.84035** (learned Task 2 onset + ongoing models; S_queue up about 0.13);
  - v2a 0.767 and v2b 0.810: Task 4 multiplicative fits, both worse than the baseline NNLS.
- **Task 4:** the additive L2 projection of each month's own prior is clearly closest to the truth. Count-based local proxies cannot rank Task 4 methods; only single-task leaderboard swaps can. Next probes: least-squares projection of a prior rescaled to the counts (v2c is built), and a geometric-mean prior across months.
- **Decided 2026-10-01:** Task 2 uses all released data timestamped at or before T, including the masked-layer observation at T (about 68% of links), as the binding rules allow. The conflicting organizer code comment is ignored.

- **Reference point:** an entry scoring 0.869 was ranked 26th on 2026-09-29. The board still mixes in pre-rebuild scores, which are not comparable.
- **Priorities:**
  - Tasks 1 and 3 come from one submission and are worth half the score.
  - Task 2 has the most headroom.
  - Task 4 is nearly saturated.
- **Award mechanics:** to be considered for a prize, a team must be in the top 10 of the *eligible* teams on the private leaderboard. Eligible means: registered by Oct 25, code that reproduces the final submission to within 1%, and a PDF report. Many of the 226 Kaggle teams will not do this, so a solid Kaggle rank plus clean paperwork is enough.
- **Final award score:** 85% private leaderboard (min-max normalized within the top-10 pool) + 15% report.

## 2. What the scorers actually reward

### Task 3 can be scored locally, despite the docs
- In `score_task3.py`, each link's vehicle count is `N = length_km · q / v`, with the link's stations averaged first.
  - Cells that are not targets use the true observations.
  - Target cells use our Task 1 values.
- The hidden organizer boundary flows are "projected onto the released observations, so a participant who reproduces what was published satisfies the conservation identity exactly" (docstring of `load_boundary_flux`). So on every scored transition the right-hand side equals `ΔN_obs`:

  `E_LWR = Σ|ΔN_sub − ΔN_obs| / Σ|ΔN_obs|`, `S_LWR = max(0, 1 − E_LWR)`

- **What follows from this:**
  - The residual is nonzero only on 5-minute transitions that touch a target cell, where it equals the step-to-step change of our **density error**.
  - To score well, three things must hold:
    - density (q/v at link level) is accurate;
    - long links matter most, because N scales with link length;
    - predictions are continuous with the observed cells at each end of a masked run, and smooth inside the run rather than jittery.
- **Local proxy:** train ships the unmasked truth. We write a fake organizer flux file with `inflow − outflow := ΔN_obs/dt − r_on + r_off` and run the **official** `score_task3.py --split train --boundary-flux <file>`.
  - **(verify)** The historical-mean baseline should land near the organizers' 0.32–0.35 and the truth near 0.96.
- **Where the points are within Task 3:**
  - S_FD (1/3 of S_physics) is computed over all cells, mostly observed ones, so we can barely move it.
  - S_LWR (2/3) is the lever. The baseline gets about 0.04; going to near 1 is worth up to about 0.09 of S_total.

### Task 1
- **Scoring:**
  - The truth is the noisy **observation**, not the underlying state. There is a noise floor, and the best prediction is the conditional mean of the observation.
  - Flow is scored per lane (lane count from `fd_parameters`). The scales are 25 km/h for speed and 600 vph per lane for flow.
  - R1, R2 and R3 count equally per panel, even when one regime has only 4 days.
- **Masks:**
  - Cells are drawn independently per (day, time, link) at 20%, 30% or 50%, and each day belongs to exactly one regime.
  - All stations on a link are masked together.
  - A masked cell usually still has observed temporal and spatial neighbours, so this is a well-posed imputation problem.
  - The target cell's `pct_observed` is still published, so it can be used as a feature.
- **Tension with Task 3:** RMSE rewards getting q and v right separately, while Task 3 rewards getting q/v right. RMSE is quadratic, so nudging v̂ (or q̂) until q̂/v̂ matches a dedicated density prediction costs Task 1 little and gains Task 3 directly. We tune this blend on the local composite `0.35·S_state + 0.15·S_physics`.

### Task 2
- **Scoring:**
  - A link counts as queued when its underlying speed is at or below `v_cut` (from `fd_parameters`, 0.6 × free speed).
  - Only eligible cells are scored, and the score is space-time IoU per window.
  - Each panel and split has 5 onset and 5 ongoing windows, over 8 panels (D12_I405 is excluded).
  - Each window is worth 1/80 of S_queue, or 0.375% of S_total. With so few windows, the leaderboard's Task 2 signal is noisy.
- **Onset windows (half of S_queue):**
  - The queue appears *only* at T+30. The organizers confirmed this and said using it is allowed.
  - So we predict only the T+30 step, and the question is which links.
  - Every window contains a queue, so predicting nothing scores 0. Always predict at least one link.
- **Ongoing windows:** these were selected so that persistence IoU is at most 0.9, meaning the queue visibly changes. The job is to forecast the tail growing or the queue dissipating.
- **Training data:** train has 9 months of unmasked data. We replicate the organizers' window selector on it to get thousands of training windows instead of the 80 official train windows.
  - **(verify)** On the official train windows, the first queue should appear at T+30 in 39 of 40 onset windows.
- **Allowed inputs:**
  - Allowed: anything timestamped at or before T in any released file, including earlier days of the same month.
  - Not allowed: our Task 1 reconstruction, because it uses data after T. We need an automated test that the features are causal.

### Task 4
- **Scoring:** `S_ODME = 0.45·S_od + 0.25·S_link + 0.15·S_dev + 0.15·S_attr`.
- **The real problem:** the counts look exactly reproducible (the public notebook reached S_link 0.9999). So what remains is choosing path flows in the null space of A, meaning flows that leave all link counts unchanged.
- **S_dev** `= exp(−|D̂/D* − 1|)` compares how far we move from the prior with how far the truth sits from it.
  - A minimum-change projection moves less than the truth does (D̂ < D*), so the baseline's S_dev is probably well below 1.
  - Choosing deliberately how far to move from the prior can gain here.
- **Local proxy:** hold out some counted links, fit on the rest, and score the held-out counts.

## 3. Phase 0 — setup (Sep 29 – Oct 1)

**Blocking steps for you:**
1. Join the competition on Kaggle and accept the rules.
2. In Kaggle, go to Settings → API → Create New Token. Save the file as `~/.kaggle/kaggle.json` and run `chmod 600` on it.
3. Send the registration e-mail now; the deadline is Oct 25. The template is in §10.

**Setup I'll do:**
- Put the data **outside Dropbox** at `~/data/tfb/kaggle_public`, found via the `TFB_DATA` environment variable. The raw data is 13 GB and should not sync.
- Create a conda env `tfb` (Python 3.12) with pandas, pyarrow, numpy, scipy, scikit-learn, lightgbm, matplotlib and kaggle, and add torch later if needed. Pin versions from the start, for reproducibility.
- `git init` this folder. Our code goes in `src/tfb/`; `organizer-repo/` stays untouched.
- Convert each panel to dense arrays `[day, 288, station, channel]` covering speed, flow, occupancy, pct_observed, eligibility, target mask and regime, plus link-level arrays. That is about 250 MB per panel, loads in seconds, and makes every later step fast.

## 4. Phase 1 — evaluation setup and data exploration (Oct 1 – Oct 4)

The main deliverable this week is local scoring we can trust. **We never tune on the public leaderboard.**
- **Folds:** by time on train (Jun–Aug, Sep–Nov, Dec–Feb). March (public) and April (private) are generated independently of each other.
- **Task 1:** official `score_task1.py` on train, using the train masked layer, plus extra synthetic masks for more data.
- **Task 3:** fake flux plus official `score_task3.py`.
- **Task 2:** label construction, replica window selector, and an IoU scorer with the same aggregation as `score_task2.py` (window → panel × condition → panel → family).
- **Task 4:** cross-validation on held-out links.
- **One command** prints the local composite score per task and per panel.

**Data questions that decide the modelling:**
1. **Noise:** stations sharing a link give the noise σ for speed and flow, and show whether noise is independent per cell or drifts slowly. This decides how much smoothing and interpolation can gain.
2. **Generator:** do (q, k) follow the triangular fundamental diagram in `fd_parameters` (capacity multiplier 1.2)? If so, a cell-transmission-model (LWR) replica of the traffic generator is feasible.
3. **Congestion map:** where queues form by link and time of day, which bottlenecks recur, and what incidents look like.
4. **Masked runs:** how long runs of consecutive masked cells are per regime, and how often both temporal neighbours are observed.
5. **Official Task 2 windows:** the label rule, the T+30 property, persistence IoU, time of day, and how T is chosen.
6. **Task 4:** problem sizes, A·b versus the counts, whether the counts can be fitted exactly, and how the prior differs between splits.

**Submissions:**
1. Organizer baselines, as our leaderboard reference (about 0.553).
2. Quick wins:
   - Task 1: time and space interpolation.
   - Task 2: onset predicted at T+30 only, on the link most likely to break first.
   - Task 4: the baseline.

   Expected score about 0.75.

## 5. Phase 2 — core models (Oct 5 – Oct 14)

**Tasks 1 + 3**
- **v1 — interpolation blend.** For each station, combine:
  - time interpolation between the nearest observed values;
  - the adjacent stations' ratio to their historical profile;
  - the (station, weekday, time-of-day) profile as a fallback.

  Weights are fitted per gap length and regime.
- **v2 — LightGBM residual models** for speed, flow per lane, and link density. Features:
  - the same station at t±1..3, with gap lengths;
  - stations 1–3 up- and downstream at t±2, including occupancy;
  - profile mean and standard deviation;
  - the day's deviation from profile in the surrounding ±2 h;
  - link attributes (lanes, v_f, capacity, k_crit, v_cut, ramps);
  - congestion flags and the target cell's `pct_observed`.

  Training uses train days re-masked at all three rates; validation is by time fold.
- **v3 — Task 3 post-processing:**
  - make (q, v) consistent with the density model;
  - keep predictions continuous with observed neighbours at the ends of each run;
  - smooth predictions inside runs.

  Tuned on the local composite score.
- **Decision gate:** compare v2 against the noise floor from data question 1. If the gap is small, skip the neural stretch in Phase 3.

**Task 2**
- **Onset:** a "which link breaks first" classifier trained on replica onset events. Features for each link at and before T:
  - speed / v_cut and its trend;
  - flow / capacity and occupancy;
  - neighbouring links and on-ramp demand;
  - time of day and weekday;
  - how often this link has been the first to queue, from train and from earlier days this month.

  Choose the predicted link set at T+30 by maximizing expected IoU, which is not the same as thresholding probabilities.
- **Ongoing:** a (link, step) classifier using:
  - the current queue's extent and its distance to each link;
  - the tail's movement over the last 15, 30 and 60 minutes;
  - shockwave tail speed from upstream flow and density;
  - demand, time of day and the step number.

  Then choose the whole 6-step set per window by expected IoU.
- **Sanity check:** the persistence baseline on our local windows should reproduce about 0.25–0.30. If it does, the replica windows are faithful.

## 6. Phase 3 — Task 4, refinements, stretch goals (Oct 15 – Oct 24)
- **Task 4, estimator comparison on held-out links:**
  - L2 projection with a grid of λ values;
  - weights 1/b (errors proportional to flow);
  - maximum entropy (KL divergence to the prior, iterative proportional fitting);
  - lognormal MAP;
  - a gravity-model prior.

  Then pick how far to move from the prior by simulating S_ODME under the fitted noise model. Confirm with one leaderboard submission that changes only Task 4.
- **Stretch goals,** chosen by how far each task is from its ceiling:
  - Task 1: a U-Net that inpaints whole day grids (Apple GPU), ensembled with the gradient-boosted model. Also self-supervised training on the observed cells of the scored months, which is allowed because Tasks 1, 3 and 4 are offline.
  - Task 2: features from forward simulation with a cell transmission model, if data question 2 says the generator is reproducible.
- **Ensembles:** average across seeds and folds.
- **Registration e-mail:** if it hasn't gone out yet, send it now; the deadline is Oct 25.

## 7. Phase 4 — lock-in (Oct 25 – Nov 6)
- Final fit on all of train, plus the self-supervised step if adopted.
- Full clean re-run from raw data. The output must match the submitted file (the rule allows 1%).
- Pick the 2 final submissions: the most robust by local composite, and the best on the leaderboard.
- **The Kaggle cutoff is 2026-11-07 06:55 UTC (08:55 Finnish time). Aim to be done Nov 5.**

## 8. Phase 5 — package and report (Nov 6 – Nov 10)
- **Code package:**
  - single entry point `python -m tfb.pipeline`;
  - README with runtime and hardware;
  - locked environment and seeds;
  - external data declaration (none);
  - licences, all OSI-approved and allowing commercial use.
- **PDF report (15% of the award score):**
  - methods per task;
  - traffic-physics analysis;
  - error analysis across the four tasks;
  - ablations.

  It is drafted from `experiments/LOG.md`, which we keep from day 1. An optional short video can go with it.
- **E-mail** by Nov 10 23:59 AoE with the subject "Traffic Flow Bench Final Package – [Team Name]".

## 9. Submission discipline and risks
- **Budget:** 5 submissions per day. Log each one: date, git hash, config, local composite, leaderboard score.
- **Change one task at a time** against a fixed reference file. The leaderboard difference then isolates that task, weighted 0.30 for Task 2, 0.20 for Task 4, and 0.35 + 0.15 for Task 1.
- **Don't chase small public moves.** The public board is March only (80 Task 2 windows, 5 incidents), while private is April (7 incidents). Trust the 9-month local validation over small leaderboard changes.
- **Never use the leaderboard to back out validation answers.** The rules prohibit it.
- **Risks and mitigations:**
  - *The organizers change the data or rules again (they have twice).* Keep the pipeline re-runnable end to end in under about 3 hours, and check the forum twice a week.
  - *A proxy is wrong (Task 3 flux, Task 2 selector).* Calibrate each against the organizers' published baseline numbers first. Confirm on the leaderboard with single-task swaps.
  - *Irreproducible results on other hardware.* Use deterministic LightGBM settings and fixed seeds. Train any neural model on CPU, or ship its weights plus the training script.
  - *Timestamp keys drift.* Carry the released timestamp text byte for byte, never re-serialize it.
  - *Submission validity.* Validate before every upload: all rows present, no blanks, `queue_pred` binary, path flows non-negative.

## 10. Registration e-mail (send now)

To: trafficflowbench@gmail.com
Subject: Traffic Flow Bench Registration

```
Kaggle team name: [team name as shown on the leaderboard]
Team members:
  - Kaggle username: [username]
    Full name: [full name]
    Affiliation: [affiliation, or "independent"]
    E-mail: [e-mail]
Primary contact: [full name], [e-mail]
```
