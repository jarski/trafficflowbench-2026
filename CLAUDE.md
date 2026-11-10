# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Our solution for the Kaggle competition **2026 IEEE Big Data Cup – TrafficFlowBench** (https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench).

- **Read first:**
  - `docs/competition-notes.md`: prizes, deadlines, eligibility (our notes of 2026-09-29).
  - `README.md` is the final package's overview and quick start. `models/` holds the selected submission's two U-Net models and manifest.
  - `PLAN.md`: plan, status and decisions.
  - `experiments/LOG.md`: every experiment and leaderboard result, newest last.
- `experiments/LOG.md` and `experiments/REPORT_OUTLINE.md` become the PDF report, which is 15% of the award score. Log every result.
- `docs/README.md`: links to the competition pages and a summary of the organizer rulings this solution relies on, which are binding. The pages and forum threads themselves are not republished here.
- `organizer-repo/`: upstream clone of the official scorers and baselines at commit c88cddf. Not tracked by our git; never edit it. `tfb` imports its scorer helpers from `organizer-repo/src`.
- **Git:** this public repository is a single-commit snapshot of the private development repository (branch `solution`, commit df79a6d of 2026-11-10). Commit ids cited in the report, REPRODUCE.md and the log refer to that repository. `pipeline_final` records the current commit in `manifest.json`.

**Deadlines (2026):**
- Oct 25: registration e-mail. Sent by the user on Oct 19 (team `jarski`).
- Nov 6: last Kaggle submission. The Kaggle cutoff is 2026-11-07 06:55 UTC.
- Nov 10: code and report. The code must reproduce the selected final submission to within 1%.

## Environment and data

- **Python:** `~/miniconda3/envs/tfb/bin/python`, Python 3.12, conda-forge. `tfb` is installed editable from `src/`. pandas is pinned below 3 because the organizer scripts need it. There is no test suite.
- **Data** lives outside Dropbox in `~/data/tfb`; override with `TFB_DATA`. Paths are in `tfb/paths.py`.
  - `kaggle_public/`: the raw release and `--release-root` for organizer scripts. It is the rebuilt data of 2026-09-11.
  - `dense/<panel>/<split>/*.npy`: arrays shaped `[day, 288, link]`, plus `meta.json` and `days.json`. Built by `python -m tfb.dense` in 30 s.
  - `cache/`: Task 2 training frames `t2_onset_train.parquet` (v4 features) and `t2_ongoing_train.parquet` (A3b), and the Task 3 flux. **Delete or rebuild the matching cache whenever you change Task 2 feature code**; frames are only rebuilt when missing, and the model uses exactly the cache's columns. Build a variant into another file with `--out` and compare it on the bench before replacing a cache.
    - `_v1` and `_v2` are older ongoing frames.
    - `_anom` and `_b2` are onset experiment frames.
  - `cache/bench/`: out-of-fold predictions per bench tag.
  - `submissions/<version>/`: per-task CSVs, `submission.csv`, the upload `.zip`, models and `manifest.json`. `submissions/queue.log` records the scheduled submissions.
    - Since 2026-11-06 only the two selected versions, v52 and v64, keep `state.csv` and `submission.csv`. Older versions keep the zip (the uploaded file), `queue.csv`, `odme.csv`, models and manifest, so they can no longer be a Task 1 `--base` or a `compose` source.
- **Kaggle CLI 2.x** authenticates with `~/.kaggle/access_token`. The Kaggle MCP connector needs `/mcp` re-auth for most calls.

## Commands

```bash
PY=~/miniconda3/envs/tfb/bin/python
$PY -m tfb.pipeline_final --out v11           # end to end -> ~/data/tfb/submissions/v11 (about 2 h with the GPU; --fresh rebuilds caches)
$PY -m tfb.pipeline_final --out v11 --tasks 2 --base v10   # rebuild only Task 2 (3 min); copy the other tasks from v10
$PY -m tfb.compose --out v12 --state v10 --queue v11 --odme v10   # submission from per-task files of earlier versions
$PY -m tfb.kaggle_submit ~/data/tfb/submissions/v5/submission.csv -m "v5: ..."   # zip, submit, wait for score
$PY -m tfb.kaggle_submit --status             # recent scores and error text (the CLI hides errors)
$PY -m tfb.t1_model --cv                      # Task 1: train Jun–Nov, score Dec–Feb with exact scorer replicas
$PY -m tfb.t2_bench onset --tag x --seeds 5 [--frame f.parquet] [--drop a,b]   # Task 2 bench: LOMO IoU on official / selector / all windows
$PY -m tfb.t2_bench ongoing --tag x --seeds 5 [--frame f.parquet]
$PY -m tfb.t2_bench report onset --tags base,x                   # compare cached bench runs
$PY -m tfb.t2_bench causality                 # all 160 scored windows: features unchanged when data after T is blanked
$PY -m tfb.t2_bench causality --files --submission v35   # the organizers' award check: per window, every row after T removed from the files (~30 min)
$PY -m tfb.pipeline_final --out check --queue-only       # Task 2 only; also runs on a release with rows removed
$PY -m tfb.t2_onset --out f.parquet           # build a training frame into a file (same for t2_ongoing)
$PY experiments/tools/t2_variant.py --out v12 --base v10 --family ongoing --frame f.parquet --seeds 5   # change one Task 2 family only
$PY -m tfb.pipeline_t4swap --base v4 --out x --method nnls --start prior   # swap only Task 4
$PY experiments/tools/compare.py <dirA> <dirB>     # per-task diff of two submission folders
$PY experiments/tools/submit_queue.py 2026-10-04T00:03:00 <csv> "msg" [<csv> "msg" ...]   # submit after the UTC reset
```

Experiments are scripts in `experiments/` (EDA under `experiments/eda/`), run with the same interpreter.

## Architecture

The flow is: release → `tfb.dense` (dense arrays, links in **travel order**) → per-task modules → three per-task CSVs → `organizer-repo/src/merge_submissions.py` (joined on keys from `submission_key.csv`) → `pipeline_v0.validate` → `kaggle_submit.package` (zip).

`pipeline_final` composes the current best components and is the reproducibility entry point. It is deterministic: fixed seeds, `deterministic=True` in LightGBM, and it reproduced v3 exactly. `pipeline_v0`, `_v1`, `_v3` and `_t4swap` are the historical steps whose functions it reuses.

**Data facts the code relies on:**
- One detector station per link in every panel.
- S/W panels are stored reversed relative to milepost, verified from `outgoing_link_ids`, so in the arrays index + 1 is always downstream. Scorers still sort by milepost, which is why `meta.json` keeps both.
- Timestamps must stay in the released text form `YYYY-MM-DDTHH:MM:SSZ`.
- Join anything to templates by key, never by row position.

**Task 1 (`t1_interp`, `t1_model`):**
- Time interpolation plus a LightGBM residual model, with separate targets for speed and flow per lane.
- The key features are neighbours' deviations from their own straight-line interpolation across the target's gap.
- Profiles:
  - **train profiles** leave each training day out of its own profile, which removed a training-time leak;
  - **month-adapted profiles** come from the scored month's own observed cells, same day type, with the target day left out. On the leaderboard they gained +0.0014, 7× the holdout estimate.
- Predictions pass `plausible_speed`: speed ≥ 5 km/h and q/v ≤ k_jam.
- **Averaged with a second model family** (`tfb.t1_public`): a public notebook's 94-feature set (CC0) retrained with LightGBM on synthetic masks of the train truth.
  - `pipeline_final.T1_PUBLIC_WEIGHT = 0.25` is its weight at target cells. Dec–Feb holdout +0.0009, positive on all 10 panels; alone it is 0.004 weaker.
- **Averaged with a third family, a 2-D U-Net over each day's [slot × link] grid (`tfb.t1_unet`, PyTorch), at `t1_unet.WEIGHT = 0.5`:** public +0.0021 (v47; weight 0.33: +0.0018, v48), Dec–Feb holdout +0.0018 composite, positive on all 10 panels (LOG 2026-10-20/21). Alone it matches the production pair. It is the one Task 1 change whose holdout gain transferred in full.
  - **Two seeds averaged** (`t1_unet.SEEDS = (5, 6)`, their corrections averaged in `predict_split`): public +0.0002 (v50), about half the holdout's +0.0004 (LOG 2026-10-26). Weight 0.5 remains best; 0.6–0.75 lose on the holdout.
  - It trains on the GPU when there is one (about 25 min per seed on an idle Apple GPU; two separate trainings of seed 5 gave bit-identical weights, v49) and on the CPU otherwise (deterministic, ~4.5 h per seed); `TFB_UNET_DEVICE` overrides. `pipeline_final --unet-weights a.pt,b.pt` loads the saved weights of a run instead of training; the CLI `python -m tfb.t1_unet --seed 6` trains one seed on all months and checkpoints every epoch.

**Task 3** has no file of its own. It is scored on the Task 1 values: N = length·q/v, and the conservation residual equals Σ|ΔN_sub − ΔN_obs| / Σ|ΔN_obs|. The organizer flux is "projected onto the observations", so it can be rebuilt on train (`task3_proxy`).

**`tfb.metrics`:**
- `s_state` and `s_physics` replicate the official Task 1 and Task 3 scorers on dense arrays. They are verified to 6 decimals and take about 0.2 s per panel.
- `day_sel` restricts scoring to holdout days.
- Always evaluate Task 1 changes with both, because measurement noise caps S_LWR well below 1.

**Task 2 (`t2_rules`, `t2_features`, `t2_onset`, `t2_ongoing`):**
- **Labels:** observed speed ≤ 0.6 × free speed on eligible cells.
- **Replica of the organizer window selector** (`t2_rules.selector_windows`):
  - a forward scan, 6 h spacing across both conditions, persistence IoU ≤ 0.9;
  - it reproduces 39/40 official onset windows but only 14/40 ongoing ones;
  - for ongoing, only the 40 official train windows show official difficulty.
- **Onset:**
  - The queue appears only at T+30. Rows are (event, link); the predicted set maximizes expected IoU (`choose_set`).
  - Production uses the v4 features with 5 seeds.
  - `EXTRA_FEATURES` ("anomaly", "context") is off: the anomaly features cost −0.006 on the leaderboard (v6) despite helping on the bench.
  - Onset changes mostly flip low-confidence windows, so ship them only on a clear bench gain.
- **Ongoing:**
  - Rows are (window, link, step) within 12 links of any history queue (`choose_cells`).
  - Short gaps with no reading between queued links count as queued (`BRIDGE_SEGMENTS`).
  - The A3 features cover dissipation and tail dynamics. With 5 seeds they gained +0.0038 on the leaderboard (v8).
  - The train queue-frequency features (`qfreq_*`) are excluded via `FEATURES_DROP`; that gained +0.0029 (v13).
  - `FAR_QFREQ` (recurrent far links) and `STEP_SHRINK` are off: far links lost −0.0012 on the leaderboard despite +0.025 on the bench.
  - **Averaged with a second model** (`tfb.t2_public`): the all-link classifier of a public notebook (anubisyy assets, CC0), ported and retrained here.
    - It uses 36 history features from T−60..T−5 and reproduces the original checkpoint exactly; it trains in about 1 minute.
    - On our candidate cells p = (1 − w) p_ours + w p_public with `t2_public.WEIGHT` w = 0.35; off them (ours is 0 there) p = `t2_public.OFF_SCALE` × p_public = 0.45 p_public; then `choose_window` over every link.
    - Public leaderboard: the 0.5 / 0.5 average gained +0.0027 (v40/v41; on our candidate cells only +0.0013, v39). **Tuning the two weights separately gained another +0.0008 (v58/v63, LOG 2026-10-29/30):** the public model must weigh less on our candidate cells (0.35; 0.5 → 0, 0.65 with off-scale 0.65 → −0.0026) and keep its scale off them (0.45–0.5; 0.35 → −0.0008, 0.65 → −0.0007). The expected-IoU set depends on calibration, and ours is calibrated on its candidate cells.
    - This is a leaderboard-tuned choice; the train bench cannot rank it. v52 (0.5 / 0.5) is the hedge pick.
    - `t2_public.WEIGHT = 0` turns it off.
  - **A convolutional corridor model (`tfb.t2_cnn`, PyTorch) is implemented but off** (`WEIGHT = 0`): it beats our LightGBM on the train bench by 0.01–0.03 IoU and loses 0.02–0.04 on validation; without static/time channels it is neutral (v46 +0.0001). Experiments use `TFB_CNN_DEVICE=mps`; production would train on the CPU (deterministic).
- **Production Task 2 path:** `pipeline_v1.panel_context` and `window_features`, the only test-time feature code; `python -m tfb.t2_bench causality` checks it.
- **Award check (thread 749601):** the organizers rerun our code with every row stamped after each window's origin removed, and the Task 2 predictions must not change.
  - `causality --files` performs exactly that for all 160 windows.
  - `tfb.dense` therefore accepts releases with rows, days or files absent. Keep it that way; arrays from the complete release are unchanged.
- **Inputs at test time:** the complete window history T−60..T−5 plus the masked layer at T.
- **Scores add up exactly across families:** onset and ongoing windows are disjoint and aggregation is linear, so combining two measured family changes needs no submission.

**Task 4 (`t4`):**
- The hidden truth is the organizer NNLS projection (λ = 0.05) of each split's own released prior. That is what we submit; S_ODME ≈ 0.993.
- `experiments/t4_hypothesis.py` predicted every probed variant's leaderboard delta to within 0.007.
- Count-based local checks mis-rank Task 4 methods. Do not change Task 4 without a strong reason.

## Rules and decisions that constrain code

- **Task 2 is online:** use only released data timestamped ≤ T, in any file, including the masked-layer observation at T (confirmed by the organizers in thread 745700, Oct 15). Never feed Task 1 reconstructions into Task 2, because they use data after T.
  - Fitting, model selection and thresholds count as use too (thread 748789). A Task 2 model may be refitted on data before T, including earlier days of the same split. April windows may use all March files.
  - Only window origins ≤ T may be used. Conditioning on the documented window-selection rule (6 h spacing, maximum persistence IoU) is allowed.
  - The report must explain design choices with evidence, preferably results on the train split (thread 749601). Leaderboard results are confirmation, not the main evidence.
- **Tasks 1 and 4 are offline:** any released data in the split may be used.
- Exploiting the T+30 onset regularity is explicitly allowed (thread 742350). Inferring validation or private labels from the leaderboard is prohibited.
- **Submissions:**
  - 5 per UTC day, uploaded as `.zip`; Kaggle rejects `.gz`.
  - Change one task at a time against the current best, so the delta isolates it.
  - Zeroing a task's predictions measures that task's exact validation score; this was done for Task 2.
- **Current best public and final candidate:** v64 = 0.87721, the full current `pipeline_final` configuration run fresh from commit e89d9c4 (byte-identical to the Task 2 probe v63; award check passed; **private 0.86769**). The hedge pick v52 = 0.87643 (private 0.86647) is the same configuration with the untuned 0.5 / 0.5 ongoing average, run fresh from commit c22e05d and reproduced byte for byte twice (v51, v53 in a clean environment):
  - Task 1: 2 seeds, self-supervised rows, capped density adjustment, no train-profile features, 300k extra importance-weighted congested training cells per panel; averaged with `t1_public` at weight 0.25 (+0.0006, v42) and with the 2-seed U-Net average at weight 0.5 (+0.0021 for one seed, v47; +0.0002 for the second, v50);
  - Task 2: onset v4 features with 5 seeds; ongoing A3b without recurrence features, plus the causal estimate of links masked at T, with 10 seeds, averaged with the public all-link model (weight 0.35 on our candidate cells, scale 0.45 off them);
  - Task 4: NNLS.
- **Task 2 information set:** `t2_features.USE_T_SLOT = True` (≤ T, including the observation at T), allowed per thread 745700. Switching to < T would cost about −0.004 total on the bench.
- **Onset is frozen** at the v4 features with 5 seeds. Every feature change lost 0.004–0.008 on validation (LOG 2026-10-08).
- **The scored months are shifted (LOG 2026-10-07).** Train months are near copies of each other (congestion-pattern correlation 0.955). Validation and private correlate only 0.84 with them.
  - Leave-one-month-out CV and the bench cannot see this shift. Changes built on train recurrence, train profiles or train priors helped the bench and lost on the leaderboard three times; removing such features gained.
  - Judge any change that relies on train-month patterns on the public leaderboard; use the bench for shift-free mechanics only.
  - For the ongoing family the train bench cannot rank model families at all (LOG 2026-10-20): the public local model loses on the bench and wins on validation, the CNN the reverse.
  - Prefer current-dynamics features and adaptation to the scored month.
  - On validation, Task 2 onset is limited by non-recurrent onsets, which no model predicts from data ≤ T (LOG 2026-10-02).
  - The current plan is PLAN.md §0; results are in LOG 2026-10-06.
