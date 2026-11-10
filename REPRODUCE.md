# Reproducing my TrafficFlowBench submission

One command rebuilds the selected final submission from the released data. Every step uses fixed seeds and no external data. All LightGBM models are deterministic (`deterministic=True`); the one neural network (the Task 1 U-Net) trains on the GPU, where repeated trainings on my machine gave bit-identical weights; PyTorch does not guarantee that on other hardware, so the two trained models of the selected submission are shipped in `models/` and can be loaded instead of retraining (section 4).

## 1. Environment
- **Hardware used:** Apple M4 Pro (14 CPU cores, 20-core GPU), 48 GB RAM, macOS 15. The GPU (PyTorch MPS backend) is used only to train the Task 1 U-Net, two seeds of about 25 minutes each; without a GPU the same training runs on the CPU in about 4.5 hours per seed. About 16 GB of RAM is enough.
- **Software:** `conda env create -f environment.yml`, then `conda activate tfb` and `pip install -e .` from this folder. Exact versions are in `environment.yml`: Python 3.12.14, numpy 2.5.3, pandas 2.3.3, pyarrow 25.0.0, scipy 1.18.1, scikit-learn 1.9.1, lightgbm 4.7.0, pytorch 2.14.1 (the conda-forge build, which includes the MPS backend on macOS; used by the Task 1 U-Net `tfb.t1_unet` and by the optional `tfb.t2_cnn`, which is off).

## 2. Data and organizer code
1. Download the competition data (the release rebuilt on 2026-09-11) and unpack it to `~/data/tfb/kaggle_public`. Set `TFB_DATA` to use a different root; the folder must then contain `kaggle_public/`.
2. Clone the organizer repository next to this README, at the commit I used:
   ```bash
   git clone https://github.com/jacky850/trafficflowbench-public organizer-repo
   git -C organizer-repo checkout c88cddf
   ```
   I use its `merge_submissions.py` and its scorer helpers. I never modify it.

## 3. Run
```bash
python -m tfb.pipeline_final --out final --fresh
```
- **Runtime:** about 2 hours on the hardware above (Task 1 LightGBM models 45 min, U-Net training 25 min per seed for two seeds on the GPU, Task 1 prediction 12 min, Tasks 2 and 4 under 10 min). Without a GPU, each U-Net seed trains on the CPU in about 4.5 hours; loading the shipped weights (section 4) skips that step.
- **Steps:**
  1. Dense arrays: preprocessing that writes the release into `[day, 288, link]` arrays under `~/data/tfb/dense`.
  2. Task 2 training frames.
  3. Task 1 models (two LightGBM families, then the U-Net, which is the slow step) and predictions.
  4. Task 2 models and predictions.
  5. Task 4 path flows.
  6. Merge with the organizer script, validate every row, and zip.
- **Output:** `~/data/tfb/submissions/final/submission.zip`, which is the uploaded file, plus `manifest.json` (git commit, package versions, configuration, runtime) and every trained model (LightGBM boosters as `.txt`, the two U-Net seeds as `t1_unet.pt` and `t1_unet_2.pt`).
- `--fresh` rebuilds all intermediate caches. Without it, cached Task 2 training frames are reused, which gives the same result faster.

## 4. Seeds and determinism
- **Task 1:** LightGBM seeds 11 and 12, averaged. Training-cell sampling uses numpy seed 0, and the self-supervised rows use numpy seed 2. The public feature set is fitted with LightGBM seed 0 on numpy-seeded masks.
- **Task 1 U-Net:** torch seeds 5 and 6 (`t1_unet.SEEDS`; each seed fixes the weights and the per-epoch training masks), one model per seed, their corrections averaged. The pipeline trains them on the GPU when PyTorch finds one (`TFB_UNET_DEVICE` overrides; `cpu` forces the deterministic CPU path). Four separate GPU trainings of each seed on my machine produced bit-identical weights (LOG 2026-10-21 to 2026-10-31), but PyTorch does not guarantee bit-reproducibility on every GPU, and the CPU kernels differ from the GPU ones; the U-Net enters the Task 1 values only as half of an average at the target cells, so a slightly different training stays far inside the 1% tolerance. For an exact reproduction, load the two models of the selected run v64, shipped in `models/`, instead of training:
  ```bash
  python -m tfb.pipeline_final --out final --fresh --unet-weights models/t1_unet.pt,models/t1_unet_2.pt
  ```
- **Task 2:** onset seeds 7–11 and ongoing seeds 7–16, averaged per family. Training-window sampling uses seed 0.
- **Task 4:** a deterministic NNLS solve, with no randomness.
- **Check:** re-running the pipeline reproduced every submitted version through v46 exactly, row for row (LOG entries from 2026-10-01 onward). The final configuration was run twice: v51 (Task 1 from the separately trained U-Net weights) and v52 (the command above, from the clean commit, both seeds trained inside the pipeline). The two are byte-identical in all three files, and the U-Net weights are bit-identical (LOG 2026-10-28). A third run in a fresh environment created from `environment.yml` with the commands of section 1 (v53) was again byte-identical, including the U-Net weights (LOG 2026-10-29). The selected submission v64 (commit e89d9c4, which changed only the two ongoing blend weights) came from a fourth fresh run and is byte-identical in Tasks 1 and 4 to those three, and in Task 2 to the probe that first used its weights (LOG 2026-10-31).

## 5. Declarations
- **External data:** none. Only the released competition files are used.
- **Public code:** two modules port code from the public Kaggle dataset `anubisyy/trafficflowbench-086517-public-assets` (CC0-1.0). Both are retrained here on the train split; no checkpoint is downloaded.
  - `src/tfb/t2_public.py` ports the all-link ongoing queue model (`queue_ongoing.py`). Its ongoing probabilities are averaged with mine: weight 0.35 on the cells my model scores (within 12 links of a history queue) and, scaled by 0.45, alone on the other links.
  - `src/tfb/t1_public.py` ports the Task 1 feature set (`state_model.py`, `prepare_state.py`), fitted with LightGBM. Its predictions get weight 0.25 at the target cells.
- **Task 2 is online.** Every forecast uses only released data timestamped at or before the window's origin T, including the observation at T (thread 745700). Models are fitted on the train split only, which precedes every scored window.
- **Licenses:**
  - numpy, pandas, scipy and scikit-learn: BSD-3-Clause;
  - pyarrow: Apache-2.0;
  - lightgbm: MIT;
  - pytorch: BSD-3-Clause;
  - organizer repository: MIT;
  - the ported public model code: CC0-1.0;
  - my code: MIT (`LICENSE`).

## 6. Task 2 award check (thread 749601)
The organizers rerun the code with every row stamped after each window's origin removed and require identical Task 2 predictions. Two ways to do that:
- **All 160 windows in one command** (about 30 minutes):
  ```bash
  python -m tfb.t2_bench causality --files --submission final
  ```
  For each scored window this:
  1. writes a copy of the release in which every row stamped after the window's origin is removed from every file: mainline and ramp states, window histories, `window_index.csv` (by origin) and the templates;
  2. rebuilds the dense arrays from that copy;
  3. predicts the window with the production models;
  4. requires the prediction to equal both the full-release prediction and the rows of `submissions/final/queue.csv`.
- **Manually, on a release you cut yourself:** place the cut release at `<root>/kaggle_public`, then run
  ```bash
  TFB_DATA=<root> python -m tfb.pipeline_final --out check --queue-only
  ```
  This runs Task 2 only and accepts missing rows, days or files. Each window still present in the cut `window_index.csv` gets the same rows in `<root>/submissions/check/queue.csv` as in the full run. Tasks 1 and 4 are offline tasks and are skipped.

A faster check, `python -m tfb.t2_bench causality`, blanks every value after T in the arrays and compares the features of all 160 windows.

## 7. Where things are
- `README.md`: an overview and quick start.
- `src/tfb/`: the package; `pipeline_final.py` is the single entry point.
- `models/`: the two U-Net models of the selected submission v64 (`t1_unet.pt` for seed 5, `t1_unet_2.pt` for seed 6) and its manifest, `manifest_v64.json`.
- `experiments/`: every experiment script. `experiments/LOG.md` records each result and submission; the report is built from it.
- `CLAUDE.md`: a developer guide to the code and the decisions behind it.
