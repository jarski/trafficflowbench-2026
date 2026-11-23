# TrafficFlowBench 2026: team jarski

Code of **jarski**, a one-person team (Jarno Rantala), for the 2026 IEEE Big Data Cup – TrafficFlowBench. One command rebuilds my selected final submission from the released data.

| | |
|---|---|
| Selected final submission | **v64**, Kaggle submission 57820324: public 0.87721, private 0.86769 |
| Second selection (hedge) | v52, Kaggle submission 57714111: public 0.87643, private 0.86647 |
| Entry script | `python -m tfb.pipeline_final --out final --fresh` |
| Runtime | about 2 hours on an Apple M4 Pro (14 CPU cores, 20-core GPU, 48 GB RAM) |
| Hardware | 16 GB RAM, and about 20 GB of free disk besides the 12 GB release. Everything runs on the CPU except the Task 1 U-Net, which trains on a GPU when PyTorch finds one (Apple MPS or CUDA). Without a GPU its training adds about 9 hours, or the shipped models can be loaded instead (below). Tested on macOS 15 on Apple silicon. |
| Final ranking | 33rd of the 69 teams with a final package (gold 0.928, bronze 0.902) |
| Report | `report/REPORT.pdf` |
| After the deadline | Five late submissions tested open questions: report Section 7.2, code in `experiments/late/` |
| Kaggle write-up | https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/writeups/jarski-private-0-868-code-report-and-what-hel |

## Reproduce

1. **Environment.** conda-forge, with exact versions pinned in `environment.yml`: Python 3.12.14, numpy 2.5.3, pandas 2.3.3, pyarrow 25.0.0, scipy 1.18.1, scikit-learn 1.9.1, lightgbm 4.7.0 and pytorch 2.14.1.
   ```bash
   conda env create -f environment.yml
   conda activate tfb
   pip install -e .
   ```
2. **Organizer code.** I use its `merge_submissions.py` and scorer helpers without changes.
   ```bash
   git clone https://github.com/jacky850/trafficflowbench-public organizer-repo
   git -C organizer-repo checkout c88cddf
   ```
3. **Data.** Unpack the competition data, the release rebuilt on 2026-09-11, to `~/data/tfb/kaggle_public`. To use another location, set `TFB_DATA` to a folder that contains `kaggle_public/`.
4. **Run.**
   ```bash
   python -m tfb.pipeline_final --out final --fresh
   ```
   The upload file is `~/data/tfb/submissions/final/submission.zip`. The same folder holds the three per-task CSVs, every trained model and `manifest.json` with the configuration, seeds, package versions and runtime.

**Exact reproduction of the U-Net.** Everything else is deterministic: LightGBM runs with `deterministic=True` and fixed seeds, and Task 4 is a deterministic solve. The U-Net trains on the GPU, and PyTorch does not guarantee bit-identical GPU training across machines. The two trained U-Net models of the selected run are in `models/`. To load them instead of training:
```bash
python -m tfb.pipeline_final --out final --fresh --unet-weights models/t1_unet.pt,models/t1_unet_2.pt
```
On my machine every run of the final configuration has been byte-identical to v64, including four separate trainings of the U-Net and one run in a fresh environment created from `environment.yml`.

**Task 2 award check** (forum thread 749601). For each scored window, every row stamped after the window's origin is removed from the release, and the forecast must not change. After the run above, this takes about 30 minutes:
```bash
python -m tfb.t2_bench causality --files --submission final
```

## Contents

- `src/tfb/`: the package. `pipeline_final.py` is the single entry point.
- `models/`: the two U-Net models of the selected submission and its manifest, `manifest_v64.json`.
- `REPRODUCE.md`: full reproduction details: environment, data, seeds, determinism, verification, declarations and licenses.
- `report/`: the report as PDF and Markdown source, with its figures.
- `experiments/`: every experiment script. `experiments/LOG.md` records every experiment and submission with its numbers.
- `docs/`: copies of the competition pages, the organizer rulings I relied on, and my early competition notes.
- `CLAUDE.md`: a developer guide to the code and the decisions behind it.

## Declarations

- **External data:** none. Only the released competition files are used.
- **Public code:** two modules port code from the public Kaggle dataset `anubisyy/trafficflowbench-086517-public-assets` (CC0-1.0): the all-link ongoing queue model (`tfb.t2_public`) and a Task 1 feature set (`tfb.t1_public`). Both are retrained here on the training split; no checkpoint is downloaded.
- **Task 2 information set:** every forecast uses only released data stamped at or before the window's origin T, including the observation at T. All models are fitted on the training split.
- **Use of AI:** I developed the solution with Claude Code (Anthropic), an AI coding assistant, which wrote most of the code under my direction. The pipeline contains no language model.
- **License:** MIT, see `LICENSE`. Third-party licenses are listed in `REPRODUCE.md`.

The submitted v64 was produced at git commit `e89d9c4`. Since then only comments, documentation, the report and `models/` have changed, and the late diagnostics in `experiments/late/` were added; none of this touches the pipeline. The archive's own commit is recorded in the zip file's comment.

## About this public copy

This repository is a snapshot of my private development repository at commit df79a6d (2026-11-10), squashed into one commit. Commit ids cited in the report and in REPRODUCE.md refer to that private repository: e89d9c4 produced the submitted v64, and 1243da7 is the package sent to the organizers. The pipeline code here is identical to that package. Later commits correct the report (2026-11-18 and 2026-11-23) and add the late diagnostics in `experiments/late/`. Copies of the competition pages and forum threads are not republished; `docs/README.md` links to them and summarizes the rulings relied on.
