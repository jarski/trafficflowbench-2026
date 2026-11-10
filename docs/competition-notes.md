# 2026 IEEE Big Data Cup – TrafficFlowBench

- Kaggle: https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench
- Organizer code, baselines and scorers: https://github.com/jacky850/trafficflowbench-public (cloned in `organizer-repo/`)
- Collected: 2026-09-29

## At a glance

| Item | Value |
|---|---|
| Category | Community (IEEE ITS Society TC on Travel Information and Traffic Management + RERITE) |
| Prize pool | USD 3,500: Gold 1,500 · Silver 1,000 · Bronze 500 · K–12 Student Award 500 |
| Teams so far | 226 |
| Max team size | 4 |
| Submissions per day | 5 |
| Submission type | Upload a single CSV (not a notebook-only competition) |
| Data | ~12.4 GB: 9,434 parquet + 240 CSV files; synthetic freeway data for 10 directional corridors (LA D7 / Orange County D12) at 5-minute resolution |

## Key dates (2026, AoE unless noted)

| Date | Milestone |
|---|---|
| **Oct 25** | **Registration e-mail required for prizes** → trafficflowbench@gmail.com, subject "Traffic Flow Bench Registration" (team name, Kaggle usernames, full names, affiliations, e-mails, primary contact) |
| **Nov 6** (Kaggle shows 2026-11-07 06:55 UTC) | Final Kaggle submission deadline |
| **Nov 10** | Code (reproduces the final submission to within 1%) + PDF report e-mailed |
| Nov 18 | Winners announced |
| Dec 14–17 | IEEE BigData 2026 presentations, Phoenix, AZ (optional; travel at your own cost) |

## Can you join? (Finland, using AI)

- **Finland: yes.** The Kaggle general rules exclude only Crimea, DNR/LNR, Cuba, Iran, North Korea and sanctioned persons. You must be 18+.
- **AI tools: yes.** The rules explicitly allow "public external datasets, pretrained models, software, automated machine learning tools, and APIs" that are lawful, properly licensed and freely or cheaply available. If you reach the finals, disclose every external dataset, model and paid API you used.
- **Not allowed:** anything aimed at finding the hidden answers, hand-labeling withheld cells, or two teams submitting substantially the same work.
- **Final award score** = 85% private leaderboard + 15% PDF report, among teams whose code reproduces their final submission.

## The task

`S_total = 0.35·S_state + 0.30·S_queue + 0.15·S_physics + 0.20·S_ODME`

1. **Task 1 – State reconstruction:** fill in blanked speed/flow cells (offline; any data in the split may be used).
2. **Task 2 – Queue forecasting:** from 60 minutes of history, predict which links are queued at T+5…T+30 (online; only data timestamped ≤ T). In `queue_onset` windows the queue always appears only at T+30, and the organizers have said using this is allowed.
3. **Task 3 – Physics consistency:** scored on your Task 1 output (fundamental diagram + vehicle conservation).
4. **Task 4 – ODME:** estimate path flows from link counts plus a weak prior.

## History worth knowing

- **Sept 9–11:** Task 2 answers leaked through the data twice. Submissions were paused and the validation and private months were regenerated. Only submissions made after 2026-09-11 count, so leaderboard scores from before then are meaningless.
- **Public baseline:** a physics-based pipeline reaches about 0.809 on the leaderboard (thread 742980, runs on Kaggle CPU in about 45 min). A score of 0.869 was ranked about 26th on 2026-09-26, so the top of the leaderboard is higher than that.

## Folder contents

```
README.md                 this summary
docs/                     every Kaggle competition page, verbatim
  abstract.md  evaluation.md  data-description.md  submission.md
  rules.md  foundational-rules.md  prizes.md  timeline-and-registration.md
  organizers-and-judging.md  references.md
  discussions/            all 13 forum threads (as of 2026-09-29)
organizer-repo/           git clone of the official code, baselines and scorers (commit c88cddf; not tracked by our git)
```

## Not yet fetched

- **The dataset (~12.4 GB) and the leaderboard.** The Kaggle connector needs you to sign in again (`/mcp`). Otherwise, install the Kaggle CLI and accept the rules on the website, then run:
  `kaggle competitions download -c 2026-ieee-big-data-traffic-flow-bench -p data/`
