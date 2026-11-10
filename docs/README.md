# Competition pages and rulings

This solution was built for the 2026 IEEE Big Data Cup – TrafficFlowBench on Kaggle. The competition pages (overview, data description, evaluation, rules, timeline) are at https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench, and the organizers' code, baselines and scorers at https://github.com/jacky850/trafficflowbench-public (this solution pins commit c88cddf). Copies of those pages and of the forum threads are not republished here; `competition-notes.md` holds my own notes of 2026-09-29.

Organizer rulings this solution relies on (thread numbers on the competition forum):

- **742068 (2026-09-19) and 745700 (2026-10-15):** Task 2 is online. A forecast with origin T may use any released data stamped at or before T, including the observation at T itself. Conditioning on the documented window-selection rule (6-hour spacing, maximum persistence IoU) is allowed.
- **742328 and 742350 (2026-09-21):** in every `queue_onset` window the queue first appears at T+30, and using this is allowed.
- **744187 (2026-09-28):** award eligibility, the final package (code and PDF report by 2026-11-10), and the final award score (85% private leaderboard, 15% report).
- **748789 (2026-10-15):** a Task 2 model may be refitted on data stamped before T, including earlier days of the same split; April windows may use all March files. This solution does not refit on the scored splits.
- **749601 (2026-10-18):** at award review the code is rerun with every row stamped after each window's origin removed, and the Task 2 predictions must not change (`python -m tfb.t2_bench causality --files` performs that check). The report should explain design choices with evidence such as training-split results.
- **755674 (2026-11-04):** a reminder of the Task 2 time rule; teams in the review pool are also run on a newly generated month.
- **756746 (2026-11-08):** submissions made before the data rebuild of 11 September 2026 are void and excluded from the final standings, even if selected. The validation and private months were regenerated in that rebuild (threads 740522, 740536 and 740844), so only submissions made after it count.
