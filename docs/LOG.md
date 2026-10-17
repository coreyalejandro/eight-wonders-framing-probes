# Decision and run log

Newest first.

| Date | Type | Entry |
|---|---|---|
| 2026-10-17 | Build | v0.2.1: steering null over 40 random directions (`--n-random`, `scripts/run_steer_null_mac.sh`); `scripts/summarize.py` generates `results/SUMMARY.md`; results consistency test. |
| 2026-10-17 | Interpretation | Claude verified v0.2.0 numbers against the JSONs and wrote RESULTS.md sections 5–7. Probes: strong, scale-dependent. Steering: inconclusive (one random direction; off-target movement). Behavioral misreading: 0/24 in forced choice. Monitor (7B): 0.93 / 0.92 correct on held-out framings. |
| 2026-10-17 | Result | v0.2.0 runs by Codex on Corey's Mac (MPS): transfer 0.812 / 0.8479 / 0.9209 for 0.5B / 1.5B / 7B, max-stat p = 0.001 each; extraction 44 / 142 / 562 s. Commit da499df. |
| 2026-10-17 | Build | v0.2.0: steering test (`src/steer.py`) and TLC runtime monitor (`src/monitor.py`, invariant EW-FRAME-001, hash-chained evidence log). 12 tests. |
| 2026-10-17 | Design fix | Monitor now applies only in relational contexts (NOT_APPLICABLE otherwise). The smoke run showed it halting on ordinary transactional shopping, which is not a misreading. |
| 2026-10-17 | Stats fix | Primary test is now a max-statistic permutation test over template pairs, correcting template dependence and layer selection. Untrained model: p ≈ 0.07 (was 0.03 under the uncorrected test). |
| 2026-10-17 | Data | v4: expanded from 3 to 8 vocabulary-disjoint template pairs per family (4,096 prompts). With 3 pairs, an untrained model reached 0.61 transfer by chance. |
| 2026-10-17 | Result | Negative control: untrained model transfer 0.509, p = 0.16. |
| 2026-10-17 | Decision | Primary test = cross-family transfer with scenario-grouped splits and a within-scenario permutation null (200 perms). |
| 2026-10-17 | Data | Built v3: 128 scenarios × 2 families × 3 templates × 2 frames = 1,536 prompts. Scenario taken from the sovereign v2 prompt (keeps cultural specifics) and shared by both frames. |
| 2026-10-17 | Finding | v2 prompts are lexically confounded: bag-of-words held-out 0.996, length-only 0.832. Earlier probe accuracies cannot be attributed to framing. |
| 2026-05-30 | History | v2 probe run on Qwen2.5-7B (TLC 2.0); V&T statement disclosed synthetic data and random projection in the harness. |
