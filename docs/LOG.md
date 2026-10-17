# Decision and run log

Newest first.

| Date | Type | Entry |
|---|---|---|
| 2026-10-17 | Result | Negative control: untrained model transfer 0.509, p = 0.16. |
| 2026-10-17 | Decision | Primary test = cross-family transfer with scenario-grouped splits and a within-scenario permutation null (200 perms). |
| 2026-10-17 | Data | Built v3: 128 scenarios × 2 families × 3 templates × 2 frames = 1,536 prompts. Scenario taken from the sovereign v2 prompt (keeps cultural specifics) and shared by both frames. |
| 2026-10-17 | Finding | v2 prompts are lexically confounded: bag-of-words held-out 0.996, length-only 0.832. Earlier probe accuracies cannot be attributed to framing. |
| 2026-05-30 | History | v2 probe run on Qwen2.5-7B (TLC 2.0); V&T statement disclosed synthetic data and random projection in the harness. |
