# Eight Wonders of Black Shopping: Framing Probes

**Question:** When a language model reads a culturally specific consumer scenario, does it internally represent *how* the scenario is framed — relational (trust, obligation, belonging) versus transactional (price, demand, margin)? Is that one direction in activation space, or eight, one per Wonder of Black Shopping?

This is the empirical layer of the research program on **Algorithmic Hermeneutical Injustice** (AHI): models misreading relational behavior as transactional. If framing is readable from activations, it can be monitored at runtime by **[The Living Constitution 2.0](https://github.com/coreyalejandro/the-living-constitution-2.0)** (TLC), which re-runs this repository's results from a fresh clone.

Research loop: problem (AHI, from lived and narrative experience) → research (this repo) → paper → runtime monitor in TLC.

## Status (2026-10-20, v0.2.2)

| Stage | Status |
|---|---|
| Found and documented a confound in the v2 probe data | Done: bag-of-words separates v2 framings at 99.6% held-out |
| Controlled dataset v4 (4,096 prompts) | Done: identical scenarios, length-matched frames, 8 vocabulary-disjoint template pairs per family |
| Lexical controls on v4 | Done: word-counter transfer across families 44–47% (chance) |
| Statistics corrected for template dependence and layer selection | Done: max-statistic permutation test over template pairs |
| Negative control (untrained model) | Done: not significant (p ≈ 0.07) |
| Real-model probes (Qwen2.5 0.5B / 1.5B / 7B) | Done: cross-family transfer 0.81 / 0.85 / 0.92, p = 0.001, 16/16 template pairs; see RESULTS.md |
| Steering test on Frontin' vignettes | Run against 40 random directions: no causal effect shown (p = 0.32, 0.07, 0.46); see RESULTS.md section 6 |
| TLC runtime monitor (ConstitutionalInvariant `EW-FRAME-001`) | Calibrated: 7B sorts held-out framings 92–93% correctly, HALT precision 0.94; evidence chains intact |

## The research loop

1. **Problem:** Algorithmic Hermeneutical Injustice. Models read relational behavior (Frontin', defensive presentation management under retail surveillance) through a transactional or theft-risk lens.
2. **Research:** Is relational vs. transactional framing a direction in the model's activations? (`src/probes.py`) Does pushing along it change how the model reads Frontin'? (`src/steer.py`)
3. **Paper:** in preparation. The Eight Wonders of Black Shopping paper builds on these results.
4. **Solution:** `src/monitor.py` turns the readable direction into TLC invariant `EW-FRAME-001`. When a relational situation is read transactionally, it triggers TLC halt authority and asks for the narrative baseline. Every decision goes to a hash-chained evidence log.

## Reproduce

```bash
make reproduce                                   # dataset, controls, tests, smoke run of probes + steering + monitor; no downloads, CPU only
make full MODEL=Qwen/Qwen2.5-7B TAG=qwen7b       # real model: probes, steering, monitor calibration + demo
./scripts/run_full_mac.sh                        # 0.5B, 1.5B and 7B in one go
```

## The test

A probe is trained on activations for template family A and tested on family B, whose content words do not overlap with A's. Surface wording cannot carry over, so transfer above the permutation null means the model represents the framing itself. Splits are grouped by scenario so no scenario appears in both training and test.

See [RESULTS.md](RESULTS.md) for numbers, [docs/LOG.md](docs/LOG.md) for the decision log, `results/PROVENANCE.json` for file hashes.
