# Eight Wonders Framing Probes

**Question:** When a language model reads a culturally specific consumer scenario, does it internally represent *how* the scenario is framed — relational (trust, obligation, belonging) versus transactional (price, demand, margin)? Is that one direction in activation space, or eight, one per Wonder of Black Shopping?

This is the empirical layer of the research program on **Algorithmic Hermeneutical Injustice** (AHI): models misreading relational behavior as transactional. If framing is readable from activations, it can be monitored at runtime by **The Living Constitution** (TLC).

Research loop: problem (AHI, from lived and narrative experience) → research (this repo) → paper → runtime monitor in TLC.

## Status (2026-10-17, v0.2.0)

| Stage | Status |
|---|---|
| Found and documented a confound in the v2 probe data | Done: bag-of-words separates v2 framings at 99.6% held-out |
| Controlled dataset v4 (4,096 prompts) | Done: identical scenarios, length-matched frames, 8 vocabulary-disjoint template pairs per family |
| Lexical controls on v4 | Done: word-counter transfer across families 44–47% (chance) |
| Statistics corrected for template dependence and layer selection | Done: max-statistic permutation test over template pairs |
| Negative control (untrained model) | Done: not significant (p ≈ 0.07) |
| Steering test on Frontin' vignettes | Built and smoke-tested; real-model run pending |
| TLC runtime monitor (ConstitutionalInvariant `EW-FRAME-001`) | Built and smoke-tested; real-model calibration pending |
| Real-model runs (Qwen2.5 0.5B / 1.5B / 7B) | Pending: `scripts/run_full_mac.sh` |

## The research loop

1. **Problem:** Algorithmic Hermeneutical Injustice. Models read relational behavior (Frontin', defensive presentation management under retail surveillance) through a transactional or theft-risk lens.
2. **Research:** Is relational vs. transactional framing a direction in the model's activations? (`src/probes.py`) Does pushing along it change how the model reads Frontin'? (`src/steer.py`)
3. **Paper:** written from the committed results.
4. **Solution:** `src/monitor.py` turns the validated direction into TLC invariant `EW-FRAME-001`. When a relational situation is read transactionally, it triggers TLC halt authority and asks for the narrative baseline. Every decision goes to a hash-chained evidence log.

## Reproduce

```bash
make reproduce                                   # dataset, controls, tests, smoke run of probes + steering + monitor; no downloads, CPU only
make full MODEL=Qwen/Qwen2.5-7B TAG=qwen7b       # real model: probes, steering, monitor calibration + demo
./scripts/run_full_mac.sh                        # 0.5B, 1.5B and 7B in one go
```

## The test

A probe is trained on activations for template family A and tested on family B, whose content words do not overlap with A's. Surface wording cannot carry over, so transfer above the permutation null means the model represents the framing itself. Splits are grouped by scenario so no scenario appears in both training and test.

See [RESULTS.md](RESULTS.md) for numbers, [docs/LOG.md](docs/LOG.md) for the decision log, `results/PROVENANCE.json` for file hashes.
