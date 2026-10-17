# Eight Wonders Framing Probes

**Question:** When a language model reads a culturally specific consumer scenario, does it internally represent *how* the scenario is framed — relational (trust, obligation, belonging) versus transactional (price, demand, margin)? Is that one direction in activation space, or eight, one per Wonder of Black Shopping?

This is the empirical layer of the research program on **Algorithmic Hermeneutical Injustice** (AHI): models misreading relational behavior as transactional. If framing is readable from activations, it can be monitored at runtime by **The Living Constitution** (TLC).

Research loop: problem (AHI, from lived and narrative experience) → research (this repo) → paper → runtime monitor in TLC.

## Status (2026-10-17)

| Stage | Status |
|---|---|
| Found and documented a confound in the v2 probe data | Done — a bag-of-words classifier separates v2 framings at 99.6% held-out |
| Controlled dataset v3 (1,536 prompts) | Done — identical scenarios, length-matched frames, two vocabulary-disjoint template families |
| Lexical controls on v3 | Done — word-counter transfer across families 47–49% (chance) |
| Pipeline + negative control (untrained model) | Done — untrained model transfer 51%, p = 0.16 |
| Real-model runs (Qwen2.5 0.5B / 1.5B / 7B) | Pending — run `scripts/run_full_mac.sh` |
| Steering on Frontin' vignettes | Planned after real-model results |
| TLC runtime monitor | Planned after steering |

## Reproduce

```bash
make reproduce                                   # dataset, controls, tests, smoke run — no downloads, CPU only
make full MODEL=Qwen/Qwen2.5-7B TAG=qwen7b       # real model (Apple Silicon / GPU)
./scripts/run_full_mac.sh                        # 0.5B, 1.5B and 7B in one go
```

## The test

A probe is trained on activations for template family A and tested on family B, whose content words do not overlap with A's. Surface wording cannot carry over, so transfer above the permutation null means the model represents the framing itself. Splits are grouped by scenario so no scenario appears in both training and test.

See [RESULTS.md](RESULTS.md) for numbers, [docs/LOG.md](docs/LOG.md) for the decision log, `results/PROVENANCE.json` for file hashes.
