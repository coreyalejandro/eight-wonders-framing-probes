# Eight Wonders framing probes — one-command reproduction.
#   make reproduce              -> dataset, lexical controls, tests, smoke run (no downloads)
#   make full MODEL=Qwen/Qwen2.5-7B TAG=qwen7b   -> real model run (GPU/Apple Silicon recommended)
PY ?= python3
VENV := .venv
BIN := $(VENV)/bin
MODEL ?= Qwen/Qwen2.5-7B
TAG ?= qwen7b
PERMS ?= 200

.PHONY: reproduce install data baseline test smoke full provenance

reproduce: install data baseline test smoke provenance
	@echo "Done. Controls in results/lexical_baseline.json; smoke run in results/probe_results_tiny.json"

install:
	$(PY) -m venv $(VENV)
	$(BIN)/pip install -q -r requirements.txt

data:
	$(BIN)/python src/build_dataset.py data/prompts_v2.csv data/prompts_v3.jsonl

baseline:
	$(BIN)/python src/lexical_baseline.py > results/lexical_baseline.json

test:
	$(BIN)/python -m pytest -q -p no:cacheprovider tests --junitxml=results/junit.xml

smoke:
	$(BIN)/python -m src.extract --model tiny-random --out results/acts_tiny.npz --batch 64
	$(BIN)/python -m src.probes --acts results/acts_tiny.npz --tag tiny --perms 50

full:
	$(BIN)/python -m src.extract --model $(MODEL) --out results/acts_$(TAG).npz --batch 8
	$(BIN)/python -m src.probes --acts results/acts_$(TAG).npz --tag $(TAG) --perms $(PERMS)
	$(BIN)/python scripts/provenance.py > results/PROVENANCE.json

provenance:
	$(BIN)/python scripts/provenance.py > results/PROVENANCE.json
