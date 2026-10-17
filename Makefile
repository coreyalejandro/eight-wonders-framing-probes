# Eight Wonders framing probes — one-command reproduction.
#   make reproduce              -> dataset, lexical controls, tests, smoke run (no downloads)
#   make full MODEL=Qwen/Qwen2.5-7B TAG=qwen7b   -> real model run (GPU/Apple Silicon recommended)
PY ?= python3
VENV := .venv
BIN := $(VENV)/bin
MODEL ?= Qwen/Qwen2.5-7B
TAG ?= qwen7b
PERMS ?= 1000

.PHONY: reproduce install data baseline test smoke full steer monitor provenance

reproduce: install data baseline test smoke provenance
	@echo "Done. Controls: results/lexical_baseline.json. Smoke (untrained model): probe_results_tiny.json, steering_tiny.json, monitor_tiny.json"

install:
	$(PY) -m venv $(VENV)
	$(BIN)/pip install -q -r requirements.txt

data:
	$(BIN)/python src/build_dataset.py data/prompts_v2.csv data/prompts_v4.jsonl

baseline:
	$(BIN)/python src/lexical_baseline.py > results/lexical_baseline.json

test:
	$(BIN)/python -m pytest -q -p no:cacheprovider tests --junitxml=results/junit.xml

smoke:
	$(BIN)/python -m src.extract --model tiny-random --out results/acts_tiny.npz --batch 128
	$(BIN)/python -m src.probes --acts results/acts_tiny.npz --tag tiny --perms 1000
	$(BIN)/python -m src.steer --model tiny-random --acts results/acts_tiny.npz --probe-results results/probe_results_tiny.json --tag tiny --n-random 5
	$(BIN)/python -m src.monitor calibrate --acts results/acts_tiny.npz --probe-results results/probe_results_tiny.json --tag tiny
	rm -f results/evidence_tiny.jsonl
	$(BIN)/python -m src.monitor run --model tiny-random --config results/monitor_tiny.json --vignettes --evidence results/evidence_tiny.jsonl > results/monitor_demo_tiny.txt

full:
	$(BIN)/python -m src.extract --model $(MODEL) --out results/acts_$(TAG).npz --batch 8
	$(BIN)/python -m src.probes --acts results/acts_$(TAG).npz --tag $(TAG) --perms $(PERMS)
	$(MAKE) steer monitor MODEL=$(MODEL) TAG=$(TAG)
	$(BIN)/python scripts/provenance.py > results/PROVENANCE.json

steer:
	$(BIN)/python -m src.steer --model $(MODEL) --acts results/acts_$(TAG).npz --probe-results results/probe_results_$(TAG).json --tag $(TAG)

monitor:
	$(BIN)/python -m src.monitor calibrate --acts results/acts_$(TAG).npz --probe-results results/probe_results_$(TAG).json --tag $(TAG)
	rm -f results/evidence_$(TAG).jsonl
	$(BIN)/python -m src.monitor run --model $(MODEL) --config results/monitor_$(TAG).json --vignettes --evidence results/evidence_$(TAG).jsonl > results/monitor_demo_$(TAG).txt

provenance:
	$(BIN)/python scripts/provenance.py > results/PROVENANCE.json
