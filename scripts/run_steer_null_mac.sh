#!/usr/bin/env bash
# v0.2.1: re-run only the steering test (40 random directions) for the three models.
# Reuses results/acts_<tag>.npz from the v0.2.0 run; re-extracts only if an activation file is missing.
set -euo pipefail
cd "$(dirname "$0")/.."
for pair in "Qwen/Qwen2.5-0.5B qwen05b" "Qwen/Qwen2.5-1.5B qwen15b" "Qwen/Qwen2.5-7B qwen7b"; do
  set -- $pair
  if [ ! -f "results/acts_$2.npz" ]; then
    .venv/bin/python -m src.extract --model "$1" --out "results/acts_$2.npz" --batch 8
  fi
  start=$(date +%s)
  make steer MODEL="$1" TAG="$2"
  echo "$2 steering seconds: $(( $(date +%s) - start ))"
done
.venv/bin/python scripts/summarize.py > /dev/null
.venv/bin/python scripts/provenance.py > results/PROVENANCE.json
