#!/usr/bin/env bash
# Real-model run for Apple Silicon (MPS). Requires the model in the local Hugging Face cache or network access.
# Runs three model sizes so results show whether the effect grows with scale.
set -euo pipefail
make install data baseline test
for spec in "Qwen/Qwen2.5-0.5B qwen05b" "Qwen/Qwen2.5-1.5B qwen15b" "Qwen/Qwen2.5-7B qwen7b"; do
  set -- $spec
  echo "=== $1 ==="; make full MODEL="$1" TAG="$2"
done
make provenance
echo "Commit results/*.json, results/*.png, results/PROVENANCE.json. Do not commit results/acts_*.npz (large) unless asked."
