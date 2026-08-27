#!/usr/bin/env bash
# Run the MulTaBench benchmark entry point.
# Usage: bash analysis/run_official_benchmark.sh DATASET_NAME WANDB_PROJECT
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 DATASET_NAME WANDB_PROJECT" >&2
  exit 2
fi

REPO="MulTaBench"
DATASET_NAME="$1"
PROJECT="$2"

cd "$REPO"
for fold in 0 1 2 3 4; do
  for model in tabm cat light tabpfnv2 tabpfnv2p5; do
    # These are the official benchmark.py states:
    # no_text = structured only; text_only = text only; all = joint frozen;
    # ft = joint Target-Aware Representation / E5 LoRA for text datasets.
    for state in no_text text_only all ft; do
      ./.venv/bin/python benchmark.py \
        --model "$model" \
        --dataset_name "$DATASET_NAME" \
        --fold "$fold" \
        --multimodal_state "$state" \
        --project "$PROJECT"
    done
  done
done
