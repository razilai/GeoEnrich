#!/usr/bin/env bash
# Frozen-only (--no-tar) joint-signal screen over the 5 prompt-variant datasets.
set -euo pipefail
cd "$(dirname "$0")/../.."  # repo root (script lives in experiments/screen/)

SCREEN=experiments/screen
OUT=$SCREEN/results.txt
: > "$OUT"

# Heavy env (tabstar/torch) lives in MulTaBench/.venv, the one init.sh builds.
# The eval imports tabstar at module top, so --no-tar still needs this interp.
PY=MulTaBench/.venv/bin/python

for csv in "$SCREEN"/df2_*.csv; do
  echo "########## $csv ##########" | tee -a "$OUT"
  "$PY" -m airbnb_surroundings.eval \
    --csv "$csv" --image-folder . --no-image --no-tar \
    --out "$SCREEN/report_$(basename "$csv" .csv).csv" 2>&1 | tee -a "$OUT"
  echo | tee -a "$OUT"
done

echo "done -> $OUT"
