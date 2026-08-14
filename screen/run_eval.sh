#!/usr/bin/env bash
# Frozen-only (--no-tar) joint-signal screen over the 5 prompt-variant datasets.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=screen/results.txt
: > "$OUT"

for csv in screen/df2_*.csv; do
  echo "########## $csv ##########" | tee -a "$OUT"
  python run_multabench_eval.py \
    --csv "$csv" --image-folder . --no-image --no-tar \
    --out "screen/report_$(basename "$csv" .csv).csv" 2>&1 | tee -a "$OUT"
  echo | tee -a "$OUT"
done

echo "done -> $OUT"
