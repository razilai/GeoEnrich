#!/usr/bin/env bash
# Ship the dataset CSVs to the vast.ai GPU box (host alias `vast` in ~/.ssh/config)
# where the TAR eval runs. The CSVs are NOT git-tracked (too big / churny), so the
# code travels via git and the data travels via this rsync.
#
# Build the dataset locally first (main.py + describe.py), then:
#     ./sync.sh
#
# Remote dir defaults to the repo root on vast, so the scripts' relative paths
# (airbnb.csv, airbnb_described.csv, ...) resolve. Override:
#     REMOTE=vast:/some/path/ ./sync.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

REMOTE="${REMOTE:-vast:/workspace/multabench/}"

command -v rsync >/dev/null || { echo "❌ rsync not found"; exit 1; }

# raw + generated variants; only the ones that exist get sent.
csvs=(airbnb.csv airbnb_vanilla.csv airbnb_enriched.csv airbnb_described.csv)
present=()
for f in "${csvs[@]}"; do [ -f "$f" ] && present+=("$f"); done
[ ${#present[@]} -gt 0 ] || { echo "❌ no CSVs found — build the dataset first"; exit 1; }

echo "📤 rsync -> $REMOTE"
printf '   %s\n' "${present[@]}"
rsync --archive --checksum --progress "${present[@]}" "$REMOTE"
echo "✅ synced ${#present[@]} CSV(s) to $REMOTE"
