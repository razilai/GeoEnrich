#!/usr/bin/env bash
# Ship the dataset CSVs to the vast.ai GPU box (host alias `vast` in ~/.ssh/config)
# where the TAR eval runs. The CSVs are NOT git-tracked (too big / churny), so the
# code travels via git and the data travels via this rsync.
#
# Build the dataset locally first (airbnb_surroundings.build + .describe), then:
#     scripts/sync.sh
#
# Remote dir defaults to the repo root on vast; --relative preserves the
# data/ + artifacts/ layout so the remote stages find their inputs. Override:
#     REMOTE=vast:/some/path/ scripts/sync.sh
set -euo pipefail

# run from the repo root (parent of scripts/) so the relative paths below resolve
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

REMOTE="${REMOTE:-vast:/workspace/multabench/}"

command -v rsync >/dev/null || { echo "❌ rsync not found"; exit 1; }

# raw input + generated variants; only the ones that exist get sent.
csvs=(data/processed/airbnb.csv data/processed/airbnb_vanilla.csv \
      data/processed/airbnb_enriched.csv data/processed/airbnb_described.csv)
present=()
for f in "${csvs[@]}"; do [ -f "$f" ] && present+=("$f"); done
[ ${#present[@]} -gt 0 ] || { echo "❌ no CSVs found — build the dataset first"; exit 1; }

echo "📤 rsync -> $REMOTE"
printf '   %s\n' "${present[@]}"
# --relative (-R): keep the data/ + artifacts/ prefixes on the remote side
rsync --archive --relative --checksum --progress "${present[@]}" "$REMOTE"
echo "✅ synced ${#present[@]} CSV(s) to $REMOTE"
