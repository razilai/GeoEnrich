#!/usr/bin/env bash
# Ship the described evaluation CSVs (one per prompt variant) to the vast.ai GPU box (host alias `vast` in
# ~/.ssh/config), where the TAR eval runs. They are NOT git-tracked (too big / churny),
# so code travels via git and these data files travel via rsync.
#
# Build the dataset locally first (src.build + .describe), then:
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

shopt -s nullglob
CSVS=(data/processed/airbnb_described_*.csv)
[ ${#CSVS[@]} -gt 0 ] || { echo "❌ no data/processed/airbnb_described_*.csv — run the describe stage first"; exit 1; }

echo "📤 rsync -> $REMOTE"
printf '   %s\n' "${CSVS[@]}"
# --relative (-R): keep the data/ + artifacts/ prefixes on the remote side
rsync --archive --relative --checksum --progress "${CSVS[@]}" "$REMOTE"
echo "✅ synced ${#CSVS[@]} file(s) to $REMOTE"
