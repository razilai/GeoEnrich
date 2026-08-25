#!/usr/bin/env bash
# Ship the described evaluation CSV to the vast.ai GPU box (host alias `vast` in
# ~/.ssh/config), where the TAR eval runs. It is NOT git-tracked (too big / churny),
# so code travels via git and this one data file travels via rsync.
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

CSV=data/processed/airbnb_described.csv
[ -f "$CSV" ] || { echo "❌ $CSV not found — run the describe stage first"; exit 1; }

echo "📤 rsync -> $REMOTE"
printf '   %s\n' "$CSV"
# --relative (-R): keep the data/ + artifacts/ prefixes on the remote side
rsync --archive --relative --checksum --progress "$CSV" "$REMOTE"
echo "✅ synced $CSV to $REMOTE"
