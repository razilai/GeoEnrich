#!/usr/bin/env bash
# Run on YOUR machine. Fetch benchmark results back from the GPU VM.
#   scripts/pull.sh
#   SSH_HOST=vast REMOTE_DIR=/workspace/geoenrich scripts/pull.sh
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SSH_HOST="${SSH_HOST:-vast}"
REMOTE_DIR="${REMOTE_DIR:-/workspace/geoenrich}"
mkdir -p "$ROOT/results"
rsync --archive --progress "$SSH_HOST:$REMOTE_DIR/results/" "$ROOT/results/"
echo "✅ results pulled into results/"
