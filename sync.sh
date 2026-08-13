#!/usr/bin/env bash
# Copy MulTaBench's filled-in credentials up to the repo root, so the top-level
# orchestrator project also has them. Run AFTER ./init.sh (which scaffolds
# MulTaBench/.env) and after you've filled in MulTaBench/.env.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$HERE/MulTaBench/.env"
DEST="$HERE/.env"

command -v rsync >/dev/null || { echo "❌ rsync not found"; exit 1; }
[ -f "$SRC" ] || { echo "❌ source not found: $SRC — run ./init.sh, then fill it in"; exit 1; }

rsync --archive --checksum "$SRC" "$DEST"
echo "✅ synced $SRC → $DEST"
