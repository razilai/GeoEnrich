#!/usr/bin/env bash
# Bootstrap a GPU VM from this checkout, then transfer MulTaBench credentials
# and the described dataset CSVs to it. The only sync script in the repo:
# credentials travel one way (here -> VM), code travels via Git, data via rsync.
#
# Build the dataset locally first (src.build + .describe), then:
#   scripts/sync.sh
#   SSH_HOST=vast REMOTE_DIR=/workspace/geoenrich scripts/sync.sh
#   REPO_URL=git@github.com:me/GeoEnrich.git scripts/sync.sh
#
# Requirements:
#   - SSH_HOST (default: vast) is an SSH alias configured in ~/.ssh/config.
#   - The VM can reach REPO_URL and has git + uv installed. init.sh may use
#     sudo to install Ubuntu build dependencies when they are missing.

set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_ENV="$ROOT/MulTaBench/.env"
SSH_HOST="${SSH_HOST:-vast}"
REMOTE_DIR="${REMOTE_DIR:-/workspace/geoenrich}"
REPO_URL="${REPO_URL:-$(git -C "$ROOT" config --get remote.origin.url || true)}"

command -v git >/dev/null || { echo "❌ git not found" >&2; exit 1; }
command -v rsync >/dev/null || { echo "❌ rsync not found" >&2; exit 1; }
command -v ssh >/dev/null || { echo "❌ ssh not found" >&2; exit 1; }

if [[ ! -f "$SOURCE_ENV" ]]; then
    echo "❌ source credentials file not found: $SOURCE_ENV" >&2
    echo "Create it from MulTaBench/.env.example and fill in the credentials first." >&2
    exit 1
fi

if [[ -z "$REPO_URL" ]]; then
    echo "❌ no repository URL configured; set REPO_URL=..." >&2
    exit 1
fi

# Do not overwrite a checkout that belongs to a different repository. An
# existing matching checkout is reused as-is; this script never pulls, resets,
# or otherwise changes its Git history.
echo "🔌 connecting to $SSH_HOST and preparing $REMOTE_DIR"
ssh "$SSH_HOST" bash -s -- "$REMOTE_DIR" "$REPO_URL" <<'REMOTE_SETUP'
set -euo pipefail

remote_dir="$1"
repo_url="$2"
parent_dir="$(dirname -- "$remote_dir")"

mkdir -p -- "$parent_dir"

if [[ -e "$remote_dir" && ! -d "$remote_dir/.git" ]]; then
    echo "❌ $remote_dir exists but is not a Git checkout; refusing to overwrite it." >&2
    exit 1
fi

if [[ ! -d "$remote_dir/.git" ]]; then
    echo "📥 cloning $repo_url -> $remote_dir"
    git clone "$repo_url" "$remote_dir"
else
    current_url="$(git -C "$remote_dir" config --get remote.origin.url || true)"
    if [[ "$current_url" != "$repo_url" ]]; then
        echo "❌ $remote_dir is a checkout of $current_url, not $repo_url; refusing to modify it." >&2
        exit 1
    fi
    echo "✅ existing matching checkout found — reusing it"
fi

cd -- "$remote_dir"
echo "🐍 running ./init.sh on ${HOSTNAME:-remote host}"
bash ./init.sh
REMOTE_SETUP

REMOTE_ENV="$SSH_HOST:$REMOTE_DIR/MulTaBench/.env"
echo "📤 syncing credentials $SOURCE_ENV -> $REMOTE_ENV"
# Preserve the file while stripping group/other access on the remote side.
rsync --archive --checksum --protect-args --chmod=go-rwx "$SOURCE_ENV" "$REMOTE_ENV"

cd "$ROOT"
shopt -s nullglob
CSVS=(data/processed/airbnb_described_*.csv)
if [ ${#CSVS[@]} -eq 0 ]; then
    echo "⚠️  no data/processed/airbnb_described_*.csv — run the describe stage, then re-run to ship data"
else
    echo "📤 syncing ${#CSVS[@]} dataset CSV(s) -> $SSH_HOST:$REMOTE_DIR/"
    printf '   %s\n' "${CSVS[@]}"
    # --relative keeps the data/processed/ prefix so remote stages find their inputs
    rsync --archive --relative --checksum --progress "${CSVS[@]}" "$SSH_HOST:$REMOTE_DIR/"
fi
echo "✅ VM initialized, credentials and data synced"
