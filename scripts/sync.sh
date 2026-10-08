#!/usr/bin/env bash
# Run on YOUR machine. Bootstraps a GPU VM (clone/fast-forward the repo, run
# init.sh there), then sends everything Git does not carry: MulTaBench
# credentials, the cleaned listings and the described dataset CSVs.
# Code travels via Git (push first), credentials and data via rsync.
# Afterwards, on the VM:  scripts/screen.sh     (and locally: scripts/pull.sh)
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

# Do not overwrite a checkout that belongs to a different repository. A matching
# checkout is fast-forwarded (never reset or merged), so pushed code reaches the VM.
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
    echo "✅ existing matching checkout found — fast-forwarding"
    git -C "$remote_dir" pull --ff-only
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
[[ -f data/processed/airbnb.csv ]] || { echo "❌ data/processed/airbnb.csv missing (run clean)" >&2; exit 1; }
CSVS=(data/processed/airbnb.csv data/processed/airbnb_described_*.csv)
if [ ${#CSVS[@]} -eq 1 ]; then
    echo "⚠️  no data/processed/airbnb_described_*.csv — run the describe stage, then re-run to ship data"
else
    echo "📤 syncing ${#CSVS[@]} dataset CSV(s) -> $SSH_HOST:$REMOTE_DIR/"
    printf '   %s\n' "${CSVS[@]}"
    # --relative keeps the data/processed/ prefix so remote stages find their inputs
    rsync --archive --relative --checksum --progress "${CSVS[@]}" "$SSH_HOST:$REMOTE_DIR/"
fi
echo "✅ VM initialized, credentials and data synced"
echo "Next: ssh $SSH_HOST \"cd $REMOTE_DIR && scripts/screen.sh\""
