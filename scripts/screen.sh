#!/usr/bin/env bash
# Run on the GPU VM, from anywhere. Whole screen pipeline, offline, resumable:
#   arms -> stage -> [probe] -> bench screen -> report screen
#
#   scripts/screen.sh                    # every prompt with a described corpus
#   scripts/screen.sh --prompt 08 --prompt 16
#   scripts/screen.sh --probe            # also time one TAR + one frozen run first
#   DEVICE=cuda:1 scripts/screen.sh
#
# Survives SSH drops: re-launches itself inside tmux when available.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -z "${TMUX:-}" && -z "${SCREEN_NO_TMUX:-}" ]] && command -v tmux >/dev/null; then
    echo "🪟 running inside tmux session 'screen' (attach: tmux attach -t screen)"
    exec tmux new-session -A -s screen "SCREEN_NO_TMUX=1 $(printf '%q ' "$0" "$@"); read -rp 'done — press enter'"
fi

PROBE=0
PROMPTS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
    --probe) PROBE=1; shift ;;
    --prompt) PROMPTS+=("$2"); shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

[[ -f data/processed/airbnb.csv ]] || { echo "❌ data/processed/airbnb.csv missing — run scripts/sync.sh locally" >&2; exit 1; }
if [[ ${#PROMPTS[@]} -eq 0 ]]; then
    for f in data/processed/airbnb_described_*.csv; do
        [[ -e "$f" ]] || { echo "❌ no described corpora — run scripts/sync.sh locally" >&2; exit 1; }
        id="${f##*airbnb_described_}"; PROMPTS+=("${id%.csv}")
    done
fi
echo "prompts: ${PROMPTS[*]}"

PFLAGS=()
for p in "${PROMPTS[@]}"; do PFLAGS+=(--prompt "$p"); done

for p in "${PROMPTS[@]}"; do uv run arms --prompt "$p"; done
uv run stage "${PFLAGS[@]}"
[[ $PROBE -eq 1 ]] && uv run bench --grid probe --confirm --device "${DEVICE:-cuda}" "${PFLAGS[@]}"
uv run bench --grid screen --confirm --device "${DEVICE:-cuda}" "${PFLAGS[@]}" || echo "⚠️ some runs failed; re-run to retry them"
uv run report --grid screen
echo "✅ results/metrics_screen.csv, results/verdict_screen.csv — fetch with scripts/pull.sh"
