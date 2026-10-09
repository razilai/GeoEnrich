#!/usr/bin/env bash
# Run on the GPU VM, from anywhere. Evaluates one described corpus,
# data/processed/airbnb_described_<prompt>.csv, offline and resumably:
#   stage -> [probe] -> bench -> report
#
#   scripts/evaluate.sh --prompt 16                 # final grid: 6 splits, 120 runs
#   scripts/evaluate.sh --prompt 16 --grid screen   # split 0 only, 20 runs
#   scripts/evaluate.sh --prompt 16 --probe         # also time one TAR + one frozen run first
#   DEVICE=cuda:1 scripts/evaluate.sh --prompt 16
#
# Outputs go to results/runs_<prompt>_<grid>/, results/metrics_<prompt>_<grid>.csv and
# results/verdict_<prompt>_<grid>.csv, never to earlier runs' paths: a run with an
# existing result there would be skipped as done.
#
# Survives SSH drops: re-launches itself inside tmux when available.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -z "${TMUX:-}" && -z "${EVALUATE_NO_TMUX:-}" ]] && command -v tmux >/dev/null; then
    echo "🪟 running inside tmux session 'evaluate' (attach: tmux attach -t evaluate)"
    exec tmux new-session -A -s evaluate "EVALUATE_NO_TMUX=1 $(printf '%q ' "$0" "$@"); read -rp 'done — press enter'"
fi

PROBE=0
PROMPT=""
GRID=final
while [[ $# -gt 0 ]]; do
    case "$1" in
    --probe) PROBE=1; shift ;;
    --prompt) PROMPT="$2"; shift 2 ;;
    --grid) GRID="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done
[[ -n "$PROMPT" ]] || { echo "❌ --prompt is required" >&2; exit 2; }
CSV="data/processed/airbnb_described_${PROMPT}.csv"
[[ -f "$CSV" ]] || { echo "❌ $CSV missing — run scripts/sync.sh locally" >&2; exit 1; }

TAG="${PROMPT}_${GRID}"
OUT="results/runs_${TAG}"
echo "dataset: $CSV   grid: $GRID   runs: $OUT"

uv run stage --prompt "$PROMPT"
[[ $PROBE -eq 1 ]] && uv run bench --grid probe --prompt "$PROMPT" --confirm --device "${DEVICE:-cuda}" --output_dir "results/runs_${PROMPT}_probe"
uv run bench --grid "$GRID" --prompt "$PROMPT" --confirm --device "${DEVICE:-cuda}" --output_dir "$OUT" \
    || echo "⚠️ some runs failed; re-run to retry them"
uv run report --grid "$GRID" --output_dir "$OUT" \
    --metrics_csv "results/metrics_${TAG}.csv" --verdict_csv "results/verdict_${TAG}.csv"
echo "✅ results/metrics_${TAG}.csv, results/verdict_${TAG}.csv — fetch with scripts/pull.sh"
