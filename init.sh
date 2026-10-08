#!/usr/bin/env bash
# One-shot setup, uv-driven.
#
#   1. clone the official MulTaBench and check out the pinned commit
#   2. run MulTaBench's own init  -> builds MulTaBench/.venv (uv) with its deps
#   3. install this project's dataset-build libs (geopandas/duckdb/pydantic-ai)
#      + the src package (editable) INTO MulTaBench/.venv
#      -> build + describe all run in that one venv
#   4. `uv sync` the thin env-holder project
#   5. remove MulTaBench/.env; credentials are supplied separately with scripts/sync.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

mkdir -p data/processed

# Official repository. Override only to point at a mirror.
MULTABENCH_REPO="${MULTABENCH_REPO:-https://github.com/alanarazi7/MulTaBench}"
# Pinned master commit; see docs/adr/0001-evaluate-on-latest-multabench-master.md.
MULTABENCH_COMMIT="d88821d"
VENV_PY="$HERE/MulTaBench/.venv/bin/python"

# uv may build/download Python during setup.  Install these headers before that
# happens so the resulting interpreter includes the stdlib _lzma and _bz2 modules.
if [ -r /etc/os-release ]; then
    # shellcheck disable=SC1091
    . /etc/os-release
    if [ "${ID:-}" = "ubuntu" ]; then
        echo "📦 ensuring Ubuntu compression build dependencies (_lzma, _bz2)"
        if [ "$(id -u)" -eq 0 ]; then
            apt-get update
            apt-get install --yes liblzma-dev libbz2-dev
        elif command -v sudo >/dev/null; then
            sudo apt-get update
            sudo apt-get install --yes liblzma-dev libbz2-dev
        else
            echo "❌ Ubuntu needs liblzma-dev and libbz2-dev, but sudo is unavailable"
            exit 1
        fi
    fi
fi

command -v git >/dev/null || {
    echo "❌ git not found"
    exit 1
}
command -v uv >/dev/null || {
    echo "❌ uv not found. Install: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
}

# GPU: choose Torch's wheel index BEFORE MulTaBench resolves its requirements.
# `UV_TORCH_BACKEND` is honored by its internal `uv pip install -r`, avoiding
# the old pattern of downloading Torch once and force-reinstalling it afterward.
GPU_CAP=""
CUDA_TAG=""
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    GPU_CAP="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null |
        head -n1 | tr -d ' .')"
    case "$GPU_CAP" in
    10* | 12* | 13*) CUDA_TAG="cu128" ;; # Blackwell (sm_100/sm_120) and newer
    "") CUDA_TAG="cu128" ;;              # old nvidia-smi w/o compute_cap: assume new
    *) CUDA_TAG="cu126" ;;               # Hopper sm_90 and older
    esac
    export UV_TORCH_BACKEND="$CUDA_TAG"
    echo "🎮 GPU sm_${GPU_CAP:-?} -> selecting torch ${CUDA_TAG} during dependency install"
fi

# 1. Clone the official MulTaBench and pin it (idempotent).
if [ ! -d MulTaBench/.git ]; then
    echo "📥 cloning MulTaBench"
    git clone "$MULTABENCH_REPO" MulTaBench
else
    echo "✅ MulTaBench already present — skipping clone"
fi
# An older checkout may point at the retired fork and carry its local patch:
# re-point origin, fetch the pinned commit, and discard the stale edit.
git -C MulTaBench remote set-url origin "$MULTABENCH_REPO"
git -C MulTaBench fetch --quiet origin
git -C MulTaBench checkout --quiet --force "$MULTABENCH_COMMIT"
echo "📌 MulTaBench pinned at $(git -C MulTaBench rev-parse --short HEAD)"

# 3. Build MulTaBench's uv venv + install its deps (its init.sh is uv-based).
# It's designed to be sourced without `set -eu`; disable our hardening inside the
# subshell so its unguarded PYTHONPATH ref doesn't trip nounset.
echo "🐍 running MulTaBench/init.sh (uv venv + deps)"
(
    set +eu
    cd MulTaBench && source init.sh
)

# 3a. Install this project's package (editable) + its dataset-build libs (the
# `pipeline` extra in pyproject.toml) into the same venv, so the stages resolve
# as `-m src.*`.
echo "📦 installing this project (editable, [pipeline] extra) into MulTaBench/.venv"
uv pip install --python "$VENV_PY" -e ".[pipeline]"

# 3b. GPU: verify the wheel selected during MulTaBench's dependency install
# supports THIS GPU's compute capability. torch 2.7.1 ships only cu118/cu126/cu128 wheels:
#   cu126 -> sm_50..sm_90 ; cu128 adds sm_100/sm_120 (Blackwell, e.g. RTX 50xx).
# We verify the wheel actually carries sm_<cap> and fail loud if not — so a
# future GPU that needs a build we didn't map can't silently fall back to "no
# kernel image".
if [ -n "$CUDA_TAG" ]; then
    "$VENV_PY" - "$GPU_CAP" <<'PY'
import sys, torch
cap = sys.argv[1]
archs = torch.cuda.get_arch_list()
print(f"   torch {torch.__version__} cuda {torch.version.cuda}")
print(f"   archs {archs}")
if cap:
    want = f"sm_{cap}"
    if want not in archs:
        sys.exit(f"❌ torch wheel lacks {want} for this GPU — "
                 f"add a case for sm_{cap} in init.sh (3b) with the right cuXXX wheel")
    print(f"   ✅ {want} supported")
PY
else
    echo "💻 no NVIDIA GPU — skipping GPU torch install (CPU / --no-tar path)"
fi

# 4. Sync the thin env-holder project (creates ./.venv).
echo "🔗 uv sync"
uv sync

# 5. Never retain credentials in a freshly initialized checkout.  The source
# .env is intentionally synced separately to a configured remote with scripts/sync.sh.
if [ -e MulTaBench/.env ]; then
    rm -f -- MulTaBench/.env
    echo "🗑️  removed MulTaBench/.env — transfer credentials separately with scripts/sync.sh"
fi

cat <<'EOF'

🎉 Setup complete.

Drive one stage at a time — each runs in MulTaBench/.venv automatically:
  uv run clean          # data/raw/airbnb_nyc.csv -> data/processed/airbnb.csv (pandas; run once)
  uv run build          # -> data/processed/airbnb_enriched.csv (Overture POIs within 400m)
  uv run describe --confirm [N]   # -> data/processed/airbnb_described_<prompt>.csv
                                  # (LLM summary; spends credits, needs .env key; N = top-N test;
                                  #  --prompt 05|08|16)

  uv run stage          # described corpora -> registered datasets in the local HF cache (offline)

Benchmark stages (stage/bench/report) run with HF_HUB_OFFLINE=1; bench/report arrive
in later tickets and take the benchmark's --device flag.

To copy credentials to this checkout on the vast host, run scripts/sync.sh from the
source checkout that contains MulTaBench/.env.
EOF
