#!/usr/bin/env bash
# One-shot setup, uv-driven.
#
#   1. clone MulTaBench (fork, latest master) + patch in the local dataset id
#   2. run MulTaBench's own init  -> builds MulTaBench/.venv (uv) with its deps
#   3. install this project's dataset-build libs (geopandas/duckdb/pydantic-ai)
#      + the airbnb_surroundings package (editable) INTO MulTaBench/.venv
#      -> build + eval all run in that one venv
#   4. `uv sync` the thin env-holder project
#   5. remove MulTaBench/.env; credentials are supplied separately with sync.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

mkdir -p data/processed

MULTABENCH_REPO="https://github.com/razilai/MulTaBench"
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

command -v git >/dev/null || { echo "❌ git not found"; exit 1; }
command -v uv >/dev/null || { echo "❌ uv not found. Install: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }

# 1. Clone MulTaBench (fork, latest master).
if [ ! -d MulTaBench/.git ]; then
    echo "📥 cloning MulTaBench (latest master)"
    git clone "$MULTABENCH_REPO" MulTaBench
else
    echo "✅ MulTaBench already present — skipping clone"
fi

# 2. Patch: register the local house-price dataset id (idempotent).
echo "🩹 ensuring local dataset enum member is present"
python3 - <<'PY'
import pathlib
f = pathlib.Path("MulTaBench/multabench/datasets/all_datasets.py")
s = f.read_text()
member = 'REG_IMAGE_HOUSE_PRICE_KING_COUNTY = "local/house-price-king-county"'
if member not in s:
    anchor = 'REG_IMAGE_HOUSES_AIRBNB_SEATTLE = "airbnb/seattle/"'
    if anchor not in s:
        raise SystemExit("❌ anchor line not found — MulTaBench layout changed; update init.sh patch")
    s = s.replace(anchor, anchor
                  + "\n    # Local semi-synthetic dataset (this project) — loaded from a local CSV + image folder\n    "
                  + member)
    f.write_text(s)
    print("   patched all_datasets.py")
else:
    print("   already patched")
PY

# 3. Build MulTaBench's uv venv + install its deps (its init.sh is uv-based).
# It's designed to be sourced without `set -eu`; disable our hardening inside the
# subshell so its unguarded PYTHONPATH ref doesn't trip nounset.
echo "🐍 running MulTaBench/init.sh (uv venv + deps)"
( set +eu; cd MulTaBench && source init.sh )

# 3. Install this project's dataset-build libs into the same venv.
echo "📦 installing dataset-build libs into MulTaBench/.venv"
uv pip install --python "$VENV_PY" -r requirements.txt

# 3a. Install this project's own package (editable, no deps — they came from
# requirements.txt above) so the stages resolve as `-m airbnb_surroundings.*`.
echo "📦 installing airbnb_surroundings (editable) into MulTaBench/.venv"
uv pip install --python "$VENV_PY" --no-deps -e .

# 3b. GPU: match the torch build to THIS GPU's compute capability.
# Runs LAST of the venv installs so pytabkit's default (cu126) torch can't clobber it.
# torch 2.7.1 ships only cu118/cu126/cu128 wheels:
#   cu126 -> sm_50..sm_90 ; cu128 adds sm_100/sm_120 (Blackwell, e.g. RTX 50xx).
# We read the cap from nvidia-smi (no torch needed), pick the wheel, then VERIFY
# the wheel actually carries sm_<cap> and fail loud if not — so a future GPU that
# needs a build we didn't map can't silently fall back to "no kernel image".
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    CAP="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
           | head -n1 | tr -d ' .')"
    case "$CAP" in
        10*|12*|13*) CUDA_TAG="cu128" ;;  # Blackwell (sm_100/sm_120) and newer
        "")          CUDA_TAG="cu128" ;;  # old nvidia-smi w/o compute_cap: assume new
        *)           CUDA_TAG="cu126" ;;  # Hopper sm_90 and older
    esac
    echo "🎮 GPU sm_${CAP:-?} -> installing torch ${CUDA_TAG} (last, so it wins)"
    uv pip install --python "$VENV_PY" --upgrade --force-reinstall \
        --index-url "https://download.pytorch.org/whl/${CUDA_TAG}" \
        torch==2.7.1 torchvision==0.22.1
    "$VENV_PY" - "$CAP" <<'PY'
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
# .env is intentionally synced separately to a configured remote with sync.sh.
if [ -e MulTaBench/.env ]; then
    rm -f -- MulTaBench/.env
    echo "🗑️  removed MulTaBench/.env — transfer credentials separately with ./sync.sh"
fi

cat <<'EOF'

🎉 Setup complete.

Run the whole pipeline (build -> describe -> eval) with one command:
  uv run main
  # dispatches every stage to MulTaBench/.venv, so tabstar/torch are always found.
  # existing outputs are skipped; on a GPU box with only synced CSVs it runs the eval.
  # forward flags to the eval after --, e.g.  uv run main -- --no-tar  (no GPU).

Or drive one stage at a time — each runs in MulTaBench/.venv automatically:
  uv run clean          # data/raw/airbnb_nyc.csv -> data/processed/airbnb.csv (PySpark; run once)
  uv run build          # -> data/processed/airbnb_enriched.csv (Overture POIs within 400m)
  uv run describe 10    # -> data/processed/airbnb_described.csv (LLM summary; needs .env key; 10 = cheap test)
  uv run eval           # MulTaBench eligibility; add --no-tar for joint-signal only (no GPU)

To copy credentials to this checkout on the vast host, run ./sync.sh from the
source checkout that contains MulTaBench/.env.
EOF
