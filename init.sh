#!/usr/bin/env bash
# One-shot setup, uv-driven.
#
#   1. clone MulTaBench (pinned) + patch in the local dataset id
#   2. run MulTaBench's own init  -> builds MulTaBench/.venv (uv) with its deps
#   3. install this project's dataset-build libs (geopandas/pyrosm/pydantic-ai)
#      INTO MulTaBench/.venv -> build + eval all run in that one venv
#   4. `uv sync` the thin env-holder project
#   5. scaffold MulTaBench/.env for credentials (HF_TOKEN etc.)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

MULTABENCH_REPO="https://github.com/alanarazi7/MulTaBench"
MULTABENCH_COMMIT="599bd6a5631c96f8aef297cc4cb6e4c197ae0dca"
VENV_PY="$HERE/MulTaBench/.venv/bin/python"

command -v git >/dev/null || { echo "❌ git not found"; exit 1; }
command -v uv >/dev/null || { echo "❌ uv not found. Install: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }

# 1. Clone MulTaBench (pinned commit for reproducibility).
if [ ! -d MulTaBench/.git ]; then
    echo "📥 cloning MulTaBench @ ${MULTABENCH_COMMIT:0:8}"
    git clone "$MULTABENCH_REPO" MulTaBench
    git -C MulTaBench checkout --quiet "$MULTABENCH_COMMIT"
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

# 4. Sync the thin env-holder project (creates ./.venv).
echo "🔗 uv sync"
uv sync

# 5. Scaffold credentials file.
if [ ! -f MulTaBench/.env ]; then
    cp MulTaBench/.env.example MulTaBench/.env
    echo "📝 created MulTaBench/.env — fill in HF_TOKEN (gated DINOv3) + WANDB/KAGGLE keys"
fi

cat <<'EOF'

🎉 Setup complete.

PY=MulTaBench/.venv/bin/python

Build the dataset:
  $PY main.py            # airbnb.csv -> airbnb_enriched.csv (OSM POIs within 50m)
  $PY describe.py 10     # -> airbnb_described.csv (LLM summary; needs .env key)

Run the eval (text + tabular only — no image modality):
  $PY run_multabench_eval.py --no-image --target price \
      --csv airbnb_described.csv --image-folder /dev/null
  # add --no-tar for joint-signal only (no GPU). TAR (LoRA) needs a GPU + HF_TOKEN.
EOF
