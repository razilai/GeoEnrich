#!/usr/bin/env bash
# One-shot setup, uv-driven. After this, run the whole pipeline with:  uv run main
#
#   1. clone MulTaBench (pinned) + patch in the local house-price dataset id
#   2. run MulTaBench's own init  -> builds MulTaBench/.venv (uv) with its deps
#   3. install this pipeline's libs (Stable Diffusion etc.) INTO MulTaBench/.venv
#      -> heavy stages (gen + eval) all run in that one venv; correct CUDA torch
#   4. clone the reverse-image-search helper (optional stage, off by default)
#   5. `uv sync` the thin orchestrator project (provides the `main` entrypoint)
#   6. scaffold MulTaBench/.env for credentials (HF_TOKEN etc.)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

MULTABENCH_REPO="https://github.com/alanarazi7/MulTaBench"
MULTABENCH_COMMIT="599bd6a5631c96f8aef297cc4cb6e4c197ae0dca"
RIS_REPO="https://github.com/ramonclaudio/Google-Reverse-Image-Search.git"
RIS_COMMIT="015628982fee23bf11293b97232fb0e5ac9f41f9"
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
echo "🐍 running MulTaBench/init.sh (uv venv + deps)"
( cd MulTaBench && source init.sh )

# 4. Install this pipeline's libs into the same venv (diffusers, pillow, ...).
echo "📦 installing pipeline libs into MulTaBench/.venv"
uv pip install --python "$VENV_PY" -r requirements.txt

# 5. Clone reverse-image-search helper (optional --ris stage; off by default).
if [ ! -d reverse-img-search/.git ]; then
    echo "📥 cloning reverse-img-search @ ${RIS_COMMIT:0:8}"
    git clone "$RIS_REPO" reverse-img-search
    git -C reverse-img-search checkout --quiet "$RIS_COMMIT"
else
    echo "✅ reverse-img-search already present — skipping clone"
fi

# 6. Sync the thin orchestrator project (creates ./.venv, exposes `main`).
echo "🔗 uv sync (orchestrator project)"
uv sync

# 7. Scaffold credentials file.
if [ ! -f MulTaBench/.env ]; then
    cp MulTaBench/.env.example MulTaBench/.env
    echo "📝 created MulTaBench/.env — fill in HF_TOKEN (gated DINOv3) + WANDB/KAGGLE keys"
fi

cat <<'EOF'

🎉 Setup complete.

To run the eval end-to-end you need:
  - NVIDIA GPU + CUDA (TAR LoRA fine-tuning asserts CUDA_VISIBLE_DEVICES)
  - authorized HF_TOKEN in MulTaBench/.env (facebook/dinov3-* is a gated repo)

Run the whole pipeline:
  uv run main --limit 20 --no-eval     # smoke: 20 images + enriched CSV
  uv run main                          # full pipeline (all rows) + eval
  uv run main --no-tar                 # eval joint-signal only (no GPU)
EOF
