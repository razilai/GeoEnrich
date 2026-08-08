#!/usr/bin/env bash
# Set up everything needed to run the semi-synthetic multimodal pipeline:
#   1. clone MulTaBench (pinned) and run its own init (uv .venv + its deps)
#   2. patch MulTaBench with the local house-price dataset enum member
#   3. install this pipeline's deps (Stable Diffusion etc.) into MulTaBench/.venv
#      -> a single venv runs BOTH generation (main.py) and eval (run_multabench_eval.py)
#   4. clone the reverse-image-search helper (optional stage, off by default)
#   5. scaffold MulTaBench/.env for credentials (HF_TOKEN etc.)
#
# Usage:  ./init.sh        (or: bash init.sh)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

MULTABENCH_REPO="https://github.com/alanarazi7/MulTaBench"
MULTABENCH_COMMIT="599bd6a5631c96f8aef297cc4cb6e4c197ae0dca"
RIS_REPO="https://github.com/ramonclaudio/Google-Reverse-Image-Search.git"
RIS_COMMIT="015628982fee23bf11293b97232fb0e5ac9f41f9"
VENV_PY="$HERE/MulTaBench/.venv/bin/python"

command -v git >/dev/null || { echo "❌ git not found"; exit 1; }
if ! command -v uv >/dev/null; then
    echo "❌ uv not found. Install: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

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

# 3. Build MulTaBench's uv venv + install its deps (its init.sh handles uv/python).
echo "🐍 running MulTaBench/init.sh (uv venv + deps)"
( cd MulTaBench && source init.sh )

# 4. Install this pipeline's deps into the same venv (SD, diffusers, etc.).
echo "📦 installing pipeline deps into MulTaBench/.venv"
uv pip install --python "$VENV_PY" -r requirements.txt

# 5. Clone reverse-image-search helper (optional --ris stage; off by default).
if [ ! -d reverse-img-search/.git ]; then
    echo "📥 cloning reverse-img-search @ ${RIS_COMMIT:0:8}"
    git clone "$RIS_REPO" reverse-img-search
    git -C reverse-img-search checkout --quiet "$RIS_COMMIT"
else
    echo "✅ reverse-img-search already present — skipping clone"
fi

# 6. Scaffold credentials file.
if [ ! -f MulTaBench/.env ]; then
    cp MulTaBench/.env.example MulTaBench/.env
    echo "📝 created MulTaBench/.env — fill in HF_TOKEN (gated DINOv3) + WANDB/KAGGLE keys"
fi

cat <<EOF

🎉 Setup complete.

Requirements to actually run the eval end-to-end:
  - NVIDIA GPU + CUDA (TAR LoRA fine-tuning asserts CUDA_VISIBLE_DEVICES)
  - authorized HF_TOKEN in MulTaBench/.env (facebook/dinov3-* is a gated repo)

Run (single venv):
  PY=MulTaBench/.venv/bin/python
  \$PY main.py --limit 20 --no-ris          # smoke: generate 20 images + enrich
  \$PY main.py --no-ris                      # full generation (all rows)
  \$PY run_multabench_eval.py --csv data/house_price_multimodal.csv \\
        --image-folder images --target price --task reg
EOF
