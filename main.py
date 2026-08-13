"""Pipeline entrypoint — run the whole thing with ``uv run main``.

This is a thin orchestrator. The heavy stages import torch / diffusers / the
MulTaBench package, so they are dispatched into the MulTaBench uv venv
(``MulTaBench/.venv``, which owns the correct CUDA torch build) rather than the
orchestrator's own env. That keeps ``uv run main`` fast and avoids duplicating
a multi-GB CUDA torch install.

Stages:
  1+2. build NL descriptions & generate one image per row (Stable Diffusion)
  3.   OPTIONAL reverse-image-search enrichment (off by default)
  4.   assemble the enriched multimodal CSV
  5.   MulTaBench tri-modal curation eligibility eval

Examples:
  uv run main --limit 20 --no-eval        # smoke: generate 20 images + enrich
  uv run main                             # full pipeline (all rows) + eval
  uv run main --ris                       # also swap in reverse-search images
  uv run main --skip-gen                  # only (re)build CSV + eval existing images
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

# DINOv3 is a gated HF repo; both frozen and TAR image conditions need access.
DINO_REPO = "facebook/dinov3-vits16-pretrain-lvd1689m"


def _project_root() -> str:
    """Locate the repo root.

    ``main`` is installed as a console script, so ``__file__`` lives in
    site-packages — useless for finding data/ and MulTaBench/. uv runs the
    command from the project directory, so walk up from cwd to the dir holding
    pyproject.toml instead.
    """
    for p in (Path.cwd(), *Path.cwd().parents):
        if (p / "pyproject.toml").is_file():
            return str(p)
    return os.getcwd()


_HERE = _project_root()
VENV_PY = os.path.join(_HERE, "MulTaBench", ".venv", "bin", "python")
RAW = os.path.join(_HERE, "data", "house_price_prediction.csv")
IMAGES = os.path.join(_HERE, "images")
OUT = os.path.join(_HERE, "data", "house_price_multimodal.csv")


def _run(argv: list[str]) -> None:
    print(f"\n$ {' '.join(argv)}", flush=True)
    subprocess.run(argv, cwd=_HERE, check=True)


def _pin_gpu(no_tar: bool) -> None:
    """TAR (LoRA) conditions hard-assert ``CUDA_VISIBLE_DEVICES``. Auto-pin it when
    a GPU is visible; otherwise tell the user to pass --no-tar *before* the run
    wastes time on image generation and a text eval that later dies on the assert.
    """
    if no_tar or "CUDA_VISIBLE_DEVICES" in os.environ:
        return
    has_gpu = shutil.which("nvidia-smi") and \
        subprocess.run(["nvidia-smi", "-L"], capture_output=True).returncode == 0
    if has_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # inherited by child subprocesses
        print("ℹ️  pinned CUDA_VISIBLE_DEVICES=0 (TAR needs it)")
    else:
        sys.exit("❌ TAR requested but no GPU detected. Re-run with --no-tar for a "
                 "CPU joint-signal eval, or set CUDA_VISIBLE_DEVICES yourself.")


def _check_dino_access() -> None:
    """Fail fast if the gated DINOv3 repo isn't accessible — a fresh clone with an
    unauthorized token otherwise only discovers this deep inside the image eval,
    after image generation and the whole text eval have already run.
    """
    code = ("from dotenv import load_dotenv; load_dotenv('MulTaBench/.env');"
            f"from huggingface_hub import auth_check; auth_check({DINO_REPO!r})")
    r = subprocess.run([VENV_PY, "-c", code], cwd=_HERE, capture_output=True, text=True)
    if r.returncode != 0:
        detail = (r.stderr.strip().splitlines() or ["(no detail)"])[-1]
        sys.exit(
            f"❌ Cannot access gated HF repo {DINO_REPO}.\n"
            f"   1. Request access: https://huggingface.co/{DINO_REPO}\n"
            "   2. Put an *authorized* token in MulTaBench/.env (HF_TOKEN=...).\n"
            "      Note: the token's account must be on the allow-list — a valid\n"
            "      token without granted access still 403s.\n"
            f"   {detail}")


def main() -> None:
    p = argparse.ArgumentParser(prog="main", description="Semi-synthetic multimodal pipeline.")
    p.add_argument("--limit", type=int, default=None, help="process only first N rows")
    p.add_argument("--model-id", default=None, help="override Stable Diffusion model id")
    p.add_argument("--skip-gen", action="store_true", help="skip image generation")
    p.add_argument("--no-eval", action="store_true", help="stop after building the CSV")
    p.add_argument("--no-tar", action="store_true", help="eval joint-signal only (no GPU LoRA)")
    ris = p.add_mutually_exclusive_group()
    ris.add_argument("--ris", dest="ris", action="store_true", help="enable reverse-image-search")
    ris.add_argument("--no-ris", dest="ris", action="store_false")
    p.set_defaults(ris=False)
    args = p.parse_args()

    if not os.path.exists(VENV_PY):
        sys.exit(f"❌ {VENV_PY} not found — run ./init.sh first.")

    # Preflight the eval's env requirements *before* image generation, so a fresh
    # clone fails in seconds instead of after the slow stages. Skipped for --no-eval.
    if not args.no_eval:
        _pin_gpu(args.no_tar)
        _check_dino_access()

    limit = ["--limit", str(args.limit)] if args.limit is not None else []

    # Stage 1+2: generate images (descriptions are computed inside generation).
    if not args.skip_gen:
        gen = [VENV_PY, "-m", "pipeline.generate_images", "--csv", RAW, "--out-dir", IMAGES, *limit]
        if args.model_id:
            gen += ["--model-id", args.model_id]
        _run(gen)

    # Stage 3 (optional): reverse-image-search enrichment.
    if args.ris:
        tmp = os.path.join(_HERE, "data", "_ris_descriptions.csv")
        _run([VENV_PY, "-c",
              "import pandas as pd,sys;from pipeline.prompt_builder import add_descriptions;"
              f"d=pd.read_csv({RAW!r});"
              + (f"d=d.head({args.limit});" if args.limit is not None else "")
              + f"add_descriptions(d.reset_index(drop=True)).to_csv({tmp!r},index=False)"])
        _run([VENV_PY, "-m", "pipeline.reverse_search_enrich", "--csv", tmp,
              "--image-folder", IMAGES, *limit])

    # Stage 4: assemble the enriched CSV.
    _run([VENV_PY, "-m", "pipeline.enrich_csv", "--raw-csv", RAW,
          "--image-folder", IMAGES, "--out-csv", OUT, *limit])

    # Stage 5: eligibility eval.
    if not args.no_eval:
        ev = [VENV_PY, "run_multabench_eval.py", "--csv", OUT, "--image-folder", IMAGES,
              "--target", "price", "--task", "reg"]
        if args.no_tar:
            ev.append("--no-tar")
        _run(ev)


if __name__ == "__main__":
    main()
