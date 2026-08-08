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
import subprocess
import sys
from pathlib import Path


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
