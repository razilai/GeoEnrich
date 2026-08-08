"""Orchestrator: structured CSV -> NL prompt -> image -> (optional RIS) -> enriched CSV.

Stages:
  1. build NL descriptions from CSV rows          (pipeline.prompt_builder)
  2. generate one image per row (Stable Diffusion)(pipeline.generate_images)
  3. OPTIONAL reverse-image-search enrichment      (pipeline.reverse_search_enrich)
  4. assemble the enriched multimodal CSV          (pipeline.enrich_csv)

Run Stage 5 (MulTaBench eligibility eval) separately via run_multabench_eval.py,
which uses the MulTaBench .venv.

Examples:
  python main.py --limit 20 --no-ris          # smoke test
  python main.py                              # full 4601-row generation
  python main.py --ris                        # also swap in reverse-search images
  python main.py --skip-gen                   # only (re)build the enriched CSV
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

from pipeline.generate_images import generate_images, DEFAULT_MODEL_ID
from pipeline.enrich_csv import build_multimodal_csv
from pipeline.prompt_builder import add_descriptions

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RAW = os.path.join(_HERE, "data", "house_price_prediction.csv")
DEFAULT_IMAGES = os.path.join(_HERE, "images")
DEFAULT_OUT = os.path.join(_HERE, "data", "house_price_multimodal.csv")


def main():
    p = argparse.ArgumentParser(description="Build the semi-synthetic multimodal dataset.")
    p.add_argument("--raw-csv", default=DEFAULT_RAW)
    p.add_argument("--image-folder", default=DEFAULT_IMAGES)
    p.add_argument("--out-csv", default=DEFAULT_OUT)
    p.add_argument("--limit", type=int, default=None, help="process only first N rows")
    p.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    p.add_argument("--skip-gen", action="store_true", help="skip image generation")
    ris = p.add_mutually_exclusive_group()
    ris.add_argument("--ris", dest="ris", action="store_true", help="enable reverse-image-search")
    ris.add_argument("--no-ris", dest="ris", action="store_false")
    p.set_defaults(ris=False)
    args = p.parse_args()

    # Stage 2: generate images (Stage 1 descriptions are computed inside).
    if not args.skip_gen:
        generate_images(csv=args.raw_csv, out_dir=args.image_folder,
                        model_id=args.model_id, limit=args.limit)

    # Stage 3 (optional): reverse-image-search needs descriptions -> write a temp
    # CSV carrying them, then swap generated images for top-1 web results.
    if args.ris:
        from pipeline.reverse_search_enrich import enrich_with_reverse_search
        df = pd.read_csv(args.raw_csv)
        if args.limit is not None:
            df = df.head(args.limit)
        tmp = os.path.join(_HERE, "data", "_ris_descriptions.csv")
        add_descriptions(df.reset_index(drop=True)).to_csv(tmp, index=False)
        enrich_with_reverse_search(csv=tmp, image_folder=args.image_folder, limit=args.limit)

    # Stage 4: assemble enriched CSV.
    build_multimodal_csv(raw_csv=args.raw_csv, image_folder=args.image_folder,
                         out_csv=args.out_csv, limit=args.limit)

    print("\nNext: run the eligibility eval (MulTaBench .venv):")
    print(f"  python run_multabench_eval.py --csv {args.out_csv} "
          f"--image-folder {args.image_folder} --target price --task reg")


if __name__ == "__main__":
    main()
