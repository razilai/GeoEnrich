"""Stage 4: assemble the enriched multimodal CSV.

Adds two columns to the raw tabular data:
  - ``description``: natural-language text (text modality). By default this is the
    deterministic ``row_to_prompt`` template; pass ``caption_csv`` to instead use a
    VLM caption of the generated image (see caption_images.py) as the text, which
    decouples the text from the tabular columns.
  - ``image``: the image filename that lives in the image folder (image modality).

Also drops the leaky ``price_per_sqft`` column (= price / sqft_living), which
would trivially reveal the regression target.
"""
from __future__ import annotations

import os

import pandas as pd

from pipeline.prompt_builder import add_descriptions

LEAKY_COLUMNS = ["price_per_sqft"]


def build_multimodal_csv(
    raw_csv: str,
    image_folder: str,
    out_csv: str,
    limit: int | None = None,
    image_col: str = "image",
    require_images: bool = True,
    caption_csv: str | None = None,
) -> pd.DataFrame:
    """Create the enriched CSV and return the resulting DataFrame.

    ``image`` holds a bare filename (e.g. ``row_00001.png``) resolved against
    ``image_folder`` — matching how MulTaBench's DINO loader expects image
    columns (filename + separate image_folder).
    """
    df = pd.read_csv(raw_csv)
    if limit is not None:
        df = df.head(limit).copy()
    df = df.reset_index(drop=True)

    # Image filename convention shared with generate_images.py.
    df[image_col] = [f"row_{i:05d}.png" for i in range(len(df))]

    if caption_csv:
        # VLM caption of the generated image as the text modality (decoupled from
        # the tabular columns). Aligned on the image filename.
        caps = pd.read_csv(caption_csv).set_index(image_col)["description"]
        df["description"] = df[image_col].map(caps)
        missing_caps = df["description"].isna().sum()
        if missing_caps:
            raise ValueError(
                f"{missing_caps} row(s) have no caption in {caption_csv}; "
                "run Stage 2.5 (caption_images) over the same images first."
            )
    else:
        df = add_descriptions(df)

    if require_images:
        missing = [
            fn for fn in df[image_col]
            if not os.path.exists(os.path.join(image_folder, fn))
        ]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} image(s) missing from {image_folder}; "
                f"run Stage 2 (generate_images) first. First few: {missing[:3]}"
            )

    df = df.drop(columns=[c for c in LEAKY_COLUMNS if c in df.columns])

    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"✅ wrote {out_csv} | {len(df)} rows | cols: {list(df.columns)}")
    return df


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Build the enriched multimodal CSV.")
    p.add_argument("--raw-csv", required=True)
    p.add_argument("--image-folder", required=True)
    p.add_argument("--out-csv", required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--no-require-images", action="store_true")
    p.add_argument("--caption-csv", default=None,
                   help="use VLM captions from this CSV as the text modality")
    args = p.parse_args()

    build_multimodal_csv(
        raw_csv=args.raw_csv,
        image_folder=args.image_folder,
        out_csv=args.out_csv,
        limit=args.limit,
        require_images=not args.no_require_images,
        caption_csv=args.caption_csv,
    )
