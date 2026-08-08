"""Stage 1: turn a structured CSV row into a natural-language description.

The same string is used two ways:
  - as the text prompt fed to the image generator (Stage 2), and
  - as the free-text ``description`` column of the enriched dataset (Stage 4),
    which becomes the *text modality* for the tri-modal MulTaBench evaluation.

Keeping both derived from one template means the text and image modalities are
grounded in the same row, which is exactly what the curation criteria probe.
"""
from __future__ import annotations

import pandas as pd

# condition is an ordinal 1-5 in the King County data; map to words for fluency.
_CONDITION_WORDS = {
    1: "very poor condition",
    2: "poor condition",
    3: "average condition",
    4: "good condition",
    5: "excellent condition",
}

_VIEW_WORDS = {
    0: "",
    1: "a fair view",
    2: "a good view",
    3: "a very good view",
    4: "an excellent view",
}


def _fmt_num(value, unit: str = "") -> str:
    """Render a numeric field compactly (drop trailing .0), tolerating NaN."""
    if pd.isna(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return f"{value}{unit}"


def row_to_prompt(row: pd.Series) -> str:
    """Build a single natural-language sentence describing the property."""
    condition = _CONDITION_WORDS.get(int(row["condition"]), "average condition") \
        if not pd.isna(row.get("condition")) else "average condition"

    floors = _fmt_num(row.get("floors"))
    bedrooms = _fmt_num(row.get("bedrooms"))
    bathrooms = _fmt_num(row.get("bathrooms"))
    sqft_living = _fmt_num(row.get("sqft_living"))
    sqft_lot = _fmt_num(row.get("sqft_lot"))
    yr_built = _fmt_num(row.get("yr_built"))
    city = str(row.get("city", "")).strip()
    statezip = str(row.get("statezip", "")).strip()

    parts = [f"A {condition} {floors}-story house" if floors else f"A {condition} house"]
    if city or statezip:
        parts.append(f"in {', '.join(p for p in (city, statezip) if p)}")
    if yr_built:
        parts.append(f"built in {yr_built}")
    if bedrooms:
        parts.append(f"with {bedrooms} bedrooms")
    if bathrooms:
        parts.append(f"and {bathrooms} bathrooms")
    if sqft_living:
        parts.append(f"offering {sqft_living} sqft of living space")
    if sqft_lot:
        parts.append(f"on a {sqft_lot} sqft lot")

    # Optional distinguishing clauses.
    extras = []
    if not pd.isna(row.get("waterfront")) and int(row["waterfront"]) == 1:
        extras.append("waterfront property")
    view_word = _VIEW_WORDS.get(int(row["view"]), "") if not pd.isna(row.get("view")) else ""
    if view_word:
        extras.append(view_word)
    if not pd.isna(row.get("yr_renovated")) and int(row["yr_renovated"]) > 0:
        extras.append(f"renovated in {int(row['yr_renovated'])}")

    sentence = " ".join(parts)
    if extras:
        sentence += ", " + ", ".join(extras)
    return sentence.strip().rstrip(",") + "."


def add_descriptions(df: pd.DataFrame, col: str = "description") -> pd.DataFrame:
    """Return a copy of ``df`` with a natural-language ``description`` column."""
    out = df.copy()
    out[col] = out.apply(row_to_prompt, axis=1)
    return out


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Preview row->prompt rendering.")
    p.add_argument("--csv", required=True)
    p.add_argument("--n", type=int, default=5)
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    for i, (_, r) in enumerate(df.head(args.n).iterrows()):
        print(f"[{i}] {row_to_prompt(r)}")
