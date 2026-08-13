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

# The prompt feeds an exterior house image generator whose CLIP text encoder is
# poor at raw numerals and can only render *visible* attributes. So every numeric
# field is bucketed into a visual adjective/phrase the model can actually depict,
# instead of digits it ignores (which produced identical "incredible" houses
# regardless of the row). Interior-only counts (bedrooms/bathrooms) and pure
# digits (exact sqft, zip) are dropped — they carry no exterior signal.

# condition 1-5 -> concrete visible upkeep cues (not the abstract word "condition").
_CONDITION_WORDS = {
    1: "dilapidated and run-down, peeling paint, broken windows, overgrown yard",
    2: "worn and aging, faded paint, weathered exterior",
    3: "ordinary, with average upkeep",
    4: "well-kept, tidy and maintained",
    5: "pristine and immaculate, with manicured landscaping",
}

_VIEW_WORDS = {
    0: "",
    1: "a fair view",
    2: "a good view",
    3: "a very good view",
    4: "an excellent view",
}

# sqft_living -> visible scale word. Thresholds ~ King County quartiles.
_SQFT_THRESHOLDS = (1200, 1800, 2600, 4000)
_SQFT_WORDS = ("compact", "modest", "spacious", "large", "sprawling")

# yr_built -> architectural era (renders as style; raw year does not).
_ERA_THRESHOLDS = (1930, 1950, 1975, 2000)
_ERA_WORDS = (
    "1920s craftsman-style",
    "1940s traditional",
    "mid-century modern",
    "late-20th-century suburban",
    "contemporary new-build",
)


def _bucket(value, thresholds: tuple, labels: tuple) -> str:
    """Map a numeric value to a label by ascending thresholds. ``len(labels)`` must
    be ``len(thresholds) + 1``. Returns the middle label on NaN/missing."""
    if pd.isna(value):
        return labels[len(labels) // 2]
    for i, t in enumerate(thresholds):
        if value < t:
            return labels[i]
    return labels[-1]


def _fmt_floors(value) -> str:
    """Story count is visible; render it compactly (drop trailing .0), '' on NaN."""
    if pd.isna(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return f"{value}"


def row_to_prompt(row: pd.Series) -> str:
    """Build a single natural-language sentence describing the property, using only
    exterior-visible, model-renderable attributes."""
    condition = _CONDITION_WORDS.get(int(row["condition"]), _CONDITION_WORDS[3]) \
        if not pd.isna(row.get("condition")) else _CONDITION_WORDS[3]
    scale = _bucket(row.get("sqft_living"), _SQFT_THRESHOLDS, _SQFT_WORDS)
    era = _bucket(row.get("yr_built"), _ERA_THRESHOLDS, _ERA_WORDS)
    floors = _fmt_floors(row.get("floors"))
    city = str(row.get("city", "")).strip()

    article = "An" if condition[0].lower() in "aeiou" else "A"
    head = f"{article} {condition}, {scale}, {era} house"
    if floors:
        head = f"{article} {condition}, {scale}, {era} {floors}-story house"
    parts = [head]
    if city and city.lower() != "nan":
        parts.append(f"in {city}")

    # Optional distinguishing clauses (all exterior-visible).
    extras = []
    if not pd.isna(row.get("waterfront")) and int(row["waterfront"]) == 1:
        extras.append("waterfront property")
    view_word = _VIEW_WORDS.get(int(row["view"]), "") if not pd.isna(row.get("view")) else ""
    if view_word:
        extras.append(view_word)
    if not pd.isna(row.get("yr_renovated")) and int(row["yr_renovated"]) > 0:
        extras.append("recently renovated")

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
