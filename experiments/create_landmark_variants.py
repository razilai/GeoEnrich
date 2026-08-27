"""Create landmark-only and landmark-redacted variants of the baseline dataset."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
VARIANTS_DIR = ROOT / "data" / "processed" / "variants"
SOURCE = VARIANTS_DIR / "v1.csv"
LANDMARKS = ROOT / "airbnb_surroundings" / "landmarks.json"
LANDMARK_ONLY = VARIANTS_DIR / "v2-landmarks.csv"
LANDMARK_REDACTED = VARIANTS_DIR / "v3-redacted.csv"
TEXT_COLUMN = "surroundings_summary"


def landmark_pattern(names: list[str]) -> re.Pattern[str]:
    """Match curated landmark names, preferring longer overlapping names."""
    alternatives = "|".join(re.escape(name) for name in sorted(names, key=len, reverse=True))
    return re.compile(rf"(?<!\w)({alternatives})(?!\w)", flags=re.IGNORECASE)


def main() -> None:
    names = list(json.loads(LANDMARKS.read_text())["landmarks"])
    canonical_names = {name.casefold(): name for name in names}
    pattern = landmark_pattern(names)
    dataset = pd.read_csv(SOURCE)

    def matched_landmarks(text: str) -> list[str]:
        return [canonical_names[match.group(0).casefold()] for match in pattern.finditer(text)]

    original_text = dataset[TEXT_COLUMN].fillna("").astype(str)

    landmark_only = dataset.copy()
    landmark_only[TEXT_COLUMN] = original_text.map(
        lambda text: "; ".join(matched_landmarks(text))
    )
    landmark_only.to_csv(LANDMARK_ONLY, index=False)

    landmark_redacted = dataset.copy()
    landmark_redacted[TEXT_COLUMN] = original_text.map(
        lambda text: pattern.sub("_", text)
    )
    landmark_redacted.to_csv(LANDMARK_REDACTED, index=False)


if __name__ == "__main__":
    main()
