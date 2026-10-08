"""Derive the landmark-censored arms from a described corpus and the curated landmark list.

`landmark_only` reduces each summary to the landmark names it mentions (an empty
string where it mentions none, so no listing is dropped); `landmark_redacted`
replaces each mentioned name with a generic phrase. Only names on the curated list
(`src/landmarks.json`) are matched, longest first so a name containing another is
replaced as a whole.

The logic is the pure `build_arms`; `main` is a thin shell around it.
"""

from __future__ import annotations

import argparse
import json
import os
import re

import pandas as pd

from src import config

SUMMARY = "surroundings_summary"
LANDMARK_ONLY = "landmark_only"
LANDMARK_REDACTED = "landmark_redacted"
REDACTION = "a well-known landmark"
LANDMARKS_JSON = os.path.join(config.ROOT, "src", "landmarks.json")


def load_landmarks(path: str = LANDMARKS_JSON) -> list[str]:
    with open(path) as f:
        return list(json.load(f)["landmarks"])


def landmark_pattern(names: list[str]) -> re.Pattern[str]:
    """Match curated landmark names, preferring longer overlapping names."""
    alternatives = "|".join(re.escape(n) for n in sorted(names, key=len, reverse=True))
    return re.compile(rf"(?<!\w)({alternatives})(?!\w)", flags=re.IGNORECASE)


def build_arms(corpus: pd.DataFrame, names: list[str]) -> dict[str, pd.DataFrame]:
    """Described corpus + landmark names -> arm name -> corpus with the censored summary.

    Rows (including any without a summary) are kept one-for-one, so every arm
    shares the enriched arm's row set.
    """
    pattern = landmark_pattern(names)
    canonical = {n.casefold(): n for n in names}

    def only(text: str) -> str:
        return "; ".join(canonical[m.group(0).casefold()] for m in pattern.finditer(text))

    def redact(text: str) -> str:
        return pattern.sub(REDACTION, text)

    arms = {}
    for arm, fn in ((LANDMARK_ONLY, only), (LANDMARK_REDACTED, redact)):
        out = corpus.copy()
        has_text = out[SUMMARY].notna()
        out[SUMMARY] = out[SUMMARY].astype(object)
        out.loc[has_text, SUMMARY] = out.loc[has_text, SUMMARY].astype(str).map(fn)
        arms[arm] = out
    return arms


def arm_csv(arm: str, prompt: str) -> str:
    return os.path.join(config.PROCESSED_DIR, f"airbnb_arm_{arm}_{prompt}.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", action="append", required=True, help="prompt id(s) to censor")
    args = parser.parse_args()

    names = load_landmarks()
    for prompt in args.prompt:
        pending = [a for a in (LANDMARK_ONLY, LANDMARK_REDACTED) if not os.path.exists(arm_csv(a, prompt))]
        if not pending:
            print(f"{prompt}: arms exist, skipping")
            continue
        corpus = pd.read_csv(config.described_csv(prompt), low_memory=False)
        for arm, frame in build_arms(corpus, names).items():
            if arm in pending:
                frame.to_csv(arm_csv(arm, prompt), index=False)
                print(f"{prompt}: wrote {arm_csv(arm, prompt)}")


if __name__ == "__main__":
    main()
