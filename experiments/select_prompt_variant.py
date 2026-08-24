"""Choose a prompt-screen winner from frozen joint-signal reports.

The primary ranking metric is the median R² gain of joint frozen text over the
structured baseline across learners.  Variants within 0.01 are resolved by lower
content-only Jaccard, so surface-form churn never beats an appreciable price gain.

Run after generating the screen CSVs and their ``airbnb_surroundings.eval`` reports:
    python experiments/select_prompt_variant.py --screen-dir experiments/screen
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

TEXT_COLUMN = "surroundings_summary"
TOKEN_RE = re.compile(r"[^\W\d_]+", re.UNICODE)
STOPWORDS = frozenset(
    """
    a an and are as at be been being but by can could did do does doing for from had
    has have having he her here hers herself him himself his how i if in into is it
    its itself just me more most my myself no nor not of off on once only or other
    our ours ourselves out over own same she should so some such than that the their
    theirs them themselves then there these they this those through to too under until
    up very was we were what when where which while who whom why will with would you
    your yours yourself yourselves
    """.split()
)
BOILERPLATE = frozenset(
    {"area", "block", "city", "listing", "nearby", "neighborhood", "place", "places", "short", "walk", "walking"}
)


def content_jaccard(values: list[str], max_pairs: int = 20_000) -> float:
    token_sets = [
        frozenset(token for token in TOKEN_RE.findall(value.lower()) if token not in STOPWORDS | BOILERPLATE)
        for value in values
    ]
    pairs = list(combinations(range(len(token_sets)), 2))
    if len(pairs) > max_pairs:
        rng = np.random.default_rng(0)
        pick = rng.choice(len(pairs), size=max_pairs, replace=False)
        pairs = [pairs[int(index)] for index in pick]
    scores = [
        len(token_sets[left] & token_sets[right]) / len(token_sets[left] | token_sets[right])
        if token_sets[left] | token_sets[right]
        else 0.0
        for left, right in pairs
    ]
    return float(np.mean(scores)) if scores else 0.0


def metrics(screen_dir: Path, prompt_id: str) -> dict[str, float | str]:
    dataset = screen_dir / f"df2_{prompt_id}.csv"
    report_path = screen_dir / f"report_df2_{prompt_id}.csv"
    frame = pd.read_csv(dataset)
    if frame[TEXT_COLUMN].isna().any() or not frame[TEXT_COLUMN].astype(str).str.strip().all():
        raise ValueError(f"{dataset} has missing summaries")
    report = pd.read_csv(report_path)
    required = {"learner", "structured", "joint_text_frozen"}
    if missing := required - set(report.columns):
        raise ValueError(f"{report_path} is missing {sorted(missing)}")
    gain = report["joint_text_frozen"] - report["structured"]
    return {
        "prompt_id": prompt_id,
        "median_joint_gain": float(gain.median()),
        "mean_joint_gain": float(gain.mean()),
        "content_jaccard": content_jaccard(frame[TEXT_COLUMN].astype(str).tolist()),
        "mean_words": float(frame[TEXT_COLUMN].astype(str).str.findall(TOKEN_RE).str.len().mean()),
    }


def choose(rows: list[dict[str, float | str]], tolerance: float = 0.01) -> dict[str, float | str]:
    best_gain = max(float(row["median_joint_gain"]) for row in rows)
    close = [row for row in rows if best_gain - float(row["median_joint_gain"]) <= tolerance]
    return min(close, key=lambda row: (float(row["content_jaccard"]), float(row["mean_words"])))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen-dir", type=Path, default=Path(__file__).parent / "screen")
    parser.add_argument("--variants", nargs="+", default=["14_contrast_access", "15_contrast_local_profile", "16_contrast_anchor_profile"])
    parser.add_argument("--out", type=Path, help="JSON selection report (defaults inside --screen-dir)")
    args = parser.parse_args()
    rows = [metrics(args.screen_dir, prompt_id) for prompt_id in args.variants]
    winner = choose(rows)
    result = {"selection_rule": "max median frozen joint R2 gain; ties within 0.01 use lower content Jaccard", "variants": rows, "winner": winner}
    out = args.out or args.screen_dir / "prompt_variant_selection.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
