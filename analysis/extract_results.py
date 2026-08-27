#!/usr/bin/env python3
"""Convert an official MulTaBench W&B CSV export into the paper's tidy score CSV."""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Any


OUTPUT_FIELDS = [
    "dataset_name",
    "learner",
    "condition",
    "fold",
    "score_metric",
    "score",
    "model",
    "official_commit",
    "train_examples",
    "pca_components",
    "tune_e5",
    "runtime_seconds",
    "n_train",
    "n_test",
]
REQUIRED_FIELDS = {"dataset", "model", "fold", "test_score", "multimodal_state", "tune_e5"}
MODEL_NAMES = {
    "tabm": "tabm",
    "catboost": "cat",
    "lightgbm": "light",
    "tabpfnv2": "tabpfnv2",
    "tabpfnv2p5": "tabpfnv2p5",
}


def condition(state: str, tune_e5: str) -> str:
    normalized = state.casefold()
    if "no_text" in normalized or "no text" in normalized:
        return "structured"
    if "text_only" in normalized or "text only" in normalized:
        return "text"
    if tune_e5.casefold() in {"true", "1", "yes"}:
        return "joint_tar"
    if "all" in normalized:
        return "joint_frozen"
    raise ValueError(f"cannot map official multimodal state '{state}' to a paper condition")


def learner(model: str) -> str:
    normalized = re.sub(r"[^a-z0-9]", "", model.casefold())
    if normalized in MODEL_NAMES:
        return MODEL_NAMES[normalized]
    raise ValueError(f"cannot map official model '{model}' to a required learner")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="CSV exported from the official MulTaBench W&B project",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with args.input.open(newline="", encoding="utf-8-sig") as fh:
        raw_rows = list(csv.DictReader(fh))
    if not raw_rows:
        raise SystemExit("W&B export is empty")
    missing = REQUIRED_FIELDS.difference(raw_rows[0])
    if missing:
        raise SystemExit(f"W&B export is missing: {', '.join(sorted(missing))}")
    tidy: list[dict[str, Any]] = []
    for number, raw in enumerate(raw_rows, start=2):
        try:
            model = raw["model"]
            tidy.append({
                "dataset_name": raw["dataset"],
                "learner": learner(model),
                "condition": condition(raw["multimodal_state"], raw["tune_e5"]),
                "fold": int(raw["fold"]),
                "score_metric": "r2",
                "score": float(raw["test_score"]),
                "model": model,
                "official_commit": raw.get("git", ""),
                "train_examples": raw.get("train_examples", ""),
                "pca_components": raw.get("pca_components", ""),
                "tune_e5": raw["tune_e5"],
                "runtime_seconds": raw.get("runtime", ""),
                "n_train": raw.get("n_train", ""),
                "n_test": raw.get("n_test", ""),
            })
        except (TypeError, ValueError) as exc:
            raise SystemExit(f"could not parse W&B export row {number}: {exc}") from exc
    unique = {(r["learner"], r["condition"], r["fold"]) for r in tidy}
    if len(unique) != len(tidy):
        raise SystemExit("W&B export has duplicate learner/condition/fold rows; export one completed run per cell")
    output = args.out / "tables"
    output.mkdir(parents=True, exist_ok=True)
    tidy.sort(key=lambda r: (r["learner"], r["condition"], r["fold"]))
    with (output / "official_scores_tidy.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(tidy)
    print(f"wrote {len(tidy)} official scores to {output / 'official_scores_tidy.csv'}")


if __name__ == "__main__":
    main()
