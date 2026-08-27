#!/usr/bin/env python3
"""Create the paper's compact dataset card and a field-level appendix table."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


def tex(value: object) -> str:
    return (
        str(value)
        .replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("#", r"\#")
    )


def field_rows(csv_path: Path) -> list[dict[str, object]]:
    with csv_path.open(newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    columns = list(rows[0])
    output = []
    for column in columns:
        values = [(row.get(column) or "").strip() for row in rows]
        nonempty = [value for value in values if value]
        numeric = 0
        parsed: list[float] = []
        for value in nonempty:
            try:
                parsed.append(float(value))
                numeric += 1
            except ValueError:
                pass
        kind = "numeric" if nonempty and numeric == len(nonempty) else "string"
        row: dict[str, object] = {
            "field": column,
            "type": kind,
            "missing_n": len(values) - len(nonempty),
            "missing_pct": round(100 * (len(values) - len(nonempty)) / len(values), 2),
            "unique_n": len(set(nonempty)),
        }
        if kind == "numeric" and parsed:
            row.update(
                min=round(min(parsed), 4),
                median=round(statistics.median(parsed), 4),
                max=round(max(parsed), 4),
            )
        else:
            row.update(min="", median="", max="")
        output.append(row)
    return output


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    csv_path = Path(config["input_csv"])
    with csv_path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        columns = reader.fieldnames or []
        n_rows = sum(1 for _ in reader)
    target = config["target_column"]
    text = config["text_column"]
    structured_columns = [column for column in columns if column not in {target, text}]
    metadata = config.get("metadata", {})
    out = args.out / "tables"
    out.mkdir(parents=True, exist_ok=True)

    table_1 = [
        {
            "Records / target": (
                f"{n_rows}; {target}; {metadata['target_scale']}; {metadata['snapshot']}"
            ),
            "Modalities": f"Structured: {len(structured_columns)}; text: {text}",
            "Curation and controls": "; ".join(
                [
                    metadata["overture_release"],
                    metadata["landmark_inventory"],
                    metadata["radii"],
                    metadata["curation_controls"],
                ]
            ),
            "Artifact": metadata["release_artifact"],
        }
    ]
    table_1_fields = list(table_1[0])
    write_csv(out / "table_1_dataset_card.csv", table_1, table_1_fields)
    (out / "table_1_dataset_card.tex").write_text(
        "\\begin{tabular}{p{.23\\linewidth}p{.21\\linewidth}p{.36\\linewidth}p{.14\\linewidth}}\n"
        "\\toprule\nRecords / target & Modalities & Curation and controls & Artifact \\\\\n\\midrule\n"
        + " & ".join(tex(table_1[0][field]) for field in table_1_fields) + " \\\\\n\\bottomrule\n\\end{tabular}\n",
        encoding="utf-8",
    )
    rows = field_rows(csv_path)
    field_names = ["field", "type", "missing_n", "missing_pct", "unique_n", "min", "median", "max"]
    write_csv(out / "appendix_a1_field_inventory.csv", rows, field_names)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
