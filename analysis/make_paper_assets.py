#!/usr/bin/env python3
"""Render the paper's MulTaBench tables, curation verdict, and PDF figures."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Any

# Non-interactive, publication-ready figures. Keep Matplotlib's cache outside
# a user's home directory, which is often read-only on compute hosts.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/geoenrich-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


LEARNERS = ("tabm", "cat", "light", "tabpfnv2", "tabpfnv2p5")
LABELS = {"tabm": "TabM", "cat": "CatBoost", "light": "LightGBM", "tabpfnv2": "TabPFN v2", "tabpfnv2p5": "TabPFN v2.5"}
CONDITIONS = ("structured", "text", "joint_frozen", "joint_tar")
CONDITION_LABELS = {"structured": "Structured", "text": "Text", "joint_frozen": "Joint frozen", "joint_tar": "Joint TAR"}
COLORS = {"structured": "#4c78a8", "text": "#f58518", "joint_frozen": "#54a24b", "joint_tar": "#e45756"}


def tex(value: object) -> str:
    return str(value).replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("&", r"\&").replace("%", r"\%").replace("#", r"\#")


def mean_sem(values: list[float]) -> tuple[float, float]:
    mean = statistics.fmean(values)
    if len(values) < 2:
        return mean, 0.0
    return mean, statistics.stdev(values) / math.sqrt(len(values))


def read_scores(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError("score file is empty")
    required = {"learner", "condition", "fold", "score", "score_metric"}
    if absent := required.difference(rows[0]):
        raise ValueError(f"score file missing fields: {sorted(absent)}")
    if {row["score_metric"].casefold() for row in rows} != {"r2"}:
        raise ValueError("paper result table requires R² scores only")
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def render_pipeline_figure(path: Path, metadata: dict[str, str], text_column: str) -> None:
    ov = metadata["overture_release"]
    landmarks = metadata["landmark_inventory"]
    radii = metadata["radii"]
    fig, ax = plt.subplots(figsize=(11, 2.8))
    ax.set(xlim=(0, 11), ylim=(0, 3.1))
    ax.axis("off")
    boxes = [
        (0.15, 2.1, 1.7, 0.7, "#e8f1fb", "listing snapshot", "curated listing records"),
        (0.15, 0.25, 1.7, 0.7, "#f5ebd7", "geo sources", f"{ov}; {landmarks}"),
        (3.1, 1.15, 2.0, 0.7, "#e5f4e3", "deterministic evidence", f"spatial join; {radii}"),
        (6.25, 1.15, 1.8, 0.7, "#f4e8f6", "grounded prompt", "no price / addresses"),
        (9.15, 1.15, 1.65, 0.7, "#fce8e6", "release CSV", f"{text_column} + table + price"),
    ]
    for x, y, width, height, color, title, subtitle in boxes:
        ax.add_patch(FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.04,rounding_size=0.08", facecolor=color, edgecolor="#274c77", linewidth=1.3))
        ax.text(x + width / 2, y + .46, title, ha="center", va="center", fontsize=10, fontweight="bold")
        ax.text(x + width / 2, y + .20, textwrap.fill(subtitle, width=27), ha="center", va="center", fontsize=7.5, color="#425466")
    for start, end in [((1.85, 2.45), (3.1, 1.55)), ((1.85, .60), (3.1, 1.45)), ((5.1, 1.5), (6.25, 1.5)), ((8.05, 1.5), (9.15, 1.5))]:
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "lw": 1.5, "color": "#274c77"})
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def render_scores_figure(path: Path, summary: dict[tuple[str, str], tuple[float, float]]) -> None:
    values = [mean for mean, _ in summary.values()]
    lower = min(-0.05, min(values) - 0.03)
    upper = max(0.05, max(values) + 0.03)
    fig, ax = plt.subplots(figsize=(10, 4.7))
    for i, learner in enumerate(LEARNERS):
        y = len(LEARNERS) - 1 - i
        for j, condition in enumerate(CONDITIONS):
            mean, sem = summary[(learner, condition)]
            ax.errorbar(mean, y + (j - 1.5) * .14, xerr=sem, fmt="o", color=COLORS[condition], capsize=3, label=CONDITION_LABELS[condition] if i == 0 else None)
    ax.set(xlim=(lower, upper), yticks=range(len(LEARNERS)), yticklabels=list(reversed([LABELS[learner] for learner in LEARNERS])), xlabel="Fold-mean R²")
    ax.set_title("MulTaBench results (error bars: ±1 SEM)", loc="left", fontweight="bold")
    ax.grid(axis="x", color="#d9e1e8")
    ax.legend(ncol=2, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    scores = read_scores(args.scores)
    groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    seen: set[tuple[str, str, int]] = set()
    appendix = []
    for row in scores:
        learner, condition, fold = row["learner"], row["condition"], int(row["fold"])
        if learner not in LEARNERS or condition not in CONDITIONS:
            raise ValueError(f"unexpected learner/condition: {learner}/{condition}")
        key = (learner, condition, fold)
        if key in seen:
            raise ValueError(f"duplicate score record: {key}")
        seen.add(key)
        score = float(row["score"])
        groups[(learner, condition)].append(score)
        appendix.append({"Learner": LABELS[learner], "Condition": CONDITION_LABELS[condition], "Fold": fold, "R2": f"{score:.6f}"})
    required = {(learner, condition) for learner in LEARNERS for condition in CONDITIONS}
    if missing := required.difference(groups):
        raise ValueError(f"missing learner/condition groups: {sorted(missing)}")
    fold_counts = {len(groups[key]) for key in required}
    if fold_counts != {5}:
        raise ValueError(f"each learner/condition must contain exactly five folds, got {sorted(fold_counts)}")
    summary = {key: mean_sem(values) for key, values in groups.items()}
    table_rows = []
    verdict_rows = []
    for learner in LEARNERS:
        structured, text, frozen, tar = (summary[(learner, condition)][0] for condition in CONDITIONS)
        delta_joint = frozen - max(structured, text)
        delta_tar = tar - frozen
        joint_signal = delta_joint > 0
        tar_gain = delta_tar > 0
        both = joint_signal and tar_gain
        table_rows.append({
            "Learner": LABELS[learner], "Structured": f"{structured:.4f}", "Text": f"{text:.4f}",
            "Joint frozen": f"{frozen:.4f}", "Joint TAR": f"{tar:.4f}",
            "Delta joint": f"{delta_joint:+.4f}", "Delta TAR": f"{delta_tar:+.4f}", "Both?": "yes" if both else "no",
        })
        verdict_rows.append({"learner": learner, "label": LABELS[learner], "joint_signal": joint_signal, "tar_gain": tar_gain, "both": both, "delta_joint": delta_joint, "delta_tar": delta_tar})
    pass_count = sum(row["both"] for row in verdict_rows)
    verdict = {"criterion": "strictly positive fold-mean deltas", "joint_signal_passes": sum(row["joint_signal"] for row in verdict_rows), "tar_gain_passes": sum(row["tar_gain"] for row in verdict_rows), "both_passes": pass_count, "required": 3, "dataset_passes": pass_count >= 3, "learners": verdict_rows}

    tables = args.out / "tables"
    figures = args.out / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    table_fields = list(table_rows[0])
    write_csv(tables / "table_2_multabench_results.csv", table_rows, table_fields)
    write_csv(tables / "appendix_a4_per_fold.csv", sorted(appendix, key=lambda row: (row["Learner"], row["Condition"], row["Fold"])), ["Learner", "Condition", "Fold", "R2"])
    latex_break = " \\\\"
    tex_rows = [" & ".join(tex(row[field]) for field in table_fields) + latex_break for row in table_rows]
    (tables / "table_2_multabench_results.tex").write_text(
        "\\begin{tabular}{lrrrrrrc}\n\\toprule\n" + " & ".join(table_fields) + latex_break + "\n\\midrule\n" + "\n".join(tex_rows) + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8"
    )
    appendix_tex = [" & ".join(tex(row[field]) for field in ("Learner", "Condition", "Fold", "R2")) + latex_break for row in sorted(appendix, key=lambda row: (row["Learner"], row["Condition"], row["Fold"]))]
    (tables / "appendix_a4_per_fold.tex").write_text(
        "\\begin{tabular}{llrr}\n\\toprule\nLearner & Condition & Fold & R2 \\\\n\\midrule\n" + "\n".join(appendix_tex) + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8"
    )
    (tables / "curation_verdict.json").write_text(json.dumps(verdict, indent=2) + "\n", encoding="utf-8")
    render_pipeline_figure(figures / "figure_1_curation_pipeline.pdf", config["metadata"], config["text_column"])
    render_scores_figure(figures / "figure_2_multabench_scores.pdf", summary)
    print(f"wrote paper assets to {args.out}")


if __name__ == "__main__":
    main()
