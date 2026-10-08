"""Render the paper's tables, methods text and figures from the `report` outputs.

`assets` is a pure function of the metrics table and verdict that `report` writes: it
returns the text files (CSV and LaTeX tables, methods text) keyed by their path under
the output directory. Nothing is read from a run-history export and no number is
copied by hand; every figure in the paper is a reshaping of those two tables.
"""

from __future__ import annotations

import argparse
import json
import os
import textwrap
from pathlib import Path

import pandas as pd
from multabench.result_keys import STATUS, TEST_SCORE, RunStatus

from src import arms, config, report

# Non-interactive, publication-ready figures. Keep Matplotlib's cache outside a user's
# home directory, which is often read-only on compute hosts.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/geoenrich-matplotlib")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

LATLON = "latlon"
CONDITION_LABELS = {
    "structured": "structured",
    "text_only": "text only",
    "joint_frozen": "joint frozen",
    "joint_tar": "joint TAR",
    LATLON: "lat/lon",
}
COLORS = {
    "structured": "#4c78a8",
    "text_only": "#f58518",
    "joint_frozen": "#54a24b",
    "joint_tar": "#e45756",
    LATLON: "#7f7f7f",
}
_NUMBER_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight"}
_RESULT_FIELDS = [
    "arm", "learner", "structured", "text_only", "joint_frozen", "joint_tar",
    "joint_signal_delta", "tar_gain_delta", "joint_signal", "tar_gain", "passes", "eligible",
]  # fmt: skip


def tex(value: object) -> str:
    return (
        str(value).replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("&", r"\&")
        .replace("%", r"\%").replace("#", r"\#")
    )  # fmt: skip


def latex_table(frame: pd.DataFrame, align: str) -> str:
    rows = [" & ".join(tex(v) for v in row) + r" \\" for row in frame.itertuples(index=False)]
    header = " & ".join(tex(c) for c in frame.columns) + r" \\"
    body = "\n".join(rows)
    return f"\\begin{{tabular}}{{{align}}}\n\\toprule\n{header}\n\\midrule\n{body}\n\\bottomrule\n\\end{{tabular}}\n"


def _ok(metrics: pd.DataFrame) -> pd.DataFrame:
    return metrics[metrics[STATUS] == RunStatus.OK]


# --- Tables ------------------------------------------------------------------------
def results_table(metrics: pd.DataFrame, verdict: pd.DataFrame) -> pd.DataFrame:
    """Table 2: per arm and committee learner, the four condition scores, deltas and flags.

    The condition means, deltas and flags are the verdict's own numbers. The `latlon`
    reference is the mean score of that arm's runs, rounded like the verdict's means.
    """
    table = verdict.rename(columns={f"{c}_mean": c for c in report.CONDITIONS})[_RESULT_FIELDS]
    latlon = _ok(metrics[metrics["arm"] == LATLON]).groupby("learner")[TEST_SCORE].mean().round(3)
    if not latlon.empty:
        table.insert(2, LATLON, table["learner"].map(latlon))
    table["learner"] = pd.Categorical(table["learner"], report.COMMITTEE)
    table = table.sort_values(["arm", "learner"], ignore_index=True)
    table["learner"] = table["learner"].astype(str)
    return table


def per_split_table(metrics: pd.DataFrame) -> pd.DataFrame:
    """Appendix A4: every successful run's score and error, one row per arm, condition, learner, split."""
    ok = _ok(metrics)
    table = ok[["arm", "condition", "learner", "split", TEST_SCORE, "test_error"]]
    table = table.rename(columns={TEST_SCORE: "score", "test_error": "error"})
    return table.sort_values(["arm", "condition", "learner", "split"], ignore_index=True)


# --- Methods text ------------------------------------------------------------------
def methods_text(metrics: pd.DataFrame, verdict: pd.DataFrame) -> str:
    """The evaluation paragraph of the methods section, stated in the glossary's vocabulary."""
    ok = _ok(metrics)
    n = ok["split"].nunique()
    splits = _NUMBER_WORDS.get(n, str(n))
    text_arms = sorted(verdict["arm"].unique())
    arm_names = ", ".join(f"`{a}`" for a in [*text_arms, *([LATLON] if (metrics["arm"] == LATLON).any() else [])])
    committee = ", ".join(report.COMMITTEE)
    margin = report.MARGIN_THOUSANDTHS / 1000
    eligible = sorted(verdict.loc[verdict["eligible"], "arm"].unique())
    verdict_line = (
        "Eligible arms: " + ", ".join(f"`{a}`" for a in eligible) + "."
        if eligible
        else "No arm is eligible."
    )
    paragraphs = [
        "## Evaluation protocol",
        "Every result comes from the official MulTaBench benchmark, run unmodified. "
        f"Each arm ({arm_names}) is evaluated under four conditions (`structured`, `text_only`, "
        f"`joint_frozen`, `joint_tar`) by the five committee learners ({committee}), on "
        f"the benchmark's {splits} splits of TabArena's 67/33 scheme. "
        "TAR is the E5-small text encoder fine-tuned on the target with LoRA.",
        "A run's score is its `test_score`, the negated RMSE of the nightly-price "
        "prediction (higher is better); its error is `test_error`, the RMSE. Scores are "
        "not directly comparable to the published MulTaBench leaderboard, whose "
        "regression metric differs.",
        "A learner shows joint signal when its mean `joint_frozen` score exceeds both its "
        f"mean `structured` and its mean `text_only` score by more than {margin}, and shows "
        f"TAR gain when its mean `joint_tar` score exceeds its mean `joint_frozen` score by "
        f"more than {margin}; means are rounded to three decimals before differencing. An arm "
        f"is eligible when at least {report.QUORUM} of the {len(report.COMMITTEE)} committee "
        "learners show both. This is this "
        f"project's computation of the published criterion. {verdict_line}",
        "The `landmark_only` arm keeps only the landmark names found in each summary; "
        "the `landmark_redacted` arm replaces each matched landmark name with the generic "
        f'phrase "{arms.REDACTION}". The phrase is a deliberate change from earlier '
        "redaction, which left sentences ungrammatical; it removes which landmark is "
        "nearby while preserving that one is.",
    ]
    return "\n\n".join(paragraphs) + "\n"


# --- Figures -----------------------------------------------------------------------
def render_pipeline_figure(path: Path, metadata: dict[str, str], text_column: str) -> None:
    ov, landmarks, radii = metadata["overture_release"], metadata["landmark_inventory"], metadata["radii"]
    fig, ax = plt.subplots(figsize=(11, 2.8))
    ax.set(xlim=(0, 11), ylim=(0, 3.1))
    ax.axis("off")
    boxes = [
        (0.15, 2.1, 1.7, 0.7, "#e8f1fb", "listing snapshot", "curated listing records"),
        (0.15, 0.25, 1.7, 0.7, "#f5ebd7", "geo sources", f"{ov}; {landmarks}"),
        (3.1, 1.15, 2.0, 0.7, "#e5f4e3", "deterministic evidence", f"spatial join; {radii}"),
        (6.25, 1.15, 1.8, 0.7, "#f4e8f6", "grounded prompt", "no price / addresses"),
        (9.15, 1.15, 1.65, 0.7, "#fce8e6", "release CSV", f"{text_column} + table + price"),
    ]  # fmt: skip
    for x, y, width, height, color, title, subtitle in boxes:
        ax.add_patch(
            FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.04,rounding_size=0.08", facecolor=color, edgecolor="#274c77", linewidth=1.3)
        )  # fmt: skip
        ax.text(x + width / 2, y + 0.46, title, ha="center", va="center", fontsize=10, fontweight="bold")
        ax.text(x + width / 2, y + 0.20, textwrap.fill(subtitle, width=27), ha="center", va="center", fontsize=7.5, color="#425466")
    for start, end in [((1.85, 2.45), (3.1, 1.55)), ((1.85, 0.60), (3.1, 1.45)), ((5.1, 1.5), (6.25, 1.5)), ((8.05, 1.5), (9.15, 1.5))]:
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "lw": 1.5, "color": "#274c77"})
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def render_scores_figure(path: Path, metrics: pd.DataFrame, verdict: pd.DataFrame) -> None:
    """One panel per text arm: split-mean score of each condition per learner, ±1 SEM across splits."""
    ok = _ok(metrics)
    text_arms = sorted(verdict["arm"].unique())
    fig, axes = plt.subplots(1, len(text_arms), figsize=(5 * len(text_arms) + 1, 4.7), sharex=True, squeeze=False)
    conditions = [*report.CONDITIONS, *([LATLON] if (ok["arm"] == LATLON).any() else [])]
    for ax, arm in zip(axes[0], text_arms):
        for i, learner in enumerate(report.COMMITTEE):
            y = len(report.COMMITTEE) - 1 - i
            for j, condition in enumerate(conditions):
                if condition == LATLON:
                    scores = ok[(ok["arm"] == LATLON) & (ok["learner"] == learner)][TEST_SCORE]
                else:
                    shared = report.SHARED if condition == "structured" else arm
                    scores = ok[(ok["arm"] == shared) & (ok["condition"] == condition) & (ok["learner"] == learner)][TEST_SCORE]
                sem = scores.sem() if len(scores) > 1 else 0.0
                ax.errorbar(
                    scores.mean(), y + (j - (len(conditions) - 1) / 2) * 0.13, xerr=sem, fmt="o", color=COLORS[condition],
                    capsize=3, label=CONDITION_LABELS[condition] if i == 0 else None,
                )  # fmt: skip
        ax.set(yticks=range(len(report.COMMITTEE)), yticklabels=list(reversed(report.COMMITTEE)), xlabel="score (negated RMSE, higher is better)")
        ax.set_title(arm, loc="left", fontweight="bold")
        ax.grid(axis="x", color="#d9e1e8")
    axes[0][0].legend(ncol=2, frameon=False, loc="lower left")
    fig.suptitle("Split-mean score by condition (error bars: ±1 SEM across splits)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# --- Entry point -------------------------------------------------------------------
def assets(metrics: pd.DataFrame, verdict: pd.DataFrame) -> dict[str, str]:
    """Every text asset of the paper, keyed by its path under the output directory."""
    results = results_table(metrics, verdict)
    per_split = per_split_table(metrics)
    return {
        "tables/table_2_results.csv": results.to_csv(index=False),
        "tables/table_2_results.tex": latex_table(results.round(3), "ll" + "r" * (len(results.columns) - 4) + "ccc"),
        "tables/appendix_a4_per_split.csv": per_split.to_csv(index=False),
        "tables/appendix_a4_per_split.tex": latex_table(per_split, "lllrrr"),
        "methods.md": methods_text(metrics, verdict),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", choices=["screen", "final"], default="final")
    parser.add_argument("--metrics_csv", help="default: results/metrics_<grid>.csv")
    parser.add_argument("--verdict_csv", help="default: results/verdict_<grid>.csv")
    parser.add_argument("--out", default=os.path.join(config.RESULTS_DIR, "paper"))
    parser.add_argument("--config", type=Path, help="analysis config; adds the pipeline figure")
    args = parser.parse_args(argv)

    metrics_csv = args.metrics_csv or os.path.join(config.RESULTS_DIR, f"metrics_{args.grid}.csv")
    verdict_csv = args.verdict_csv or os.path.join(config.RESULTS_DIR, f"verdict_{args.grid}.csv")
    metrics = pd.read_csv(metrics_csv, dtype={"prompt": str})
    verdict = pd.read_csv(verdict_csv, dtype={"prompt": str})

    out = Path(args.out)
    for name, content in assets(metrics, verdict).items():
        path = out / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    (out / "figures").mkdir(parents=True, exist_ok=True)
    render_scores_figure(out / "figures" / "figure_2_scores.pdf", metrics, verdict)
    if args.config:

        cfg = json.loads(args.config.read_text(encoding="utf-8"))
        render_pipeline_figure(out / "figures" / "figure_1_curation_pipeline.pdf", cfg["metadata"], cfg["text_column"])
    print(f"wrote paper assets to {out}")


if __name__ == "__main__":
    main()
