"""Collect benchmark results into one metrics CSV and judge curation eligibility.

Two pure steps, each a function of a table, with `main` a thin shell around them:

- `metrics_table`: the official collector's rows (the benchmark's own `test_score`,
  `test_error`, timings and commit, unaltered) plus the labels that identify each row:
  arm, prompt, condition, learner and split. Nothing is computed from the metrics here
  and no metric the benchmark does not produce (no R²) is added.
- `judge`: this project's computation of MulTaBench's published curation criterion
  (see docs/adr/0001). The official repository no longer implements it.
"""

from __future__ import annotations

import argparse
import os
import re

import pandas as pd
from multabench.baselines.catboost import CatBoost
from multabench.baselines.lgbm import LightGBM
from multabench.baselines.tabm import TabM
from multabench.baselines.tabpfnv2 import TabPFNv2, TabPFNv2p5
from multabench.benchmark.collect import collect_results
from multabench.result_keys import STATUS, TEST_SCORE, RunStatus

from src import bench, config, stage

_COMMITTEE_MODELS = (LightGBM, CatBoost, TabM, TabPFNv2, TabPFNv2p5)
COMMITTEE = bench.COMMITTEE
assert COMMITTEE == tuple(m.SHORT_NAME for m in _COMMITTEE_MODELS)  # guards drift from the benchmark
SHORT_NAME = {m.MODEL_NAME: m.SHORT_NAME for m in _COMMITTEE_MODELS}

MARGIN_THOUSANDTHS = 1  # state means are compared as integer thousandths; deltas must exceed this
QUORUM = 3
CONDITIONS = ("structured", "text_only", "joint_frozen", "joint_tar")
SHARED = "shared"  # the arm label of the one structured dataset every text arm is judged against
LATLON = "latlon"

_PREFIX = "REG_TEXT_GEOENRICH_"
_TEXT_DATASET = re.compile(r"^(?P<arm>.+)_(?P<prompt>[^_]+)_(?P<kind>TEXT_ONLY|JOINT)$")
ARM_PROMPT = ["arm", "prompt"]
LABELS = [*ARM_PROMPT, "condition", "learner", "split"]


class IncompleteGrid(ValueError):
    """The results do not cover every (learner, condition, split) the verdict needs."""


# --- Metrics table -----------------------------------------------------------------
def _label(dataset: str, encoder: str | None) -> tuple[str, str | None, str]:
    """(arm, prompt, condition) of one run from its dataset name and text encoder."""
    key = dataset.removeprefix(_PREFIX)
    if key in ("STRUCTURED", "LATLON"):
        return (SHARED if key == "STRUCTURED" else LATLON), None, "structured"
    m = _TEXT_DATASET.match(key)
    if m is None:
        raise ValueError(f"unrecognised dataset name {dataset!r}")
    arm, prompt = m["arm"].lower(), m["prompt"]
    if m["kind"] == "TEXT_ONLY":
        return arm, prompt, "text_only"
    return arm, prompt, "joint_tar" if encoder == str(bench.TAR) else "joint_frozen"


def metrics_table(output_dir: str) -> pd.DataFrame:
    """One row per run result in `output_dir`, failed runs included, labelled by what was run."""
    df = collect_results(output_dir)
    labels = [_label(d, e) for d, e in zip(df["dataset"], df["text_encoder"])]
    df.insert(0, "arm", [lab[0] for lab in labels])
    df.insert(1, "prompt", [lab[1] for lab in labels])
    df.insert(2, "condition", [lab[2] for lab in labels])
    df.insert(3, "learner", df["model"].map(SHORT_NAME).fillna(df["model"]))
    df.insert(4, "split", df["fold"])
    return df.sort_values(LABELS, na_position="first", ignore_index=True)


# --- Verdict -----------------------------------------------------------------------
def _state_means(cell: pd.DataFrame, splits: tuple[int, ...], where: str) -> dict[str, int]:
    """Mean score of each (learner, condition) as integer thousandths, refusing a partial grid."""
    means = {}
    for (learner, condition), runs in cell.groupby(["learner", "condition"]):
        ok = runs[(runs[STATUS] == RunStatus.OK) & runs["split"].isin(splits)]
        if sorted(ok["split"]) != sorted(splits):
            raise IncompleteGrid(
                f"{where}: {learner}/{condition} has splits {sorted(ok['split'])}, needs {sorted(splits)}"
            )
        means[(learner, condition)] = round(round(ok[TEST_SCORE].mean(), 3) * 1000)
    for learner in COMMITTEE:
        for condition in CONDITIONS:
            if (learner, condition) not in means:
                raise IncompleteGrid(f"{where}: no {learner}/{condition} results")
    return means


def judge(table: pd.DataFrame, splits: tuple[int, ...]) -> pd.DataFrame:
    """This project's computation of MulTaBench's curation criterion over the text arms of a metrics table.

    Per (arm, prompt) and committee learner, against the shared structured baseline, or
    `latlon` for `latlon_enriched`, which also takes its text_only from the `enriched` arm: state means are rounded to three decimals
    before differencing; joint signal is `joint_frozen - max(structured, text_only)`
    and TAR gain is `joint_tar - joint_frozen`, each required to exceed 0.001; an arm
    is eligible when at least 3 of the 5 learners show both. Raises `IncompleteGrid`
    rather than averaging over fewer splits than `splits`.
    """
    rows = []
    text = table[~table["arm"].isin([SHARED, LATLON])]
    for (arm, prompt), cell in text.groupby(ARM_PROMPT):
        if arm == stage.LATLON_ARM:
            structured = table[table["arm"] == LATLON]
            enriched = table[(table["arm"] == stage.ARM) & (table["prompt"] == prompt)]
            cell = pd.concat([cell, enriched[enriched["condition"] == "text_only"]])
        else:
            structured = table[table["arm"] == SHARED]
        means = _state_means(pd.concat([structured, cell]), splits, f"{arm}/{prompt}")
        for learner in COMMITTEE:
            m = {c: means[(learner, c)] for c in CONDITIONS}
            signal = m["joint_frozen"] - max(m["structured"], m["text_only"])
            gain = m["joint_tar"] - m["joint_frozen"]
            rows.append(
                {
                    "arm": arm,
                    "prompt": prompt,
                    "learner": learner,
                    **{f"{c}_mean": m[c] / 1000 for c in CONDITIONS},
                    "joint_signal_delta": signal / 1000,
                    "tar_gain_delta": gain / 1000,
                    "joint_signal": signal > MARGIN_THOUSANDTHS,
                    "tar_gain": gain > MARGIN_THOUSANDTHS,
                }
            )
    verdict = pd.DataFrame(rows)
    if verdict.empty:
        raise IncompleteGrid("no text-arm results to judge")
    verdict["passes"] = verdict["joint_signal"] & verdict["tar_gain"]
    verdict["n_pass"] = verdict.groupby(ARM_PROMPT)["passes"].transform("sum")
    verdict["eligible"] = verdict["n_pass"] >= QUORUM
    return verdict


# --- Entry point -------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", choices=["screen", "final"], required=True)
    parser.add_argument("--output_dir", default=os.path.join(config.RESULTS_DIR, "runs"))
    parser.add_argument("--metrics_csv", help="default: results/metrics_<grid>.csv")
    parser.add_argument("--verdict_csv", help="default: results/verdict_<grid>.csv")
    parser.add_argument("--overwrite", action="store_true", help="rebuild outputs that already exist")
    args = parser.parse_args()

    splits = bench.SCREEN_SPLITS if args.grid == "screen" else bench.FINAL_SPLITS
    metrics_csv = args.metrics_csv or os.path.join(config.RESULTS_DIR, f"metrics_{args.grid}.csv")
    verdict_csv = args.verdict_csv or os.path.join(config.RESULTS_DIR, f"verdict_{args.grid}.csv")
    os.makedirs(os.path.dirname(metrics_csv), exist_ok=True)

    table = None
    if os.path.exists(metrics_csv) and not args.overwrite:
        print(f"skip: {metrics_csv} exists (--overwrite to rebuild)")
    else:
        table = metrics_table(args.output_dir)
        table.to_csv(metrics_csv, index=False)
        n_failed = int((table[STATUS] == RunStatus.ERROR).sum())
        print(f"Wrote {len(table)} runs ({n_failed} failed) to {metrics_csv}")

    if os.path.exists(verdict_csv) and not args.overwrite:
        print(f"skip: {verdict_csv} exists (--overwrite to rebuild)")
        return
    table = pd.read_csv(metrics_csv, dtype={"prompt": str}) if table is None else table
    try:
        verdict = judge(table, splits)
    except IncompleteGrid as e:
        raise SystemExit(f"cannot judge: {e}")
    verdict.to_csv(verdict_csv, index=False)
    columns = ["arm", "prompt", "learner", "joint_signal_delta", "tar_gain_delta", "joint_signal", "tar_gain"]
    print(verdict[columns].to_string(index=False))
    print(verdict.groupby(ARM_PROMPT)[["n_pass", "eligible"]].first().to_string())
    print(f"Wrote {verdict_csv}")


if __name__ == "__main__":
    main()
