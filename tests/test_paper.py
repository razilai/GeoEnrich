from __future__ import annotations

import io

import pandas as pd
import pytest
from test_report import FROZEN, uniform, with_learner, write_grid, write_run

from src import arms, bench, paper, report

SPLITS = bench.FINAL_SPLITS


@pytest.fixture
def results(tmp_path):
    """The metrics table and verdict `report` derives from a final-grid fixture."""
    runs = tmp_path / "runs"
    runs.mkdir()
    write_grid(runs, uniform(), arm="enriched", splits=SPLITS)
    weak = with_learner(uniform(frozen=-101.0), "light", joint_frozen=-90.0)
    write_grid(runs, weak, arm="landmark_only", splits=SPLITS)
    for learner in report.COMMITTEE:
        for split in SPLITS:
            write_run(runs, "latlon", FROZEN, learner, split, -95.0 - split)
    metrics = report.metrics_table(str(runs))
    return metrics, report.judge(metrics, SPLITS)


def build(results):
    metrics, verdict = results
    return paper.assets(metrics, verdict)


def csv_of(files, name) -> pd.DataFrame:
    return pd.read_csv(io.StringIO(files[name]))


def test_results_table_has_a_row_per_arm_and_learner_with_report_numbers(results) -> None:
    table = csv_of(build(results), "tables/table_2_results.csv")
    assert len(table) == 2 * 5
    row = table[(table["arm"] == "enriched") & (table["learner"] == "light")].iloc[0]
    assert row["structured"] == pytest.approx(-100.0)
    assert row["joint_frozen"] == pytest.approx(-90.0)
    assert row["joint_signal_delta"] == pytest.approx(10.0)
    assert row["tar_gain_delta"] == pytest.approx(10.0)
    assert bool(row["passes"])
    assert table[table["arm"] == "enriched"]["eligible"].all()
    assert not table[table["arm"] == "landmark_only"]["eligible"].any()


def test_latlon_reference_is_a_column_of_the_results_table(results) -> None:
    table = csv_of(build(results), "tables/table_2_results.csv")
    assert table["latlon"].tolist() == [pytest.approx(-97.5)] * 10


def test_appendix_lists_every_split_score_and_error(results) -> None:
    metrics, _ = results
    appendix = csv_of(build(results), "tables/appendix_a4_per_split.csv")
    assert len(appendix) == len(metrics)
    assert {"arm", "condition", "learner", "split", "score", "error"} <= set(appendix)
    assert set(appendix["split"]) == set(SPLITS)


def test_failed_runs_are_not_scored_in_the_appendix(results) -> None:
    metrics, verdict = results
    metrics = metrics.copy()
    metrics.loc[0, "status"] = "error"
    appendix = csv_of(paper.assets(metrics, verdict), "tables/appendix_a4_per_split.csv")
    assert len(appendix) == len(metrics) - 1


def test_latex_tables_are_emitted_and_escape_underscores(results) -> None:
    files = build(results)
    assert files["tables/table_2_results.tex"].startswith("\\begin{tabular}")
    assert "landmark\\_only" in files["tables/table_2_results.tex"]


def test_methods_state_six_splits_score_and_leaderboard_caveat(results) -> None:
    methods = build(results)["methods.md"]
    assert "six splits" in methods
    assert "negated RMSE" in methods and "test_error" in methods
    assert "not directly comparable to the published" in methods
    assert "leaderboard" in methods


def test_methods_state_the_redaction_token_as_deliberate(results) -> None:
    methods = build(results)["methods.md"]
    assert arms.REDACTION in methods
    assert "deliberate" in methods


def test_methods_drop_retired_protocol(results) -> None:
    methods = build(results)["methods.md"].casefold()
    for retired in ("r²", "r2", "five folds", "five-fold", "multimodal_state", "fold"):
        assert retired not in methods


def test_methods_use_glossary_vocabulary(results) -> None:
    methods = build(results)["methods.md"]
    for term in ("arm", "condition", "split", "score", "TAR", "joint signal", "TAR gain", "eligible"):
        assert term in methods
    assert "3 of the 5" in methods


def test_split_count_follows_the_results(results) -> None:
    metrics, verdict = results
    screen = metrics[metrics["split"] < 3]
    assert "three splits" in paper.assets(screen, verdict)["methods.md"]


def test_cli_reads_only_report_outputs(results, tmp_path) -> None:
    metrics, verdict = results
    metrics.to_csv(tmp_path / "metrics.csv", index=False)
    verdict.to_csv(tmp_path / "verdict.csv", index=False)
    out = tmp_path / "paper_assets"
    paper.main(
        ["--metrics_csv", str(tmp_path / "metrics.csv"), "--verdict_csv", str(tmp_path / "verdict.csv"), "--out", str(out)]
    )
    assert (out / "tables" / "table_2_results.csv").exists()
    assert (out / "methods.md").exists()
    assert (out / "figures" / "figure_2_scores.pdf").exists()
