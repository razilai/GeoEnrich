from __future__ import annotations

import json

import pytest

from src import bench, report, stage

DISPLAY = {
    "light": "LightGBM",
    "cat": "CatBoost",
    "tabm": "TabM",
    "tabpfnv2": "TabPFN-v2",
    "tabpfnv2p5": "TabPFN-v2p5",
}
SPLITS = (0, 1, 2)
FROZEN, TAR = str(bench.FROZEN), str(bench.TAR)

# (dataset key, text encoder) per condition of a text arm; structured is shared.
CONDITIONS = {
    "structured": ("structured", FROZEN),
    "text_only": ("{arm}_08_text_only", FROZEN),
    "joint_frozen": ("{arm}_08_joint", FROZEN),
    "joint_tar": ("{arm}_08_joint", TAR),
}


def write_run(path, key, encoder, learner, split, score, status="ok"):
    row = {
        "model": DISPLAY[learner],
        "dataset": stage.enum_name(key),
        "text_encoder": encoder,
        "image_encoder": None,
        "size": "10K",
        "fold": split,
        "status": status,
        "git": "d88821d",
        "timestamp": "2026-10-08 10:00:00",
    }
    if status == "ok":
        row.update(
            metric="rmse", test_score=score, test_error=-score, train_time_per_1k_s=1.5,
            inference_time_per_1k_s=0.5, n_train=10, n_test=5,
        )
    else:
        row["error"] = "ValueError: boom"
    name = f"{learner}_{key}_{encoder}_{split}.json"
    (path / name).write_text(json.dumps(row))


def write_grid(path, scores, arm="enriched", learners=report.COMMITTEE, splits=SPLITS, skip=()):
    """scores: condition -> learner -> mean score (constant over splits)."""
    for condition, (key, encoder) in CONDITIONS.items():
        for learner in learners:
            for split in splits:
                if (condition, learner, split) in skip:
                    continue
                score = scores[condition][learner]
                write_run(path, key.format(arm=arm), encoder, learner, split, score)


def uniform(structured=-100.0, text_only=-100.0, frozen=-90.0, tar=-80.0):
    return {
        "structured": dict.fromkeys(report.COMMITTEE, structured),
        "text_only": dict.fromkeys(report.COMMITTEE, text_only),
        "joint_frozen": dict.fromkeys(report.COMMITTEE, frozen),
        "joint_tar": dict.fromkeys(report.COMMITTEE, tar),
    }


def with_learner(scores, learner, **conditions):
    for condition, value in conditions.items():
        scores[condition] = {**scores[condition], learner: value}
    return scores


# --- metrics table -----------------------------------------------------------------
def test_one_row_per_arm_condition_learner_split(tmp_path) -> None:
    write_grid(tmp_path, uniform())
    write_grid(tmp_path, uniform(), arm="landmark_only")
    table = report.metrics_table(str(tmp_path))
    # structured is one shared dataset; each text arm adds three conditions.
    assert len(table) == (1 + 3 * 2) * 5 * 3
    key = ["arm", "condition", "learner", "split"]
    assert not table.duplicated(key + ["prompt"]).any()
    assert set(table["learner"]) == set(report.COMMITTEE)
    assert set(table["condition"]) == set(CONDITIONS)
    assert set(table.loc[table["condition"] == "structured", "arm"]) == {"shared"}


def test_latlon_arm_has_structured_condition_only(tmp_path) -> None:
    for learner in report.COMMITTEE:
        write_run(tmp_path, "latlon", FROZEN, learner, 0, -70.0)
    table = report.metrics_table(str(tmp_path))
    assert set(table["arm"]) == {"latlon"}
    assert set(table["condition"]) == {"structured"}


def test_metric_error_timings_and_commit_carried_unaltered(tmp_path) -> None:
    write_run(tmp_path, "enriched_08_joint", TAR, "light", 1, -81.123456789)
    row = report.metrics_table(str(tmp_path)).iloc[0]
    assert row["test_score"] == -81.123456789
    assert row["test_error"] == 81.123456789
    assert row["train_time_per_1k_s"] == 1.5
    assert row["inference_time_per_1k_s"] == 0.5
    assert row["git"] == "d88821d"
    assert (row["arm"], row["prompt"], row["condition"]) == ("enriched", "08", "joint_tar")
    assert "r2" not in "".join(report.metrics_table(str(tmp_path)).columns).lower()


def test_failed_run_surfaces_as_failure_row(tmp_path) -> None:
    write_run(tmp_path, "enriched_08_joint", FROZEN, "cat", 0, None, status="error")
    row = report.metrics_table(str(tmp_path)).iloc[0]
    assert row["status"] == "error"
    assert row["error"] == "ValueError: boom"
    assert row["test_score"] != row["test_score"]  # NaN


# --- verdict -----------------------------------------------------------------------
def judged(path, splits=SPLITS):
    return report.judge(report.metrics_table(str(path)), splits)


def write_condition(path, key, encoder, score, splits=SPLITS):
    for learner in report.COMMITTEE:
        for split in splits:
            write_run(path, key, encoder, learner, split, score)


def test_latlon_enriched_is_judged_against_latlon_and_the_enriched_text_only(tmp_path) -> None:
    write_grid(tmp_path, uniform(structured=-100.0, text_only=-120.0))
    write_condition(tmp_path, "latlon", FROZEN, -95.0)
    write_condition(tmp_path, "latlon_enriched_08_joint", FROZEN, -90.0)
    write_condition(tmp_path, "latlon_enriched_08_joint", TAR, -85.0)
    v = judged(tmp_path).set_index("arm").loc["latlon_enriched"]
    assert v["structured_mean"].tolist() == [-95.0] * 5
    assert v["text_only_mean"].tolist() == [-120.0] * 5
    assert v["joint_signal_delta"].tolist() == pytest.approx([5.0] * 5)
    assert v["eligible"].all()


def test_latlon_enriched_without_latlon_is_incomplete(tmp_path) -> None:
    write_grid(tmp_path, uniform())
    write_condition(tmp_path, "latlon_enriched_08_joint", FROZEN, -90.0)
    write_condition(tmp_path, "latlon_enriched_08_joint", TAR, -85.0)
    with pytest.raises(report.IncompleteGrid, match="latlon_enriched"):
        judged(tmp_path)


def test_verdict_per_arm_with_deltas(tmp_path) -> None:
    write_grid(tmp_path, uniform())
    write_grid(tmp_path, uniform(frozen=-100.0, tar=-100.0), arm="landmark_only")
    v = judged(tmp_path)
    assert set(v["arm"]) == {"enriched", "landmark_only"}
    enriched = v[v["arm"] == "enriched"].iloc[0]
    assert enriched["joint_signal_delta"] == pytest.approx(10.0)
    assert enriched["tar_gain_delta"] == pytest.approx(10.0)
    assert bool(enriched["eligible"])
    assert not v[v["arm"] == "landmark_only"]["eligible"].any()
    assert len(v) == 2 * 5


def test_joint_signal_is_against_the_better_of_structured_and_text_only(tmp_path) -> None:
    write_grid(tmp_path, uniform(structured=-100.0, text_only=-85.0, frozen=-90.0))
    v = judged(tmp_path)
    assert v["joint_signal_delta"].tolist() == [-5.0] * 5
    assert not v["joint_signal"].any()


def test_state_means_are_rounded_before_differencing(tmp_path) -> None:
    # Raw delta 0.5014 - 0.4996 = 0.0018 would pass; rounded means 0.501 - 0.500 = 0.001 do not.
    s = uniform(structured=-1.0, text_only=-1.0, frozen=0.5014, tar=0.9)
    s = with_learner(s, "light", structured=0.4996, text_only=0.4996)
    write_grid(tmp_path, s)
    row = judged(tmp_path).set_index("learner").loc["light"]
    assert row["joint_signal_delta"] == pytest.approx(0.001)
    assert not row["joint_signal"]


@pytest.mark.parametrize(
    "frozen,expected",
    [(0.5021, True), (0.50151, True), (0.5014, False), (0.5009, False), (0.5, False), (0.498, False)],
)
def test_margin_boundary_is_strictly_greater_than_one_thousandth(tmp_path, frozen, expected) -> None:
    # Means are rounded to thousandths first, so a delta of exactly 0.001 fails and 0.002 passes.
    s = uniform(structured=-1.0, text_only=-1.0, frozen=frozen, tar=frozen + 0.01)
    s = with_learner(s, "light", structured=0.5, text_only=0.5)
    s = with_learner(s, "light", joint_tar=frozen + {True: 0.002, False: 0.001}[expected])
    write_grid(tmp_path, s)
    row = judged(tmp_path).set_index("learner").loc["light"]
    assert bool(row["joint_signal"]) is expected
    assert bool(row["tar_gain"]) is expected


@pytest.mark.parametrize("n_passing,eligible", [(2, False), (3, True), (5, True), (0, False)])
def test_three_of_five_quorum(tmp_path, n_passing, eligible) -> None:
    s = uniform()
    for learner in report.COMMITTEE[n_passing:]:
        s = with_learner(s, learner, joint_frozen=-100.0, joint_tar=-100.0)
    write_grid(tmp_path, s)
    v = judged(tmp_path)
    assert (v["n_pass"] == n_passing).all()
    assert v["eligible"].all() == eligible


def test_a_learner_must_pass_both_criteria(tmp_path) -> None:
    s = uniform()
    for learner in report.COMMITTEE[:3]:  # joint signal everywhere, but no TAR gain for three
        s = with_learner(s, learner, joint_tar=-90.0)
    write_grid(tmp_path, s)
    v = judged(tmp_path)
    assert (v["n_pass"] == 2).all()
    assert not v["eligible"].any()


def test_missing_split_raises_instead_of_averaging(tmp_path) -> None:
    write_grid(tmp_path, uniform(), skip={("joint_tar", "tabm", 2)})
    with pytest.raises(report.IncompleteGrid, match="tabm"):
        judged(tmp_path)


def test_failed_run_makes_grid_incomplete(tmp_path) -> None:
    write_grid(tmp_path, uniform(), skip={("joint_frozen", "cat", 1)})
    write_run(tmp_path, "enriched_08_joint", FROZEN, "cat", 1, None, status="error")
    with pytest.raises(report.IncompleteGrid, match="cat"):
        judged(tmp_path)


def test_missing_committee_learner_raises(tmp_path) -> None:
    write_grid(tmp_path, uniform(), learners=report.COMMITTEE[:4])
    with pytest.raises(report.IncompleteGrid, match="tabpfnv2p5"):
        judged(tmp_path)


def test_missing_structured_raises(tmp_path) -> None:
    write_grid(tmp_path, uniform())
    for path in tmp_path.glob("*_structured_*"):
        path.unlink()
    with pytest.raises(report.IncompleteGrid, match="structured"):
        judged(tmp_path)


def test_screen_judges_each_prompt_against_the_shared_structured(tmp_path) -> None:
    write_grid(tmp_path, uniform())
    for condition in ("text_only", "joint_frozen", "joint_tar"):
        key, encoder = CONDITIONS[condition]
        for learner in report.COMMITTEE:
            for split in SPLITS:
                write_run(tmp_path, key.format(arm="enriched").replace("_08_", "_16_"), encoder,
                          learner, split, -100.0)
    v = judged(tmp_path)
    assert set(zip(v["arm"], v["prompt"])) == {("enriched", "08"), ("enriched", "16")}
    by_prompt = v.groupby("prompt")["eligible"].all()
    assert by_prompt["08"] and not by_prompt["16"]


def test_screen_ignores_splits_outside_the_requested_set(tmp_path) -> None:
    write_grid(tmp_path, uniform(), splits=(0, 1, 2, 3, 4, 5))
    assert judged(tmp_path, splits=(0, 1, 2))["eligible"].all()
    with pytest.raises(report.IncompleteGrid):
        judged(tmp_path, splits=(0, 1, 2, 3, 4, 5, 6))


def test_screen_grid_is_judged_over_split_zero_and_refuses_a_missing_run(tmp_path) -> None:
    assert bench.SCREEN_SPLITS == (0,)
    write_grid(tmp_path, uniform(), splits=bench.SCREEN_SPLITS)
    assert judged(tmp_path, splits=bench.SCREEN_SPLITS)["eligible"].all()
    next(tmp_path.glob("tabm_enriched_08_joint_*")).unlink()
    with pytest.raises(report.IncompleteGrid, match="tabm"):
        judged(tmp_path, splits=bench.SCREEN_SPLITS)
