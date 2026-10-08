from __future__ import annotations

import os

import pandas as pd
import pytest

from src import arms, stage
from test_stage import listings

NAMES = ["Central Park", "Central Park Zoo", "Empire State Building"]
TEXTS = [
    "Steps from Central Park Zoo and a deli.",
    "Near the Empire State Building and Central Park; try Joe's Pizza.",
    "A quiet block with a bakery.",
    None,
    "central park is close.",
    "Next to Madison Square Garden.",
]


@pytest.fixture
def corpus() -> pd.DataFrame:
    return pd.DataFrame({"index": range(6), "surroundings_summary": TEXTS})


@pytest.fixture
def built(corpus) -> dict[str, pd.DataFrame]:
    return arms.build_arms(corpus, NAMES)


def test_arms_keep_every_row(corpus, built) -> None:
    for frame in built.values():
        assert len(frame) == len(corpus)
        assert frame["index"].tolist() == corpus["index"].tolist()
        assert frame["surroundings_summary"].isna().tolist() == corpus["surroundings_summary"].isna().tolist()


def test_landmark_only_keeps_empty_string_without_landmark(built) -> None:
    summary = built["landmark_only"]["surroundings_summary"]
    assert summary[2] == "" and summary[5] == ""
    assert summary[1] == "Empire State Building; Central Park"


def test_only_curated_names_match(built) -> None:
    assert "Joe's Pizza" in built["landmark_redacted"]["surroundings_summary"][1]
    assert "Madison Square Garden" in built["landmark_redacted"]["surroundings_summary"][5]
    assert "Joe" not in built["landmark_only"]["surroundings_summary"][1]


def test_longest_name_wins(built) -> None:
    assert built["landmark_only"]["surroundings_summary"][0] == "Central Park Zoo"
    assert built["landmark_redacted"]["surroundings_summary"][0] == "Steps from a well-known landmark and a deli."


def test_redacted_removes_exactly_matched_names(built) -> None:
    summary = built["landmark_redacted"]["surroundings_summary"]
    assert summary[1] == "Near the a well-known landmark and a well-known landmark; try Joe's Pizza."
    assert summary[2] == TEXTS[2]
    assert summary[4] == "a well-known landmark is close."


def test_names_are_canonical_case(built) -> None:
    assert built["landmark_only"]["surroundings_summary"][4] == "Central Park"


def test_curated_list_loads() -> None:
    names = arms.load_landmarks()
    assert len(names) == 77 and "Empire State Building" in names


def censored(corpus: pd.DataFrame) -> dict[tuple[str, str], pd.DataFrame]:
    described = {("enriched", "05"): corpus}
    described.update({(arm, "05"): frame for arm, frame in arms.build_arms(corpus, NAMES).items()})
    return described


def test_staged_censored_arms_share_rows_with_summary_as_only_text(corpus) -> None:
    datasets = stage.build_datasets(listings(), censored(corpus))
    assert {len(frame) for frame in datasets.values()} == {5}
    for arm in ("landmark_only", "landmark_redacted"):
        joint = datasets[stage.dataset_key(arm, "05", "joint")]
        assert joint["surroundings_summary"].dtype == "string"
        assert isinstance(joint["room_type"].dtype, pd.CategoricalDtype)
        text_only = datasets[stage.dataset_key(arm, "05", "text_only")]
        assert list(text_only.columns) == ["surroundings_summary", "price"]


def test_staged_censored_feature_types(corpus, tmp_path) -> None:
    detect = pytest.importorskip("multabench.baselines.preprocessing.feature_types").detect_feature_types
    datasets = stage.build_datasets(listings(), censored(corpus))
    for arm in ("landmark_only", "landmark_redacted"):
        key = stage.dataset_key(arm, "05", "joint")
        stage.write_snapshots({key: datasets[key]}, str(tmp_path))
        path = os.path.join(stage.snapshot_dir(str(tmp_path), key), "data.parquet")
        types = detect(pd.read_parquet(path).drop(columns="price"), image_column=None)
        assert types.text_features == {"surroundings_summary"}
        assert types.categorical_features == {"room_type"}
