from __future__ import annotations

import pandas as pd
import pytest

from src import arms

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
    return pd.DataFrame({"index": range(6), "price": [100.0 + i for i in range(6)], "surroundings_summary": TEXTS})


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


def test_csv_round_trip_keeps_empty_strings_and_missing_summaries(corpus, tmp_path) -> None:
    path = str(tmp_path / "arm.csv")
    arms.build_arms(corpus, NAMES)["landmark_only"].to_csv(path, index=False)
    summary = arms.read_arm(path, corpus)["surroundings_summary"]
    assert summary[2] == "" and summary[5] == ""
    assert pd.isna(summary[3])
