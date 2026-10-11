from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import clean

FEATURES = [
    "ratings", "guests", "bathrooms", "room_type", "property_number_of_reviews",
    "is_superhost", "num_bedrooms", "minimum_nights",
]  # fmt: skip


def raw(n: int = 4, **overrides) -> pd.DataFrame:
    """Raw scrape rows as `main` reads them: every column a string, empty fields missing."""
    columns = {
        "price": ["$1,113.97"] * n,
        "review_scores_rating": ["4.8"] * n,
        "latitude": ["40.7"] * n,
        "longitude": ["-73.9"] * n,
        "accommodates": ["2"] * n,
        "bedrooms": ["1"] * n,
        "bathrooms": ["1"] * n,
        "room_type": ["Private room"] * n,
        "bathrooms_text": ["1 bath"] * n,
        "number_of_reviews": ["10"] * n,
        "host_is_superhost": ["t"] * n,
        "minimum_nights": ["30"] * n,
    }
    columns.update(overrides)
    return pd.DataFrame(columns, dtype="object")


def test_columns_and_types() -> None:
    df = clean.clean(raw())
    assert sorted(df.columns) == sorted(["price", "lat", "long", *FEATURES])
    assert df["minimum_nights"].dtype == float
    assert df["price"].tolist() == pytest.approx([np.log1p(1113.97)] * 4)


def test_bathrooms_fall_back_to_details() -> None:
    df = clean.clean(raw(n=2, bathrooms=[None, "2"], bathrooms_text=["1.5 baths", "1 bath"]))
    assert df["bathrooms"].tolist() == [1.5, 2.0]


def test_null_or_non_positive_price_is_dropped() -> None:
    assert len(clean.clean(raw(price=[None, "$0", "$10", "$20"]))) == 2


def test_rows_missing_over_a_quarter_of_features_are_dropped() -> None:
    df = clean.clean(
        raw(
            review_scores_rating=["4.8", None, None, None],
            bedrooms=["1", "1", None, None],
            minimum_nights=["30", "30", "30", None],
            bathrooms_text=["1 bath", "1 bath", "", ""],  # no bedroom count to fall back on
        )
    )
    kept = clean.drop_incomplete(df)
    assert kept.index.tolist() == [0, 1, 2]  # 0, 1 and 2 of 8 features missing; row 3 misses 3


def test_missing_coordinates_are_dropped() -> None:
    df = clean.clean(raw(latitude=["40.7", None, "40.7", "40.7"]))
    assert clean.drop_incomplete(df).index.tolist() == [0, 2, 3]


def test_cap_samples_reproducibly() -> None:
    df = clean.clean(raw(n=10, number_of_reviews=[str(i) for i in range(10)]))
    first = clean.drop_incomplete(df, max_listings=4)
    assert len(first) == 4
    assert first.index.tolist() == clean.drop_incomplete(df, max_listings=4).index.tolist()
