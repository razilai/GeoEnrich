#!/usr/bin/env python
# coding: utf-8

"""Stage 0 (pandas twin of clean.py): clean the raw NYC scrape into the tidy CSV.

Byte-for-byte the same cleaning logic as airbnb_surroundings.clean, but built on
pandas instead of PySpark so it runs without a JVM (the Spark path needs Java
8/11/17; newer JDKs drop jdk.internal.ref.Cleaner and Spark fails to start).

Maps the raw Airbnb schema to canonical columns, casts types, derives room
counts, drops incomplete/invalid rows, caps the size, and writes a single CSV
(data/processed/airbnb.csv) that build.py reads next.

    python -m airbnb_surroundings.clean_pandas   # data/raw/airbnb_nyc.csv -> data/processed/airbnb.csv
    python -m airbnb_surroundings.clean_pandas RAW.csv --output-path OUT.csv
"""

import argparse
import os

import pandas as pd

from airbnb_surroundings import config

MAX_LISTINGS = 10_000

# Spark's StringType -> BooleanType cast accepts these (case-insensitive);
# anything else becomes null. Mirrored here so is_superhost casts identically.
_TRUE_STRINGS = {"t", "true", "y", "yes", "1"}
_FALSE_STRINGS = {"f", "false", "n", "no", "0"}


def initial_selection(df: pd.DataFrame) -> pd.DataFrame:
    """Map supported Airbnb source schemas to the cleaned column names."""
    column_sources = {
        "price": ("price",),
        "ratings": ("ratings", "review_scores_rating"),
        "lat": ("lat", "latitude"),
        "long": ("long", "longitude"),
        "guests": ("guests", "accommodates"),
        # `bedrooms` is kept only to derive num_bedrooms; dropped from output in
        # transform_details. `beds` removed entirely — duplicate signal.
        "bedrooms": ("bedrooms",),
        "bathrooms": ("bathrooms",),
        "room_type": ("room_type",),
        "details": ("details", "bathrooms_text"),
        "host_rating": ("host_rating",),
        "property_number_of_reviews": (
            "property_number_of_reviews",
            "number_of_reviews",
        ),
        "is_superhost": ("is_supperhost", "is_superhost", "host_is_superhost"),
    }

    renames = {}
    for target, candidates in column_sources.items():
        source = next((name for name in candidates if name in df.columns), None)
        if source is not None:
            renames[source] = target

    if not renames:
        raise ValueError("No expected columns were found in the input data.")

    return df[list(renames.keys())].rename(columns=renames)


def _cast_double(series: pd.Series) -> pd.Series:
    """String -> double with Spark semantics: unparseable values become null.

    Forced to float64 so all-integer columns still serialize as `2.0`, matching
    Spark's DoubleType, rather than pandas' int64 `2`.
    """
    return pd.to_numeric(series, errors="coerce").astype("float64")


def _cast_boolean(series: pd.Series) -> pd.Series:
    """String -> boolean with Spark semantics: only true/false tokens map; else null."""
    normalized = series.astype("string").str.strip().str.lower()

    def to_bool(value):
        if value in _TRUE_STRINGS:
            return True
        if value in _FALSE_STRINGS:
            return False
        return None

    return normalized.map(to_bool).astype("object")


def set_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Safely cast numeric and boolean columns to their expected types."""
    double_columns = (
        "price",
        "ratings",
        "lat",
        "long",
        "guests",
        "bedrooms",
        "bathrooms",
        "host_rating",
        "property_number_of_reviews",
    )

    df = df.copy()
    for column in double_columns:
        if column in df.columns:
            series = df[column]
            if column == "price":
                # NYC prices are strings such as "$113.97"; this normalizes the
                # stored amount without converting between currencies.
                series = series.astype("string").str.replace(
                    r"[$,]", "", regex=True
                )
            df[column] = _cast_double(series)

    if "is_superhost" in df.columns:
        df["is_superhost"] = _cast_boolean(df["is_superhost"])

    return df


def transform_details(df: pd.DataFrame) -> pd.DataFrame:
    """Create bedroom and bathroom counts for each listing.

    Existing numeric columns take precedence. When they are unavailable, counts are
    derived from the human-readable listing details. The raw `bedrooms` and
    `details` sources are dropped afterward, keeping only the derived num_* columns.
    """
    df = df.copy()

    if "details" in df.columns:
        detail_text = df["details"].astype("string").fillna("")
    else:
        detail_text = pd.Series("", index=df.index, dtype="string")

    def extracted_count(pattern: str) -> pd.Series:
        # group 1 of the match, or NaN when there is no match — matching Spark's
        # regexp_extract("")-then-nullif-then-cast(double) chain.
        value = detail_text.str.extract(pattern, expand=False)
        return _cast_double(value)

    source_or_extracted = {
        "num_bedrooms": ("bedrooms", r"(?i)\b(\d+\.?\d*)\s+bedrooms?\b"),
        "num_baths": ("bathrooms", r"(?i)\b(\d+\.?\d*)\s+bath(?:s|rooms?)?\b"),
    }

    for output_column, (source_column, pattern) in source_or_extracted.items():
        extracted = extracted_count(pattern)
        if source_column in df.columns:
            # source takes precedence; fall back to the extracted count (coalesce).
            df[output_column] = df[source_column].where(
                df[source_column].notna(), extracted
            )
        else:
            df[output_column] = extracted

    # `bedrooms`/`details` were only sources for the num_* counts above — drop them
    # so they do not ship as duplicate columns in the cleaned output (`details`
    # re-parsed to a bath float duplicates num_baths/bathrooms).
    df = df.drop(columns=[c for c in ("bedrooms", "details") if c in df.columns])

    return df


def filter_valid_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows without a usable positive price."""
    return df[df["price"].notna() & (df["price"] > 0)]


def apply_basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """Run the listing-cleaning pipeline on raw data."""
    return filter_valid_prices(transform_details(set_schema(initial_selection(df))))


def load_airbnb_nyc(csv_path: str = config.NYC_SCRAPE_CSV) -> pd.DataFrame:
    """Read the raw NYC scrape CSV and return its cleaned listings DataFrame."""
    # dtype=str mirrors Spark reading every column as a string before set_schema
    # casts; pandas' default doublequote handles the "" escape the Spark reader
    # configured via quote/escape='"', and embedded newlines in quoted fields
    # (Spark's multiLine) are handled by the default C parser.
    raw_listings = pd.read_csv(csv_path, dtype=str, keep_default_na=False, na_values=[""])
    return apply_basic_cleaning(raw_listings)


def remove_nulls_and_limit(
    df: pd.DataFrame, max_listings: int = MAX_LISTINGS
) -> pd.DataFrame:
    """Remove incomplete listings, then randomly cap the result when needed.

    Nulls are removed here, before any future imputation step can be added. The
    count is taken after that removal so the 10,000-listing cap applies only to
    complete listings.
    """
    complete_listings = df.dropna(how="any")
    row_count = len(complete_listings)
    print(f"Listings after removing null values: {row_count}")

    if row_count > max_listings:
        print(f"Randomly sampling {max_listings:,} listings.")
        return complete_listings.sample(n=max_listings)

    return complete_listings


def main() -> None:
    """Run the cleaner locally and preview the cleaned NYC listings."""
    parser = argparse.ArgumentParser(
        description="Clean Airbnb NYC listings with pandas."
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        default=config.NYC_SCRAPE_CSV,
        help="Path to the raw source CSV (default: data/raw/airbnb_nyc.csv).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of cleaned listings to display (default: 5).",
    )
    parser.add_argument(
        "--output-path",
        default=config.CLEANED_CSV,
        help=(
            "Path for the cleaned CSV (default: data/processed/airbnb.csv — the "
            "file build.py reads). Written as a single CSV."
        ),
    )
    args = parser.parse_args()

    cleaned_listings = remove_nulls_and_limit(load_airbnb_nyc(args.csv_path))
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    cleaned_listings.to_csv(args.output_path, index=False)
    print(f"wrote {args.output_path}", flush=True)
    with pd.option_context("display.max_columns", None, "display.width", None):
        print(cleaned_listings.head(args.limit).to_string(index=False))


# Backwards-compatible name for callers of the previous cleaning entry point.
apply_stateless_transformations = apply_basic_cleaning


if __name__ == "__main__":
    main()
