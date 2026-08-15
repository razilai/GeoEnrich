#!/usr/bin/env python
# coding: utf-8

"""Stage 0: clean the raw NYC scrape into the tidy CSV the pipeline consumes.

Maps the raw Airbnb schema to canonical columns, casts types, derives room
counts, drops incomplete/invalid rows, caps the size, and writes a single CSV
(data/airbnb.csv) that build.py reads next.

    python -m airbnb_surroundings.clean          # data/airbnb_nyc.csv -> data/airbnb.csv
    python -m airbnb_surroundings.clean RAW.csv --output-path OUT.csv
"""

import argparse

import pyspark.sql.functions as F
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import BooleanType, DoubleType

from airbnb_surroundings import config

MAX_LISTINGS = 10_000


def initial_selection(df: DataFrame) -> DataFrame:
    """Map supported Airbnb source schemas to the cleaned column names."""
    column_sources = {
        "price": ("price",),
        "ratings": ("ratings", "review_scores_rating"),
        "lat": ("lat", "latitude"),
        "long": ("long", "longitude"),
        "guests": ("guests", "accommodates"),
        "beds": ("beds",),
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

    selected = []
    for target, candidates in column_sources.items():
        source = next((name for name in candidates if name in df.columns), None)
        if source is not None:
            selected.append(F.col(source).alias(target))

    if not selected:
        raise ValueError("No expected columns were found in the input data.")

    return df.select(*selected)


def set_schema(df: DataFrame) -> DataFrame:
    """Safely cast numeric and boolean columns to their expected types."""
    dtype_mapping = {
        "price": DoubleType(),
        "ratings": DoubleType(),
        "lat": DoubleType(),
        "long": DoubleType(),
        "guests": DoubleType(),
        "beds": DoubleType(),
        "bedrooms": DoubleType(),
        "bathrooms": DoubleType(),
        "host_number_of_reviews": DoubleType(),
        "host_rating": DoubleType(),
        "property_number_of_reviews": DoubleType(),
        "is_superhost": BooleanType(),
    }

    for column, data_type in dtype_mapping.items():
        if column in df.columns:
            expression = F.col(column)
            if column == "price":
                # NYC prices are strings such as "$113.97"; this normalizes the
                # stored amount without converting between currencies.
                expression = F.regexp_replace(expression, r"[$,]", "")
            df = df.withColumn(
                column,
                expression.cast(data_type),
            )

    return df


def transform_details(df: DataFrame) -> DataFrame:
    """Create bed, bedroom, bathroom, and total-room counts for each listing.

    Existing numeric columns take precedence. When they are unavailable, counts are
    derived from the human-readable listing details.
    """
    detail_columns = [column for column in ("details",) if column in df.columns]
    detail_text = (
        F.concat_ws(" ", *(F.col(column) for column in detail_columns))
        if detail_columns
        else F.lit("")
    )

    def extracted_count(pattern: str):
        value = F.regexp_extract(detail_text, pattern, 1)
        return F.nullif(value, F.lit("")).cast("double")

    source_or_extracted = {
        "num_bedrooms": ("bedrooms", r"(?i)\b(\d+\.?\d*)\s+bedrooms?\b"),
        "num_baths": ("bathrooms", r"(?i)\b(\d+\.?\d*)\s+bath(?:s|rooms?)?\b"),
    }

    for output_column, (source_column, pattern) in source_or_extracted.items():
        source_value = (
            F.col(source_column) if source_column in df.columns else F.lit(None)
        )
        df = df.withColumn(
            output_column, F.coalesce(source_value, extracted_count(pattern))
        )

    return df.withColumn(
        "num_rooms",
        F.coalesce(F.col("num_bedrooms"), F.lit(0))
        + F.coalesce(F.col("num_baths"), F.lit(0)),
    )


def filter_valid_prices(df: DataFrame) -> DataFrame:
    """Remove rows without a usable positive price."""
    return df.filter(F.col("price").isNotNull() & (F.col("price") > 0))


def apply_basic_cleaning(df: DataFrame) -> DataFrame:
    """Run the listing-cleaning pipeline on raw data."""
    return filter_valid_prices(transform_details(set_schema(initial_selection(df))))


def load_airbnb_nyc(spark: SparkSession, csv_path: str = config.NYC_SCRAPE_CSV) -> DataFrame:
    """Read the raw NYC scrape CSV and return its cleaned listings DataFrame."""
    raw_listings = (
        spark.read.option("header", True)
        .option("multiLine", True)
        .option("quote", '"')
        .option("escape", '"')
        .csv(csv_path)
    )
    return apply_basic_cleaning(raw_listings)


def remove_nulls_and_limit(
    df: DataFrame, max_listings: int = MAX_LISTINGS
) -> DataFrame:
    """Remove incomplete listings, then randomly cap the result when needed.

    Nulls are removed here, before any future imputation step can be added. The
    count is taken after that removal so the 10,000-listing cap applies only to
    complete listings.
    """
    complete_listings = df.dropna(how="any")
    row_count = complete_listings.count()
    print(f"Listings after removing null values: {row_count}")

    if row_count > max_listings:
        print(f"Randomly sampling {max_listings:,} listings.")
        return complete_listings.orderBy(F.rand()).limit(max_listings)

    return complete_listings


def main() -> None:
    """Run the cleaner locally and preview the cleaned NYC listings."""
    parser = argparse.ArgumentParser(
        description="Clean Airbnb NYC listings with PySpark."
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        default=config.NYC_SCRAPE_CSV,
        help="Path to the raw source CSV (default: data/airbnb_nyc.csv).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of cleaned listings to display (default: 5).",
    )
    parser.add_argument(
        "--output-path",
        default=config.RAW_CSV,
        help=(
            "Path for the cleaned CSV (default: data/airbnb.csv — the file "
            "build.py reads). Written as a single CSV, not a Spark part-dir."
        ),
    )
    args = parser.parse_args()

    spark = (
        SparkSession.builder.master("local[*]")
        .appName("airbnb-nyc-cleaning")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    try:
        cleaned_listings = remove_nulls_and_limit(
            load_airbnb_nyc(spark, args.csv_path)
        )
        # single CSV (not a Spark part-dir) so build.py can read it directly.
        # Dataset is capped at MAX_LISTINGS, so collecting to pandas is safe.
        cleaned_listings.toPandas().to_csv(args.output_path, index=False)
        print(f"wrote {args.output_path}", flush=True)
        cleaned_listings.show(args.limit, truncate=False)
    finally:
        spark.stop()


# Backwards-compatible name for callers of the previous cleaning entry point.
apply_stateless_transformations = apply_basic_cleaning


if __name__ == "__main__":
    main()
