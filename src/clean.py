"""Stage 0: clean the raw NYC scrape into the tidy CSV the pipeline consumes.

Maps the raw Airbnb schema to canonical columns, casts types, derives room
counts, drops incomplete/invalid rows, caps the size, and writes a single CSV
(data/processed/airbnb.csv) that build.py reads next.

    python -m src.clean   # data/raw/airbnb_nyc.csv -> data/processed/airbnb.csv
    python -m src.clean RAW.csv --output-path OUT.csv
"""

import argparse
import os

import pandas as pd

from src import config

MAX_LISTINGS = 10_000

# Canonical column -> raw column names it may come from; the first one present wins.
# `bedrooms` and `details` only feed the derived room counts and are dropped after.
COLUMN_SOURCES = {
    "price": ["price"],
    "ratings": ["ratings", "review_scores_rating"],
    "lat": ["lat", "latitude"],
    "long": ["long", "longitude"],
    "guests": ["guests", "accommodates"],
    "bedrooms": ["bedrooms"],
    "bathrooms": ["bathrooms"],
    "room_type": ["room_type"],
    "details": ["details", "bathrooms_text"],
    "host_rating": ["host_rating"],
    "property_number_of_reviews": ["property_number_of_reviews", "number_of_reviews"],
    "is_superhost": ["is_supperhost", "is_superhost", "host_is_superhost"],
}

NUMERIC_COLUMNS = [
    "price", "ratings", "lat", "long", "guests", "bedrooms", "bathrooms",
    "host_rating", "property_number_of_reviews",
]

# Case-insensitive boolean tokens; anything else becomes null.
BOOLEAN_TOKENS = {
    **dict.fromkeys(["t", "true", "y", "yes", "1"], True),
    **dict.fromkeys(["f", "false", "n", "no", "0"], False),
}

# Derived count -> (numeric column that takes precedence, fallback regex over `details`).
ROOM_COUNTS = {
    "num_bedrooms": ("bedrooms", r"(?i)\b(\d+\.?\d*)\s+bedrooms?\b"),
    "num_baths": ("bathrooms", r"(?i)\b(\d+\.?\d*)\s+bath(?:s|rooms?)?\b"),
}


def to_float(series: pd.Series) -> pd.Series:
    """Parse to float64 (so integers serialize as `2.0`); unparseable values become NaN."""
    return pd.to_numeric(series, errors="coerce").astype("float64")


def select_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the supported raw columns, renamed to their canonical names."""
    renames = {}
    for target, candidates in COLUMN_SOURCES.items():
        source = next((c for c in candidates if c in df.columns), None)
        if source is not None:
            renames[source] = target
    if not renames:
        raise ValueError("No expected columns were found in the input data.")
    return df[list(renames)].rename(columns=renames)


def cast_types(df: pd.DataFrame) -> pd.DataFrame:
    """Cast numeric columns to float64 and is_superhost to boolean."""
    df = df.copy()
    # Prices are strings such as "$1,113.97"; strip the symbol and thousands separators.
    df["price"] = df["price"].astype("string").str.replace(r"[$,]", "", regex=True)
    for column in df.columns.intersection(NUMERIC_COLUMNS):
        df[column] = to_float(df[column])
    if "is_superhost" in df:
        tokens = df["is_superhost"].astype("string").str.strip().str.lower()
        df["is_superhost"] = tokens.map(BOOLEAN_TOKENS).astype("object")
    return df


def derive_room_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Add num_bedrooms/num_baths: the numeric column if set, else parsed from `details`."""
    details = df.get("details", pd.Series(pd.NA, index=df.index)).astype("string")
    counts = {}
    for target, (source, pattern) in ROOM_COUNTS.items():
        parsed = to_float(details.str.extract(pattern, expand=False))
        counts[target] = df[source].fillna(parsed) if source in df else parsed
    return df.assign(**counts).drop(columns=["bedrooms", "details"], errors="ignore")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Run the cleaning steps on raw listings and keep those with a positive price."""
    df = derive_room_counts(cast_types(select_columns(df)))
    return df[df["price"] > 0]  # NaN compares False, so null prices go too


def drop_incomplete(df: pd.DataFrame, max_listings: int = MAX_LISTINGS) -> pd.DataFrame:
    """Drop listings with any null, then randomly sample down to `max_listings`."""
    df = df.dropna()
    print(f"Listings after removing null values: {len(df)}")
    if len(df) > max_listings:
        print(f"Randomly sampling {max_listings:,} listings.")
        df = df.sample(n=max_listings)
    return df


def main() -> None:
    """Clean the raw scrape, write the CSV, and preview the first rows."""
    parser = argparse.ArgumentParser(description="Clean Airbnb NYC listings.")
    parser.add_argument("csv_path", nargs="?", default=config.NYC_SCRAPE_CSV,
                        help="Raw source CSV (default: data/raw/airbnb_nyc.csv).")
    parser.add_argument("--limit", type=int, default=5,
                        help="Number of cleaned listings to preview (default: 5).")
    parser.add_argument("--output-path", default=config.CLEANED_CSV,
                        help="Cleaned CSV (default: data/processed/airbnb.csv, read by build.py).")
    args = parser.parse_args()

    # Read every column as a string and treat only empty fields as missing
    # ("NA", "null", ... stay strings); cast_types does all type conversion.
    raw = pd.read_csv(args.csv_path, dtype=str, keep_default_na=False, na_values=[""])
    listings = drop_incomplete(clean(raw))

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    listings.to_csv(args.output_path, index=False)
    print(f"wrote {args.output_path}", flush=True)
    with pd.option_context("display.max_columns", None, "display.width", None):
        print(listings.head(args.limit).to_string(index=False))


if __name__ == "__main__":
    main()
