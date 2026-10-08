"""Shared paths and enrichment constants for the pipeline.

All paths are absolute, anchored to the repo root (override with $PROJECT_ROOT),
so every stage resolves the same files regardless of the working directory.
"""

import os

ROOT = os.environ.get("PROJECT_ROOT") or os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

# --- Directories -------------------------------------------------------------

DATA_DIR = os.path.join(ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
ARTIFACTS_DIR = os.path.join(ROOT, "artifacts")
RESULTS_DIR = os.path.join(ROOT, "results")

# --- Datasets (in pipeline order) -------------------------------------------

NYC_SCRAPE_CSV = os.path.join(RAW_DIR, "airbnb_nyc.csv")
CLEANED_CSV = os.path.join(PROCESSED_DIR, "airbnb.csv")
VANILLA_CSV = os.path.join(PROCESSED_DIR, "airbnb_vanilla.csv")
ENRICHED_CSV = os.path.join(PROCESSED_DIR, "airbnb_enriched.csv")


def described_csv(prompt_id):
    """Described dataset for one prompt variant (prompts.toml), one file per
    variant so runs never overwrite each other's corpus."""
    return os.path.join(PROCESSED_DIR, f"airbnb_described_{prompt_id}.csv")

# --- Other outputs -----------------------------------------------------------

BATCH_JSON = os.path.join(ARTIFACTS_DIR, "batch_result.json")

# --- Enrichment parameters (see build.py) -----------------------------------

# Capture radii in meters (Euclidean). 450m ~ 570m walk on the Manhattan grid.
RADIUS = 450
DOORSTEP = 150
# Landmarks are matched by distance to their geometry rather than by POI name,
# and reach farther since a famous landmark remains a price signal at 800m.
LANDMARK_RADIUS = 800

MIN_CONF = 0.6  # minimum Overture confidence score for a POI
NYC_UTM = 32618  # EPSG code of a metric CRS for NYC

# https://docs.overturemaps.org/release-calendar/
OVERTURE_RELEASE = "2026-08-19.0"

# Overture top-level taxonomy groups kept as POI categories; the rest add noise
# without describing a neighbourhood. "lodging" includes competing short-term
# rentals, a possible target leak kept intentionally.
OVERTURE_GROUPS = (
    "food_and_drink",
    "shopping",
    "arts_and_entertainment",
    "cultural_and_historic",
    "sports_and_recreation",
    "lodging",
    "travel_and_transportation",
)
