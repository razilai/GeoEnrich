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

# Overture level-2 taxonomy categories (taxonomy.hierarchy[2]) kept as POI
# categories, keyed by their level-1 group; each category is counted separately
# and every other place is discarded. The hand-picked list, with NYC counts, is
# analysis/overture_taxonomy_l2.txt. air_transport_facility_or_service is left
# out: in NYC it is mostly airline offices and mis-pinned "airport" records.
OVERTURE_TAXONOMY = {
    "arts_and_entertainment": ("museum", "nightlife_venue", "performing_arts_venue"),
    "cultural_and_historic": ("historic_site",),
    "food_and_drink": (
        "alcoholic_beverage_venue",
        "casual_eatery",
        "non_alcoholic_beverage_venue",
        "restaurant",
    ),
    "health_care": ("hospital",),
    "lodging": ("hotel",),
    "services_and_business": ("corporate_or_business_office",),
    "shopping": (
        "convenience_store",
        "department_store",
        "discount_store",
        "fashion_and_apparel_store",
        "food_and_beverage_store",
    ),
    "sports_and_recreation": ("park", "sport_or_fitness_facility"),
    "travel_and_transportation": ("ground_transport_facility_or_service", "parking"),
}
OVERTURE_CATEGORIES = tuple(c for cats in OVERTURE_TAXONOMY.values() for c in cats)
# Level-2 names are unique across level-1 groups, so each maps to one parent.
CATEGORY_GROUP = {c: g for g, cats in OVERTURE_TAXONOMY.items() for c in cats}
