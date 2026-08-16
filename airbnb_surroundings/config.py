"""Shared paths + enrichment tuning constants for the pipeline.

Single source of truth so build/describe/eval agree on where the raw input and
generated artifacts live. Paths are absolute (anchored to the repo root), so a
stage works no matter which directory it's launched from.
"""

import os

# Repo root. Anchored to this file's editable-install location
# (airbnb_surroundings/config.py -> two levels up), overridable via env for
# odd deployments.
ROOT = os.environ.get("PROJECT_ROOT") or os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_DIR = os.path.join(ROOT, "data")  # gitignored dataset tree
RAW_DIR = os.path.join(DATA_DIR, "raw")  # untouched source scrapes
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")  # pipeline-generated datasets
ARTIFACTS_DIR = os.path.join(ROOT, "artifacts")  # misc scratch: audit dumps, logs
RESULTS_DIR = os.path.join(ROOT, "results")  # gitignored eval reports

# dataset stage I/O (data/raw -> data/processed)
NYC_SCRAPE_CSV = os.path.join(RAW_DIR, "airbnb_nyc.csv")  # raw scrape (clean.py input)
CLEANED_CSV = os.path.join(PROCESSED_DIR, "airbnb.csv")  # cleaned listings (build.py input)
VANILLA_CSV = os.path.join(PROCESSED_DIR, "airbnb_vanilla.csv")  # tabular-only
ENRICHED_CSV = os.path.join(PROCESSED_DIR, "airbnb_enriched.csv")  # + POI JSON
DESCRIBED_CSV = os.path.join(PROCESSED_DIR, "airbnb_described.csv")  # + LLM prose

# non-dataset outputs
BATCH_JSON = os.path.join(ARTIFACTS_DIR, "batch_result.json")  # raw LLM batch dump (audit)
EVAL_REPORT_CSV = os.path.join(RESULTS_DIR, "eval_report.csv")  # eval.py output

# enrichment tuning (see build.py)
RADIUS = 450  # meters — outer capture radius, Euclidean (~570m walk at Manhattan
# grid detour ~1.27; hedonic lit centers ~650m walk / 500m+ buffers)
DOORSTEP = 150  # meters — inner horizon; counts split at <=150m (block) vs <=400m (walk)
MIN_CONF = 0.6  # Overture confidence gate — replaces hand-maintained junk filters
# Curated landmarks (landmarks.json) are the highest-variance token, so the channel
# reaches farther than the density buckets: a famous landmark 800m out is still a
# price signal. Landmarks are matched by geometric distance to their own geocoded OSM
# geometry (not POI names), so there is no fuzzy/confidence gate — a listing gets a
# landmark iff it is within LANDMARK_RADIUS of that landmark's geometry edge.
LANDMARK_RADIUS = 800  # meters — curated-landmark capture radius (>= RADIUS)
NYC_UTM = 32618  # metric CRS for NYC so buffer() is in real meters

# Overture Places release; bump when a newer one ships:
# https://docs.overturemaps.org/release-calendar/
OVERTURE_RELEASE = "2026-07-22.0"
