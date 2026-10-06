"""Enrich NYC Airbnb listings with their surroundings from Overture Maps Places.

Each listing gains a `surroundings` JSON column (input for describe.py):
    cats       top-level taxonomy group -> [count <= RADIUS, nearest m]
    landmarks  [[name, metres], ...] curated landmarks within LANDMARK_RADIUS, nearest first

Places are categorised only by their Overture top-level taxonomy group; only the
landmarks are curated by hand. Listings with no POI within RADIUS are dropped.
"""

import argparse
import functools
import json
import os
import sys

import duckdb
import geopandas as gpd
import pandas as pd
from shapely import wkt

from airbnb_surroundings import config
from airbnb_surroundings.config import (
    LANDMARK_RADIUS,
    MIN_CONF,
    NYC_UTM,
    OVERTURE_GROUPS,
    RADIUS,
)

WGS84 = 4326
PLACES_PATH = (
    f"s3://overturemaps-us-west-2/release/{config.OVERTURE_RELEASE}/"
    "theme=places/type=place/*"
)
LANDMARKS_JSON = os.path.join(os.path.dirname(__file__), "landmarks.json")

# The bbox.* filters let DuckDB skip Parquet row groups, so only NYC is downloaded.
PLACES_SQL = """
    SELECT taxonomy.hierarchy[1] AS "group",
           ST_AsWKB(geometry)    AS geometry
    FROM read_parquet($path)
    WHERE bbox.xmin <= $maxx AND bbox.xmax >= $minx
      AND bbox.ymin <= $maxy AND bbox.ymax >= $miny
      AND confidence >= $min_conf
      AND list_contains($groups, taxonomy.hierarchy[1])
"""

# Identifier and leaky columns; they would pollute the eval's tabular baseline.
DROP_COLS = ["id", "name", "host_id", "host_name", "license", "last_review"]
BBOX_PAD_DEG = 0.02  # ~2 km, so listings at the edge still see their full RADIUS

log = functools.partial(print, flush=True)


# --------------------------------------------------------------------------- #
# Per-listing aggregation
# --------------------------------------------------------------------------- #
def aggregate_surroundings(nearby: pd.DataFrame) -> dict[str, list[int]]:
    """Per-group [count <= RADIUS, nearest m] for one listing's nearby places.

    `nearby` has one row per place with `group` and `dist` (metres). Groups are
    ordered nearest first.
    """
    stats = (
        nearby.groupby("group", sort=False)["dist"]
        .agg(count="size", nearest="min")
        .sort_values("nearest", kind="stable")
    )
    return {
        str(group): [int(row["count"]), int(round(row["nearest"]))]
        for group, row in stats.iterrows()
    }


# --------------------------------------------------------------------------- #
# Data sources
# --------------------------------------------------------------------------- #
def load_listings() -> pd.DataFrame:
    """Cleaned listings with `latitude`/`longitude` and a stable `index` key
    (describe.py caches on it)."""
    df = pd.read_csv(config.CLEANED_CSV, low_memory=False)
    df = df.rename(columns={"lat": "latitude", "long": "longitude"})
    df = df.drop(columns=DROP_COLS, errors="ignore")
    df.insert(0, "index", df.index)
    return df


def connect() -> duckdb.DuckDBPyConnection:
    """DuckDB able to read the public Overture bucket anonymously."""
    con = duckdb.connect()
    for extension in ("httpfs", "spatial"):
        con.install_extension(extension)
        con.load_extension(extension)
    con.execute("SET s3_region = 'us-west-2';")
    return con


def load_pois(
    con: duckdb.DuckDBPyConnection, listings: pd.DataFrame
) -> gpd.GeoDataFrame:
    """Overture places around the listings, projected to NYC_UTM."""
    df = con.execute(
        PLACES_SQL,
        {
            "path": PLACES_PATH,
            "minx": listings.longitude.min() - BBOX_PAD_DEG,
            "miny": listings.latitude.min() - BBOX_PAD_DEG,
            "maxx": listings.longitude.max() + BBOX_PAD_DEG,
            "maxy": listings.latitude.max() + BBOX_PAD_DEG,
            "min_conf": MIN_CONF,
            "groups": list(OVERTURE_GROUPS),
        },
    ).df()
    pois = gpd.GeoDataFrame(
        df.drop(columns="geometry"),
        geometry=gpd.GeoSeries.from_wkb(df["geometry"].map(bytes), crs=WGS84),
    )
    return pois[pois.geometry.notna()].to_crs(NYC_UTM).reset_index(drop=True)


def load_landmarks() -> list[tuple[str, object]]:
    """Curated `(name, geometry)` pairs from landmarks.json, in NYC_UTM.

    Landmarks are matched by distance to their own geometry rather than by POI
    name: famous names are shared by many unrelated Overture places.
    """
    with open(LANDMARKS_JSON, encoding="utf-8") as f:
        entries = json.load(f)["landmarks"]
    geometries = gpd.GeoSeries(
        [wkt.loads(entry["wkt"]) for entry in entries.values()], crs=WGS84
    ).to_crs(NYC_UTM)
    return list(zip(entries, geometries))


# --------------------------------------------------------------------------- #
# Spatial matching (all distances in metres; polygons measure to their edge)
# --------------------------------------------------------------------------- #
def nearby_pois(points: gpd.GeoSeries, pois: gpd.GeoDataFrame) -> pd.DataFrame:
    """Every (listing, POI) pair within RADIUS, indexed by listing, with `dist`."""
    circles = gpd.GeoDataFrame(geometry=points.buffer(RADIUS))
    pairs = gpd.sjoin(circles, pois, predicate="intersects")
    pairs["dist"] = (
        points.loc[pairs.index]
        .distance(pois.geometry.loc[pairs["index_right"]], align=False)
        .to_numpy()
    )
    pairs = pairs.sort_values("dist")
    return pairs[pairs["dist"] <= RADIUS]


def nearby_landmarks(points: gpd.GeoSeries) -> dict[int, list[list]]:
    """Listing -> [[landmark, metres], ...] within LANDMARK_RADIUS, nearest first."""
    found: dict[int, list[list]] = {listing: [] for listing in points.index}
    for name, geometry in load_landmarks():
        distances = points.distance(geometry)
        within = distances[distances <= LANDMARK_RADIUS].round().astype(int)
        for listing, metres in within.items():
            found[listing].append([name, int(metres)])
    return {
        listing: sorted(landmarks, key=lambda landmark: landmark[1])
        for listing, landmarks in found.items()
    }


def surroundings(listings: pd.DataFrame, pois: gpd.GeoDataFrame) -> dict[int, dict]:
    """Surroundings record per listing that has at least one POI within RADIUS."""
    points = gpd.GeoSeries(
        gpd.points_from_xy(listings.longitude, listings.latitude),
        index=listings.index,
        crs=WGS84,
    ).to_crs(NYC_UTM)
    landmarks = nearby_landmarks(points)

    records = {}
    for listing, nearby in nearby_pois(points, pois).groupby(level=0):
        records[listing] = {
            "cats": aggregate_surroundings(nearby),
            "landmarks": landmarks[listing],
        }
    return records


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--enriched-out",
        default=os.environ.get("BUILD_ENRICHED_OUT", config.ENRICHED_CSV),
        help="output CSV for the enriched JSON (defaults to the production path)",
    )
    args = parser.parse_args()

    listings = load_listings()
    log(
        f"{len(listings)} NYC listings — querying Overture {config.OVERTURE_RELEASE} "
        f"({len(OVERTURE_GROUPS)} top-level groups)"
    )
    pois = load_pois(connect(), listings)
    if pois.empty:
        sys.exit("no POIs returned from Overture — check release id / S3 access")
    log(f"{len(pois)} POIs")

    records = surroundings(listings, pois)
    kept = listings.loc[listings.index.isin(list(records))]
    log(f"dropped {len(listings) - len(kept)} empty listings, kept {len(kept)}")

    # Coordinates would leak location -> price into the model.
    enriched = kept.drop(columns=["latitude", "longitude"]).assign(
        surroundings=[json.dumps(records[i], ensure_ascii=False) for i in kept.index]
    )

    os.makedirs(config.PROCESSED_DIR, exist_ok=True)
    enriched.drop(columns=["surroundings", "index"]).to_csv(
        config.VANILLA_CSV, index=False
    )
    enriched.to_csv(args.enriched_out, index=False)
    log(f"done -> {config.VANILLA_CSV}, {args.enriched_out}")


if __name__ == "__main__":
    main()
