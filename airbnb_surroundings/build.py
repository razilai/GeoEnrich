"""Enrich Airbnb listings (New York City) with nearby POIs (Overture Maps Places).

POIs come from Overture Maps Places — OSM + Foursquare merged, deduped, with a
standardized category per place and a confidence score. We read straight from
the public S3 Parquet via DuckDB (no local extracts), filter by the NYC bounding
box + confidence, then for every listing list the POIs whose geometry falls
within RADIUS meters, tagged with a coarse proximity band. One new column:
    - surroundings : JSON array of {category, name?, prox} dicts, ready to feed
                     an LLM for a text summary
Listings with zero POIs in radius are dropped (count used internally, never
persisted as a column).

Quality over raw OSM: Overture ships clean categories (no benches/hydrants to
strip) and a confidence field, so cleaning collapses to a confidence gate.
"""

import collections
import csv
import json
import os
import sys

import duckdb
import geopandas as gpd
import pandas as pd

from airbnb_surroundings import config
from airbnb_surroundings.config import (
    DOORSTEP,
    MAX_POIS,
    MIN_CONF,
    NAME_MIN_CONF,
    NYC_UTM,
    PER_CATEGORY,
    RADIUS,
    SHORTWALK_RESERVE,
)


# Overture Places S3 Parquet (public, requester-anonymous, region us-west-2).
OVERTURE = (
    f"s3://overturemaps-us-west-2/release/{config.OVERTURE_RELEASE}/"
    "theme=places/type=place/*"
)

# Overture top-level category groups kept as price/character signal. Everything
# else (professional/medical/beauty/religious/financial/government/home/auto/...)
# says little about a listing's neighbourhood feel and floods the sample, so it's
# filtered out at query time.
KEEP_GROUPS = {
    "eat_and_drink",
    "retail",
    "arts_and_entertainment",
    "attractions_and_activities",
    "active_life",
}
# transit is the strongest locational driver but sits in the mixed `travel` group
# (alongside parking, car rental, tours) — cherry-pick just the transit leaves.
TRANSIT_LEAVES = {
    "transportation",
    "public_transportation",
    "train_station",
    "metro_station",
    "subway_station",
    "bus_station",
    "bus_stop",
    "light_rail_station",
    "ferry_terminal",
    "tram_station",
    "airport",
}
# Leaves where the POI's NAME carries signal (a specific landmark reads premium:
# "9/11 Memorial", "American Museum of Natural History"). Everywhere else the name
# is noise — a restaurant/shop's identity is its category, not "Dough Vale". Kept
# deliberately tight: Overture's broad landmark_and_historical_building / art_gallery
# leaves are ~90% minor (co-ops, tiny galleries), so they're excluded. Combined
# with NAME_MIN_CONF (Overture ships no fame/wikidata field, confidence is the
# best available proxy).
NAME_LEAVES = {
    "museum", "art_museum", "history_museum", "science_museum", "childrens_museum",
    "contemporary_art_museum", "monument", "memorial", "memorial_park",
    "national_park", "botanical_garden", "zoo", "aquarium", "observatory",
    "castle", "stadium", "arena", "concert_hall", "opera_house",
}
_TAXONOMY_CSV = os.path.join(os.path.dirname(__file__), "overture_categories.csv")


def _leaf_groups():
    """leaf category code -> top-level Overture group, from the vendored taxonomy."""
    m = {}
    with open(_TAXONOMY_CSV, encoding="utf-8-sig") as f:
        for row in csv.reader(f, delimiter=";"):
            if len(row) < 2 or row[0].strip() == "Category code":
                continue
            m[row[0].strip()] = row[1].strip().strip("[]").split(",")[0].strip()
    return m


_LEAF_GROUP = _leaf_groups()


def allowed_categories():
    """Leaf codes worth keeping: any leaf whose group is in KEEP_GROUPS, plus the
    hand-picked transit leaves."""
    keep = set(TRANSIT_LEAVES)
    keep |= {c for c, g in _LEAF_GROUP.items() if g in KEEP_GROUPS}
    return keep


def keeps_name(cat, conf):
    """True if this POI's NAME is signal: a notable-category place mapped
    confidently enough to be the real thing (not a mis-tagged co-op)."""
    return cat in NAME_LEAVES and conf >= NAME_MIN_CONF

# leaky / ID columns dropped up front (in memory) if present, so no output
# variant ever carries them — they pollute the eval's structured baseline.
# Source airbnb.csv is untouched. A fresh `index` column is the stable
# per-listing key (used by describe.py's incremental cache).
DROP_COLS = ["id", "name", "host_id", "host_name", "license", "last_review"]


def connect():
    """DuckDB with spatial + httpfs, anonymous read of the public Overture bucket."""
    con = duckdb.connect()
    con.execute("INSTALL spatial; LOAD spatial;")
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute("SET s3_region='us-west-2';")
    return con


def load_pois(con, bbox, utm, categories):
    """POIs inside `bbox` (lon/lat) above the confidence gate, restricted to the
    price-relevant `categories`, in metric CRS.

    bbox is [xmin, ymin, xmax, ymax]. The bbox.* struct columns give Parquet
    row-group pushdown, so only the relevant slices are scanned over the wire.
    """
    xmin, ymin, xmax, ymax = bbox
    # category codes are simple [a-z_] identifiers; quote for SQL all the same
    cats = ", ".join("'" + c.replace("'", "''") + "'" for c in sorted(categories))
    q = f"""
        SELECT names.primary            AS name,
               categories.primary       AS category,
               confidence,
               ST_AsText(geometry) AS wkt
        FROM read_parquet('{OVERTURE}')
        WHERE bbox.xmin >= {xmin} AND bbox.xmax <= {xmax}
          AND bbox.ymin >= {ymin} AND bbox.ymax <= {ymax}
          AND confidence >= {MIN_CONF}
          AND categories.primary IN ({cats})
    """
    df = con.execute(q).df()
    if df.empty:
        return df
    g = gpd.GeoDataFrame(
        df.drop(columns="wkt"),
        geometry=gpd.GeoSeries.from_wkt(df["wkt"], crs=4326),
    )
    return g[g.geometry.notna()].to_crs(utm).reset_index(drop=True)


def main():
    df = pd.read_csv(config.CLEANED_CSV, low_memory=False).reset_index(drop=True)
    df = df.rename(columns={"lat": "latitude", "long": "longitude"})
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    df.insert(0, "index", df.index)  # stable per-listing key for describe.py's cache

    con = connect()
    pad = 0.02  # ~2km, comfortably covers the 400m radius at the edges
    bbox = [
        df.longitude.min() - pad,
        df.latitude.min() - pad,
        df.longitude.max() + pad,
        df.latitude.max() + pad,
    ]
    cats = allowed_categories()
    print(f"{len(df)} NYC listings — querying Overture "
          f"({len(cats)} price-relevant categories)", flush=True)
    pois = load_pois(con, bbox, NYC_UTM, cats)
    if pois.empty:
        sys.exit("no POIs returned from Overture — check release id / S3 access")
    print(f"{len(pois)} POIs", flush=True)

    # candidate POIs within RADIUS: buffer the point + intersect. Keep the
    # unbuffered points too, to measure the real listing->POI distance.
    gpts = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(df.longitude, df.latitude),
        index=df.index,
        crs=4326,
    ).to_crs(NYC_UTM)
    buf = gpts.copy()
    buf["geometry"] = buf.geometry.buffer(RADIUS)

    joined = gpd.sjoin(buf, pois, predicate="intersects", how="inner")
    # exact distance, vectorized. Point-to-geometry uses the EDGE for polygon
    # POIs (a park's boundary, not its centroid).
    lp = gpts.geometry.loc[joined.index].to_numpy()
    pg = pois.geometry.loc[joined["index_right"]].to_numpy()
    joined["dist"] = (
        gpd.GeoSeries(lp, crs=NYC_UTM)
        .distance(gpd.GeoSeries(pg, crs=NYC_UTM), align=False)
        .to_numpy()
    )

    # noise cut at range: unnamed POIs beyond the doorstep say little
    joined["nm"] = joined["name"].fillna("")
    keep = (joined["nm"] != "") | (joined["dist"] <= DOORSTEP)
    joined = joined[keep].sort_values("dist")

    labels = {}  # listing index -> list[dict] of POI records within RADIUS
    for lid, grp in joined.groupby(level=0):
        seen, per_cat, picked = set(), collections.Counter(), []
        for nm, cat, d, conf in zip(
            grp["nm"], grp["category"], grp["dist"], grp["confidence"]
        ):
            if nm and nm in seen:  # nearest copy per name
                continue
            if per_cat[cat] >= PER_CATEGORY:  # variety: cap copies per category
                continue
            if nm:
                seen.add(nm)
            per_cat[cat] += 1
            # minimal record: category + distance carry the signal. Keep the name
            # only for notable places (landmarks/museums), where it IS the signal.
            rec = {"category": cat, "dist_m": round(float(d))}
            if nm and keeps_name(cat, conf):
                rec["name"] = nm
            picked.append(rec)  # already distance-sorted

        # band stratification: reserve slots for short-walk POIs (parks, landmarks,
        # transit at range) so the global cap doesn't fill entirely at the doorstep.
        short = [p for p in picked if p["dist_m"] > DOORSTEP][:SHORTWALK_RESERVE]
        door = [p for p in picked if p["dist_m"] <= DOORSTEP][: MAX_POIS - len(short)]
        labels[lid] = sorted(door + short, key=lambda p: p["dist_m"])

    recs = df.index.map(labels)  # list[dict] per listing, or NaN
    n_pois = [len(x) if isinstance(x, list) else 0 for x in recs]  # local only, never a column
    df["surroundings"] = [
        json.dumps(x, ensure_ascii=False) if isinstance(x, list) else "[]" for x in recs
    ]

    # drop listings with no POIs within RADIUS
    before = len(df)
    df = df[[n > 0 for n in n_pois]]
    print(f"dropped {before - len(df)} empty listings, kept {len(df)}", flush=True)

    # lat/long are spent now (used only for POI matching) and would leak
    # location -> price into the model, so drop them from every output.
    df = df.drop(columns=["latitude", "longitude"])

    # two variants: vanilla = tabular only (no bulky JSON, model-ready);
    # json = vanilla + the surroundings POI JSON (input for describe.py).
    # vanilla is a clean modelling baseline -> also drop the internal `index`
    # key. enriched keeps `index` because describe.py caches on it.
    os.makedirs(config.PROCESSED_DIR, exist_ok=True)
    df.drop(columns=["surroundings", "index"]).to_csv(config.VANILLA_CSV, index=False)
    df.to_csv(config.ENRICHED_CSV, index=False)
    print(f"done -> {config.VANILLA_CSV}, {config.ENRICHED_CSV}", flush=True)


if __name__ == "__main__":
    main()
