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

import json
import os
import sys

import duckdb
import geopandas as gpd
import pandas as pd

from airbnb_surroundings import config
from airbnb_surroundings.config import DOORSTEP, MAX_POIS, MIN_CONF, NYC_UTM, RADIUS


def band(d):
    return "doorstep" if d <= DOORSTEP else "short walk"


# Overture Places S3 Parquet (public, requester-anonymous, region us-west-2).
OVERTURE = (
    f"s3://overturemaps-us-west-2/release/{config.OVERTURE_RELEASE}/"
    "theme=places/type=place/*"
)

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


def load_pois(con, bbox, utm):
    """POIs inside `bbox` (lon/lat) above the confidence gate, in metric CRS.

    bbox is [xmin, ymin, xmax, ymax]. The bbox.* struct columns give Parquet
    row-group pushdown, so only the relevant slices are scanned over the wire.
    """
    xmin, ymin, xmax, ymax = bbox
    q = f"""
        SELECT names.primary            AS name,
               categories.primary       AS category,
               confidence,
               ST_AsText(ST_GeomFromWKB(geometry)) AS wkt
        FROM read_parquet('{OVERTURE}')
        WHERE bbox.xmin >= {xmin} AND bbox.xmax <= {xmax}
          AND bbox.ymin >= {ymin} AND bbox.ymax <= {ymax}
          AND confidence >= {MIN_CONF}
          AND categories.primary IS NOT NULL
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
    print(f"{len(df)} NYC listings — querying Overture", flush=True)
    pois = load_pois(con, bbox, NYC_UTM)
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
        recs, seen = [], set()
        for nm, cat, d in zip(grp["nm"], grp["category"], grp["dist"]):
            if nm:  # nearest copy per name (dedup any residual doubles)
                if nm in seen:
                    continue
                seen.add(nm)
            rec = {"category": cat, "prox": band(d)}
            if nm:
                rec["name"] = nm
            recs.append(rec)
            if len(recs) >= MAX_POIS:
                break
        labels[lid] = recs

    recs = df.index.map(labels)  # list[dict] per listing, or NaN
    n_pois = [len(x) if isinstance(x, list) else 0 for x in recs]  # local only, never a column
    df["surroundings"] = [
        json.dumps(x, ensure_ascii=False) if isinstance(x, list) else "[]" for x in recs
    ]

    # drop listings with no POIs within RADIUS
    before = len(df)
    df = df[[n > 0 for n in n_pois]]
    print(f"dropped {before - len(df)} empty listings, kept {len(df)}", flush=True)

    # two variants: vanilla = tabular only (no bulky JSON, model-ready);
    # json = vanilla + the surroundings POI JSON (input for describe.py).
    os.makedirs(config.PROCESSED_DIR, exist_ok=True)
    df.drop(columns=["surroundings"]).to_csv(config.VANILLA_CSV, index=False)
    df.to_csv(config.ENRICHED_CSV, index=False)
    print(f"done -> {config.VANILLA_CSV}, {config.ENRICHED_CSV}", flush=True)


if __name__ == "__main__":
    main()
