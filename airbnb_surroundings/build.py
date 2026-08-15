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

import csv
import json
import os
import sys

import duckdb
import geopandas as gpd
import pandas as pd
from rapidfuzz import fuzz as rf_fuzz
from rapidfuzz import process as rf_process

from airbnb_surroundings import config
from airbnb_surroundings.config import (
    DOORSTEP,
    FUZZ_MIN,
    LANDMARK_MIN_CONF,
    LANDMARK_TARGET,
    MIN_CONF,
    NYC_UTM,
    RADIUS,
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
    "accommodation",  # hotel/lodging density = tourist-district signal. NOTE:
    # includes vacation_rental / service_apartments (competitor short-term rentals)
    # — a potential target leak, kept intentionally at the user's request.
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
_TAXONOMY_CSV = os.path.join(os.path.dirname(__file__), "overture_categories.csv")
_LANDMARKS_JSON = os.path.join(os.path.dirname(__file__), "landmarks.json")


def _norm_name(s):
    """Normalize a place name for landmark matching: lowercase, strip, drop 'the'."""
    s = (s or "").lower().strip()
    return s[4:] if s.startswith("the ") else s


def _load_landmarks():
    """Curated landmark names (landmarks.json) → (canonical list, normalized list).

    Overture buries icons like the Empire State Building in the generic
    landmark_and_historical_building leaf (confidence-1.0 apartment towers sit in
    the same leaf), so category+confidence alone can't surface them. Names are
    fuzzy-matched against this hand list; intended to be LLM-maintained later.
    """
    with open(_LANDMARKS_JSON, encoding="utf-8") as f:
        names = json.load(f)["landmarks"]
    return names, [_norm_name(n) for n in names]


LANDMARKS, _LANDMARK_NORMS = _load_landmarks()


def _leaf_paths():
    """leaf category code -> full Overture taxonomy path [group, subcat, ..., leaf]."""
    m = {}
    with open(_TAXONOMY_CSV, encoding="utf-8-sig") as f:
        for row in csv.reader(f, delimiter=";"):
            if len(row) < 2 or row[0].strip() == "Category code":
                continue
            m[row[0].strip()] = [x.strip() for x in row[1].strip().strip("[]").split(",")]
    return m


_LEAF_PATH = _leaf_paths()
_LEAF_GROUP = {leaf: p[0] for leaf, p in _LEAF_PATH.items()}  # leaf -> top-level group


def allowed_categories():
    """Leaf codes worth keeping: any leaf whose group is in KEEP_GROUPS, plus the
    hand-picked transit leaves."""
    keep = set(TRANSIT_LEAVES)
    keep |= {c for c, g in _LEAF_GROUP.items() if g in KEEP_GROUPS}
    return keep


def is_landmark(nm, cat, conf):
    """If this POI is a curated landmark, return its CANONICAL name, else None.

    Fuzzy name match (rapidfuzz WRatio >= FUZZ_MIN) against landmarks.json, gated
    by high confidence (>= LANDMARK_MIN_CONF) and the attractions_and_activities
    group, so a low-confidence or same-named non-attraction (e.g. "Empire State
    Building Gym") is not matched. Returns the clean list name, not the OSM
    variant, so output is consistent and dedupes across spellings."""
    if conf < LANDMARK_MIN_CONF or _LEAF_GROUP.get(cat) != "attractions_and_activities":
        return None
    m = rf_process.extractOne(
        _norm_name(nm), _LANDMARK_NORMS, scorer=rf_fuzz.WRatio, score_cutoff=FUZZ_MIN
    )
    return LANDMARKS[m[2]] if m else None


# --- signal buckets -------------------------------------------------------
# Roll the 853 fine leaves up to ~13 price-relevant buckets. Deterministic:
# every leaf lands by exact group / 2nd-level / set membership (no fuzzy match).
# Hand-curated pieces are only: this routing, GROCERY_LEAVES, PARK_SUBS,
# CULTURE_SUBS (grocery hides under the `shopping` 2nd-level, so the taxonomy
# alone can't separate it). Everything else auto-routes from the taxonomy.
GROCERY_LEAVES = {
    "grocery_store", "asian_grocery_store", "indian_grocery_store",
    "international_grocery_store", "japanese_grocery_store", "korean_grocery_store",
    "kosher_grocery_store", "mexican_grocery_store", "organic_grocery_store",
    "russian_grocery_store", "specialty_grocery_store", "ethical_grocery",
    "supermarket", "convenience_store", "delicatessen", "farmers_market",
    "public_market", "health_market", "seafood_market", "butcher", "greengrocer",
    "health_food_store", "bodega",
}
PARK_SUBS = {  # attractions 2nd-level values that are green/open space
    "park", "botanical_garden", "beach", "plaza", "trail", "national_park",
    "state_park", "memorial_park",
}
CULTURE_SUBS = {"museum", "art_gallery", "cultural_center"}


def bucket(leaf):
    """Map a fine leaf category to its coarse signal bucket."""
    g = _LEAF_GROUP.get(leaf, leaf)
    p = _LEAF_PATH.get(leaf, [leaf])
    sub = p[1] if len(p) > 1 else g
    if leaf in TRANSIT_LEAVES:
        return "transit"
    if leaf in GROCERY_LEAVES:
        return "grocery"
    if g == "accommodation":
        return "lodging"
    if g == "eat_and_drink":
        return {"bar": "nightlife", "cafe": "cafe"}.get(sub, "dining")
    if g == "retail":
        return "pharmacy" if sub in ("pharmacy", "drugstore") else "shopping"
    if g == "active_life":
        return "fitness_sport"
    if g == "arts_and_entertainment":
        return "entertainment"
    if g == "attractions_and_activities":
        if sub in PARK_SUBS:
            return "park_green"
        if sub in CULTURE_SUBS:
            return "culture"
        return "landmark"
    return g

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

    joined["nm"] = joined["name"].fillna("")
    joined = joined.sort_values("dist")  # nearest-first (nearest-per-bucket, landmark order)

    # Aggregate per listing into a compact signal record (no per-POI list, no
    # caps — density is the signal, not truncated):
    #   cats:      bucket -> [count<=150m, count<=400m, nearest_m]
    #   landmarks: [[name, dist_m], ...]  (major landmarks only, name-bearing)
    labels = {}
    for lid, grp in joined.groupby(level=0):
        cats = {}  # bucket -> [c150, c400, nearest_m]
        lm = {}  # curated landmark canonical name -> nearest dist_m
        fb = {}  # uncurated named attraction (raw name) -> nearest dist_m
        for cat, d, nm, conf in zip(
            grp["category"], grp["dist"], grp["nm"], grp["confidence"]
        ):
            b = bucket(cat)
            dm = round(float(d))
            e = cats.get(b)
            if e is None:
                e = cats[b] = [0, 0, dm]
            e[1] += 1  # within RADIUS (all joined rows are)
            if d <= DOORSTEP:
                e[0] += 1
            if dm < e[2]:
                e[2] = dm
            canon = is_landmark(nm, cat, conf) if nm else None
            if canon and (canon not in lm or dm < lm[canon]):
                lm[canon] = dm
            # top-up pool: named attractions that aren't on the curated list
            elif nm and _LEAF_GROUP.get(cat) == "attractions_and_activities":
                if nm not in fb or dm < fb[nm]:
                    fb[nm] = dm
        landmarks = sorted(([n, dd] for n, dd in lm.items()), key=lambda x: x[1])
        # fill empty/thin landmark slots with nearest named attractions so the
        # high-signal proper-noun channel is not wasted (curated names first)
        if len(landmarks) < LANDMARK_TARGET:
            extra = sorted(([n, dd] for n, dd in fb.items()), key=lambda x: x[1])
            landmarks += extra[: LANDMARK_TARGET - len(landmarks)]
        labels[lid] = {"cats": cats, "landmarks": landmarks}

    recs = [labels.get(i, {"cats": {}, "landmarks": []}) for i in df.index]
    df["surroundings"] = [json.dumps(x, ensure_ascii=False) for x in recs]

    # drop listings with no POIs within RADIUS
    before = len(df)
    df = df[[bool(x["cats"]) for x in recs]]
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
