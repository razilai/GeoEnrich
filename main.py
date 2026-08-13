"""Enrich Airbnb listings with OSM POIs within a 50m radius.

Data spans 4 metros (Paris / NYC / SF / Amsterdam). Each maps to one .pbf.
For every listing we list the OSM POIs whose geometry falls within RADIUS
meters, then write one new column:
    - surroundings_50m : JSON array of cleaned POI tag dicts (descriptive
                         tags only), ready to feed an LLM for a text summary
Listings with zero POIs in radius are dropped (count used internally, never
persisted as a column).
"""

import json
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
from pyrosm import OSM

RADIUS = 50  # meters

# center = rough metro center (for nearest-city assignment)
# utm    = metric CRS so buffer() is in real meters
CITIES = {
    "france": dict(pbf="data/france.pbf", center=(48.86, 2.35), utm=32631),
    "nyc": dict(pbf="data/nyc.pbf", center=(40.71, -74.00), utm=32618),
    "norcal": dict(pbf="data/norcal.pbf", center=(37.77, -122.42), utm=32610),
    "holland": dict(pbf="data/holland.pbf", center=(52.37, 4.90), utm=32631),
}

ONLY = sys.argv[1] if len(sys.argv) > 1 else None  # optional: test one city


# pyrosm metadata columns — everything else is a real OSM tag
_META = {
    "visible",
    "tags",
    "lat",
    "changeset",
    "lon",
    "timestamp",
    "version",
    "id",
    "geometry",
    "osm_type",
}

# tags that tell an LLM what a place *is* / what it's like near the property.
# everything else (refs, ids, edit metadata, contact info, wikidata, ...) is
# dropped — it adds tokens and noise without shaping the free-text summary.
KEEP_TAGS = {
    "name",
    "amenity",
    "shop",
    "tourism",
    "leisure",
    "craft",
    "office",
    "healthcare",
    "sport",
    "cuisine",
    "outdoor_seating",
    "brand",
    "description",
    "wheelchair",
}

# a POI's category comes from the first of these that's present
CATEGORY_KEYS = ("amenity", "shop", "tourism", "leisure", "craft", "office")

# category values that are street furniture / infrastructure — say nothing
# about the neighbourhood feel, so the whole POI is dropped
JUNK_VALUES = {
    "yes",
    "parking",
    "parking_space",
    "parking_entrance",
    "bicycle_parking",
    "motorcycle_parking",
    "bench",
    "waste_basket",
    "waste_disposal",
    "recycling",
    "vending_machine",
    "atm",
    "drinking_water",
    "post_box",
    "telephone",
    "charging_station",
    "bicycle_repair_station",
    "grit_bin",
    "hunting_stand",
    "clock",
    "fire_hydrant",
    "surveillance",
    "shelter",
    "street_lamp",
    "toilets",
}


def clean_poi(tags):
    """Reduce a raw tag dict to property-relevant fields, or None to drop it."""
    cat = next((tags[k] for k in CATEGORY_KEYS if k in tags), None)
    if cat is None or cat in JUNK_VALUES:
        return None
    kept = {k: v for k, v in tags.items() if k in KEEP_TAGS and v not in JUNK_VALUES}
    return kept or None


def poi_records(pois):
    """One cleaned tag dict per POI (or None for street furniture / empty).

    pyrosm promotes common tags to columns and dumps the rest into a JSON
    string in `tags`; we merge both, then keep only descriptive fields.
    """
    tag_cols = [c for c in pois.columns if c not in _META]
    base = pois[tag_cols].to_dict("records")
    extra = pois["tags"].tolist()

    out = []
    for b, ex in zip(base, extra):
        tags = {k: v for k, v in b.items() if pd.notna(v)}
        if isinstance(ex, str) and ex:
            try:
                tags.update(json.loads(ex))
            except ValueError:
                pass
        out.append(clean_poi(tags))
    return out


# leaky / ID columns dropped up front (in memory) so no output variant ever
# carries them — they pollute the eval's structured baseline (high-cardinality
# codes ≈ row ids). Source airbnb.csv is untouched. `index` is kept as the
# stable per-listing key (used by describe.py's incremental cache).
DROP_COLS = ["id", "name", "host_id", "host_name", "license", "last_review"]


def main():
    df = pd.read_csv("airbnb.csv", low_memory=False).reset_index(drop=True)
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    # assign each listing to the nearest metro center (cities are far apart,
    # so squared lat/lon distance is enough to separate them cleanly)
    names = list(CITIES)
    centers = np.array([CITIES[n]["center"] for n in names])
    pts = df[["latitude", "longitude"]].to_numpy()
    d2 = ((pts[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    df["city"] = [names[i] for i in d2.argmin(axis=1)]

    labels = {}  # listing index -> list[dict] of POI records within RADIUS

    for name, cfg in CITIES.items():
        if ONLY and name != ONLY:
            continue
        sub = df[df.city == name]
        if sub.empty:
            continue
        print(f"[{name}] {len(sub)} listings — loading {cfg['pbf']}", flush=True)

        # only parse the PBF around the listings (crops during read, not after)
        pad = 0.02  # ~2km, comfortably covers the 50m radius at the edges
        bbox = [
            sub.longitude.min() - pad,
            sub.latitude.min() - pad,
            sub.longitude.max() + pad,
            sub.latitude.max() + pad,
        ]
        pois = OSM(cfg["pbf"], bounding_box=bbox).get_pois(
            custom_filter={
                "amenity": True,
                "shop": True,
                "tourism": True,
                "leisure": True,
            }
        )
        if pois is None or pois.empty:
            print(f"[{name}] no POIs", flush=True)
            continue
        pois = pois[pois.geometry.notna()].copy()
        pois["poi"] = poi_records(pois)
        pois = pois[pois["poi"].notna()]  # drop street furniture / empty
        pois = pois[["poi", "geometry"]].to_crs(cfg["utm"])
        print(f"[{name}] {len(pois)} POIs", flush=True)

        # listings -> 50m buffers in the metric CRS (need ALL POIs in radius,
        # not just the nearest, so buffer + intersects, not sjoin_nearest)
        g = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(sub.longitude, sub.latitude),
            index=sub.index,
            crs=4326,
        ).to_crs(cfg["utm"])
        g["geometry"] = g.geometry.buffer(RADIUS)

        joined = gpd.sjoin(g, pois, predicate="intersects", how="left")
        matched = joined[joined["poi"].notna()]
        agg = matched.groupby(level=0)["poi"].apply(list)
        labels.update(agg.to_dict())
        print(f"[{name}] matched {len(agg)} listings", flush=True)

    recs = df.index.map(labels)  # list[dict] per listing, or NaN
    n_pois = [len(x) if isinstance(x, list) else 0 for x in recs]  # local only, never a column
    df["surroundings_50m"] = [
        json.dumps(x, ensure_ascii=False) if isinstance(x, list) else "[]" for x in recs
    ]

    # drop listings with no POIs within RADIUS
    before = len(df)
    df = df[[n > 0 for n in n_pois]]
    print(f"dropped {before - len(df)} empty listings, kept {len(df)}", flush=True)

    # two variants: vanilla = tabular only (no bulky JSON, model-ready);
    # json = vanilla + the surroundings_50m POI JSON (input for describe.py).
    vanilla = "airbnb_vanilla.csv"
    enriched = "airbnb_enriched.csv"
    df.drop(columns=["surroundings_50m"]).to_csv(vanilla, index=False)
    df.to_csv(enriched, index=False)
    print(f"done -> {vanilla}, {enriched}", flush=True)


if __name__ == "__main__":
    main()
