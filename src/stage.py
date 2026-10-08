"""Stage the cleaned listings and described corpora as registered MulTaBench datasets.

Every arm/condition pair is one dataset id: a typed `data.parquet` plus
`metadata.json`, laid out by hand in the local Hugging Face cache so the benchmark
reads it offline (nothing is uploaded, no token is used). The benchmark reads
feature types from the stored dtypes, so they are declared here, never inferred:
`room_type` categorical, `is_superhost` boolean, every other tabular column float
and the summary string. An object dtype would be classified as text.

The pipeline logic is the pure `build_datasets`; `main` is a thin shell around it.
"""

from __future__ import annotations

import argparse
import json
import os
import re

import pandas as pd

from src import config

TARGET = "price"
CATEGORICAL = ("room_type",)
BOOLEAN = ("is_superhost",)
COORDINATES = ("latitude", "longitude")
SUMMARY = "surroundings_summary"
INDEX = "index"

# `lat`/`long` as written by clean.py; build.py renames them the same way.
_COORDINATE_RENAMES = {"lat": "latitude", "long": "longitude"}

ARM = "enriched"  # the only arm staged by this verb until the censored arms land
HF_ORG = "multabench"  # namespace the benchmark's hub module resolves ids under
REVISION = "0" * 40  # fixed snapshot name; delete the cache entry to restage
DATA_PARQUET = "data.parquet"
METADATA_JSON = "metadata.json"
SOURCE = "local: GeoEnrich-NYC (src/stage.py)"

# Anchors in the benchmark's dataset registry (multabench/datasets/all_datasets.py).
_ENUM_END = '\n\n\n_IMAGE_PREFIXES = ('
_SOURCES_END = '\n}\n\n\nfor _d in MulTaBenchDatasetID:'


# --- Dataset construction (pure) ---------------------------------------------------
def _typed(df: pd.DataFrame) -> pd.DataFrame:
    """Declare column types explicitly; see the module docstring."""
    out = {}
    for col in df.columns:
        if col in CATEGORICAL:
            out[col] = df[col].astype("category")
        elif col in BOOLEAN:
            out[col] = df[col].astype("boolean" if df[col].isna().any() else bool)
        elif col == SUMMARY:
            out[col] = df[col].astype("string")
        else:
            out[col] = df[col].astype(float)
    return pd.DataFrame(out, index=df.index)


def dataset_key(arm: str, prompt: str, condition: str) -> str:
    return f"{arm}_{prompt}_{condition}"


def build_datasets(
    listings: pd.DataFrame, described: dict[str, pd.DataFrame]
) -> dict[str, pd.DataFrame]:
    """Cleaned listings + one described corpus per prompt -> dataset key -> typed frame.

    `listings` is the cleaned frame (coordinates kept, positional index). Each
    described corpus carries the listing `index` and its summary. All datasets share
    the rows that have a summary in every corpus, in listing order, so the
    benchmark's positional splits line up across arms.
    """
    listings = listings.rename(columns=_COORDINATE_RENAMES)
    rows = None
    for corpus in described.values():
        kept = set(corpus.loc[corpus[SUMMARY].notna(), INDEX])
        rows = kept if rows is None else rows & kept
    rows = sorted(rows or set())
    base = listings.loc[rows].reset_index(drop=True)
    tabular = [c for c in base.columns if c not in COORDINATES]

    datasets = {
        "latlon": _typed(base),
        "structured": _typed(base[tabular]),
    }
    for prompt, corpus in described.items():
        summary = (
            corpus.drop_duplicates(INDEX).set_index(INDEX).loc[rows, SUMMARY].reset_index(drop=True)
        )
        joint = base[tabular].assign(**{SUMMARY: summary})
        datasets[dataset_key(ARM, prompt, "text_only")] = _typed(joint[[SUMMARY, TARGET]])
        datasets[dataset_key(ARM, prompt, "joint")] = _typed(joint)
    return datasets


# --- Registry ----------------------------------------------------------------------
def enum_name(key: str) -> str:
    """Registry member name; the REG_TEXT_ prefix is what makes the benchmark treat it as text."""
    return "REG_TEXT_GEOENRICH_" + re.sub(r"\W", "_", key).upper()


def repo_name(key: str) -> str:
    return "geoenrich-" + re.sub(r"\W", "-", key)


def patch_registry(source: str, keys: list[str]) -> str:
    """Add each key's id and provenance entry to the registry source; a no-op once present.

    Raises when the registry layout no longer has the expected anchors.
    """
    for anchor in (_ENUM_END, _SOURCES_END):
        if source.count(anchor) != 1:
            raise RuntimeError(f"registry anchor not found, benchmark layout changed: {anchor.strip()!r}")
    members, sources = [], []
    for key in keys:
        name = enum_name(key)
        if f"    {name} = " not in source:
            members.append(f'    {name} = "{repo_name(key)}"')
            sources.append(f'    MulTaBenchDatasetID.{name}: "{SOURCE}",')
    if not members:
        return source
    source = source.replace(_ENUM_END, "\n" + "\n".join(members) + _ENUM_END)
    return source.replace(_SOURCES_END, "\n" + "\n".join(sources) + _SOURCES_END)


# --- Local Hugging Face cache --------------------------------------------------------
def snapshot_dir(cache_dir: str, key: str) -> str:
    repo = f"datasets--{HF_ORG}--{repo_name(key)}"
    return os.path.join(cache_dir, repo, "snapshots", REVISION)


def write_snapshots(datasets: dict[str, pd.DataFrame], cache_dir: str) -> list[str]:
    """Lay each dataset out as a hub snapshot; returns the keys written (existing ones are skipped)."""
    written = []
    for key, frame in datasets.items():
        snapshot = snapshot_dir(cache_dir, key)
        if all(os.path.exists(os.path.join(snapshot, f)) for f in (DATA_PARQUET, METADATA_JSON)):
            continue
        os.makedirs(snapshot, exist_ok=True)
        frame.to_parquet(os.path.join(snapshot, DATA_PARQUET), index=False)
        meta = {"target": TARGET, "image_col": None, "task_type": "reg"}
        with open(os.path.join(snapshot, METADATA_JSON), "w") as f:
            json.dump(meta, f)
        refs = os.path.join(os.path.dirname(os.path.dirname(snapshot)), "refs")
        os.makedirs(refs, exist_ok=True)
        with open(os.path.join(refs, "main"), "w") as f:
            f.write(REVISION)
        written.append(key)
    return written


# --- Entry point -----------------------------------------------------------------------
def hub_cache() -> str:
    from huggingface_hub import constants

    return constants.HF_HUB_CACHE


def registry_path() -> str:
    return os.path.join(config.ROOT, "MulTaBench", "multabench", "datasets", "all_datasets.py")


def prompts_with_corpus() -> list[str]:
    prefix, suffix = "airbnb_described_", ".csv"
    names = os.listdir(config.PROCESSED_DIR)
    return sorted(n[len(prefix) : -len(suffix)] for n in names if n.startswith(prefix) and n.endswith(suffix))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", action="append", help="prompt id(s) to stage (default: every described corpus)")
    args = parser.parse_args()

    prompts = args.prompt or prompts_with_corpus()
    if not prompts:
        raise SystemExit("no described corpus found; run describe first")
    described = {}
    for prompt in prompts:
        corpus = pd.read_csv(config.described_csv(prompt), low_memory=False)
        if INDEX not in corpus:
            raise SystemExit(f"{config.described_csv(prompt)} has no `{INDEX}` column; re-run describe")
        described[prompt] = corpus
    datasets = build_datasets(pd.read_csv(config.CLEANED_CSV), described)

    path = registry_path()
    with open(path) as f:
        source = f.read()
    patched = patch_registry(source, list(datasets))
    if patched != source:
        with open(path, "w") as f:
            f.write(patched)
    written = write_snapshots(datasets, hub_cache())
    print(f"{len(datasets)} datasets registered, {len(written)} newly staged in {hub_cache()}")


if __name__ == "__main__":
    main()
