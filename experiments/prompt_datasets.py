"""Generate N prompt-variant datasets over ONE fixed set of Amsterdam listings.

Screen: which enrichment prompt yields the strongest joint-signal, BEFORE paying
to enrich the full dataset. Every output has identical rows and identical tabular
columns — the ONLY difference is the `surroundings_summary` text (prompt varies).

Reuses describe.py end-to-end (POI->user-block, batching, caching, cost print):
we monkeypatch `describe.INSTRUCTIONS` per prompt and point its IN/OUT csvs at one
fixed sample, so all variants see the exact same listings.

Does NOT run build.py / re-enrich POIs — consumes an existing enriched CSV as-is.
Never runs on the full dataset unless you pass --all (explicit, costs $$).

Output (one per prompt), same schema as the evaluated df2.csv:
    <outdir>/df2_<promptid>.csv

Usage:
    python prompt_datasets.py --sample 800
    python prompt_datasets.py --sample 800 --prompts 03_neighborhood_character 05_local_guide
    python prompt_datasets.py --all                       # full set — costs money
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tomllib

import pandas as pd

from airbnb_surroundings import config, describe

_HERE = os.path.dirname(os.path.abspath(__file__))

# Columns of the evaluated df2.csv, in order. `price` is the eval target; the rest
# are the clean tabular features. Dropped: `index` (id) and `surroundings` (JSON,
# already stripped by describe), plus `details` — the listing's own free text, a
# leaky confound for the surroundings-text channel we're screening. Every variant
# is apples-to-apples: identical rows and identical tabular columns, only the
# `surroundings_summary` text differs.
DF2_COLS = [
    "price", "room_type", "ratings", "guests", "beds", "bedrooms", "bathrooms",
    "property_number_of_reviews", "is_superhost", "num_bedrooms", "num_baths",
    "num_rooms",
]

# Default screen set: three complementary Overture evidence views. Every variant
# uses the same listings and structured columns, but receives a different slice
# of the surrounding-place record rather than a paraphrase prompt.
DEFAULT_PROMPTS = [
    "14_contrast_access", "15_contrast_local_profile", "16_contrast_anchor_profile",
]


def load_prompts(path: str) -> dict[str, dict]:
    with open(path, "rb") as f:
        toml = tomllib.load(f)
    return {k.split(".", 1)[-1] if "." in k else k: v
            for k, v in toml.get("prompts", {}).items()}


def stratified_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """~n rows, proportional per stratum (keeps the mix representative).

    Stratifies on `neighbourhood` if present, else `room_type`; falls back to a
    plain random draw if neither exists. Sorted by `index` for a stable key.
    """
    if n >= len(df):
        return df
    strat = next((c for c in ("neighbourhood", "room_type") if c in df.columns), None)
    if strat is None:
        samp = df.sample(n=n, random_state=seed)
    else:
        samp = df.groupby(strat, group_keys=False).sample(
            frac=n / len(df), random_state=seed)
        # Fractional group sampling rounds per group, so it can miss the requested
        # total by a row or two. Fill from the unsampled rows (or trim deterministically)
        # to make --sample N mean exactly N records.
        if len(samp) < n:
            remaining = df.loc[~df.index.isin(samp.index)]
            extra = remaining.sample(n=n - len(samp), random_state=seed)
            samp = pd.concat([samp, extra])
        elif len(samp) > n:
            samp = samp.sample(n=n, random_state=seed)
    sort_col = "index" if "index" in df.columns else df.columns[0]
    return samp.sort_values(sort_col).reset_index(drop=True)


def enrich_variant(
    pid: str, prompt: dict, sample_csv: str, outdir: str, reference_csv: str | None
) -> str:
    """Run describe.py on a fixed sample with one configured view and prompt."""
    described = os.path.join(outdir, f"{pid}.described.csv")
    view = prompt.get("view", "character_deviation")
    if view not in describe._VIEWS:
        sys.exit(f"prompt '{pid}' has unsupported view '{view}'")

    # Patch describe's module state: same sample in, per-prompt cache out, and a
    # deliberately distinct Overture information view.
    reference_stat = os.stat(reference_csv) if reference_csv else None
    cache_material = "\0".join(
        [
            prompt["system"].strip(),
            view,
            describe._VIEW_CACHE_VERSIONS.get(view, "v1"),
            os.path.abspath(reference_csv or sample_csv),
            str(reference_stat.st_size if reference_stat else 0),
            str(reference_stat.st_mtime_ns if reference_stat else 0),
        ]
    )
    cache_tag = hashlib.sha256(cache_material.encode()).hexdigest()[:16]
    previous = (
        describe.INSTRUCTIONS,
        describe.IN_CSV,
        describe.OUT_CSV,
        describe.SURR_VIEW,
        describe.REFERENCE_CSV,
        describe.CACHE_TAG,
    )
    describe.INSTRUCTIONS = prompt["system"].strip()
    describe.IN_CSV = sample_csv
    describe.OUT_CSV = described
    describe.SURR_VIEW = view
    describe.REFERENCE_CSV = reference_csv
    describe.CACHE_TAG = cache_tag
    argv = sys.argv
    sys.argv = ["describe"]           # k=None -> enrich every row of the sample
    try:
        describe.main()
    finally:
        sys.argv = argv
        (
            describe.INSTRUCTIONS,
            describe.IN_CSV,
            describe.OUT_CSV,
            describe.SURR_VIEW,
            describe.REFERENCE_CSV,
            describe.CACHE_TAG,
        ) = previous

    # describe drops `index` on write; row order is describe's deterministic
    # density sort — identical across variants (same sample), so no realignment.
    out = pd.read_csv(described)
    # Preview inputs can contain a reduced structured schema. Keep every canonical
    # field that is present; evaluation can use that stable subset directly.
    selected_cols = [c for c in DF2_COLS if c in out.columns]
    missing = [c for c in DF2_COLS if c not in out.columns]
    if missing:
        print(f"[{pid}] preview is missing structured columns: {missing}", flush=True)
    df2 = out[selected_cols + ["surroundings_summary"]]
    df2_path = os.path.join(outdir, f"df2_{pid}.csv")
    df2.to_csv(df2_path, index=False)
    return df2_path


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--sample", type=int, help="stratified N-row screen (cheap)")
    g.add_argument("--all", action="store_true", help="full dataset — costs money")
    p.add_argument("--in", dest="in_csv", default=config.ENRICHED_CSV)
    p.add_argument(
        "--reference-in",
        help="full enriched corpus for citywide deviation percentiles (defaults to --in)",
    )
    p.add_argument("--prompts", nargs="+", default=DEFAULT_PROMPTS)
    p.add_argument("--toml", default=os.path.join(_HERE, "prompts.toml"))
    p.add_argument("--outdir", default=os.path.join(_HERE, "screen"))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    prompts = load_prompts(args.toml)

    chosen = {}
    for pid in args.prompts:
        if pid not in prompts:
            sys.exit(f"prompt id '{pid}' not in {args.toml} ({sorted(prompts)})")
        if prompts[pid].get("fewshot"):
            sys.exit(f"prompt '{pid}' is few-shot (needs demo block) — not supported here")
        if "view" not in prompts[pid]:
            sys.exit(f"prompt '{pid}' has no Overture input view")
        chosen[pid] = prompts[pid]

    df = pd.read_csv(args.in_csv, low_memory=False)
    if "index" not in df.columns:
        sys.exit(f"{args.in_csv} has no `index` column (needed as the stable listing key)")

    # ONE fixed sample, reused for every prompt -> identical listings across variants
    sample = df if args.all else stratified_sample(df, args.sample, args.seed)
    sample_csv = os.path.join(args.outdir, "_sample_enriched.csv")
    # Do not overwrite a user-supplied source preview when screening a smaller
    # subset from it; write that derived subset alongside the source instead.
    if os.path.abspath(sample_csv) == os.path.abspath(args.in_csv) and len(sample) < len(df):
        sample_csv = os.path.join(args.outdir, f"_sample_{len(sample)}_enriched.csv")
    sample.to_csv(sample_csv, index=False)
    print(f"sample: {len(sample)} listings (of {len(df)}) -> {sample_csv}", flush=True)
    print(f"prompts: {list(chosen)}", flush=True)
    print(f"LLM calls this run: up to {len(sample) * len(chosen)} "
          f"({len(sample)} listings x {len(chosen)} prompts)\n", flush=True)

    paths = []
    for pid, prompt in chosen.items():
        print(f"=== {pid} ===", flush=True)
        paths.append(
            enrich_variant(
                pid, prompt, sample_csv, args.outdir, args.reference_in or args.in_csv
            )
        )

    print(f"\n{len(paths)} prompt-variant datasets — same listings, "
          "only surroundings_summary differs:")
    for pth in paths:
        print(f"  {pth}", flush=True)


if __name__ == "__main__":
    main()
