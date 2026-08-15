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
    python prompt_datasets.py --sample 800 --prompts 01_flatten_control 05_price_implication
    python prompt_datasets.py --all                       # full set — costs money
"""
from __future__ import annotations

import argparse
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

# Default screen set: spans the enrichment-style spectrum — dry factual control
# -> neighbourhood character (chosen) -> chain-of-thought -> local-guide role ->
# booking trade-offs. All are system-prompt-only, so describe's user block (the
# deviation render) is untouched. Ids must exist in prompts.toml.
DEFAULT_PROMPTS = [
    "01_factual_control", "03_neighborhood_character", "04_cot_salient",
    "05_local_guide", "07_booking_tradeoffs",
]


def load_prompts(path: str) -> dict[str, str]:
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
    sort_col = "index" if "index" in df.columns else df.columns[0]
    return samp.sort_values(sort_col).reset_index(drop=True)


def enrich_variant(pid: str, system: str, sample_csv: str, outdir: str) -> str:
    """Run describe.py on the fixed sample with `system` as the prompt; return df2 path."""
    described = os.path.join(outdir, f"{pid}.described.csv")

    # patch describe's module state: same sample in, per-prompt cache out, new prompt
    describe.INSTRUCTIONS = system.strip()
    describe.IN_CSV = sample_csv
    describe.OUT_CSV = described
    argv = sys.argv
    sys.argv = ["describe"]           # k=None -> enrich every row of the sample
    try:
        describe.main()
    finally:
        sys.argv = argv

    # describe drops `index` on write; row order is describe's deterministic
    # density sort — identical across variants (same sample), so no realignment.
    out = pd.read_csv(described)
    missing = [c for c in DF2_COLS if c not in out.columns]
    if missing:
        sys.exit(f"[{pid}] described CSV missing columns {missing}")
    df2 = out[DF2_COLS + ["surroundings_summary"]]
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
        chosen[pid] = prompts[pid]["system"]

    df = pd.read_csv(args.in_csv, low_memory=False)
    if "index" not in df.columns:
        sys.exit(f"{args.in_csv} has no `index` column (needed as the stable listing key)")

    # ONE fixed sample, reused for every prompt -> identical listings across variants
    sample = df if args.all else stratified_sample(df, args.sample, args.seed)
    sample_csv = os.path.join(args.outdir, "_sample_enriched.csv")
    sample.to_csv(sample_csv, index=False)
    print(f"sample: {len(sample)} listings (of {len(df)}) -> {sample_csv}", flush=True)
    print(f"prompts: {list(chosen)}", flush=True)
    print(f"LLM calls this run: up to {len(sample) * len(chosen)} "
          f"({len(sample)} listings x {len(chosen)} prompts)\n", flush=True)

    paths = []
    for pid, system in chosen.items():
        print(f"=== {pid} ===", flush=True)
        paths.append(enrich_variant(pid, system, sample_csv, args.outdir))

    print("\n5 prompt-variant datasets — same listings, only surroundings_summary differs:")
    for pth in paths:
        print(f"  {pth}", flush=True)


if __name__ == "__main__":
    main()
