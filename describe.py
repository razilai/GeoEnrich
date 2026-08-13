"""Summarise each listing's surroundings with an LLM (via OpenRouter).

Reads airbnb_enriched.csv (produced by main.py — needs the `surroundings_50m`
column of cleaned POI JSON), asks the model for a short free-text description
of the area, and writes a `surroundings_summary` column to airbnb_described.csv.

Config comes from .env:
    OPENROUTER_API_KEY  — your OpenRouter key
    LLM_MODEL           — model slug, e.g. openai/gpt-4o-mini

Incremental: reuses summaries already in airbnb_described.csv (keyed by listing
`id`) and only calls the LLM for rows still missing one. Rerunning is free once
all summaries exist — so the eval pipeline can consume the CSV without any LLM
inference. Checkpoints every 200 calls, so a crash resumes instead of restarting.

Usage:
    python describe.py         # fill summaries for every listing that has POIs
    python describe.py 10      # only the top 10 by POI count — cheap test run
"""

import json
import os
import sys

import pandas as pd
from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.providers.openrouter import OpenRouterProvider

load_dotenv()

MODEL = os.environ.get("LLM_MODEL", "openai/gpt-4o-mini")
API_KEY = os.environ.get("OPENROUTER_API_KEY")

IN_CSV = "airbnb_enriched.csv"
OUT_CSV = "airbnb_described.csv"

INSTRUCTIONS = (
    "You describe the surroundings of a short-stay rental for a traveller "
    "deciding where to book. You are given the listing's neighbourhood and a "
    "JSON list of points of interest within 50 metres (raw OSM tags). Write "
    "2-4 sentences of natural, specific prose: what the immediate area is like, "
    "what's on the doorstep (cafes, bars, shops, sights, green space), and the "
    "overall feel (lively / quiet / touristy / residential). Name a few notable "
    "places by name. Only use what the data supports — do not invent anything. "
    "Plain text only: no lists, no headings, no markdown."
)


def build_agent():
    if not API_KEY or API_KEY == "sk-or-v1-REPLACE_ME":
        sys.exit("OPENROUTER_API_KEY not set — edit .env")
    model = OpenRouterModel(MODEL, provider=OpenRouterProvider(api_key=API_KEY))
    return Agent(model, output_type=str, instructions=INSTRUCTIONS)


def prompt_for(row):
    return (
        f"Neighbourhood: {row.get('neighbourhood')}\n"
        f"Room type: {row.get('room_type')}\n"
        f"POIs within 50m:\n{row['surroundings_50m']}"
    )


def load_cache():
    """index -> summary from a prior run — so we never re-pay for an existing description.

    Keyed on `index` (the stable per-listing id); `id` was dropped as a leaky column.
    """
    if not os.path.exists(OUT_CSV):
        return {}
    prev = pd.read_csv(OUT_CSV, low_memory=False)
    if "index" not in prev or "surroundings_summary" not in prev:
        return {}
    prev = prev[prev["surroundings_summary"].notna()]
    return dict(zip(prev["index"], prev["surroundings_summary"]))


def save(df):
    # final variant: LLM prose replaces the raw POI JSON — drop surroundings_50m.
    df.drop(columns=["surroundings_50m"]).to_csv(OUT_CSV, index=False)


def main():
    k = int(sys.argv[1]) if len(sys.argv) > 1 else None

    df = pd.read_csv(IN_CSV, low_memory=False)
    # densest surroundings first (cheap test runs hit the richest listings).
    # count derived from JSON on the fly — never a persisted column.
    n_pois = df["surroundings_50m"].map(lambda s: len(json.loads(s)))
    df = df.iloc[n_pois.argsort()[::-1]]
    if k is not None:
        df = df.head(k)

    # reuse any summary already computed in a prior OUT_CSV; only NULLs hit the LLM.
    df["surroundings_summary"] = df["index"].map(load_cache())
    todo = df[df["surroundings_summary"].isna()]
    print(f"{len(df)} listings — {len(todo)} need the LLM, "
          f"{len(df) - len(todo)} cached (model {MODEL})", flush=True)

    if not todo.empty:
        agent = build_agent()  # only touch the API (+ require a key) when work remains
        for i, (idx, row) in enumerate(todo.iterrows(), 1):
            res = agent.run_sync(prompt_for(row))
            df.at[idx, "surroundings_summary"] = res.output
            print(f"[{i}/{len(todo)}] {row.get('neighbourhood')} (idx {row['index']})", flush=True)
            if i % 200 == 0:
                save(df)  # checkpoint — a crash resumes from here, not from scratch

    save(df)
    print(f"done -> {OUT_CSV}", flush=True)


if __name__ == "__main__":
    main()
