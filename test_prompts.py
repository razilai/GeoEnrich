"""Try 10 JSON->text enrichment prompts on 3 listings; write results for review.

Goal: keep the description factual/grounded in the listed POI names, but let the
model's world knowledge surface high/low-value signal (e.g. a place next to the
Rijksmuseum reads premium). Spectrum spans a dry attribute-flatten control (the
research-favoured baseline) through role, landmark-salience, hedonic price-signal,
chain-of-thought, structured-tag, desirability-score and few-shot variants.

Run:  .venv/bin/python test_prompts.py   ->  prompt_experiments.json
"""
import asyncio
import json
import os
from collections import Counter

import pandas as pd
from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.providers.openrouter import OpenRouterProvider

load_dotenv()
MODEL = os.environ.get("EXP_MODEL", "openai/gpt-4o-mini")  # sync + fast for the trial
API_KEY = os.environ["OPENROUTER_API_KEY"]
LISTINGS = [1671, 567, 0]  # premium+dense / famous-landmark-but-sparse / budget-mundane


def facts(pois):
    """Deduped place-type counts + name list (attribute-flatten input, shared by all)."""
    seen, uniq = set(), []
    for p in pois:
        nm = p.get("name") if isinstance(p, dict) else None
        if nm and nm in seen:
            continue
        if nm:
            seen.add(nm)
        uniq.append(p)
    cats = Counter(p.get("amenity") or p.get("shop") or p.get("tourism") or p.get("leisure")
                   for p in uniq if isinstance(p, dict))
    names = sorted(seen)
    return dict(cats), names


def user_block(pois):
    cats, names = facts(pois)
    return f"Place types (count): {cats}\nPlaces: {', '.join(names) or '(none)'}"


GROUND = ("Name only places from the list, copied exactly; never invent a place. "
          "Plain text, no markdown.")

# few-shot demonstrations (teach the value mapping explicitly)
FEWSHOT = (
    "Example A —\n"
    "Place types (count): {'museum': 1, 'cafe': 1}\nPlaces: Rijksmuseum, Café Blom\n"
    "Output: Sits beside the Rijksmuseum, one of the country's major museums, with "
    "Café Blom alongside — a prestigious, high-demand cultural location.\n\n"
    "Example B —\n"
    "Place types (count): {'dentist': 1}\nPlaces: Zeeburg Tandartsen\n"
    "Output: Only a dentist, Zeeburg Tandartsen, sits nearby — a quiet, ordinary "
    "residential spot with little of draw.\n\n"
    "Now do the same for:\n"
)

PROMPTS = [
    dict(id="01_flatten_control",
         name="Flatten control (dry)",
         desc="Research-favoured baseline: factual types+names, no judgment. Control.",
         system="You convert a list of nearby places into one plain factual sentence: "
                "the main place types plus a few named examples. No opinions or "
                "adjectives of quality. " + GROUND),
    dict(id="02_local_expert",
         name="Local-expert role",
         desc="Role prompt: an Amsterdam insider judges high-end vs budget from the names.",
         system="You are a long-time Amsterdam local who knows which streets are "
                "prestigious and which are ordinary. In two sentences, name what's "
                "nearby and say whether these surroundings suggest a high-end, mid, or "
                "budget area, justified by the specific named places. " + GROUND),
    dict(id="03_landmark_salience",
         name="Landmark salience",
         desc="Factual, but flags any famous landmark/museum/market and its significance.",
         system="Describe the nearby places by type with named examples. If any is a "
                "well-known landmark, museum, market, park or attraction, say so and "
                "note in a few words why it matters for the area. " + GROUND),
    dict(id="04_desirability_verdict",
         name="Fact + tier verdict",
         desc="One factual sentence, then a clause: premium / average / budget.",
         system="Write one factual sentence naming the nearby places, then one short "
                "clause stating whether the immediate location is premium, average, or "
                "budget for a short-stay rental, based on what is nearby. " + GROUND),
    dict(id="05_price_implication",
         name="Hedonic price-signal",
         desc="Ties surroundings to what they imply about rental price level, with why.",
         system="Describe what is nearby using the named places, and state what it "
                "implies about the area's rental price level (higher or lower than "
                "average) and why, referencing specific named places. " + GROUND),
    dict(id="06_cot_then_desc",
         name="Reasoning then summary",
         desc="Notes notable places + implication first, then a 2-sentence summary.",
         system="First write a line 'REASONING:' noting any notable or prestigious "
                "named places and what they imply about desirability. Then a line "
                "'SUMMARY:' with two factual sentences naming places and their value "
                "implication. " + GROUND),
    dict(id="07_tagged_output",
         name="Prose + machine tag",
         desc="1-2 factual sentences, then a parseable [tier|anchors|character] tag.",
         system="Write 1-2 factual sentences naming nearby places. Then append one tag "
                "line exactly as: [tier: premium|mid|budget | anchors: <named notable "
                "places or 'none'> | character: <2-4 words>]. " + GROUND),
    dict(id="08_desirability_score",
         name="Desirability score /5",
         desc="Description + explicit 1-5 location-desirability score with justification.",
         system="Describe the nearby named places in one sentence. Then add "
                "'Desirability: X/5 — <justification>', where famous landmarks, museums "
                "and markets raise the score and mundane services lower it, justified by "
                "the specific named places. " + GROUND),
    dict(id="09_few_shot",
         name="Few-shot value mapping",
         desc="Two worked examples (landmark->premium, dentist->budget) then the target.",
         system="Describe the nearby places in 1-2 sentences, naming them and conveying "
                "whether the spot is prestigious/high-demand or ordinary/low-draw, in the "
                "style of the examples. " + GROUND,
         fewshot=True),
    dict(id="10_tourist_vs_resident",
         name="Tourist vs residential",
         desc="Says whether area reads tourist-prominent (landmarks) or everyday, with draws.",
         system="Name the nearby places by type, then state whether the area reads as "
                "tourist-prominent (famous landmarks or attractions nearby) or "
                "everyday-residential, naming the specific draws that make it so. " + GROUND),
]


async def main():
    df = pd.read_csv("airbnb_enriched.csv", low_memory=False).set_index("index")
    provider = OpenRouterProvider(api_key=API_KEY)
    model = OpenRouterModel(MODEL, provider=provider)
    sem = asyncio.Semaphore(10)

    listing_meta, pois_by_idx = [], {}
    for idx in LISTINGS:
        row = df.loc[idx]
        pois = json.loads(row["surroundings_50m"])
        pois_by_idx[idx] = pois
        _, names = facts(pois)
        listing_meta.append(dict(index=int(idx), price=int(row["price"]),
                                 neighbourhood=row["neighbourhood"], n_named=len(names),
                                 names=names))

    async def run(system, user):
        agent = Agent(model, output_type=str, instructions=system,
                      model_settings={"temperature": 0.4})
        async with sem:
            res = await agent.run(user)
        return res.output

    out = {"model": MODEL, "listings": listing_meta, "prompts": []}
    for p in PROMPTS:
        tasks = []
        for idx in LISTINGS:
            ub = user_block(pois_by_idx[idx])
            if p.get("fewshot"):
                ub = FEWSHOT + ub
            tasks.append(run(p["system"], ub))
        results = await asyncio.gather(*tasks)
        out["prompts"].append(dict(
            id=p["id"], name=p["name"], description=p["desc"], system=p["system"],
            results={str(idx): r for idx, r in zip(LISTINGS, results)}))
        print(f"✓ {p['id']}", flush=True)

    out_path = os.environ.get("EXP_OUT", "prompt_experiments.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
