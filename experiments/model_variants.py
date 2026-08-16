"""Render the SAME 10 random listings with ONE prompt across 3 model variants.

Companion to prompt_datasets.py, but the axis is flipped: prompt is FIXED
(08_local_guide_exact, needs SURR_VIEW=deviation_exact) and the MODEL varies.
Goal: see how much the surroundings prose moves with the model alone — cheap,
before paying to enrich the full set with any one of them.

Same rendering as the real pipeline: reuses describe.prompt_for under
SURR_VIEW=deviation_exact so the USER block each model sees is byte-identical to
what describe.py would send. Only the model (and its thinking config) differs.

Thinking models: two of the three slugs reason before answering. They are handled
differently from the plain model:
  - reasoning ENABLED (openrouter effort=medium); the plain model has it disabled,
  - temperature is NOT sent (many reasoning models pin it to 1 / reject <1); the
    plain model keeps the pipeline's 0.4,
  - the chain-of-thought lands in a separate ThinkingPart, so res.output stays
    clean prose — we capture the reasoning alongside for inspection, it never
    leaks into the summary.

Models are read from slugs.txt (one slug per line). A `:batch` suffix is stripped
here — this is a 30-call interactive inspection, so it runs synchronous, not via
the async batch endpoint.

Run:  python experiments/model_variants.py            -> model_variants.json
      N_LISTINGS=10 EXP_SEED=0 python experiments/model_variants.py
"""
from __future__ import annotations

import asyncio
import json
import os
import tomllib

import pandas as pd
from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse, ThinkingPart
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.providers.openrouter import OpenRouterProvider

from airbnb_surroundings import config, describe

load_dotenv()

_HERE = os.path.dirname(os.path.abspath(__file__))
API_KEY = os.environ["OPENROUTER_API_KEY"]
SLUGS_TXT = os.path.join(_HERE, "slugs.txt")
PROMPTS_TOML = os.path.join(_HERE, "prompts.toml")
PROMPT_ID = os.environ.get("EXP_PROMPT", "08_local_guide_exact")
SURR_VIEW = "deviation_exact"          # the view 08's system prompt is written for
N_LISTINGS = int(os.environ.get("N_LISTINGS", "10"))
SEED = int(os.environ.get("EXP_SEED", "0"))

# Which slugs reason before answering. Gemini 3.x flash and GPT-5.x are thinking
# models; gemini-2.5-flash-lite is a plain (hybrid) model we run with thinking off.
def is_thinking(base_slug: str) -> bool:
    s = base_slug.lower()
    return "gpt-5" in s or ("gemini-3" in s and "lite" not in s)


def read_slugs(path: str) -> list[str]:
    with open(path) as f:
        return [ln.strip() for ln in f if ln.strip()]


def load_prompt_system(pid: str) -> str:
    with open(PROMPTS_TOML, "rb") as f:
        toml = tomllib.load(f)
    prompts = toml.get("prompts", {})
    if pid not in prompts:
        raise SystemExit(f"prompt '{pid}' not in {PROMPTS_TOML} ({sorted(prompts)})")
    return prompts[pid]["system"].strip()


def model_settings_for(thinking: bool) -> dict:
    """Reasoning config + temperature policy per model kind (see module docstring)."""
    if thinking:
        # reason, but keep the raw CoT out of the answer text (still captured via
        # ThinkingPart); no temperature — let the reasoning model use its default.
        return {"openrouter_reasoning": {"effort": "medium"}}
    return {
        "temperature": float(os.environ.get("LLM_TEMPERATURE", "0.4")),
        "openrouter_reasoning": {"enabled": False},
    }


async def run_one(model, settings, system, user) -> tuple[str, str, dict]:
    """One call; return (prose, reasoning_text, usage). Reasoning is a separate
    ThinkingPart, so `res.output` is the clean summary with no CoT bleed."""
    agent = Agent(model, output_type=str, instructions=system, model_settings=settings)
    res = await agent.run(user)
    reasoning = "".join(
        p.content
        for m in res.all_messages() if isinstance(m, ModelResponse)
        for p in m.parts if isinstance(p, ThinkingPart)
    )
    u = res.usage() if callable(res.usage) else res.usage
    usage = {
        "input_tokens": getattr(u, "input_tokens", None) or getattr(u, "request_tokens", 0),
        "output_tokens": getattr(u, "output_tokens", None) or getattr(u, "response_tokens", 0),
    }
    return res.output, reasoning, usage


async def main():
    df = pd.read_csv(config.ENRICHED_CSV, low_memory=False)
    describe.load_reference(df)         # corpus percentiles for the deviation_exact view
    describe.SURR_VIEW = SURR_VIEW      # prompt_for reads this at call time

    sample = df.sample(n=min(N_LISTINGS, len(df)), random_state=SEED)
    sample = sample.sort_values("index").reset_index(drop=True)

    system = load_prompt_system(PROMPT_ID)
    slugs = read_slugs(SLUGS_TXT)
    provider = OpenRouterProvider(api_key=API_KEY)

    # render the USER block once per listing — identical across every model
    listings = []
    for _, row in sample.iterrows():
        listings.append(dict(
            index=int(row["index"]), price=int(row["price"]),
            user_block=describe.prompt_for(row)))

    print(f"prompt: {PROMPT_ID} (view={SURR_VIEW})", flush=True)
    print(f"listings: {[l['index'] for l in listings]} (seed {SEED})", flush=True)

    out = {"prompt_id": PROMPT_ID, "view": SURR_VIEW, "seed": SEED,
           "system": system, "listings": [
               {k: l[k] for k in ("index", "price", "user_block")} for l in listings],
           "models": []}

    sem = asyncio.Semaphore(8)
    for slug in slugs:
        base = slug[:-len(":batch")] if slug.endswith(":batch") else slug
        thinking = is_thinking(base)
        settings = model_settings_for(thinking)
        model = OpenRouterModel(base, provider=provider)
        kind = "thinking" if thinking else "plain"
        print(f"\n=== {slug}  [{kind}]  (calling as {base}) ===", flush=True)

        async def call(l):
            async with sem:
                prose, reasoning, usage = await run_one(model, settings, system, l["user_block"])
            print(f"  idx {l['index']}  ({usage['output_tokens']} out tok)", flush=True)
            return dict(index=l["index"], price=l["price"], summary=prose,
                        reasoning=reasoning or None, usage=usage)

        results = await asyncio.gather(*(call(l) for l in listings))
        out["models"].append(dict(slug=slug, base=base, kind=kind,
                                  settings=settings, results=results))

    out_path = os.environ.get("EXP_OUT", os.path.join(_HERE, "model_variants.json"))
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nwrote {out_path}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
