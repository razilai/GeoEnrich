"""Summarise each listing's surroundings with an LLM (via OpenRouter).

Reads artifacts/airbnb_enriched.csv (from build.py — needs the `surroundings`
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
    python -m airbnb_surroundings.describe      # fill summaries for every listing with POIs
    python -m airbnb_surroundings.describe 10   # only top 10 by POI count — cheap test run
"""

import asyncio
import json
import os
import random
import re
import sys
from collections import Counter

import pandas as pd
from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.providers.openrouter import OpenRouterProvider

load_dotenv()

MODEL = os.environ.get("LLM_MODEL", "openai/gpt-4o-mini")
API_KEY = os.environ.get("OPENROUTER_API_KEY")

from airbnb_surroundings import config

IN_CSV = os.environ.get("DESC_IN", config.ENRICHED_CSV)
OUT_CSV = os.environ.get("DESC_OUT", config.DESCRIBED_CSV)

# concurrent in-flight LLM calls (raise if OpenRouter rate limit allows, lower on 429)
CONCURRENCY = int(os.environ.get("LLM_CONCURRENCY", "32"))
# checkpoint every CHUNK completed listings — a crash resumes from the last save
CHUNK = 200
# retry transient errors (429 rate-limit, 5xx) with exponential backoff + jitter
MAX_RETRIES = 6
BACKOFF_BASE = 2.0  # seconds: 2, 4, 8, 16, ... capped at BACKOFF_CAP
BACKOFF_CAP = 60.0
RETRY_STATUS = {429, 500, 502, 503, 504}

# JSON->text description (experiment prompt "03_landmark_salience"): a reliable,
# grounded summary of the named POIs by type, flagging any landmark/museum/market
# with a few words on why it matters. NO numeric score — dropped the old
# "Desirability: X/5" tail because gemini-3.1-flash-lite collapsed it to 3/4 on
# every batch listing (no usable signal); the description itself carries the
# neighbourhood character, and value can be modelled downstream from price.
INSTRUCTIONS = (
    "You convert a list of nearby places into a short, reliable description of "
    "the surroundings. In one or two sentences, name the main place types with a "
    "few named examples, grouped naturally (dining, retail, nightlife, culture). "
    "If any listed place is a well-known landmark, museum, market, park or "
    "attraction, say so and note in a few words why it matters for the area.\n"
    "Rules:\n"
    "- Name only places from the list, copied exactly. Never invent a place. If "
    "the list is empty, say the area has no notable named places nearby.\n"
    "- Do NOT rate, score, rank or price the location. No numbers, stars, or "
    "tiers (premium/mid/budget) — describe only.\n"
    "- You may note rough proximity with the given band (on the doorstep, a "
    "short walk away). Do NOT state exact metres or radius.\n"
    "- Plain text, no lists, headings, or markdown."
)

_NAME_RE = re.compile(r"[A-Z][\w&'’]+(?:\s+[A-Z][\w&'’]+)*")

# regenerate a draft that names places not in the POI data. 0 for this prompt: the
# landmark "why it matters" clause adds world-knowledge commentary (proper nouns not
# in the list), so the strict closed-world name check would misfire and regen rows.
GROUND_RETRIES = int(os.environ.get("GROUND_RETRIES", "0"))

# live token/call accounting across every LLM call (incl. retries) for cost estimation
USAGE = {"calls": 0, "in": 0, "out": 0, "cost": 0.0}


def _poi_names(pois):
    return sorted({p["name"] for p in pois if isinstance(p, dict) and p.get("name")})


def _category(p):
    """Place type shown to the LLM. Overture's `category` is already the rich,
    standardized leaf (e.g. 'italian_restaurant', 'coffee_shop') — cuisine/sub-
    type is baked in. Just humanize the underscores."""
    cat = p.get("category") if isinstance(p, dict) else None
    return cat.replace("_", " ") if cat else None


def ungrounded(summary, pois):
    """Capitalised place-names in `summary` that match no POI name (hallucinated)."""
    allowed = {n.lower() for n in _poi_names(pois)}
    cands = [c for c in _NAME_RE.findall(summary) if len(c) > 3]
    return [
        c for c in cands if not any(c.lower() in n or n in c.lower() for n in allowed)
    ]


def build_agent():
    if not API_KEY or API_KEY == "sk-or-v1-REPLACE_ME":
        sys.exit("OPENROUTER_API_KEY not set — edit .env")
    model = OpenRouterModel(MODEL, provider=OpenRouterProvider(api_key=API_KEY))
    # low temperature: the description should be stable and grounded, not creative.
    return Agent(
        model,
        output_type=str,
        instructions=INSTRUCTIONS,
        model_settings={"temperature": 0.4},
    )


def prompt_for(row, note=""):
    """Closed-world prompt: place-type counts + a name allow-list grouped by band."""
    pois = json.loads(row["surroundings"])
    # dedupe by name — keep first per name; unnamed POIs pass through.
    seen, uniq = set(), []
    for p in pois:
        nm = p.get("name") if isinstance(p, dict) else None
        if nm and nm in seen:
            continue
        if nm:
            seen.add(nm)
        uniq.append(p)
    pois = uniq
    cats = Counter(c for p in pois if isinstance(p, dict) and (c := _category(p)))
    # names grouped by proximity band, so the model can place them ("on the
    # doorstep" vs "a short walk away") without inventing exact distances.
    bands = {"doorstep": [], "short walk": []}
    for p in pois:
        nm = p.get("name") if isinstance(p, dict) else None
        if nm:
            bands.setdefault(p.get("prox", "short walk"), []).append(nm)
    lines = []
    for b in ("doorstep", "short walk"):
        if bands[b]:
            lines.append(f"{b}:")
            lines += [f"  - {n}" for n in sorted(set(bands[b]))]
    allowed = "\n".join(lines) or "(none — name nothing)"
    return (
        f"Place types (count): {dict(cats)}\n"
        f"Places by proximity (use only these names, copied exactly):\n{allowed}{note}"
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
    # final variant: LLM prose replaces the raw POI JSON — drop surroundings.
    df.drop(columns=["surroundings"]).to_csv(OUT_CSV, index=False)


async def call_with_retry(agent, prompt):
    """One LLM call, retrying transient 429/5xx with exponential backoff + jitter."""
    for attempt in range(MAX_RETRIES + 1):
        try:
            res = await agent.run(prompt)
            raw = getattr(res, "usage", None)
            u = (
                raw() if callable(raw) else raw
            )  # method in some versions, attr in others
            USAGE["calls"] += 1
            USAGE["in"] += (
                getattr(u, "input_tokens", None) or getattr(u, "request_tokens", 0) or 0
            )
            USAGE["out"] += (
                getattr(u, "output_tokens", None)
                or getattr(u, "response_tokens", 0)
                or 0
            )
            return res
        except ModelHTTPError as e:
            if e.status_code not in RETRY_STATUS or attempt == MAX_RETRIES:
                raise
            delay = min(BACKOFF_BASE * 2**attempt, BACKOFF_CAP)
            delay += random.uniform(0, delay)  # full jitter — spread the retry storm
            print(
                f"  {e.status_code} rate-limited, retry {attempt + 1}/{MAX_RETRIES} "
                f"in {delay:.1f}s",
                flush=True,
            )
            await asyncio.sleep(delay)


BATCH_URL = "https://openrouter.ai/api/beta/batches"
BATCH_POLL = float(os.environ.get("BATCH_POLL", "15"))  # seconds between status polls
# OpenRouter reserves credits up front = requests × max completion tokens. Cap the
# per-request completion and cap the batch size so the reservation stays small enough
# to clear against the account balance (an over-large reservation returns 402).
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "2000"))  # requests per submitted batch
BATCH_MAX_TOKENS = int(os.environ.get("BATCH_MAX_TOKENS", "400"))  # summary is short


async def _submit_batch(c, headers, rows, df):
    """Submit one batch of `rows`, poll to completion, parse results into df."""
    reqs = [
        {
            "custom_id": str(row["index"]),
            "body": {
                "messages": [
                    {"role": "system", "content": INSTRUCTIONS},
                    {"role": "user", "content": prompt_for(row)},
                ],
                "temperature": 0.4,
                "max_tokens": BATCH_MAX_TOKENS,  # caps the up-front credit reservation
                "usage": {"include": True},  # ask OpenRouter to return usage.cost
            },
        }
        for _, row in rows
    ]

    # endpoint + model MUST precede requests in the JSON (API stream-parses; else 400)
    payload = {"endpoint": "/v1/chat/completions", "model": MODEL, "requests": reqs}
    r = await c.post(BATCH_URL, headers=headers, content=json.dumps(payload))
    r.raise_for_status()
    bid = r.json()["id"]
    print(f"batch {bid} submitted: {len(reqs)} requests", flush=True)
    misses = 0
    while True:
        await asyncio.sleep(BATCH_POLL)
        g = await c.get(f"{BATCH_URL}/{bid}", headers=headers)
        if g.status_code == 404:  # eventual consistency right after submit
            misses += 1
            print(f"  poll 404 ({misses}) — batch not yet queryable", flush=True)
            if misses > 8:
                g.raise_for_status()
            continue
        g.raise_for_status()
        j = g.json()
        st = j.get("status")
        print(f"  status={st}", flush=True)
        if st in ("completed", "failed", "expired", "cancelled"):
            break

    # raw dump for inspection / cost audit
    with open("batch_result.json", "w") as f:
        json.dump(j, f)
    if st != "completed":
        print(f"batch ended {st} — see batch_result.json", flush=True)
        return

    items = (
        j.get("results")
        or j.get("output")
        or j.get("responses")
        or j.get("requests")
        or []
    )
    by_id = {str(r["index"]): idx for idx, r in rows}
    for it in items:
        cid = it.get("custom_id")
        resp = it.get("response") or {}
        body = resp.get("body") or {}
        if not body or cid not in by_id:
            continue
        content = body["choices"][0]["message"]["content"]
        df.at[by_id[cid], "surroundings_summary"] = content
        u = body.get("usage") or {}
        USAGE["calls"] += 1
        USAGE["in"] += u.get("prompt_tokens", 0)
        USAGE["out"] += u.get("completion_tokens", 0)
        USAGE["cost"] += float(u.get("cost", 0) or 0)


async def run_batch(rows, df):
    """Submit `rows` as OpenRouter async batches (:batch models, ~50% price).

    Split into BATCH_SIZE-sized sub-batches so the up-front credit reservation stays
    small; df is checkpointed after each sub-batch so a crash resumes mid-run.
    One request per listing, no per-row grounding regen (batch is single-pass);
    Fills df in place and accumulates USAGE (tokens + real usage.cost).
    """
    import httpx

    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    n_batches = (len(rows) + BATCH_SIZE - 1) // BATCH_SIZE
    async with httpx.AsyncClient(timeout=120) as c:
        for b in range(n_batches):
            sub = rows[b * BATCH_SIZE : (b + 1) * BATCH_SIZE]
            print(f"--- sub-batch {b + 1}/{n_batches} ({len(sub)} rows) ---", flush=True)
            await _submit_batch(c, headers, sub, df)
            save(df)  # checkpoint — a crash resumes from here, not from scratch
    print(
        f"batch parsed {USAGE['calls']} results, "
        f"reported usage.cost=${USAGE['cost']:.4f}",
        flush=True,
    )


async def run_chunk(agent, rows, df, done, total):
    """Fire all rows in the chunk concurrently, bounded by CONCURRENCY.

    Returns (done, n_failed). A row that exhausts retries is left as NaN so a
    later run resumes it; it never cancels the sibling calls.
    """
    sem = asyncio.Semaphore(CONCURRENCY)
    lock = asyncio.Lock()

    async def one(idx, row):
        nonlocal done
        pois = json.loads(row["surroundings"])
        async with sem:
            # regenerate while the draft names places absent from the POI data
            note = ""
            for attempt in range(GROUND_RETRIES + 1):
                res = await call_with_retry(agent, prompt_for(row, note))
                bad = ungrounded(res.output, pois)
                if not bad or attempt == GROUND_RETRIES:
                    break
                note = (
                    f"\n\nYour previous draft named places NOT in the list: "
                    f"{bad}. Rewrite naming ONLY the listed places."
                )
        df.at[idx, "surroundings_summary"] = res.output
        async with lock:
            done += 1
            print(
                f"[{done}/{total}] idx {row['index']}",
                flush=True,
            )

    # return_exceptions: a single row's failure must not cancel the rest of the batch
    results = await asyncio.gather(
        *(one(idx, row) for idx, row in rows), return_exceptions=True
    )
    failed = [r for r in results if isinstance(r, Exception)]
    for e in failed:
        print(f"  row failed after retries: {e!r}", flush=True)
    return done, len(failed)


def main():
    k = int(sys.argv[1]) if len(sys.argv) > 1 else None

    df = pd.read_csv(IN_CSV, low_memory=False)
    # densest surroundings first (cheap test runs hit the richest listings).
    # count derived from JSON on the fly — never a persisted column.
    n_pois = df["surroundings"].map(lambda s: len(json.loads(s)))
    df = df.iloc[n_pois.argsort()[::-1]]
    if k is not None:
        df = df.head(k)

    # reuse any summary already computed in a prior OUT_CSV; only NULLs hit the LLM.
    # object dtype: an all-NaN map() yields float64, which rejects string .at[] writes
    df["surroundings_summary"] = df["index"].map(load_cache()).astype("object")
    todo = df[df["surroundings_summary"].isna()]
    print(
        f"{len(df)} listings — {len(todo)} need the LLM, "
        f"{len(df) - len(todo)} cached (model {MODEL})",
        flush=True,
    )

    if not todo.empty and MODEL.endswith(":batch"):
        if not API_KEY or API_KEY == "sk-or-v1-REPLACE_ME":
            sys.exit("OPENROUTER_API_KEY not set — edit .env")
        asyncio.run(run_batch(list(todo.iterrows()), df))
        save(df)
        miss = df["surroundings_summary"].isna().sum()
        if miss:
            print(f"WARNING: {miss} listings still missing a summary", flush=True)
    elif not todo.empty:
        agent = build_agent()  # only touch the API (+ require a key) when work remains
        rows = list(todo.iterrows())
        total = len(rows)

        async def run_all():
            done = failed = 0
            try:
                # process in CHUNK-sized batches so we still checkpoint mid-run
                for start in range(0, total, CHUNK):
                    chunk = rows[start : start + CHUNK]
                    done, n_fail = await run_chunk(agent, chunk, df, done, total)
                    failed += n_fail
                    save(df)  # checkpoint — a crash resumes from here, not from scratch
            finally:
                # persist whatever completed, even on an unexpected error
                save(df)
            return failed

        failed = asyncio.run(run_all())
        if failed:
            print(
                f"WARNING: {failed} listings still missing a summary — "
                f"rerun to retry them",
                flush=True,
            )

    save(df)
    n = max(USAGE["calls"], 1)
    done_n = int(df["surroundings_summary"].notna().sum())
    print(f"done -> {OUT_CSV}", flush=True)
    print(
        f"USAGE calls={USAGE['calls']} in_tok={USAGE['in']} out_tok={USAGE['out']} "
        f"| per_listing: calls={USAGE['calls'] / max(done_n, 1):.2f} "
        f"in={USAGE['in'] // n} out={USAGE['out'] // n}",
        flush=True,
    )

    # cost: gpt-4o-mini batch = 50% of $0.15/$0.60 per 1M in/out tokens
    BATCH_IN, BATCH_OUT = 0.075 / 1e6, 0.30 / 1e6
    tok_cost = USAGE["in"] * BATCH_IN + USAGE["out"] * BATCH_OUT
    reported = USAGE.get("cost", 0.0)
    per = (reported or tok_cost) / max(done_n, 1)
    FULL = 4687  # enriched rows with POIs = full-dataset size
    print(
        f"COST this run: reported=${reported:.4f}  token-est=${tok_cost:.4f}  "
        f"(${per * 1000:.3f}/1k listings)",
        flush=True,
    )
    print(f"EXTRAPOLATED full {FULL} listings ≈ ${per * FULL:.2f}", flush=True)


if __name__ == "__main__":
    main()
