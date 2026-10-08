"""Summarise each listing's surroundings with an LLM (via OpenRouter).

Reads the enriched CSV (build.py's `surroundings` POI JSON column), renders each
listing's JSON as a short evidence block under one of several "views", asks the
model for a one-or-two-sentence description of the area, and writes it to the
`surroundings_summary` column of the described CSV.

Environment (.env):
    OPENROUTER_API_KEY   OpenRouter key
    LLM_MODEL            model slug, e.g. openai/gpt-4o-mini; a `:batch` suffix
                         uses OpenRouter's async batch API (~50% price)
    LLM_THINKING         false/0/no/off/none explicitly disables model reasoning
    LLM_TEMPERATURE      sampling temperature (default 0.4)
    LLM_CONCURRENCY      concurrent live calls (default 32; lower on 429s)
    DESC_IN, DESC_OUT    input / output CSV paths (output defaults per prompt)
    DESC_REFERENCE_CSV   optional full corpus for the citywide percentiles
    DESC_CACHE_TAG       cache fingerprint (default: hash of prompt + view)

The prompt variant (view + system instruction) comes from prompts.toml and is
chosen with --prompt; each variant writes its own airbnb_described_<id>.csv.

Runs are incremental: summaries are cached per listing `index` in a sidecar file,
only rows still missing one hit the LLM, and progress is checkpointed after every
chunk, so a crash resumes instead of restarting.

Usage:
    python -m src.describe                # every listing, default prompt
    python -m src.describe --prompt 08    # a specific prompt variant
    python -m src.describe 10             # 10 random listings — cheap test run
"""

import argparse
import asyncio
import functools
import hashlib
import itertools
import json
import os
import random
import re
import sys
import tomllib

import httpx
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.providers.openrouter import OpenRouterProvider

from src import config

load_dotenv()
log = functools.partial(print, flush=True)
env = os.environ.get

MODEL = env("LLM_MODEL", "openai/gpt-4o-mini")
API_KEY = env("OPENROUTER_API_KEY")
_PLACEHOLDER_KEY = "sk-or-v1-REPLACE_ME"
THINKING_DISABLED = env("LLM_THINKING", "").strip().lower() in {
    "0",
    "false",
    "no",
    "off",
    "none",
}
TEMPERATURE = float(env("LLM_TEMPERATURE", "0.4"))

# Run configuration. Experiments override these module attributes (and OUT_CSV /
# SURR_VIEW / INSTRUCTIONS, set by use_prompt below) before calling run().
IN_CSV = env("DESC_IN", config.ENRICHED_CSV)
# Lets a small prompt-screen sample keep genuinely citywide percentiles.
REFERENCE_CSV = env("DESC_REFERENCE_CSV")
# Stops one prompt variant from reusing drafts written by another.
CACHE_TAG = env("DESC_CACHE_TAG")

SUMMARY = "surroundings_summary"

CONCURRENCY = int(env("LLM_CONCURRENCY", "32"))
CHUNK = 200  # live-mode checkpoint interval, in listings

MAX_RETRIES = 6
BACKOFF_BASE = 2.0  # seconds, doubled per attempt up to BACKOFF_CAP
BACKOFF_CAP = 60.0
RETRY_STATUS = {429, 500, 502, 503, 504}

# Regenerations allowed while a draft names places absent from the landmarks. Off by
# default: landmark commentary legitimately adds proper nouns the closed-world
# check would flag.
GROUND_RETRIES = int(env("GROUND_RETRIES", "0"))

BATCH_URL = "https://openrouter.ai/api/beta/batches"
BATCH_POLL = float(env("BATCH_POLL", "20"))  # seconds between polls
# OpenRouter reserves requests × max_tokens credits up front and rejects an
# oversized reservation with 402, so both are capped.
BATCH_SIZE = int(env("BATCH_SIZE", "3000"))
BATCH_MAX_TOKENS = int(env("BATCH_MAX_TOKENS", "400"))
_BATCH_TERMINAL_STATES = {"completed", "failed", "expired", "cancelled"}
_BATCH_MAX_404S = 8  # a fresh batch is briefly not queryable


# --- prompt -----------------------------------------------------------------------
PROMPTS_TOML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts.toml")

with open(PROMPTS_TOML, "rb") as _f:
    _toml = tomllib.load(_f)
DEFAULT_PROMPT = _toml["default"]
PROMPTS = _toml["prompts"]


def use_prompt(prompt_id):
    """Select a prompt variant: its view, its instruction, and its own output CSV
    (DESC_OUT still overrides)."""
    global SURR_VIEW, INSTRUCTIONS, OUT_CSV
    prompt = PROMPTS[prompt_id]
    SURR_VIEW = prompt["view"]
    INSTRUCTIONS = prompt["system"].strip()
    OUT_CSV = env("DESC_OUT", config.described_csv(prompt_id))


use_prompt(DEFAULT_PROMPT)


# --- surroundings JSON ------------------------------------------------------------
# Schema written by build.py; every category is an Overture level-2 taxonomy
# category (config.OVERTURE_CATEGORIES), called a `group` below:
#   cats:      {group: [count<=450m, nearest_m]}
#   landmarks: [[name, dist_m], ...]  (nearest first)
_WITHIN_450M, _NEAREST_M = 0, 1


def _count(cats, group):
    """Places of `group` within 450 m (0 when absent)."""
    return cats.get(group, (0, 0))[_WITHIN_450M]


def _total(cats):
    """Places of every kept category within 450 m."""
    return sum(values[_WITHIN_450M] for values in cats.values())


# Plain nouns for the Overture codes, which the model otherwise paraphrases
# unpredictably or copies verbatim. Worded after each category's NYC level-3 mix
# (e.g. ground transport is rail, taxi and transit in roughly equal parts).
_LABELS = {
    "museum": "museums",
    "nightlife_venue": "nightclubs",
    "performing_arts_venue": "music and theatre venues",
    "historic_site": "historic sites",
    "alcoholic_beverage_venue": "bars",
    "casual_eatery": "casual eateries",
    "non_alcoholic_beverage_venue": "coffee shops and juice bars",
    "restaurant": "restaurants",
    "hospital": "hospitals",
    "hotel": "hotels",
    "corporate_or_business_office": "offices",
    "convenience_store": "convenience stores",
    "department_store": "department stores",
    "discount_store": "discount stores",
    "fashion_and_apparel_store": "clothing and fashion stores",
    "food_and_beverage_store": "grocery and food stores",
    "park": "parks and playgrounds",
    "sport_or_fitness_facility": "gyms and fitness studios",
    "ground_transport_facility_or_service": "rail, transit and taxi services",
    "parking": "parking",
}


def _label(group):
    """Plain words for an Overture category code, e.g. restaurant -> restaurants."""
    return _LABELS.get(group, group.replace("_", " "))


# Landmark distance bands in air metres, tight near and wide far: the landmark
# price premium is steepest at the doorstep.
_LM_BANDS = [
    (100, "right by"),
    (250, "a couple minutes from"),
    (500, "a short walk from"),
    (float("inf"), "about 10 minutes from"),
]
# Only the nearest few landmarks are shown.
_LM_CAP = 3

# Given many standouts, the model averages them into generic "dense area" prose,
# so only the sharpest few are shown.
_DEV_CAP = 5
_PRIMARY_CAP = 3

# A standout must sit in the outer 15% of the corpus on either side; anything
# nearer the middle is padding, not a contrast.
_MIN_EXTREMITY = 0.35
# "More than usual" needs at least this many places: in sparse categories one or
# two places would otherwise rank near the top of the city.
_MIN_TOP_COUNT = 3

# Doorstep cues only for categories whose nearness says something; restaurants
# and shops are on the doorstep almost everywhere.
_PROXIMITY_GROUPS = {
    "park",
    "ground_transport_facility_or_service",
    "museum",
    "hotel",
    "nightlife_venue",
    "historic_site",
}
_PROXIMITY_CAP = 1  # the instructions allow at most one proximity cue

_NAME_RE = re.compile(r"[A-Z][\w&'’]+(?:\s+[A-Z][\w&'’]+)*")


def ungrounded(summary, surr):
    """Capitalised place-names in `summary` that match no supplied landmark
    (hallucinated)."""
    allowed = {name.lower() for name, _ in surr.get("landmarks", [])}
    return [
        c
        for c in _NAME_RE.findall(summary)
        if len(c) > 3 and not any(c.lower() in n or n in c.lower() for n in allowed)
    ]


# --- corpus reference distributions ---------------------------------------------
# Filled by load_reference(); the deviation views rank a listing against these.
_REF = {}  # group -> sorted within-450m counts across the corpus
_REF_TOTAL = np.array([])  # sorted within-450m totals across the corpus
_PRESENT = {}  # group -> share of listings with at least one


def load_reference(df):
    """Build the corpus distributions (call once before rendering prompts)."""
    global _REF_TOTAL
    _REF.clear()
    _PRESENT.clear()
    cats = [json.loads(s).get("cats", {}) for s in df["surroundings"]]
    totals = np.array([_total(c) for c in cats])
    _REF_TOTAL = np.sort(totals)
    for group in {g for c in cats for g in c}:
        counts = np.array([_count(c, group) for c in cats])
        _REF[group] = np.sort(counts)
        _PRESENT[group] = float((counts > 0).mean())


# Word bands for _deviation_band, checked in order; percentiles in between are typical.
_HIGH_BANDS = [
    (0.95, "far more than most neighbourhoods"),
    (0.80, "well above average"),
    (0.65, "above average"),
]
_LOW_BANDS = [
    (0.05, "almost none"),
    (0.20, "well below average"),
    (0.35, "below average"),
]

# Overall busyness by citywide quintile of the total place count.
_DENSITY_BANDS = [
    (0.2, "very quiet"),
    (0.4, "quiet"),
    (0.6, "average"),
    (0.8, "busy"),
    (float("inf"), "very busy"),
]


def _percentile(ref, value):
    """Mid-rank share of the sorted `ref` below `value`, ties counted half: a
    count most listings share (often 0 or 1) lands mid-pack instead of jumping
    to the top. None without a reference."""
    if ref is None or not len(ref):
        return None
    below = np.searchsorted(ref, value, side="left")
    at_or_below = np.searchsorted(ref, value, side="right")
    return (below + at_or_below) / 2 / len(ref)


def _notably_absent(group):
    """Absence only counts where the group is usually present."""
    return _PRESENT.get(group, 0) >= 0.6


def _rank(group, count, pct):
    """(extremity, "top/bottom X% of NYC neighbourhoods") or None if roughly
    typical: a self-describing citywide rank the model can read at face value."""
    if pct is None:
        return None
    extremity = abs(pct - 0.5)
    if extremity < _MIN_EXTREMITY:
        return None
    if count <= 0:
        if not _notably_absent(group):
            return None
        return extremity, "none nearby (most NYC neighbourhoods have some)"
    if pct >= 0.5:
        if count < _MIN_TOP_COUNT:
            return None
        return extremity, f"top {max(1, round((1 - pct) * 100))}% of NYC neighbourhoods"
    return extremity, f"bottom {max(1, round(pct * 100))}% of NYC neighbourhoods"


def _deviation_rank(group, cats):
    """Rank of the group's count of places versus the corpus."""
    count = _count(cats, group)
    return _rank(group, count, _percentile(_REF.get(group), count))


# Word bands for _contrast_band, checked in order; they split the same outer 15%
# that _rank keeps, so a percentile never reaches the prompt to be quoted.
_CONTRAST_HIGH = [
    (0.99, "more than almost anywhere in NYC"),
    (0.95, "far more than most neighbourhoods"),
    (0.0, "more than most neighbourhoods"),
]
_CONTRAST_LOW = [
    (0.05, "very few"),
    (1.0, "fewer than most neighbourhoods"),
]


def _contrast_band(group, cats):
    """The standouts _deviation_rank keeps, phrased as fixed word bands: equal
    ranks read alike and the model has no number to copy."""
    count = _count(cats, group)
    pct = _percentile(_REF.get(group), count)
    ranked = _rank(group, count, pct)
    if ranked is None or count <= 0:
        return ranked
    if pct >= 0.5:
        phrase = next(p for cutoff, p in _CONTRAST_HIGH if pct >= cutoff)
    else:
        phrase = next(p for cutoff, p in _CONTRAST_LOW if pct <= cutoff)
    return ranked[0], phrase


def _deviation_band(group, cats):
    """(extremity, band word) versus the corpus, or None if roughly typical: the
    same signal as _deviation_rank, coarsened into six word bands."""
    count = _count(cats, group)
    pct = _percentile(_REF.get(group), count)
    if pct is None:
        return None
    if count <= 0:
        return (0.5, "none nearby") if _notably_absent(group) else None
    phrase = None
    if count >= _MIN_TOP_COUNT:
        phrase = next((p for cutoff, p in _HIGH_BANDS if pct >= cutoff), None)
    phrase = phrase or next((p for cutoff, p in _LOW_BANDS if pct <= cutoff), None)
    return (abs(pct - 0.5), phrase) if phrase else None


def _deviations(surr, classify):
    """(extremity, group, phrase) for every group `classify` flags as atypical,
    strongest first and at most one per Overture level-1 group, so three food
    categories never all say "food". Walks every reference group, so notable
    absence counts too."""
    cats = surr.get("cats", {})
    found = sorted(
        ((r[0], group, r[1]) for group in _REF if (r := classify(group, cats))),
        key=lambda d: (-d[0], _label(d[1])),
    )
    seen, kept = set(), []
    for deviation in found:
        parent = config.CATEGORY_GROUP.get(deviation[1], deviation[1])
        if parent not in seen:
            seen.add(parent)
            kept.append(deviation)
    return kept


# --- evidence lines ---------------------------------------------------------------
def _proximity_phrase(metres):
    if metres <= 50:
        return "on the doorstep"
    return "steps away" if metres <= 150 else "a short walk"


def _section(header, lines, empty):
    return "\n".join([header, *(lines or [empty])])


def _deviation_lines(deviations):
    return [f"- {_label(group)}: {phrase}" for _, group, phrase in deviations]


def _density_line(surr):
    pct = _percentile(_REF_TOTAL, _total(surr.get("cats", {})))
    if pct is None:
        return None
    word = next(w for cutoff, w in _DENSITY_BANDS if pct < cutoff)
    return (
        f"Overall: {word}, by how many places are within a short walk "
        "compared with other NYC neighbourhoods."
    )


def _landmark_line(surr):
    by_band = {}  # landmarks arrive nearest-first, so each band stays ordered
    for name, metres in surr.get("landmarks", [])[:_LM_CAP]:
        phrase = next(p for cutoff, p in _LM_BANDS if metres <= cutoff)
        by_band.setdefault(phrase, []).append(name)
    if not by_band:
        return "There are no well-known landmarks nearby."
    lines = [f"- {p}: {', '.join(by_band[p])}" for _, p in _LM_BANDS if p in by_band]
    return "\n".join(
        ["Well-known places nearby (copy names exactly, do not add any):", *lines]
    )


def _proximity_lines(surr, excluded_groups: set[str], cap=_PROXIMITY_CAP):
    """Telling groups (_PROXIMITY_GROUPS) with a place within the doorstep
    radius, nearest first.

    Groups already among the primary contrasts are skipped, so the prose gains a
    second axis (e.g. a park on the doorstep beside an office district) instead
    of restating a primary contrast.
    """
    close = sorted(
        (values[_NEAREST_M], group)
        for group, values in surr.get("cats", {}).items()
        if group in _PROXIMITY_GROUPS
        and group not in excluded_groups
        and values[_NEAREST_M] <= config.DOORSTEP
    )
    return [
        f"- {_label(group)}: {_proximity_phrase(metres)}"
        for metres, group in close[:cap]
    ]


# --- views: the same JSON rendered as different evidence --------------------------
def _standouts_view(surr, header, classify):
    standouts = _deviation_lines(_deviations(surr, classify)[:_DEV_CAP])
    return "\n".join(
        [
            _section(header, standouts, "- (unremarkable — typical across the board)"),
            _landmark_line(surr),
        ]
    )


def _view_deviation(surr):
    return _standouts_view(
        surr,
        "How this neighbourhood compares with a typical New York neighbourhood "
        "(only the ways it stands out are listed):",
        _deviation_band,
    )


def _view_deviation_exact(surr):
    return _standouts_view(
        surr,
        "How this neighbourhood ranks among all New York neighbourhoods "
        "(citywide percentile; only standouts listed):",
        _deviation_rank,
    )


def _evidence_rng(surr):
    """RNG seeded by the listing's surroundings: the same listing always gets the
    same shuffle, so prompts are reproducible across runs."""
    digest = hashlib.sha256(json.dumps(surr, sort_keys=True).encode()).digest()
    return random.Random(int.from_bytes(digest[:8]))


def _view_price_relevant_profile(surr):
    """Overall busyness, the strongest citywide contrasts (each category ranked
    independently by its count, phrased as word bands), plus one non-redundant
    doorstep cue.

    Sections, and the contrasts within theirs, come in a per-listing shuffled
    order, so the prose does not inherit one fixed evidence structure.
    """
    rng = _evidence_rng(surr)
    primary = _deviations(surr, _contrast_band)[:_PRIMARY_CAP]
    contrasts = _deviation_lines(primary)
    rng.shuffle(contrasts)
    sections = [
        _density_line(surr),
        _section(
            "Primary evidence — strongest citywide contrasts:",
            contrasts,
            "- (unremarkable — typical across the measured place types)",
        ),
        _section(
            "Independent proximity evidence (use only if it adds a new idea):",
            _proximity_lines(surr, {group for _, group, _ in primary}),
            "- (no additional proximity evidence)",
        ),
        _landmark_line(surr),
    ]
    sections = [s for s in sections if s]
    rng.shuffle(sections)
    return "\n\n".join(sections)


_VIEWS = {
    "deviation": _view_deviation,
    "deviation_exact": _view_deviation_exact,
    "character_deviation": _view_deviation_exact,
    "price_relevant_profile": _view_price_relevant_profile,
}
# Bump a view's version whenever its rendered evidence changes: prompt screens add
# it to their cache fingerprint, so a corrected renderer can't reuse old drafts.
_VIEW_CACHE_VERSIONS = {
    "deviation": "v4",
    "deviation_exact": "v5",
    "character_deviation": "v5",
    "price_relevant_profile": "v8",
}


def prompt_for(row):
    """Render a listing's surroundings JSON under the selected SURR_VIEW."""
    return _VIEWS[SURR_VIEW](json.loads(row["surroundings"]))


# --- cache and output -------------------------------------------------------------
def _prompt_fingerprint():
    """Short hash of the active instruction and view, so editing either starts a
    fresh cache instead of resuming with stale summaries."""
    material = "\0".join(
        [INSTRUCTIONS, SURR_VIEW, _VIEW_CACHE_VERSIONS.get(SURR_VIEW, "v1")]
    )
    return hashlib.sha256(material.encode()).hexdigest()[:12]


def _cache_csv():
    """Sidecar checkpoint path, named per OUT_CSV and CACHE_TAG (default: the
    prompt fingerprint) so different outputs or prompts never share a cache."""
    tag = CACHE_TAG or _prompt_fingerprint()
    return os.path.join(
        config.ARTIFACTS_DIR, f"{os.path.basename(OUT_CSV)}.{tag}.cache"
    )


def load_cache():
    """index -> summary from earlier runs, so no description is paid for twice.

    Keyed on the stable per-listing `index`, which only the sidecar cache keeps.
    """
    path = _cache_csv()
    if not os.path.exists(path):
        return {}
    prev = pd.read_csv(path, low_memory=False)
    if "index" not in prev or SUMMARY not in prev:
        return {}
    prev = prev[prev[SUMMARY].notna()]
    return dict(zip(prev["index"], prev[SUMMARY]))


def save(df):
    """Checkpoint to the sidecar cache, keeping `index` for incremental resume."""
    os.makedirs(config.ARTIFACTS_DIR, exist_ok=True)
    df.drop(columns=["surroundings"], errors="ignore").to_csv(_cache_csv(), index=False)


def publish(df):
    """Write the final dataset: prose replaces the POI JSON. `index` stays, so
    `stage` can align the corpus with the cleaned listings; it drops it again."""
    os.makedirs(config.PROCESSED_DIR, exist_ok=True)
    df.drop(columns=["surroundings"], errors="ignore").to_csv(
        OUT_CSV, index=False
    )


# --- live LLM calls ---------------------------------------------------------------
def _require_api_key():
    if not API_KEY or API_KEY == _PLACEHOLDER_KEY:
        sys.exit("OPENROUTER_API_KEY not set — edit .env")


def build_agent():
    _require_api_key()
    model = OpenRouterModel(MODEL, provider=OpenRouterProvider(api_key=API_KEY))
    settings = {"temperature": TEMPERATURE}
    if THINKING_DISABLED:
        settings["thinking"] = False
    return Agent(
        model, output_type=str, instructions=INSTRUCTIONS, model_settings=settings
    )


async def call_with_retry(agent, prompt):
    """One LLM call, retrying transient 429/5xx with exponential backoff + full jitter."""
    for attempt in range(MAX_RETRIES + 1):
        try:
            return await agent.run(prompt)
        except ModelHTTPError as e:
            if e.status_code not in RETRY_STATUS or attempt == MAX_RETRIES:
                raise
            delay = min(BACKOFF_BASE * 2**attempt, BACKOFF_CAP)
            delay += random.uniform(0, delay)
            log(
                f"  {e.status_code} rate-limited, retry {attempt + 1}/{MAX_RETRIES} "
                f"in {delay:.1f}s"
            )
            await asyncio.sleep(delay)


async def _summarise(agent, row):
    """Generate a summary, regenerating while it names places absent from the landmarks."""
    surr = json.loads(row["surroundings"])
    prompt = request = prompt_for(row)
    for attempt in range(GROUND_RETRIES + 1):
        output = (await call_with_retry(agent, request)).output
        bad = ungrounded(output, surr)
        if not bad or attempt == GROUND_RETRIES:
            return output
        request = (
            f"{prompt}\n\nYour previous draft named places NOT in the list: "
            f"{bad}. Rewrite naming ONLY the listed places."
        )


async def run_live(agent, rows, df):
    """Summarise `rows` concurrently (bounded by CONCURRENCY), checkpointing after
    every CHUNK. A row that exhausts its retries stays NaN for a later run to
    resume, without cancelling its siblings."""
    sem = asyncio.Semaphore(CONCURRENCY)
    done = itertools.count(1)

    async def one(idx, row):
        async with sem:
            df.at[idx, SUMMARY] = await _summarise(agent, row)
        log(f"[{next(done)}/{len(rows)}] idx {row['index']}")

    try:
        for start in range(0, len(rows), CHUNK):
            chunk = rows[start : start + CHUNK]
            results = await asyncio.gather(
                *(one(*r) for r in chunk), return_exceptions=True
            )
            for e in results:
                if isinstance(e, Exception):
                    log(f"  row failed after retries: {e!r}")
            save(df)
    finally:
        save(df)  # keep whatever completed, even on an unexpected error


# --- OpenRouter batch API -----------------------------------------------------------
async def _poll_batch(client, batch_id):
    """Poll until the batch reaches a terminal state; return its final status JSON."""
    misses = 0
    while True:
        await asyncio.sleep(BATCH_POLL)
        resp = await client.get(f"{BATCH_URL}/{batch_id}")
        if resp.status_code == 404 and misses < _BATCH_MAX_404S:
            misses += 1
            log(f"  poll 404 ({misses}) — batch not yet queryable")
            continue
        resp.raise_for_status()
        status = resp.json()
        log(f"  status={status.get('status')}")
        if status.get("status") in _BATCH_TERMINAL_STATES:
            return status


_THINK_BLOCK = re.compile(r"<(think|thinking)>.*?</\1>", re.S | re.I)


def _final_text(body):
    """The answer text of a chat completion, or None when it isn't a clean final
    answer. Reasoning lives in message.reasoning, never content; but reasoning can
    exhaust max_tokens and leave content empty or cut off (finish_reason=length),
    and some models inline <think> tags — neither may reach the CSV."""
    choice = body["choices"][0]
    if choice.get("finish_reason") not in (None, "stop"):
        return None
    text = _THINK_BLOCK.sub("", choice["message"].get("content") or "").strip()
    if not text or re.search(r"</?think(ing)?>", text, re.I):
        return None
    return text


def _apply_batch_results(batch, rows, df):
    """Write each returned summary into df."""
    keys = ("results", "output", "responses", "requests")
    items = next((batch[k] for k in keys if batch.get(k)), [])
    df_index_by_id = {str(row["index"]): idx for idx, row in rows}
    skipped = 0
    for item in items:
        body = (item.get("response") or {}).get("body")
        if body and (idx := df_index_by_id.get(item.get("custom_id"))) is not None:
            if text := _final_text(body):
                df.at[idx, SUMMARY] = text
            else:
                skipped += 1
    if skipped:
        log(f"{skipped} batch results unusable (truncated/empty) — rerun to retry them")


async def _submit_batch(client, rows, df):
    """Submit one batch of `rows`, wait for it, and parse the results into df."""
    settings = {"temperature": TEMPERATURE, "max_tokens": BATCH_MAX_TOKENS}
    if THINKING_DISABLED:
        settings["reasoning"] = {"effort": "none"}
    requests = [
        {
            "custom_id": str(row["index"]),
            "body": {
                "messages": [
                    {"role": "system", "content": INSTRUCTIONS},
                    {"role": "user", "content": prompt_for(row)},
                ],
                **settings,
            },
        }
        for _, row in rows
    ]
    # The API stream-parses: endpoint and model must precede requests, else 400.
    payload = {"endpoint": "/v1/chat/completions", "model": MODEL, "requests": requests}
    resp = await client.post(BATCH_URL, content=json.dumps(payload))
    resp.raise_for_status()
    batch_id = resp.json()["id"]
    log(f"batch {batch_id} submitted: {len(requests)} requests")

    batch = await _poll_batch(client, batch_id)
    os.makedirs(config.ARTIFACTS_DIR, exist_ok=True)
    with open(config.BATCH_JSON, "w") as f:  # raw dump for inspection
        json.dump(batch, f)
    if batch.get("status") == "completed":
        _apply_batch_results(batch, rows, df)
    else:
        log(f"batch ended {batch.get('status')} — see {config.BATCH_JSON}")


async def resume_batch(rows, df, batch_id):
    """Wait for an already-submitted batch and apply its results (no new submit)."""
    _require_api_key()
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    async with httpx.AsyncClient(headers=headers, timeout=120) as client:
        batch = await _poll_batch(client, batch_id)
    os.makedirs(config.ARTIFACTS_DIR, exist_ok=True)
    with open(config.BATCH_JSON, "w") as f:
        json.dump(batch, f)
    if batch.get("status") == "completed":
        _apply_batch_results(batch, rows, df)
    else:
        log(f"batch ended {batch.get('status')} — see {config.BATCH_JSON}")


async def run_batch(rows, df):
    """Summarise `rows` via OpenRouter async batches (:batch models, ~50% price).

    Splits into BATCH_SIZE sub-batches to keep the up-front credit reservation
    small, checkpointing after each. Batches are single-pass: no grounding regen.
    """
    _require_api_key()
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    n_batches = -(-len(rows) // BATCH_SIZE)
    async with httpx.AsyncClient(headers=headers, timeout=120) as client:
        for b, start in enumerate(range(0, len(rows), BATCH_SIZE), 1):
            sub = rows[start : start + BATCH_SIZE]
            log(f"--- sub-batch {b}/{n_batches} ({len(sub)} rows) ---")
            await _submit_batch(client, sub, df)
            save(df)


# --- entry point ------------------------------------------------------------------
# Fixed so a k-row test run always draws the same listings and reruns hit the cache.
SAMPLE_SEED = 0


def _select_rows(df, k):
    """A seeded random sample of k listings (all when None), so a test run sees
    quiet and busy areas alike."""
    n = len(df) if k is None else min(k, len(df))
    return df.sample(n=n, random_state=SAMPLE_SEED)


def run(k=None, resume=None):
    """Describe k random listings (all when None) with the active prompt."""
    df = pd.read_csv(IN_CSV, low_memory=False)
    ref = df
    if REFERENCE_CSV:
        ref = pd.read_csv(REFERENCE_CSV, usecols=["surroundings"], low_memory=False)
        log(f"reference distribution: {len(ref)} listings from {REFERENCE_CSV}")
    load_reference(ref)
    df = _select_rows(df, k)

    # object dtype: an all-NaN map() yields float64, which rejects string writes
    df[SUMMARY] = df["index"].map(load_cache()).astype("object")
    rows = list(df[df[SUMMARY].isna()].iterrows())
    log(
        f"{len(df)} listings — {len(rows)} need the LLM, "
        f"{len(df) - len(rows)} cached (model {MODEL}, view {SURR_VIEW})"
    )

    if rows:
        if resume:
            asyncio.run(resume_batch(rows, df, resume))
        elif MODEL.endswith(":batch"):
            asyncio.run(run_batch(rows, df))
        else:
            asyncio.run(run_live(build_agent(), rows, df))
        if n_missing := df[SUMMARY].isna().sum():
            log(
                f"WARNING: {n_missing} listings still missing a summary — rerun to retry them"
            )

    save(df)
    publish(df)
    log(f"done -> {OUT_CSV}")


def main():
    p = argparse.ArgumentParser(
        description="Summarise listing surroundings with an LLM."
    )
    p.add_argument("k", nargs="?", type=int, help="only k random listings")
    p.add_argument(
        "--prompt",
        choices=sorted(PROMPTS),
        default=DEFAULT_PROMPT,
        help=f"prompt variant from prompts.toml (default {DEFAULT_PROMPT})",
    )
    p.add_argument(
        "--resume",
        metavar="BATCH_ID",
        help="collect an already-submitted batch instead of submitting a new one "
        "(same k and --prompt as the original run)",
    )
    p.add_argument(
        "--confirm",
        action="store_true",
        help="required: describe issues paid LLM calls",
    )
    args = p.parse_args()
    if not args.confirm:
        p.error("describe spends LLM credits; re-run with --confirm to proceed")
    use_prompt(args.prompt)
    run(args.k, args.resume)


if __name__ == "__main__":
    main()
