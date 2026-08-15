"""Summarise each listing's surroundings with an LLM (via OpenRouter).

Reads data/processed/airbnb_enriched.csv (from build.py — needs the `surroundings`
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

import numpy as np
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
# Which JSON facet drives the text — the axis for the joint-signal screen.
# Same enrichment JSON, different information view (see the _view_* renders).
SURR_VIEW = os.environ.get("SURR_VIEW", "deviation")

_RULES = (
    "\nRules:\n"
    "- Use only the summary; never invent or add a place; copy any listed name exactly.\n"
    "- Do NOT rate, score, tier, or price the location, and give no counts or metres.\n"
    "- Keep it calibrated: little of something is quiet or limited, not lively.\n"
    "- Plain text, no lists, headings, or markdown."
)
_VIEW_INSTRUCTIONS = {
    "deviation": "You describe what makes a listing's location distinctive from a "
    "summary of how its block compares with a typical New York block (only the "
    "standout differences are given) and any well-known places. In one or two "
    "sentences, say what kind of area it is and what stands out." + _RULES,
    "proximity": "You describe a listing's location from a summary of how close "
    "each kind of place is. In one or two sentences, convey how convenient and "
    "walkable it is — what is on the doorstep versus a walk away — and note transit "
    "access and any well-known place." + _RULES,
    "composition": "You describe a listing's location from a summary of the MIX of "
    "nearby places (which types dominate). In one or two sentences, say what kind of "
    "area this is from that mix — e.g. dining-and-nightlife, retail-heavy, quiet "
    "residential, culture-rich — and any well-known place." + _RULES,
    "landmarks": "You describe a listing's location from the well-known places near "
    "it. In one or two sentences, say what those places suggest about the area "
    "(prestige, tourist draw, cultural character). If there are none, say the area "
    "has no notable landmarks nearby." + _RULES,
    "combined": "You describe what makes a listing's location distinctive from a "
    "summary covering how its block compares with a typical New York block, how "
    "close each kind of place is, the overall mix, and any well-known places. In one "
    "or two sentences, capture the area's character, convenience, and any landmark."
    + _RULES,
}
INSTRUCTIONS = _VIEW_INSTRUCTIONS.get(SURR_VIEW, _VIEW_INSTRUCTIONS["deviation"])

_NAME_RE = re.compile(r"[A-Z][\w&'’]+(?:\s+[A-Z][\w&'’]+)*")

# regenerate a draft that names places not in the POI data. 0 for this prompt: the
# landmark "why it matters" clause adds world-knowledge commentary (proper nouns not
# in the list), so the strict closed-world name check would misfire and regen rows.
GROUND_RETRIES = int(os.environ.get("GROUND_RETRIES", "0"))

# live token/call accounting across every LLM call (incl. retries) for cost estimation
USAGE = {"calls": 0, "in": 0, "out": 0, "cost": 0.0}


# surroundings schema: {"cats": {bucket: [count<=150m, count<=400m, nearest_m]},
#                       "landmarks": [[name, dist_m], ...]}
def _poi_names(surr):
    return sorted({n for n, _ in surr.get("landmarks", [])})


def ungrounded(summary, surr):
    """Capitalised place-names in `summary` that match no landmark name (hallucinated)."""
    allowed = {n.lower() for n in _poi_names(surr)}
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


_DISPLAY = {
    "dining": "dining", "cafe": "cafes", "nightlife": "bars & nightlife",
    "grocery": "grocery stores", "shopping": "shops", "pharmacy": "pharmacies",
    "fitness_sport": "gyms & sports", "park_green": "parks & green space",
    "culture": "museums & galleries", "landmark": "landmarks & sights",
    "entertainment": "entertainment venues", "lodging": "hotels",
    "transit": "subway & transit",
}

# The signal is DEVIATION from a typical block, not presence: ~98% of listings
# have dining/shops/grocery, so listing them says nothing. We compare each
# bucket's count against the corpus distribution and surface only the standouts.
_REF = {}  # bucket -> sorted np.array of within-400m counts across the corpus
_PRESENT = {}  # bucket -> fraction of listings with the bucket present


def load_reference(df):
    """Build the corpus count distribution per bucket (call once before prompting)."""
    cats = df["surroundings"].map(lambda s: json.loads(s).get("cats", {}))
    for b in {k for c in cats for k in c}:
        arr = np.array([c.get(b, [0, 0, 0])[1] for c in cats])
        _REF[b] = np.sort(arr)
        _PRESENT[b] = float((arr > 0).mean())


def _relative(bucket, v):
    """How this bucket's count compares with the corpus — or None if ~typical.
    Finer bands (top third / bottom third, six levels) so more blocks carry a
    distinguishing signal instead of collapsing to a shared coarse pattern."""
    arr = _REF.get(bucket)
    if arr is None or len(arr) == 0:
        return None
    if v <= 0:  # absence is signal only where the bucket is usually present
        return "none nearby" if _PRESENT.get(bucket, 0) >= 0.6 else None
    pct = np.searchsorted(arr, v, side="left") / len(arr)
    if pct >= 0.95:
        return "far more than most blocks"
    if pct >= 0.80:
        return "well above average"
    if pct >= 0.65:
        return "above average"
    if pct <= 0.05:
        return "almost none"
    if pct <= 0.20:
        return "well below average"
    if pct <= 0.35:
        return "below average"
    return None  # middle third ~typical → omit


_REL_ORDER = {
    "far more than most blocks": 0, "well above average": 1, "above average": 2,
    "none nearby": 3, "almost none": 4, "well below average": 5, "below average": 6,
}


def _prox_word(m):
    return "on the doorstep" if m <= 50 else "steps away" if m <= 150 else "a short walk"


def _label(b):
    return _DISPLAY.get(b, b.replace("_", " "))


def _sec(header, lines, empty):
    return header + "\n" + "\n".join(lines or [empty])


def _landmark_line(surr):
    names = ", ".join(n for n, _ in surr.get("landmarks", [])) or "(none)"
    return f"Well-known places nearby (copy names exactly, do not add any): {names}"


# --- facet renders (same JSON, different information view) --------------------
def _dev_lines(surr):
    cats = surr.get("cats", {})
    items = []
    for b in _REF:  # iterate all buckets so notable ABSENCE is caught too
        rel = _relative(b, cats.get(b, [0, 0, 0])[1])
        if rel:
            items.append((_REL_ORDER[rel], _label(b), rel))
    items.sort()
    return [f"- {label}: {rel}" for _, label, rel in items]


def _prox_lines(surr):
    cats = surr.get("cats", {})
    return [f"- {_label(b)}: {_prox_word(v[2])}"
            for b, v in sorted(cats.items(), key=lambda kv: kv[1][2])]


def _comp_lines(surr):
    cats = surr.get("cats", {})
    total = sum(v[1] for v in cats.values())
    lines = []
    for b, v in sorted(cats.items(), key=lambda kv: -kv[1][1]):
        s = v[1] / total if total else 0
        # finer share bands -> more distinct mixes carry signal
        w = ("mostly" if s >= 0.30 else "lots of" if s >= 0.18 else "plenty of"
             if s >= 0.10 else "some" if s >= 0.05 else "a little" if s >= 0.02 else None)
        if w:
            lines.append(f"- {w} {_label(b)}")
    return lines


def _view_deviation(surr):
    return (_sec("How this block compares with a typical New York block "
                 "(only the ways it stands out are listed):", _dev_lines(surr),
                 "- (unremarkable — typical across the board)")
            + "\n" + _landmark_line(surr))


def _view_proximity(surr):
    return (_sec("How close each kind of place is:", _prox_lines(surr),
                 "- (nothing nearby)")
            + "\n" + _landmark_line(surr))


def _view_composition(surr):
    return (_sec("The mix of places nearby (share of everything around):",
                 _comp_lines(surr), "- (little of anything)")
            + "\n" + _landmark_line(surr))


def _view_landmarks(surr):
    return _landmark_line(surr)


def _view_combined(surr):
    return "\n\n".join([
        _sec("How this block compares with a typical New York block:",
             _dev_lines(surr), "- (typical across the board)"),
        _sec("How close each kind of place is:", _prox_lines(surr), "- (nothing nearby)"),
        _sec("The mix of places nearby:", _comp_lines(surr), "- (little of anything)"),
    ]) + "\n\n" + _landmark_line(surr)


_VIEWS = {
    "deviation": _view_deviation, "proximity": _view_proximity,
    "composition": _view_composition, "landmarks": _view_landmarks,
    "combined": _view_combined,
}


def prompt_for(row, note=""):
    """Render the enrichment JSON under the selected SURR_VIEW facet. Raw counts
    stay in the JSON for the tabular channel; each view exposes a different slice
    of signal the counts can't cheaply give the model."""
    surr = json.loads(row["surroundings"])
    return _VIEWS.get(SURR_VIEW, _view_deviation)(surr) + note


def _cache_csv():
    """Sidecar checkpoint path (in the misc artifacts dir). Named per OUT_CSV so
    prompt-screen runs that repoint OUT_CSV don't share a cache. It keeps the
    `index` key that the published dataset intentionally drops."""
    return os.path.join(config.ARTIFACTS_DIR, os.path.basename(OUT_CSV) + ".cache")


def load_cache():
    """index -> summary from a prior run — so we never re-pay for an existing description.

    Keyed on `index` (the stable per-listing id); `id` was dropped as a leaky column.
    Read from the sidecar cache, since the published CSV no longer carries `index`.
    """
    path = _cache_csv()
    if not os.path.exists(path):
        return {}
    prev = pd.read_csv(path, low_memory=False)
    if "index" not in prev or "surroundings_summary" not in prev:
        return {}
    prev = prev[prev["surroundings_summary"].notna()]
    return dict(zip(prev["index"], prev["surroundings_summary"]))


def save(df):
    """Checkpoint to the sidecar cache — keeps `index` for incremental resume."""
    os.makedirs(config.ARTIFACTS_DIR, exist_ok=True)
    cols = [c for c in ("surroundings",) if c in df.columns]
    df.drop(columns=cols).to_csv(_cache_csv(), index=False)


def publish(df):
    """Write the final dataset. LLM prose replaces the POI JSON; drop the internal
    `index` key so the published CSV has no leaky id column and renders no index."""
    os.makedirs(config.PROCESSED_DIR, exist_ok=True)
    drop = [c for c in ("surroundings", "index") if c in df.columns]
    df.drop(columns=drop).to_csv(OUT_CSV, index=False)


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
    os.makedirs(config.ARTIFACTS_DIR, exist_ok=True)
    with open(config.BATCH_JSON, "w") as f:
        json.dump(j, f)
    if st != "completed":
        print(f"batch ended {st} — see {config.BATCH_JSON}", flush=True)
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
        surr = json.loads(row["surroundings"])
        async with sem:
            # regenerate while the draft names places absent from the POI data
            note = ""
            for attempt in range(GROUND_RETRIES + 1):
                res = await call_with_retry(agent, prompt_for(row, note))
                bad = ungrounded(res.output, surr)
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
    load_reference(df)  # corpus distribution for the relative (deviation) prompt
    # densest surroundings first (cheap test runs hit the richest listings).
    # total POIs = sum of within-400m counts across buckets; derived on the fly.
    def _total(s):
        return sum(v[1] for v in json.loads(s).get("cats", {}).values())

    n_pois = df["surroundings"].map(_total)
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

    save(df)  # final checkpoint (cache, with index)
    publish(df)  # clean published dataset (no index / lat / long)
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
