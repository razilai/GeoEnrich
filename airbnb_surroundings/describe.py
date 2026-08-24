"""Summarise each listing's surroundings with an LLM (via OpenRouter).

Reads data/processed/airbnb_enriched.csv (from build.py — needs the `surroundings`
column of cleaned POI JSON), asks the model for a short free-text description
of the area, and writes a `surroundings_summary` column to airbnb_described.csv.

Config comes from .env:
    OPENROUTER_API_KEY  — your OpenRouter key
    LLM_MODEL           — model slug, e.g. openai/gpt-4o-mini
    LLM_THINKING        — set to false to explicitly disable model reasoning

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


def _thinking_disabled() -> bool:
    """Whether to explicitly request OpenRouter's no-reasoning mode."""
    return os.environ.get("LLM_THINKING", "").strip().lower() in {
        "0", "false", "no", "off", "none",
    }

from airbnb_surroundings import config

IN_CSV = os.environ.get("DESC_IN", config.ENRICHED_CSV)
OUT_CSV = os.environ.get("DESC_OUT", config.DESCRIBED_CSV)
# An optional full enriched corpus used only to compute citywide reference
# distributions. Prompt screens can therefore enrich a small fixed sample while
# retaining genuinely citywide percentile language in deviation-based views.
REFERENCE_CSV = os.environ.get("DESC_REFERENCE_CSV")
# Prompt screens set this to a stable fingerprint of the prompt/view/reference
# configuration. It prevents one variant from reusing drafts made with an earlier
# version of that configuration while preserving normal incremental production runs.
CACHE_TAG = os.environ.get("DESC_CACHE_TAG")

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
# LOCKED to the 08 variant (deviation_exact): local-guide voice over exact citywide
# percentiles. The prompt screen picked it, so the pipeline no longer branches on an
# env var — the other views/instructions are kept only for reference in this module.
SURR_VIEW = "deviation_exact"

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
    # local-guide voice over exact citywide percentiles (was experiments prompt 08).
    # Self-contained: carries its own rules (do not quote raw numbers/percentiles),
    # so it does NOT append _RULES. Pair with SURR_VIEW=deviation_exact input view.
    "deviation_exact": "You are a knowledgeable local helping a renter picture a "
    "listing's location. You are given exact figures for how its block ranks among "
    'all New York blocks: each standout amenity as a citywide percentile — "top X% '
    'of NYC blocks" means it has more of that than most blocks (the smaller the X, '
    'the denser), "bottom X%" means it has unusually few. In one or two grounded '
    "sentences, translate these ranks into natural, calibrated language — say what "
    "kind of area it is and where it has unusually many or few things, letting the "
    "rank set how strongly you phrase it (top 5% = far more than usual; top 30% = "
    "somewhat more; bottom 10% = unusually few). Do not quote the raw numbers, "
    "percentiles, counts, or metres in your answer. Be honest: a block low on "
    "something is quiet or limited, not vibrant. Use only what is given; never invent "
    "or add a place; copy any listed name exactly. Plain "
    "text, no markdown.",
    "named_destinations": "You describe a listing's location through a short, "
    "grounded set of named nearby places and well-known landmarks. In one or two "
    "sentences, explain what these destinations suggest about the area and its "
    "convenience. Name only places in the supplied summary, copy names exactly, and "
    "do not invent places, price, scores, counts, or distances. Plain text, no markdown.",
    "access_mix": "You describe a listing's local access and place mix from the "
    "nearby place types and distance bands provided. In one or two grounded sentences, "
    "say what is convenient or limited and what kind of activity dominates. Use only "
    "the supplied summary; do not invent places, price, scores, counts, or distances. "
    "Plain text, no markdown.",
    "overture_environment": "You are a knowledgeable local helping a renter picture "
    "a listing's environment. From the supplied nearby places and place types, write "
    "one or two grounded sentences about what is immediately distinctive or convenient. "
    "Mention at most two listed names, copy them exactly, and do not add places. Do not "
    "quote counts or distances, or mention price, tier, or rating. Plain text, no markdown.",
    "deviation_anchored_environment": "You are a knowledgeable local helping a renter "
    "picture a listing's location. The citywide contrasts are the primary evidence; "
    "named anchors are optional grounding, not evidence that an ordinary business "
    "defines the area. In one or two grounded sentences, first translate one or two "
    "strongest contrasts into the block's character. Only then, if a meaningful anchor "
    "is supplied, use at most one such name to make that character concrete. Do not use "
    "ordinary business names; supplied names are limited to curated landmarks and major "
    "rail, ferry, or airport infrastructure. Do not quote percentiles, counts, or "
    "metres, or mention price, tier, or rating. Use only what is supplied and copy any "
    "used name exactly. Plain text, no markdown.",
}
INSTRUCTIONS = _VIEW_INSTRUCTIONS[SURR_VIEW]  # 08 variant, locked

_NAME_RE = re.compile(r"[A-Z][\w&'’]+(?:\s+[A-Z][\w&'’]+)*")

# regenerate a draft that names places not in the POI data. 0 for this prompt: the
# landmark "why it matters" clause adds world-knowledge commentary (proper nouns not
# in the list), so the strict closed-world name check would misfire and regen rows.
GROUND_RETRIES = int(os.environ.get("GROUND_RETRIES", "0"))

# live token/call accounting across every LLM call (incl. retries) for cost estimation
USAGE = {"calls": 0, "in": 0, "out": 0, "cost": 0.0}


# surroundings schema: {
#   "cats": {coarse_bucket: [count<=150m, count<=450m, nearest_m]},
#   "fine_cats": {overture_leaf: [count<=150m, count<=450m, nearest_m]},
#   "pois": [{"name", "category", "bucket", "distance_m", "ring"}, ...],
#   "landmarks": [[name, dist_m], ...],
# }
def _poi_names(surr):
    landmarks = {n for n, _ in surr.get("landmarks", [])}
    overture_places = {p["name"] for p in surr.get("pois", []) if p.get("name")}
    return sorted(landmarks | overture_places)


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
    # LLM_TEMPERATURE overrides for sweeps.
    settings = {"temperature": float(os.environ.get("LLM_TEMPERATURE", "0.4"))}
    if _thinking_disabled():
        settings["thinking"] = False
    return Agent(
        model,
        output_type=str,
        instructions=INSTRUCTIONS,
        model_settings=settings,
    )


_DISPLAY = {
    "dining": "dining",
    "cafe": "cafes",
    "nightlife": "bars & nightlife",
    "grocery": "grocery stores",
    "shopping": "shops",
    "fitness_sport": "gyms & sports",
    "park_green": "parks & green space",
    "culture": "museums & galleries",
    "landmark": "attractions",
    "entertainment": "entertainment venues",
    "lodging": "hotels",
    "transit": "subway & transit",
}

# The signal is DEVIATION from a typical block, not presence: ~98% of listings
# have dining/shops/grocery, so listing them says nothing. We compare each
# bucket's count against the corpus distribution and surface only the standouts.
_REF = {}  # bucket -> sorted np.array of within-450m counts across the corpus
_PRESENT = {}  # bucket -> fraction of listings with the bucket present
_FINE_REF = {}  # Overture leaf -> sorted corpus count (schema-v2 inputs only)

# Fine Overture leaves that add a distinct residential or visitor-access signal.
# Ordinary shops/services are deliberately absent: their names and mere presence add
# lexical variety but little independent price signal after coarse density is known.
_PROFILE_FINE_CLASSES = {
    "rapid transit": (
        "transit",
        {
            "train_station", "metro_station", "subway_station", "light_rail_station",
            "tram_station", "ferry_terminal", "airport",
        },
    ),
    "parks and open space": (
        "park_green",
        {
            "park", "botanical_garden", "beach", "plaza", "trail", "national_park",
            "state_park", "memorial_park",
        },
    ),
    "cultural venues": (
        "culture",
        {
            "museum", "art_gallery", "cultural_center", "performing_arts_theater",
            "movie_theater", "theater",
        },
    ),
    "grocery and markets": (
        "grocery",
        {
            "supermarket", "grocery_store", "organic_grocery_store", "farmers_market",
            "public_market", "health_market",
        },
    ),
    "hotels": ("lodging", {"hotel", "resort_hotel", "boutique_hotel", "motel"}),
    "sports facilities": (
        "fitness_sport", {"stadium", "sports_center", "sports_club", "gym", "swimming_pool"}
    ),
}


def load_reference(df):
    """Build the corpus count distribution per bucket (call once before prompting)."""
    _REF.clear()
    _PRESENT.clear()
    _FINE_REF.clear()
    cats = df["surroundings"].map(lambda s: json.loads(s).get("cats", {}))
    for b in {k for c in cats for k in c}:
        arr = np.array([c.get(b, [0, 0, 0])[1] for c in cats])
        _REF[b] = np.sort(arr)
        _PRESENT[b] = float((arr > 0).mean())
    fine_cats = df["surroundings"].map(lambda s: json.loads(s).get("fine_cats", {}))
    for category in {k for values in fine_cats for k in values}:
        arr = np.array([values.get(category, [0, 0, 0])[1] for values in fine_cats])
        _FINE_REF[category] = np.sort(arr)


def _relative(bucket, v):
    """(band word, extremity) for how this bucket compares with the corpus, or
    None if ~typical. Extremity is |percentile - 0.5| so callers can rank the
    sharpest deviations first. Finer bands (six levels) keep the gradation in
    prose; absence (where the bucket is usually present) scores max extremity."""
    arr = _REF.get(bucket)
    if arr is None or len(arr) == 0:
        return None
    if v <= 0:  # absence is signal only where the bucket is usually present
        return ("none nearby", 0.5) if _PRESENT.get(bucket, 0) >= 0.6 else None
    pct = np.searchsorted(arr, v, side="left") / len(arr)
    ext = abs(pct - 0.5)
    if pct >= 0.95:
        return "far more than most blocks", ext
    if pct >= 0.80:
        return "well above average", ext
    if pct >= 0.65:
        return "above average", ext
    if pct <= 0.05:
        return "almost none", ext
    if pct <= 0.20:
        return "well below average", ext
    if pct <= 0.35:
        return "below average", ext
    return None  # middle third ~typical → omit


# Cram 9 standouts into "one or two sentences" and the model averages them to a
# generic "dense area" — the band distinctions collapse. Surface only the few
# sharpest so it writes the distinctive thing (keeps the most price-relevant
# deviations, kills the cross-block prose collapse).
_DEV_CAP = 5


def _prox_word(m):
    return (
        "on the doorstep" if m <= 50 else "steps away" if m <= 150 else "a short walk"
    )


def _label(b):
    return _DISPLAY.get(b, b.replace("_", " "))


def _sec(header, lines, empty):
    return header + "\n" + "\n".join(lines or [empty])


# Landmark distance bands (air metres, matching build.py dist_m). Convex spacing:
# landmark price premium is steepest at the doorstep, so bands are tight near and
# wide far — resolution where the signal lives, not evenly across the 800m reach.
_LM_BANDS = [
    (100, "right by"),
    (250, "a couple minutes from"),
    (500, "a short walk from"),
    (float("inf"), "about 10 minutes from"),
]


def _landmark_line(surr):
    groups = {}  # band phrase -> [names]; landmarks arrive nearest-first (build.py)
    for n, m in surr.get("landmarks", []):
        for cutoff, phrase in _LM_BANDS:
            if m <= cutoff:
                groups.setdefault(phrase, []).append(n)
                break
    lines = [
        f"- {phrase}: {', '.join(groups[phrase])}"
        for _, phrase in _LM_BANDS
        if phrase in groups
    ]
    if not lines:
        return "There are no well-known landmarks nearby."
    return _sec(
        "Well-known places nearby (copy names exactly, do not add any):", lines, ""
    )


_RING_LABELS = {
    "doorstep": "on the doorstep",
    "nearby": "nearby",
    "walk": "a short walk away",
}


def _fine_label(category: str) -> str:
    """Make an Overture leaf code readable without changing its meaning."""
    return category.replace("_", " ")


def _named_poi_lines(surr):
    """Render the deterministic Overture representatives grouped by distance ring."""
    grouped: dict[str, list[str]] = {}
    for poi in surr.get("pois", []):
        name = poi.get("name")
        if not name:
            continue
        label = _fine_label(str(poi.get("category", poi.get("bucket", "place"))))
        grouped.setdefault(str(poi.get("ring", "walk")), []).append(f"{name} ({label})")
    return [
        f"- {_RING_LABELS[ring]}: {', '.join(grouped[ring])}"
        for ring in ("doorstep", "nearby", "walk")
        if ring in grouped
    ]


_MEANINGFUL_TRANSIT_CATEGORIES = {
    "train_station",
    "metro_station",
    "subway_station",
    "light_rail_station",
    "tram_station",
    "ferry_terminal",
    "airport",
}


def _meaningful_anchor_lines(surr, cap=2):
    """Return only named POIs that can genuinely anchor area character.

    The representative-POI selection deliberately covers every amenity bucket, so
    it often includes a useful-but-generic local business. Those names add lexical
    variety but should not be allowed to define a block's price-relevant character.
    Curated landmarks are rendered separately. Overture names are limited to major
    rail/ferry/airport infrastructure: a nearby business, gallery, or even a small
    park is too weak a signal to define a listing's location in prose.
    """
    anchors = []
    seen = set()
    for poi in surr.get("pois", []):
        name = poi.get("name")
        if not name or poi.get("category") not in _MEANINGFUL_TRANSIT_CATEGORIES:
            continue
        key = " ".join(str(name).split()).casefold()
        if not key or key in seen:
            continue
        seen.add(key)
        label = _fine_label(str(poi.get("category", poi["bucket"])))
        ring = _RING_LABELS.get(str(poi.get("ring", "walk")), "nearby")
        anchors.append(f"- {ring}: {name} ({label})")
        if len(anchors) == cap:
            break
    return anchors


def _fine_access_lines(surr, cap=8):
    """Select useful fine Overture categories by local density then proximity."""
    items = []
    for category, values in surr.get("fine_cats", {}).items():
        doorstep, total, nearest_m = values
        items.append((total, doorstep, nearest_m, category))
    items.sort(key=lambda item: (-item[0], -item[1], item[2], item[3]))
    return [
        f"- {_fine_label(category)}: {_prox_word(nearest_m)}"
        for _, _, nearest_m, category in items[:cap]
    ]


def _profile_fine_lines(surr, excluded_buckets: set[str], cap=2):
    """Pick independent, price-relevant fine Overture evidence.

    A fine class already represented by a primary coarse contrast is skipped. This
    gives the prose a second axis (for example, park access beside retail density)
    instead of restating that a dense commercial block has shops nearby.
    """
    fine = surr.get("fine_cats", {})
    choices = []
    for label, (bucket, leaves) in _PROFILE_FINE_CLASSES.items():
        if bucket in excluded_buckets:
            continue
        observed = [
            (leaf, values) for leaf, values in fine.items()
            if leaf in leaves and values[1] > 0
        ]
        if not observed:
            continue

        def rank(item):
            leaf, values = item
            arr = _FINE_REF.get(leaf)
            pct = (
                np.searchsorted(arr, values[1], side="left") / len(arr)
                if arr is not None and len(arr)
                else 0.5
            )
            return (abs(pct - 0.5), -values[2], leaf)

        leaf, values = max(observed, key=rank)
        arr = _FINE_REF.get(leaf)
        pct = (
            np.searchsorted(arr, values[1], side="left") / len(arr)
            if arr is not None and len(arr)
            else 0.5
        )
        score = abs(pct - 0.5) + (0.25 if values[2] <= 150 else 0.0)
        choices.append((score, values[2], label))
    choices.sort(key=lambda item: (-item[0], item[1], item[2]))
    return [
        f"- {label}: {_prox_word(nearest_m)}"
        for _, nearest_m, label in choices[:cap]
    ]


# --- facet renders (same JSON, different information view) --------------------
def _dev_lines(surr, cap=_DEV_CAP):
    cats = surr.get("cats", {})
    items = []
    for b in _REF:  # iterate all buckets so notable ABSENCE is caught too
        r = _relative(b, cats.get(b, [0, 0, 0])[1])
        if r:
            rel, ext = r
            items.append((ext, _label(b), rel))
    items.sort(key=lambda t: (-t[0], t[1]))  # sharpest deviation first
    return [f"- {label}: {rel}" for _, label, rel in items[:cap]]


def _dev_exact(bucket, v):
    """(extremity, exact-relative phrase) vs the corpus, or None if ~typical.
    Same signal as _relative but numeric — the citywide percentile, phrased as a
    self-describing rank (top/bottom X% of NYC blocks) so the LLM reads it at face
    value with no scale to misinterpret. Absence is surfaced only where the bucket
    is usually present."""
    arr = _REF.get(bucket)
    if arr is None or len(arr) == 0:
        return None
    pct = np.searchsorted(arr, v, side="left") / len(arr)
    ext = abs(pct - 0.5)
    p = round(pct * 100)
    if v <= 0:
        if _PRESENT.get(bucket, 0) < 0.6:
            return None  # absent-and-usually-absent → no signal
        return ext, "none nearby (most NYC blocks have some)"
    if ext < 0.15:
        return None  # middle third ~typical → omit (mirrors the banded view)
    if pct >= 0.5:
        return ext, f"top {100 - p}% of NYC blocks"
    return ext, f"bottom {p}% of NYC blocks"


def _dev_lines_exact(surr, cap=_DEV_CAP):
    cats = surr.get("cats", {})
    items = []
    for b in _REF:  # iterate all buckets so notable ABSENCE is caught too
        r = _dev_exact(b, cats.get(b, [0, 0, 0])[1])
        if r:
            ext, txt = r
            items.append((ext, _label(b), txt))
    items.sort(key=lambda t: (-t[0], t[1]))  # sharpest deviation first
    return [f"- {label}: {txt}" for _, label, txt in items[:cap]]


def _prox_lines(surr):
    cats = surr.get("cats", {})
    return [
        f"- {_label(b)}: {_prox_word(v[2])}"
        for b, v in sorted(cats.items(), key=lambda kv: kv[1][2])
    ]


def _comp_lines(surr):
    cats = surr.get("cats", {})
    total = sum(v[1] for v in cats.values())
    lines = []
    for b, v in sorted(cats.items(), key=lambda kv: -kv[1][1]):
        s = v[1] / total if total else 0
        # finer share bands -> more distinct mixes carry signal
        w = (
            "mostly"
            if s >= 0.30
            else "lots of"
            if s >= 0.18
            else "plenty of"
            if s >= 0.10
            else "some"
            if s >= 0.05
            else "a little"
            if s >= 0.02
            else None
        )
        if w:
            lines.append(f"- {w} {_label(b)}")
    return lines


def _view_deviation(surr):
    return (
        _sec(
            "How this block compares with a typical New York block "
            "(only the ways it stands out are listed):",
            _dev_lines(surr),
            "- (unremarkable — typical across the board)",
        )
        + "\n"
        + _landmark_line(surr)
    )


def _view_deviation_exact(surr):
    return (
        _sec(
            "How this block ranks among all New York blocks "
            "(citywide percentile; only standouts listed):",
            _dev_lines_exact(surr),
            "- (unremarkable — typical across the board)",
        )
        + "\n"
        + _landmark_line(surr)
    )


def _view_named_destinations(surr):
    return (
        _sec(
            "Selected named places nearby (copy names exactly, do not add any):",
            _named_poi_lines(surr),
            "- (no selected named places nearby)",
        )
        + "\n"
        + _landmark_line(surr)
    )


def _view_access_mix(surr):
    return (
        _sec(
            "Nearby place types and access:",
            _fine_access_lines(surr),
            "- (no nearby place types recorded)",
        )
        + "\n"
        + _landmark_line(surr)
    )


def _view_overture_environment(surr):
    """Compact combined Overture context for the minimalist production candidate."""
    return "\n\n".join(
        [
            _sec(
                "Selected named places nearby (copy names exactly, do not add any):",
                _named_poi_lines(surr),
                "- (no selected named places nearby)",
            ),
            _sec(
                "Nearby place types and access:",
                _fine_access_lines(surr, cap=6),
                "- (no nearby place types recorded)",
            ),
            _landmark_line(surr),
        ]
    )


def _view_deviation_anchored_environment(surr):
    """Strong citywide contrast signal, grounded only by meaningful local anchors."""
    return "\n\n".join(
        [
            _sec(
                "Primary evidence — how this block ranks among all New York blocks "
                "(citywide percentile; only strongest contrasts listed):",
                _dev_lines_exact(surr, cap=4),
                "- (unremarkable — typical across the measured place types)",
            ),
            _sec(
                "Optional meaningful Overture anchors (ground the description only; "
                "do not let ordinary businesses define the area):",
                _meaningful_anchor_lines(surr),
                "- (no named rail, ferry, or airport anchor selected)",
            ),
            _landmark_line(surr),
        ]
    )


def _view_price_relevant_profile(surr):
    """Middle-ground view: coarse contrasts plus non-redundant fine access."""
    cats = surr.get("cats", {})
    ranked = []
    for bucket in _REF:
        result = _dev_exact(bucket, cats.get(bucket, [0, 0, 0])[1])
        if result:
            ranked.append((result[0], bucket))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    selected_buckets = {bucket for _, bucket in ranked[:3]}
    return "\n\n".join(
        [
            _sec(
                "Primary evidence — strongest citywide contrasts:",
                _dev_lines_exact(surr, cap=3),
                "- (unremarkable — typical across the measured place types)",
            ),
            _sec(
                "Independent price-relevant access evidence (use only if it adds a new idea):",
                _profile_fine_lines(surr, selected_buckets, cap=2),
                "- (no additional selected access evidence)",
            ),
            _sec(
                "Optional meaningful anchors (copy a used name exactly):",
                _meaningful_anchor_lines(surr, cap=1),
                "- (no named rail, ferry, or airport anchor selected)",
            ),
            _landmark_line(surr),
        ]
    )


def _view_proximity(surr):
    return (
        _sec(
            "How close each kind of place is:", _prox_lines(surr), "- (nothing nearby)"
        )
        + "\n"
        + _landmark_line(surr)
    )


def _view_composition(surr):
    return (
        _sec(
            "The mix of places nearby (share of everything around):",
            _comp_lines(surr),
            "- (little of anything)",
        )
        + "\n"
        + _landmark_line(surr)
    )


def _view_landmarks(surr):
    return _landmark_line(surr)


def _view_combined(surr):
    return (
        "\n\n".join(
            [
                _sec(
                    "How this block compares with a typical New York block:",
                    _dev_lines(surr),
                    "- (typical across the board)",
                ),
                _sec(
                    "How close each kind of place is:",
                    _prox_lines(surr),
                    "- (nothing nearby)",
                ),
                _sec(
                    "The mix of places nearby:",
                    _comp_lines(surr),
                    "- (little of anything)",
                ),
            ]
        )
        + "\n\n"
        + _landmark_line(surr)
    )


_VIEWS = {
    "deviation": _view_deviation,
    "deviation_exact": _view_deviation_exact,
    "character_deviation": _view_deviation_exact,
    "named_destinations": _view_named_destinations,
    "access_mix": _view_access_mix,
    "overture_environment": _view_overture_environment,
    "deviation_anchored_environment": _view_deviation_anchored_environment,
    "price_relevant_profile": _view_price_relevant_profile,
    "proximity": _view_proximity,
    "composition": _view_composition,
    "landmarks": _view_landmarks,
    "combined": _view_combined,
}
# Bump a view's value whenever its rendered evidence changes. Prompt screens add it
# to their cache fingerprint, so a corrected renderer cannot reuse old LLM drafts.
_VIEW_CACHE_VERSIONS = {
    "deviation_anchored_environment": "v2",
    "price_relevant_profile": "v1",
}


def prompt_for(row, note=""):
    """Render the enrichment JSON under the selected SURR_VIEW facet. Raw counts
    stay in the JSON for the tabular channel; each view exposes a different slice
    of signal the counts can't cheaply give the model."""
    surr = json.loads(row["surroundings"])
    return _VIEWS[SURR_VIEW](surr) + note  # 08 variant, locked


def _cache_csv():
    """Sidecar checkpoint path (in the misc artifacts dir). Named per OUT_CSV so
    prompt-screen runs that repoint OUT_CSV don't share a cache. It keeps the
    `index` key that the published dataset intentionally drops."""
    suffix = f".{CACHE_TAG}" if CACHE_TAG else ""
    return os.path.join(config.ARTIFACTS_DIR, os.path.basename(OUT_CSV) + suffix + ".cache")


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
BATCH_POLL = float(os.environ.get("BATCH_POLL", "20"))  # seconds between status polls
# OpenRouter reserves credits up front = requests × max completion tokens. Cap the
# per-request completion and cap the batch size so the reservation stays small enough
# to clear against the account balance (an over-large reservation returns 402).
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "3000"))  # requests per submitted batch
BATCH_MAX_TOKENS = int(os.environ.get("BATCH_MAX_TOKENS", "400"))  # summary is short


async def _submit_batch(c, headers, rows, df):
    """Submit one batch of `rows`, poll to completion, parse results into df."""
    batch_settings = {
        "temperature": float(os.environ.get("LLM_TEMPERATURE", "0.4")),
        "max_tokens": BATCH_MAX_TOKENS,  # caps the up-front credit reservation
        "usage": {"include": True},  # ask OpenRouter to return usage.cost
    }
    if _thinking_disabled():
        batch_settings["reasoning"] = {"effort": "none"}

    reqs = [
        {
            "custom_id": str(row["index"]),
            "body": {
                "messages": [
                    {"role": "system", "content": INSTRUCTIONS},
                    {"role": "user", "content": prompt_for(row)},
                ],
                **batch_settings,
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
            print(
                f"--- sub-batch {b + 1}/{n_batches} ({len(sub)} rows) ---", flush=True
            )
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
    if REFERENCE_CSV:
        reference_df = pd.read_csv(REFERENCE_CSV, usecols=["surroundings"], low_memory=False)
        print(f"reference distribution: {len(reference_df)} listings from {REFERENCE_CSV}", flush=True)
    else:
        reference_df = df
    load_reference(reference_df)  # corpus distribution for the relative (deviation) prompt

    # densest surroundings first (cheap test runs hit the richest listings).
    # total POIs = sum of within-450m counts across buckets; derived on the fly.
    def _total(s):
        return sum(v[1] for v in json.loads(s).get("cats", {}).values())

    # DESC_SAMPLE=random draws a random k (seed DESC_SEED) for a representative
    # distinctiveness check; default keeps densest-first for cheap rich test runs.
    if os.environ.get("DESC_SAMPLE") == "random":
        n = k if k is not None else len(df)
        df = df.sample(
            n=min(n, len(df)), random_state=int(os.environ.get("DESC_SEED", "0"))
        )
    else:
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
