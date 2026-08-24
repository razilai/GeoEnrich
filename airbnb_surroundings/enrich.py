"""Full-corpus enrichment pinned to prompt variant 16.

Run via ``uv run enrich``.  Model and credentials are inherited from the
current environment; this module deliberately fixes every dataset/prompt choice
so a resumed run cannot accidentally mix prompt variants.
"""

from airbnb_surroundings import config, describe


PROMPT_16 = """Write one or two grounded, neutral sentences that combine the strongest citywide contrasts with one independent access or fine place-type cue. If a supplied landmark or major rail, ferry, or airport name makes the location more specific, add at most one such name; never use ordinary business names. Keep each fact distinct rather than listing amenities. Do not begin with “This area”, “This block”, or “is characterized by”. Do not mention price, tier, rating, percentiles, counts, or metres, and do not use promotional or unsupported language. Plain text, no markdown."""


def main() -> None:
    """Enrich the full fine-schema corpus with the fixed prompt-16 configuration."""
    describe.IN_CSV = config.ENRICHED_FINE_CSV
    describe.OUT_CSV = config.DESCRIBED_16_CSV
    describe.REFERENCE_CSV = config.ENRICHED_FINE_CSV
    describe.SURR_VIEW = "price_relevant_profile"
    describe.INSTRUCTIONS = PROMPT_16
    describe.CACHE_TAG = "prompt16_price_relevant_profile_v1"
    describe.main()


if __name__ == "__main__":
    main()
