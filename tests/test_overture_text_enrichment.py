from __future__ import annotations

import json
import unittest

import numpy as np
import pandas as pd

from src import build, describe


class OvertureAggregationTest(unittest.TestCase):
    def test_groups_get_count_and_nearest_distance(self) -> None:
        nearby = pd.DataFrame(
            [
                {"group": "hotel", "dist": 310.4},
                {"group": "restaurant", "dist": 35.2},
                {"group": "restaurant", "dist": 55.0},
                {"group": "hotel", "dist": 420.0},
            ]
        )

        cats = build.aggregate_surroundings(nearby)

        self.assertEqual(cats, {"restaurant": [2, 35], "hotel": [2, 310]})
        self.assertEqual(list(cats), ["restaurant", "hotel"])


class DescriptionViewsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.surr = {
            "cats": {
                "restaurant": [3, 45],
                "hotel": [2, 250],
                "museum": [1, 120],
            },
            "landmarks": [["Central Park", 350]],
        }

    def test_grounding_allows_only_supplied_landmarks(self) -> None:
        self.assertEqual(describe.ungrounded("Near Central Park.", self.surr), [])
        self.assertEqual(describe.ungrounded("Near Alpha Cafe.", self.surr), ["Near Alpha Cafe"])

    def test_proximity_skips_primary_groups_and_far_places(self) -> None:
        lines = describe._proximity_lines(self.surr, {"restaurant"})
        self.assertEqual(lines, ["- museums: steps away"])

    def test_proximity_ignores_categories_near_almost_everywhere(self) -> None:
        self.assertEqual(describe._proximity_lines(self.surr, set()), ["- museums: steps away"])

    def test_price_profile_uses_only_kept_categories(self) -> None:
        describe.load_reference(pd.DataFrame({"surroundings": [json.dumps(self.surr)]}))
        profile = describe._view_price_relevant_profile(self.surr)
        self.assertIn("Central Park", profile)
        for line in profile.splitlines():
            if line.startswith("- ") and ":" in line and "Central Park" not in line:
                label = line[2:].split(":")[0]
                kept = {describe._label(c) for c in describe.config.OVERTURE_CATEGORIES}
                self.assertIn(label, kept)

    def test_price_profile_shuffle_is_reproducible_and_varies(self) -> None:
        describe.load_reference(pd.DataFrame({"surroundings": [json.dumps(self.surr)]}))
        profile = describe._view_price_relevant_profile
        self.assertEqual(profile(self.surr), profile(dict(self.surr)))
        firsts = set()
        for i in range(20):
            surr = {**self.surr, "landmarks": [[f"Place {i}", 350]]}
            firsts.add(profile(surr).split("\n", 1)[0])
        self.assertGreater(len(firsts), 1)

    def test_every_kept_category_has_a_plain_label(self) -> None:
        self.assertEqual(set(describe._LABELS), set(describe.config.OVERTURE_CATEGORIES))

    def test_midrank_keeps_a_shared_count_mid_pack(self) -> None:
        ref = np.array([0] * 8 + [1, 2])
        self.assertAlmostEqual(describe._percentile(ref, 0), 0.4)
        self.assertAlmostEqual(describe._percentile(ref, 2), 0.95)

    def test_rank_needs_enough_places_and_never_says_top_zero(self) -> None:
        self.assertIsNone(describe._rank("museum", 2, 0.99))
        self.assertEqual(describe._rank("museum", 5, 0.999)[1], "top 1% of NYC neighbourhoods")
        self.assertIsNone(describe._rank("museum", 5, 0.7))

    def test_contrast_bands_keep_rank_selection_without_numbers(self) -> None:
        ref = np.arange(1000)  # count == its own rank
        describe._REF["museum"] = ref
        band = lambda count: describe._contrast_band("museum", {"museum": [count, 0]})
        self.assertEqual(band(995)[1], "more than almost anywhere in NYC")
        self.assertEqual(band(960)[1], "far more than most neighbourhoods")
        self.assertEqual(band(900)[1], "more than most neighbourhoods")
        self.assertIsNone(band(500))
        self.assertEqual(band(120)[1], "fewer than most neighbourhoods")
        self.assertEqual(band(30)[1], "very few")

    def test_deviations_keep_one_category_per_level1_group(self) -> None:
        surr = {"cats": {"restaurant": [9, 10], "casual_eatery": [9, 10]}}
        typical = {"cats": {"restaurant": [1, 10], "casual_eatery": [1, 10]}}
        corpus = [json.dumps(typical)] * 9 + [json.dumps(surr)]
        describe.load_reference(pd.DataFrame({"surroundings": corpus}))
        groups = [g for _, g, _ in describe._deviations(surr, describe._deviation_rank)]
        self.assertEqual(groups, ["casual_eatery"])  # both food_and_drink; one kept

    def test_landmarks_are_capped_to_the_nearest(self) -> None:
        surr = {"landmarks": [[f"Place {i}", 100 * i] for i in range(1, 8)]}
        line = describe._landmark_line(surr)
        self.assertIn("Place 3", line)
        self.assertNotIn("Place 4", line)

    def test_cache_tag_separates_prompt_variant_checkpoints(self) -> None:
        previous = describe.CACHE_TAG
        try:
            describe.CACHE_TAG = "prompt-fingerprint"
            self.assertTrue(describe._cache_csv().endswith(".prompt-fingerprint.cache"))
        finally:
            describe.CACHE_TAG = previous

    def test_official_prompts_have_views_and_own_outputs(self) -> None:
        self.assertIn(describe.DEFAULT_PROMPT, describe.PROMPTS)
        previous = (describe.SURR_VIEW, describe.INSTRUCTIONS, describe.OUT_CSV)
        try:
            outputs, caches = set(), set()
            for prompt_id, prompt in describe.PROMPTS.items():
                self.assertIn(prompt["view"], describe._VIEWS)
                describe.use_prompt(prompt_id)
                self.assertEqual(describe.SURR_VIEW, prompt["view"])
                outputs.add(describe.OUT_CSV)
                caches.add(describe._cache_csv())
            self.assertEqual(len(outputs), len(describe.PROMPTS))
            self.assertEqual(len(caches), len(describe.PROMPTS))
        finally:
            describe.SURR_VIEW, describe.INSTRUCTIONS, describe.OUT_CSV = previous

    def test_banded_view_hides_percentiles(self) -> None:
        describe.load_reference(
            pd.DataFrame(
                {
                    "surroundings": [
                        json.dumps({"cats": {"shopping": [n, 100]}}) for n in range(1, 21)
                    ]
                }
            )
        )
        view = describe._view_deviation({"cats": {"shopping": [20, 100]}, "landmarks": []})
        self.assertIn("- shopping: far more than most neighbourhoods", view)
        self.assertNotIn("%", view)


if __name__ == "__main__":
    unittest.main()
