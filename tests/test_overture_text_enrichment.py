from __future__ import annotations

import json
import unittest

import pandas as pd

from src import build, describe


class OvertureAggregationTest(unittest.TestCase):
    def test_groups_get_count_and_nearest_distance(self) -> None:
        nearby = pd.DataFrame(
            [
                {"group": "shopping", "dist": 310.4},
                {"group": "food_and_drink", "dist": 35.2},
                {"group": "food_and_drink", "dist": 55.0},
                {"group": "shopping", "dist": 420.0},
            ]
        )

        cats = build.aggregate_surroundings(nearby)

        self.assertEqual(cats, {"food_and_drink": [2, 35], "shopping": [2, 310]})
        self.assertEqual(list(cats), ["food_and_drink", "shopping"])


class DescriptionViewsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.surr = {
            "cats": {
                "food_and_drink": [3, 45],
                "shopping": [2, 250],
                "arts_and_entertainment": [1, 120],
            },
            "landmarks": [["Central Park", 350]],
        }

    def test_grounding_allows_only_supplied_landmarks(self) -> None:
        self.assertEqual(describe.ungrounded("Near Central Park.", self.surr), [])
        self.assertEqual(describe.ungrounded("Near Alpha Cafe.", self.surr), ["Near Alpha Cafe"])

    def test_proximity_skips_primary_groups_and_far_places(self) -> None:
        lines = describe._proximity_lines(self.surr, {"food_and_drink"})
        self.assertEqual(lines, ["- arts and entertainment: steps away"])

    def test_price_profile_uses_only_top_level_groups(self) -> None:
        describe.load_reference(pd.DataFrame({"surroundings": [json.dumps(self.surr)]}))
        profile = describe._view_price_relevant_profile(self.surr)
        self.assertIn("Central Park", profile)
        for line in profile.splitlines():
            if line.startswith("- ") and ":" in line and "Central Park" not in line:
                label = line[2:].split(":")[0]
                self.assertIn(label.replace(" ", "_"), describe.config.OVERTURE_GROUPS)

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
        self.assertIn("- shopping: far more than most blocks", view)
        self.assertNotIn("%", view)


if __name__ == "__main__":
    unittest.main()
