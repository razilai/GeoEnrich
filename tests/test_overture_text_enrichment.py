from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from airbnb_surroundings import build, describe
from experiments.prompt_datasets import stratified_sample


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_PATH = ROOT / "experiments" / "datasets-analysis" / "analyze_text_columns.py"
spec = importlib.util.spec_from_file_location("text_analysis", ANALYSIS_PATH)
text_analysis = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = text_analysis
spec.loader.exec_module(text_analysis)


class OvertureAggregationTest(unittest.TestCase):
    def test_representatives_are_deduplicated_and_bounded(self) -> None:
        nearby = pd.DataFrame(
            [
                {"name": "  Alpha Cafe ", "category": "cafe", "dist": 35},
                {"name": "alpha cafe", "category": "cafe", "dist": 55},
                {"name": "Beta Station", "category": "subway_station", "dist": 110},
                {"name": "Gamma Gallery", "category": "art_gallery", "dist": 240},
                {"name": "Delta Books", "category": "bookstore", "dist": 310},
                {"name": "Epsilon Gym", "category": "gym", "dist": 350},
                {"name": "Zeta Market", "category": "supermarket", "dist": 420},
                {"name": "Eta Cinema", "category": "movie_theater", "dist": 430},
            ]
        )

        cats, fine_cats, places = build.aggregate_surroundings(nearby)

        self.assertEqual(cats[build.bucket("cafe")], [2, 2, 35])
        self.assertEqual(fine_cats["cafe"], [2, 2, 35])
        self.assertLessEqual(len(places), build.MAX_POI_EXAMPLES)
        self.assertEqual(sum(place["name"].casefold() == "alpha cafe" for place in places), 1)
        self.assertEqual(places[0]["ring"], "doorstep")
        self.assertTrue({"nearby", "walk"} & {place["ring"] for place in places})

    def test_distance_rings_cover_the_configured_radius(self) -> None:
        self.assertEqual(build.distance_ring(150), "doorstep")
        self.assertEqual(build.distance_ring(151), "nearby")
        self.assertEqual(build.distance_ring(301), "walk")


class DescriptionViewsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.surr = {
            "cats": {},
            "fine_cats": {
                "cafe": [1, 3, 45],
                "bookstore": [0, 2, 250],
            },
            "pois": [
                {"name": "Alpha Cafe", "category": "cafe", "bucket": "cafe", "distance_m": 45, "ring": "doorstep"},
                {"name": "Beta Books", "category": "bookstore", "bucket": "shopping", "distance_m": 250, "ring": "nearby"},
            ],
            "landmarks": [["Central Park", 350]],
        }

    def test_new_views_render_only_their_intended_evidence(self) -> None:
        named = describe._view_named_destinations(self.surr)
        access = describe._view_access_mix(self.surr)
        self.assertIn("Alpha Cafe (cafe)", named)
        self.assertIn("Central Park", named)
        self.assertIn("bookstore: a short walk", access)
        self.assertNotIn("Alpha Cafe", access)

    def test_anchored_view_filters_out_ordinary_business_names(self) -> None:
        anchors = describe._meaningful_anchor_lines(self.surr)
        anchored = describe._view_deviation_anchored_environment(self.surr)
        self.assertEqual(anchors, [])
        self.assertIn("Central Park", anchored)
        self.assertNotIn("Alpha Cafe", anchored)
        self.assertNotIn("Beta Books", anchored)

    def test_anchored_view_keeps_named_rapid_transit(self) -> None:
        self.surr["pois"].append(
            {
                "name": "North Station",
                "category": "subway_station",
                "bucket": "transit",
                "distance_m": 100,
                "ring": "doorstep",
            }
        )
        anchors = describe._meaningful_anchor_lines(self.surr)
        self.assertIn("North Station (subway station)", anchors[0])

    def test_grounding_allows_selected_overture_names(self) -> None:
        self.assertNotIn("Alpha Cafe", describe.ungrounded("Alpha Cafe is nearby.", self.surr))

    def test_anchored_view_is_available_to_the_prompt_runner(self) -> None:
        self.assertIn("deviation_anchored_environment", describe._VIEWS)

    def test_price_profile_uses_independent_fine_access(self) -> None:
        self.surr["fine_cats"] = {
            "subway_station": [1, 2, 100],
            "cafe": [3, 8, 40],
        }
        describe._FINE_REF.clear()
        describe._FINE_REF["subway_station"] = np.array([0, 0, 1, 1, 2])
        lines = describe._profile_fine_lines(self.surr, {"cafe"})
        self.assertEqual(lines, ["- rapid transit: steps away"])
        self.assertIn("price_relevant_profile", describe._VIEWS)

    def test_cache_tag_separates_prompt_variant_checkpoints(self) -> None:
        previous = describe.CACHE_TAG
        try:
            describe.CACHE_TAG = "prompt-fingerprint"
            self.assertTrue(describe._cache_csv().endswith(".prompt-fingerprint.cache"))
        finally:
            describe.CACHE_TAG = previous


class TextAnalysisTest(unittest.TestCase):
    def test_content_tokens_drop_scaffolding(self) -> None:
        tokens = text_analysis.tokenize("The nearby block has Central Park and 42 cafes.", content=True)
        self.assertEqual(tokens, ["central", "park", "cafes"])

    def test_jaccard_is_meaned_over_summary_pairs_within_a_column(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "datasets"
            folder = root / "variants" / "v2-o"
            folder.mkdir(parents=True)
            (folder / "metadata.json").write_text(
                json.dumps({"slug": "v2-o", "target": "price", "text_encoding_columns": ["summary"]})
            )
            pd.DataFrame(
                {"price": [1, 2], "summary": ["Central Park museum", "Central Park gallery"]}
            ).to_csv(folder / "data.csv", index=False)

            records = text_analysis.load_records(root)
            summary = text_analysis.build_summary_rows(records, max_pairs=20)

        self.assertEqual(len(summary), 1)
        self.assertEqual(summary[0]["summary_pair_count"], "1")
        self.assertAlmostEqual(float(summary[0]["jaccard"]), 0.5)


class PromptDatasetSamplingTest(unittest.TestCase):
    def test_stratified_sample_returns_the_requested_size(self) -> None:
        df = pd.DataFrame(
            {
                "index": range(10),
                "room_type": ["private"] * 7 + ["entire"] * 3,
            }
        )
        sample = stratified_sample(df, n=5, seed=0)
        self.assertEqual(len(sample), 5)
        self.assertEqual(sample["index"].nunique(), 5)


if __name__ == "__main__":
    unittest.main()
