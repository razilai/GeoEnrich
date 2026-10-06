from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

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
