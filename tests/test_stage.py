from __future__ import annotations

import os
import types

import numpy as np
import pandas as pd
import pytest

from src import stage

REGISTRY = os.path.join(
    os.path.dirname(__file__), "..", "MulTaBench", "multabench", "datasets", "all_datasets.py"
)
TABULAR = [
    "price", "ratings", "guests", "bathrooms", "room_type", "property_number_of_reviews",
    "is_superhost", "num_bedrooms", "num_baths",
]  # fmt: skip


def described(order=(5, 1, 3, 4, 0, 2), missing: tuple[int, ...] = ()) -> pd.DataFrame:
    """A described CSV as read back: described in density order, not listing order."""
    n = len(order)
    return pd.DataFrame(
        {
            "index": list(order),
            "price": [100.0 + i for i in order],
            "ratings": [4.5] * n,
            "guests": [2.0] * n,
            "bathrooms": [1.0] * n,
            "room_type": ["Private room", "Entire home/apt"] * (n // 2),
            "property_number_of_reviews": [10.0] * n,
            "is_superhost": [True, False] * (n // 2),
            "num_bedrooms": [1.0] * n,
            "num_baths": [1.0] * n,
            "surroundings_summary": [None if i in missing else f"summary {i}" for i in order],
        }
    )


@pytest.fixture
def datasets() -> dict[str, pd.DataFrame]:
    return stage.build_datasets(described(missing=(4,)), "16")


def roundtrip(key: str, frame: pd.DataFrame, tmp_path) -> pd.DataFrame:
    stage.write_snapshots({key: frame}, str(tmp_path))
    return pd.read_parquet(os.path.join(stage.snapshot_dir(str(tmp_path), key), "data.parquet"))


def test_dataset_keys(datasets) -> None:
    assert set(datasets) == {"structured", "enriched_16_text_only", "enriched_16_joint"}


def test_rows_with_a_summary_in_listing_order(datasets) -> None:
    for frame in datasets.values():
        assert frame["price"].tolist() == [100.0, 101.0, 102.0, 103.0, 105.0]
    assert datasets["enriched_16_joint"]["surroundings_summary"].tolist() == [
        f"summary {i}" for i in (0, 1, 2, 3, 5)
    ]


def test_target_is_the_described_csv_price_unchanged() -> None:
    corpus = described()
    corpus["price"] = np.log1p(corpus["price"])
    for frame in stage.build_datasets(corpus, "16").values():
        assert frame["price"].tolist() == np.log1p([100.0, 101.0, 102.0, 103.0, 104.0, 105.0]).tolist()


def test_columns_per_condition(datasets) -> None:
    assert list(datasets["structured"].columns) == TABULAR
    assert list(datasets["enriched_16_text_only"].columns) == ["surroundings_summary", "price"]
    assert list(datasets["enriched_16_joint"].columns) == [*TABULAR, "surroundings_summary"]
    for frame in datasets.values():
        assert "index" not in frame


def test_declared_dtypes(datasets) -> None:
    joint = datasets["enriched_16_joint"]
    assert isinstance(joint["room_type"].dtype, pd.CategoricalDtype)
    assert joint["is_superhost"].dtype == bool
    assert joint["surroundings_summary"].dtype == "string"
    for col in ("price", "guests", "property_number_of_reviews", "ratings"):
        assert joint[col].dtype == float, col


def test_benchmark_feature_types_per_dataset(datasets, tmp_path) -> None:
    detect = pytest.importorskip(
        "multabench.baselines.preprocessing.feature_types"
    ).detect_feature_types
    tabular_numeric = {
        "ratings", "guests", "bathrooms", "property_number_of_reviews",
        "is_superhost", "num_bedrooms", "num_baths",
    }  # fmt: skip
    expect = {
        "structured": (tabular_numeric, {"room_type"}, set()),
        "enriched_16_text_only": (set(), set(), {"surroundings_summary"}),
        "enriched_16_joint": (tabular_numeric, {"room_type"}, {"surroundings_summary"}),
    }
    for key, (numerical, categorical, text) in expect.items():
        x = roundtrip(key, datasets[key], tmp_path).drop(columns="price")
        types = detect(x, image_column=None)
        assert types.numerical_features == numerical, key
        assert types.categorical_features == categorical, key
        assert types.text_features == text, key
        assert not types.date_features and not types.image_features


def test_missing_superhost_stays_boolean_like() -> None:
    corpus = described()
    corpus["is_superhost"] = [True, None, False, True, False, True]
    out = stage.build_datasets(corpus, "16")["structured"]
    assert out["is_superhost"].isna().sum() == 1
    assert pd.api.types.is_bool_dtype(out["is_superhost"])


def test_registered_ids_resolve_offline_to_local_snapshot(datasets, tmp_path, monkeypatch) -> None:
    hub = pytest.importorskip("multabench.datasets.hub")
    hf = pytest.importorskip("huggingface_hub")
    from huggingface_hub import constants

    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", True)
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path))
    stage.write_snapshots(datasets, str(tmp_path))
    for key in datasets:
        stand_in = types.SimpleNamespace(value=stage.repo_name(key))
        path = hf.snapshot_download(repo_id=hub.hf_repo_id(stand_in), repo_type="dataset")
        assert path == stage.snapshot_dir(str(tmp_path), key)
        assert os.path.exists(os.path.join(path, "data.parquet"))
        assert os.path.exists(os.path.join(path, "metadata.json"))


def test_metadata(datasets, tmp_path) -> None:
    import json

    stage.write_snapshots(datasets, str(tmp_path))
    with open(os.path.join(stage.snapshot_dir(str(tmp_path), "structured"), "metadata.json")) as f:
        assert json.load(f) == {"target": "price", "image_col": None, "task_type": "reg"}


def test_restaging_does_no_work(datasets, tmp_path) -> None:
    assert sorted(stage.write_snapshots(datasets, str(tmp_path))) == sorted(datasets)
    assert stage.write_snapshots(datasets, str(tmp_path)) == []


@pytest.fixture
def registry_source() -> str:
    if not os.path.exists(REGISTRY):
        pytest.skip("MulTaBench checkout absent")
    with open(REGISTRY) as f:
        return f.read()


def test_patch_registry_registers_text_ids(registry_source, datasets) -> None:
    patched = stage.patch_registry(registry_source, list(datasets))
    namespace: dict = {}
    exec(compile(patched, "all_datasets.py", "exec"), namespace)  # its own asserts run too
    ids, modality = namespace["MulTaBenchDatasetID"], namespace["dataset_modality"]
    for key in datasets:
        member = ids[stage.enum_name(key)]
        assert member.value == stage.repo_name(key)
        assert modality(member).value == "text"


def test_patch_registry_is_idempotent(registry_source, datasets) -> None:
    once = stage.patch_registry(registry_source, list(datasets))
    assert once != registry_source
    assert stage.patch_registry(once, list(datasets)) == once


def test_patch_registry_fails_loudly_without_anchor(registry_source) -> None:
    with pytest.raises(RuntimeError, match="anchor"):
        stage.patch_registry(registry_source.replace("_IMAGE_PREFIXES", "_PREFIXES"), ["latlon"])


def test_restaging_after_row_set_change_rewrites(tmp_path) -> None:
    full = stage.build_datasets(described(), "16")
    fewer = stage.build_datasets(described(missing=(5,)), "16")
    stage.write_snapshots(full, str(tmp_path))
    assert sorted(stage.write_snapshots(fewer, str(tmp_path))) == sorted(fewer)
    staged = pd.read_parquet(os.path.join(stage.snapshot_dir(str(tmp_path), "structured"), "data.parquet"))
    assert len(staged) == 5


def test_restaging_after_target_change_rewrites(tmp_path) -> None:
    raw = described()
    logged = raw.assign(price=np.log1p(raw["price"]))
    stage.write_snapshots(stage.build_datasets(raw, "16"), str(tmp_path))
    assert len(stage.write_snapshots(stage.build_datasets(logged, "16"), str(tmp_path))) == 3


@pytest.mark.parametrize("anchor", ["\n}\n\n\nfor _d in MulTaBenchDatasetID:", "\n\n\n_IMAGE_PREFIXES = ("])
def test_patch_registry_requires_both_anchors(registry_source, anchor) -> None:
    assert anchor in registry_source
    with pytest.raises(RuntimeError, match="anchor"):
        stage.patch_registry(registry_source.replace(anchor, "\n#\n"), ["latlon"])
