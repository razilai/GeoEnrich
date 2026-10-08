from __future__ import annotations

import os
import types

import pandas as pd
import pytest

from src import stage

REGISTRY = os.path.join(
    os.path.dirname(__file__), "..", "MulTaBench", "multabench", "datasets", "all_datasets.py"
)


def listings(n: int = 6) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "price": [100.0 + i for i in range(n)],
            "ratings": [4.5] * n,
            "lat": [40.7 + i / 100 for i in range(n)],
            "long": [-73.9] * n,
            "guests": [2] * n,
            "bathrooms": [1.0] * n,
            "room_type": ["Private room", "Entire home/apt"] * (n // 2),
            "property_number_of_reviews": [10] * n,
            "is_superhost": [True, False] * (n // 2),
            "num_bedrooms": [1.0] * n,
            "num_baths": [1.0] * n,
        }
    )


def corpus(indices: list[int], tag: str = "a", missing: tuple[int, ...] = ()) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "index": indices,
            "surroundings_summary": [None if i in missing else f"{tag} summary {i}" for i in indices],
        }
    )


@pytest.fixture
def datasets() -> dict[str, pd.DataFrame]:
    # Described in density order, not listing order; listing 4 has no summary in "08".
    return stage.build_datasets(
        listings(),
        {"05": corpus([5, 1, 3, 4, 0]), "08": corpus([3, 1, 5, 4, 0], "b", missing=(4,))},
    )


def roundtrip(key: str, frame: pd.DataFrame, tmp_path) -> pd.DataFrame:
    stage.write_snapshots({key: frame}, str(tmp_path))
    return pd.read_parquet(os.path.join(stage.snapshot_dir(str(tmp_path), key), "data.parquet"))


def test_dataset_keys(datasets) -> None:
    assert set(datasets) == {
        "latlon",
        "structured",
        "enriched_05_text_only",
        "enriched_05_joint",
        "enriched_08_text_only",
        "enriched_08_joint",
    }


def test_all_datasets_share_one_row_set_in_listing_order(datasets) -> None:
    assert {len(frame) for frame in datasets.values()} == {4}
    targets = {key: frame["price"].tolist() for key, frame in datasets.items()}
    assert all(t == [100.0, 101.0, 103.0, 105.0] for t in targets.values())
    assert datasets["enriched_05_joint"]["surroundings_summary"].tolist() == [
        f"a summary {i}" for i in (0, 1, 3, 5)
    ]


def test_only_latlon_carries_coordinates(datasets) -> None:
    for key, frame in datasets.items():
        has = {"latitude", "longitude"} <= set(frame.columns)
        assert has == (key == "latlon"), key
        assert "lat" not in frame and "long" not in frame


def test_summary_only_in_text_conditions(datasets) -> None:
    for key, frame in datasets.items():
        assert ("surroundings_summary" in frame) == (key.startswith("enriched")), key
    assert list(datasets["enriched_05_text_only"].columns) == ["surroundings_summary", "price"]


def test_declared_dtypes(datasets) -> None:
    joint = datasets["enriched_05_joint"]
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
        "latlon": (tabular_numeric | {"latitude", "longitude"}, {"room_type"}, set()),
        "structured": (tabular_numeric, {"room_type"}, set()),
        "enriched_05_text_only": (set(), set(), {"surroundings_summary"}),
        "enriched_05_joint": (tabular_numeric, {"room_type"}, {"surroundings_summary"}),
    }
    for key, (numerical, categorical, text) in expect.items():
        x = roundtrip(key, datasets[key], tmp_path).drop(columns="price")
        types = detect(x, image_column=None)
        assert types.numerical_features == numerical, key
        assert types.categorical_features == categorical, key
        assert types.text_features == text, key
        assert not types.date_features and not types.image_features


def test_missing_superhost_stays_boolean_like() -> None:
    frame = listings()
    frame["is_superhost"] = [True, None, False, True, False, True]
    out = stage.build_datasets(frame, {"05": corpus(list(range(6)))})["structured"]
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
    with open(os.path.join(stage.snapshot_dir(str(tmp_path), "latlon"), "metadata.json")) as f:
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
