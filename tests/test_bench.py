from __future__ import annotations

import subprocess
import sys

import pytest

from src import bench, stage

PROMPTS = ["05", "08", "16"]
ARMS = ["enriched", "landmark_only", "landmark_redacted"]


def names(runs):
    return {r.dataset for r in runs}


def encoders_of(runs, dataset):
    return sorted(str(r.text_encoder) for r in runs if r.dataset == dataset)


def test_screen_grid_counts() -> None:
    grid = bench.screen_grid(PROMPTS)
    runs = bench.plan_runs(grid)
    assert len(grid.pairs) == 10
    assert len(runs) == grid.total == 50
    assert sum(r.text_encoder == bench.TAR for r in runs) == 15
    assert {r.fold for r in runs} == {0}
    assert {r.model for r in runs} == set(bench.COMMITTEE)


def test_final_grid_counts() -> None:
    grid = bench.final_grid([(a, "08") for a in ARMS])
    runs = bench.plan_runs(grid)
    assert len(grid.pairs) == 11
    assert len(runs) == grid.total == 330
    assert sum(r.text_encoder == bench.TAR for r in runs) == 90
    assert {r.fold for r in runs} == {0, 1, 2, 3, 4, 5}


@pytest.mark.parametrize(
    "grid", [bench.screen_grid(PROMPTS), bench.final_grid([(a, "08") for a in ARMS])]
)
def test_baselines_only_frozen_and_joint_once_per_encoder(grid) -> None:
    runs = bench.plan_runs(grid)
    for key in ("structured", "latlon"):
        name = stage.enum_name(key)
        assert set(encoders_of(runs, name)) <= {str(bench.FROZEN)}
    for dataset in names(runs):
        if dataset.endswith("_JOINT"):
            counts = {
                enc: sum(r.dataset == dataset and str(r.text_encoder) == enc for r in runs)
                for enc in (str(bench.FROZEN), str(bench.TAR))
            }
            per_cell = len(grid.learners) * len(grid.splits)
            assert counts == {str(bench.FROZEN): per_cell, str(bench.TAR): per_cell}
        elif dataset.endswith("_TEXT_ONLY"):
            assert set(encoders_of(runs, dataset)) == {str(bench.FROZEN)}


def test_latlon_enriched_adds_only_joint_frozen_and_tar() -> None:
    grid = bench.final_grid([("enriched", "16"), ("latlon_enriched", "16")])
    assert grid.pairs[-2:] == (
        ("latlon_enriched_16_joint", bench.FROZEN),
        ("latlon_enriched_16_joint", bench.TAR),
    )
    assert grid.total == 7 * 5 * 6


def test_commands_use_device_and_official_entry_point() -> None:
    run = bench.plan_runs(bench.probe_grid("enriched", "08"))[0]
    cmd = bench.run_command(run, "python", "/out", "cuda:1")
    assert cmd[1] == "benchmark.py"
    assert cmd[-2:] == ["--device", "cuda:1"]
    assert bench.offline_env()["HF_HUB_OFFLINE"] == "1"


def test_probe_plans_one_tar_and_one_frozen() -> None:
    runs = bench.plan_runs(bench.probe_grid("enriched", "08"))
    assert len(runs) == 2
    assert sorted(str(r.text_encoder) for r in runs) == ["e5-small", "e5-small-tar"]
    assert len(names(runs)) == 1


def test_bench_refuses_without_confirmation(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: calls.append(a))
    monkeypatch.setattr(sys, "argv", ["bench", "--grid", "probe"])
    with pytest.raises(SystemExit) as exc:
        bench.main()
    assert exc.value.code not in (0, None)
    assert calls == []


def test_execute_skips_existing_and_continues_after_failure(tmp_path, monkeypatch) -> None:
    runs = bench.plan_runs(bench.probe_grid("enriched", "08"))
    (tmp_path / f"{runs[0].name}.json").write_text("{}")
    launched = []

    def fake(cmd, **kwargs):
        launched.append(cmd)
        return subprocess.CompletedProcess(cmd, 1)

    monkeypatch.setattr(subprocess, "run", fake)
    assert bench.execute(runs, str(tmp_path), "cpu", str(tmp_path)) == 1
    assert len(launched) == 1
