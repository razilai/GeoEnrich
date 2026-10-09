from __future__ import annotations

import subprocess
import sys

import pytest

from src import bench, report

def names(runs):
    return {r.dataset for r in runs}


def test_screen_grid_counts() -> None:
    grid = bench.screen_grid("16")
    runs = bench.plan_runs(grid)
    assert len(runs) == grid.total == 4 * 5 * 1
    assert sum(r.text_encoder == bench.TAR for r in runs) == 5
    assert {r.fold for r in runs} == {0}
    assert {r.model for r in runs} == set(bench.COMMITTEE)


def test_final_grid_is_the_four_conditions_of_one_dataset() -> None:
    grid = bench.final_grid("16")
    assert grid.pairs == (
        ("structured", bench.FROZEN),
        ("enriched_16_text_only", bench.FROZEN),
        ("enriched_16_joint", bench.FROZEN),
        ("enriched_16_joint", bench.TAR),
    )
    runs = bench.plan_runs(grid)
    assert len(runs) == grid.total == 120
    assert sum(r.text_encoder == bench.TAR for r in runs) == 30
    assert {r.fold for r in runs} == {0, 1, 2, 3, 4, 5}


@pytest.mark.parametrize("grid", [bench.screen_grid("16"), bench.final_grid("16")])
def test_runs_are_judgeable_by_report(grid) -> None:
    labels = {report._label(r.dataset, str(r.text_encoder)) for r in bench.plan_runs(grid)}
    assert labels == {
        ("shared", None, "structured"),
        ("enriched", "16", "text_only"),
        ("enriched", "16", "joint_frozen"),
        ("enriched", "16", "joint_tar"),
    }


def test_commands_use_device_and_official_entry_point() -> None:
    run = bench.plan_runs(bench.probe_grid("08"))[0]
    cmd = bench.run_command(run, "python", "/out", "cuda:1")
    assert cmd[1] == "benchmark.py"
    assert cmd[-2:] == ["--device", "cuda:1"]
    assert bench.offline_env()["HF_HUB_OFFLINE"] == "1"


def test_probe_plans_one_tar_and_one_frozen() -> None:
    runs = bench.plan_runs(bench.probe_grid("08"))
    assert len(runs) == 2
    assert sorted(str(r.text_encoder) for r in runs) == ["e5-small", "e5-small-tar"]
    assert len(names(runs)) == 1


def test_bench_refuses_without_confirmation(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: calls.append(a))
    monkeypatch.setattr(sys, "argv", ["bench", "--grid", "probe", "--prompt", "16"])
    with pytest.raises(SystemExit) as exc:
        bench.main()
    assert exc.value.code not in (0, None)
    assert calls == []


def test_execute_skips_existing_and_continues_after_failure(tmp_path, monkeypatch) -> None:
    runs = bench.plan_runs(bench.probe_grid("08"))
    (tmp_path / f"{runs[0].name}.json").write_text("{}")
    launched = []

    def fake(cmd, **kwargs):
        launched.append(cmd)
        return subprocess.CompletedProcess(cmd, 1)

    monkeypatch.setattr(subprocess, "run", fake)
    assert bench.execute(runs, str(tmp_path), "cpu", str(tmp_path)) == 1
    assert len(launched) == 1
