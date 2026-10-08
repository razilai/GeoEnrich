from __future__ import annotations

import socket
import sys

import pytest

import run
from src import describe


def test_suite_blocks_sockets() -> None:
    with pytest.raises(RuntimeError):
        socket.create_connection(("example.com", 80))
    with pytest.raises(RuntimeError):
        socket.socket().connect(("127.0.0.1", 9))


def test_describe_refuses_without_confirmation(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(describe, "run", lambda k: calls.append(k))
    monkeypatch.setattr(sys, "argv", ["describe", "10"])
    with pytest.raises(SystemExit) as exc:
        describe.main()
    assert exc.value.code not in (0, None)
    assert calls == []


def test_describe_runs_with_confirmation(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(describe, "run", lambda k: calls.append(k))
    monkeypatch.setattr(sys, "argv", ["describe", "--confirm", "10"])
    describe.main()
    assert calls == [10]


@pytest.mark.parametrize("stage", ["stage", "bench", "report"])
def test_benchmark_stages_force_hf_offline(stage) -> None:
    assert run.stage_env(stage)["HF_HUB_OFFLINE"] == "1"


def test_build_does_not_force_offline(monkeypatch) -> None:
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    assert "HF_HUB_OFFLINE" not in run.stage_env("build")


def test_launcher_no_longer_pins_gpu() -> None:
    assert not hasattr(run, "pin_gpu")
    assert "CUDA_VISIBLE_DEVICES" not in open(run.__file__).read()
