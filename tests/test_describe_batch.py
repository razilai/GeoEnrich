from __future__ import annotations

import asyncio
import json

import httpx
import pandas as pd

from src import describe


def completed(batch: dict) -> dict:
    """A finished batch echoing one summary per submitted request."""
    results = [
        {
            "custom_id": r["custom_id"],
            "response": {
                "body": {
                    "choices": [
                        {"finish_reason": "stop", "message": {"content": f"summary {r['custom_id']}"}}
                    ]
                }
            },
        }
        for r in batch["requests"]
    ]
    return {"status": "completed", "results": results}


def fake_api(events: list[str], fail_submit: int | None = None) -> httpx.MockTransport:
    """Batch API stub: every batch stays pending until all of them are submitted."""
    batches: dict[str, dict] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            n = len(events)
            events.append("submit")
            if n == fail_submit:
                return httpx.Response(402, json={"error": "insufficient credits"})
            batch_id = f"b{n}"
            batches[batch_id] = json.loads(request.content)
            return httpx.Response(200, json={"id": batch_id})
        batch_id = request.url.path.rsplit("/", 1)[1]
        events.append(f"poll {batch_id}")
        if events.count("submit") < 3:
            return httpx.Response(200, json={"status": "in_progress"})
        return httpx.Response(200, json=completed(batches[batch_id]))

    return httpx.MockTransport(handler)


def run_batches(monkeypatch, tmp_path, fail_submit: int | None = None):
    events: list[str] = []
    transport = fake_api(events, fail_submit)
    monkeypatch.setattr(describe, "_batch_client", lambda: httpx.AsyncClient(transport=transport))
    monkeypatch.setattr(describe, "BATCH_SIZE", 2)
    monkeypatch.setattr(describe, "BATCH_POLL", 0)
    monkeypatch.setattr(describe, "prompt_for", lambda row: "evidence")
    monkeypatch.setattr(describe, "save", lambda df: None)
    monkeypatch.setattr(describe.config, "ARTIFACTS_DIR", str(tmp_path))
    monkeypatch.setattr(describe.config, "BATCH_JSON", str(tmp_path / "batch_result.json"))
    df = pd.DataFrame({"index": range(5), describe.SUMMARY: [None] * 5}, dtype="object")
    asyncio.run(describe.run_batch(list(df.iterrows()), df))
    return events, df


def test_sub_batches_are_all_submitted_before_any_is_awaited(monkeypatch, tmp_path) -> None:
    events, df = run_batches(monkeypatch, tmp_path)
    assert events[:3] == ["submit"] * 3
    assert df[describe.SUMMARY].tolist() == [f"summary {i}" for i in range(5)]
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "batch_result_b0.json", "batch_result_b1.json", "batch_result_b2.json",
    ]  # fmt: skip


def test_a_failed_submit_leaves_its_rows_and_spares_the_others(monkeypatch, tmp_path) -> None:
    _, df = run_batches(monkeypatch, tmp_path, fail_submit=1)
    assert df[describe.SUMMARY].tolist() == ["summary 0", "summary 1", None, None, "summary 4"]
