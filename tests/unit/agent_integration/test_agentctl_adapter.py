from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from polylogue.agent_integration import agentctl_adapter as adapter


def _request() -> dict[str, object]:
    return {
        "schema": 1,
        "request_id": str(uuid4()),
        "correlation_id": str(uuid4()),
        "operation": adapter.OPERATION,
        "owner": adapter.OWNER,
        "principal": "test",
        "arguments": {},
    }


def test_status_route_preserves_ids_and_returns_one_source_binding(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "archive"
    root.mkdir()
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(root))

    class FakePolylogue:
        def __init__(self, **kwargs: object) -> None:
            assert kwargs["archive_root"] == root

        async def __aenter__(self) -> FakePolylogue:
            return self

        async def __aexit__(self, *args: object) -> None:
            return None

        async def stats(self) -> SimpleNamespace:
            return SimpleNamespace(
                session_count=2,
                message_count=3,
                word_count=4,
                origins={"codex-session": 2},
                tags={},
                last_sync=None,
            )

    monkeypatch.setattr(adapter, "Polylogue", FakePolylogue)
    request = _request()
    response = adapter._read_status(request)

    assert response["ok"] is True
    assert response["request_id"] == request["request_id"]
    assert response["correlation_id"] == request["correlation_id"]
    assert response["owner"] == adapter.OWNER
    assert len(response["source_bindings"]) == 1
    binding = response["source_bindings"][0]
    assert binding["source_ref"] == adapter.SOURCE_REF
    assert binding["root_digest"].startswith("sha256:")
    json.dumps(response)


def test_status_rejects_noncanonical_operation() -> None:
    request = _request()
    request["operation"] = "polylogue.archive.write"
    with pytest.raises(ValueError, match="unsupported owner operation"):
        adapter._read_status(request)
