"""Composed transcript reads never ask a continuation for rows past the caller's bound."""

from __future__ import annotations

from typing import cast

import pytest

from polylogue.cli import archive_query
from polylogue.cli.operation_kernel import OperationFailedError, OperationRequest
from polylogue.config import Config


def test_session_window_continuation_keeps_remaining_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fails if a continuation drops the remaining bound and asks for the oversized, unrequested sixth row."""
    monkeypatch.setattr(archive_query, "_SESSION_READ_WINDOW", 4)
    requests: list[OperationRequest] = []

    def read(
        _config: Config, request: OperationRequest, **_kwargs: object
    ) -> tuple[dict[str, object], dict[str, object]]:
        requests.append(request)
        resumed = request.payload.get("continuation") is not None
        if resumed and request.payload.get("limit") != 1:
            raise OperationFailedError("unrequested_rows", "read exceeded the selected message bound")
        start, count = (4, 1) if resumed else (0, 4)
        return {
            "session": {"session_id": "w12", "messages": [{"position": i} for i in range(start, start + count)]},
            "complete": False,
            "outcome": {"state": "ok", "reason": None, "detail": {}},
            "lineage_complete": True,
            "lineage_truncation_reason": None,
            "continuation": "fixture-after-five" if resumed else "fixture-after-four",
        }, {}

    monkeypatch.setattr(archive_query, "dispatch_read", read)
    result = archive_query._read_session_windows(
        cast(Config, object()), "session:w12", daemon_disabled=True, message_limit=5
    )
    assert [request.payload.get("limit") for request in requests] == [4, 1]
    assert result["messages"] == [{"position": i} for i in range(5)]


def test_session_window_resumes_the_original_page_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    """Continuation pages preserve the requested bound and exact row sequence."""
    monkeypatch.setattr(archive_query, "_SESSION_READ_WINDOW", 4)
    requests: list[tuple[int, object]] = []

    def read(
        _config: Config, request: OperationRequest, **_kwargs: object
    ) -> tuple[dict[str, object], dict[str, object]]:
        size = request.payload.get("limit")
        assert isinstance(size, int)
        cursor = request.payload.get("continuation")
        start = int(str(cursor).removeprefix("after-")) if cursor is not None else 0
        requests.append((size, cursor))
        end = min(start + size, 7)
        return {
            "session": {"session_id": "w12", "messages": [{"position": i} for i in range(start, end)]},
            "complete": end == 7,
            "outcome": {"state": "ok", "reason": None, "detail": {}},
            "lineage_complete": True,
            "lineage_truncation_reason": None,
            "continuation": f"after-{end}" if end < 7 else None,
        }, {}

    monkeypatch.setattr(archive_query, "dispatch_read", read)
    result = archive_query._read_session_windows(cast(Config, object()), "session:w12", daemon_disabled=True)
    assert requests == [(4, None), (4, "after-4")]
    assert result["messages"] == [{"position": i} for i in range(7)]
