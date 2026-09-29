"""Bounded tests for the daemon attachment library (polylogue-q54dt)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest

from polylogue.daemon.http import DaemonAPIHandler


class _PagedArchive:
    """Small facade double exposing only the declared bounded read."""

    def __init__(self, rows: list[tuple[object, str, str | None]]) -> None:
        self.rows = rows
        self.calls: list[dict[str, object]] = []

    async def _get_attachment_library_page(self, **kwargs: object) -> list[tuple[object, str, str | None]]:
        self.calls.append(kwargs)
        limit = cast(int, kwargs["limit"])
        offset = cast(int, kwargs["offset"])
        return self.rows[offset : offset + limit]

    def filter(self, *_args: object, **_kwargs: object) -> object:
        raise AssertionError("attachment library must not enumerate session summaries")

    async def get_session(self, *_args: object, **_kwargs: object) -> object:
        raise AssertionError("attachment library must not hydrate sessions")

    async def get_session_summaries(self, session_ids: list[str]) -> dict[str, object]:
        return {
            session_id: SimpleNamespace(display_label="Synthesized session", title="Opening prompt")
            for session_id in session_ids
        }


def _attachment(index: int) -> object:
    return SimpleNamespace(
        id=f"att-{index}",
        session_id=f"session-{index}",
        message_id=f"message-{index}",
        name=f"file-{index}.txt",
        mime_type="text/plain",
        size_bytes=10,
        availability=SimpleNamespace(state="available", can_fetch=False),
    )


def _handler() -> Any:
    return cast(Any, DaemonAPIHandler.__new__(DaemonAPIHandler))


@pytest.mark.asyncio
async def test_attachment_library_uses_one_bounded_read_without_session_walk() -> None:
    """An offset page never calls the retired archive-wide session walk."""

    archive = _PagedArchive([(_attachment(index), f"title-{index}", "codex") for index in range(4)])

    payload = cast(
        dict[str, object],
        await _handler()._do_attachment_library(
            archive,
            limit=1,
            offset=2,
            mime_filter="",
            state_filter="",
            session_filter="",
        ),
    )

    assert [item["attachment_id"] for item in cast(list[dict[str, object]], payload["items"])] == ["att-2"]
    assert archive.calls == [{"limit": 2, "offset": 2, "mime_filter": "", "session_filter": "", "state_filter": ""}]
    assert payload["total"] is None
    assert payload["total_lower_bound"] == 3


@pytest.mark.asyncio
async def test_attachment_library_pages_concatenate_exactly_once() -> None:
    """Three matches at limit two appear once across two bounded pages."""

    archive = _PagedArchive([(_attachment(index), f"title-{index}", "codex") for index in range(3)])
    handler = _handler()

    first = cast(
        dict[str, object],
        await handler._do_attachment_library(
            archive, limit=2, offset=0, mime_filter="", state_filter="", session_filter=""
        ),
    )
    second = cast(
        dict[str, object],
        await handler._do_attachment_library(
            archive, limit=2, offset=2, mime_filter="", state_filter="", session_filter=""
        ),
    )

    ids = [
        str(item["attachment_id"]) for page in (first, second) for item in cast(list[dict[str, object]], page["items"])
    ]
    assert ids == ["att-0", "att-1", "att-2"]
    assert first["total"] is None
    assert first["total_is_exact"] is False
    assert first["total_lower_bound"] == 2
    assert second["total"] == 3
    assert second["total_is_exact"] is True
    assert [call["limit"] for call in archive.calls] == [3, 3]


@pytest.mark.asyncio
async def test_attachment_library_uses_canonical_summary_label() -> None:
    """Heuristic prompt titles never become attachment-library headings.

    ANTI-VACUITY: rendering the SQL row's raw title instead of the canonical
    summary label changes the emitted session title to "Opening prompt".
    """
    archive = _PagedArchive([(_attachment(0), "Opening prompt", "codex")])
    payload = cast(
        dict[str, object],
        await _handler()._do_attachment_library(
            archive, limit=1, offset=0, mime_filter="", state_filter="", session_filter=""
        ),
    )

    item = cast(list[dict[str, object]], payload["items"])[0]
    assert item["session_title"] == "Synthesized session"


class _ArchiveReaderHandler:
    """Request double for the archive-reader branch of the library route."""

    def __init__(self, params: dict[str, int]) -> None:
        self.params = params
        self.sent: list[tuple[object, object]] = []

    def _get_int(self, _params: object, name: str, default: int) -> int:
        return self.params.get(name, default)

    def _send_json(self, status: object, payload: object) -> None:
        self.sent.append((status, payload))


def _archive_reader_page(
    monkeypatch: pytest.MonkeyPatch, rows: list[tuple[object, str, str | None]], **params: int
) -> dict[str, object]:
    from contextlib import contextmanager
    from pathlib import Path

    import polylogue.archive.query.transaction as transaction
    import polylogue.daemon.http as daemon_http
    import polylogue.operations.http_read_models as read_models
    from polylogue.daemon.route_families import read_query

    @contextmanager
    def _context(*_args: object, **_kwargs: object) -> Any:
        yield object()

    def _page(_archive: object, *, limit: int, offset: int, **_filters: object) -> list[tuple[object, str, str | None]]:
        return rows[offset : offset + limit]

    monkeypatch.setattr(daemon_http, "_web_reader_archive_root", lambda: Path("/archive"))
    monkeypatch.setattr(transaction, "archive_read_context", _context)
    monkeypatch.setattr(read_models, "read_attachment_library_page", _page)
    handler = _ArchiveReaderHandler(params)
    read_query._handle_attachment_library(handler, {})
    [(_status, payload)] = handler.sent
    return cast(dict[str, object], payload)


def test_archive_reader_library_empty_page_past_the_end_claims_no_count(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two matches read at offset 100 on the archive-reader route.

    Anti-vacuity: publishing ``offset + len(entries)`` as an exact total for
    an untruncated page reports ``total=100, total_is_exact=true`` here.
    """
    rows = [(_attachment(index), f"title-{index}", "codex") for index in range(2)]

    payload = _archive_reader_page(monkeypatch, rows, limit=10, offset=100)

    assert payload["items"] == []
    assert payload["total"] is None
    assert payload["total_is_exact"] is False
    assert "total_lower_bound" not in payload
    assert _archive_reader_page(monkeypatch, rows, limit=10, offset=1)["total"] == 2
