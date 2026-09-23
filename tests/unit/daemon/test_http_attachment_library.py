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
