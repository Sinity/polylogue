"""``/api/paste-browser`` serves one page from one bounded declared read.

polylogue-q54dt. The handler used to call ``poly.filter().list_summaries()``
and then ``poly.get_session`` for every session in the archive, on an
interactive kernel slot, for every request. It now compiles one terminal
query-unit expression over the message-grain ``has_paste`` predicate and
pushes ``offset``/``limit`` into SQL.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, cast

import pytest

from polylogue.api import Polylogue
from polylogue.daemon.http import DaemonAPIHandler
from tests.infra.storage_records import SessionBuilder

_DIFF = "@@ -1 +1 @@\n-old\n+new"


def _seed(index_db: Path, *, empty_sessions: int, paste_messages: int) -> None:
    """Seed ``empty_sessions`` paste-free sessions, then one with pastes.

    Sorting puts the empty sessions first, so a handler that still walked the
    archive would have to load every one of them before reaching a match.
    """

    for index in range(empty_sessions):
        (
            SessionBuilder(index_db, f"empty-{index:03d}")
            .provider("claude-code")
            .add_message("typed", role="user", text="typed by hand")
            .save()
        )
    builder = SessionBuilder(index_db, "zz-pasted").provider("claude-code")
    for index in range(paste_messages):
        builder = builder.add_message(f"paste-{index}", role="user", text=f"{_DIFF}\npaste {index}")
    builder.save()
    with sqlite3.connect(index_db) as conn:
        conn.execute("UPDATE messages SET has_paste = 1 WHERE native_id LIKE 'paste-%'")


def _handler() -> Any:
    return cast(Any, DaemonAPIHandler.__new__(DaemonAPIHandler))


def _forbid_session_walk(polylogue: Polylogue, monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the retired full-archive walk fail loudly if it comes back."""

    def _refuse(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("the paste browser must not enumerate or hydrate sessions")

    monkeypatch.setattr(type(polylogue), "filter", _refuse, raising=True)
    monkeypatch.setattr(type(polylogue), "get_session", _refuse, raising=True)


@pytest.mark.asyncio
async def test_page_past_the_offset_does_not_walk_the_archive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Serving an offset page touches neither ``list_summaries`` nor ``get_session``.

    Concrete input: 12 paste-free sessions sort ahead of the only
    paste-bearing session; the request asks for offset 1, limit 1.

    Anti-vacuity: restoring ``poly.filter().list_summaries()`` plus the
    per-session ``poly.get_session`` loop trips the monkeypatched refusal, so
    this test is red for exactly the shape acceptance criterion 3 rejects.
    """
    index_db = tmp_path / "archive" / "index.db"
    _seed(index_db, empty_sessions=12, paste_messages=3)

    polylogue = Polylogue(archive_root=index_db.parent, db_path=index_db)
    try:
        _forbid_session_walk(polylogue, monkeypatch)
        payload = cast(
            dict[str, object],
            await _handler()._do_paste_browser(polylogue, limit=1, offset=1),
        )
    finally:
        await polylogue.close()

    items = cast(list[dict[str, object]], payload["items"])
    assert len(items) == 1
    assert str(items[0]["message_id"]).endswith(":paste-1")


@pytest.mark.asyncio
async def test_pages_concatenate_each_match_exactly_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Three matches at limit 2: page one, then page two, each item once.

    Anti-vacuity: an offset applied after loading (the retired shape) or a
    non-deterministic row order lets an item repeat across pages or vanish,
    and the exact-sequence assertion is red.
    """
    index_db = tmp_path / "archive" / "index.db"
    _seed(index_db, empty_sessions=2, paste_messages=3)

    polylogue = Polylogue(archive_root=index_db.parent, db_path=index_db)
    try:
        _forbid_session_walk(polylogue, monkeypatch)
        first = cast(dict[str, object], await _handler()._do_paste_browser(polylogue, limit=2, offset=0))
        second = cast(dict[str, object], await _handler()._do_paste_browser(polylogue, limit=2, offset=2))
    finally:
        await polylogue.close()

    def _ids(payload: dict[str, object]) -> list[str]:
        return [str(item["message_id"]) for item in cast(list[dict[str, object]], payload["items"])]

    assert [identifier.rsplit(":", 1)[-1] for identifier in _ids(first) + _ids(second)] == [
        "paste-0",
        "paste-1",
        "paste-2",
    ]
    # polylogue-q54dt acceptance criterion 1, already landed: a filled page
    # publishes no total, and the final page publishes the exact one.
    assert first["total"] is None
    assert first["total_is_exact"] is False
    assert first["total_lower_bound"] == 2
    assert second["total"] == 3
    assert second["total_is_exact"] is True
