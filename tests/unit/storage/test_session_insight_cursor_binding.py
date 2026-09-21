"""The session-insight batch loaders decode through the declared column specs.

Anti-vacuity for this module: restoring a per-row mapper (a hand-written
``SessionRecord(...)`` field list in ``rebuild.py``, or ``_row_to_message`` /
``_row_to_content_block`` called directly inside the batch loops) turns
``test_batch_loaders_plan_each_cursor_once`` red, because the spec's per-row
``row_to_record_kwargs`` starts firing again.  Mis-binding a column position
in ``bind_record_mapper`` turns ``test_session_batch_decodes_declared_fields``
red, because the adjacent text columns carry deliberately distinct values.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

from polylogue.storage.derived.session.rebuild import load_marker_blocks_sync, load_sync_batch
from polylogue.storage.sqlite.archive_tiers.column_spec import TableColumnSpec
from polylogue.storage.sqlite.connection import open_connection
from tests.infra.storage_records import make_message, make_session, store_records

_ORIGIN = "codex-session"


def _sid(native_id: str) -> str:
    return f"{_ORIGIN}:{native_id}"


def _seed(conn: sqlite3.Connection, native_id: str, *, count: int) -> str:
    """One session whose adjacent TEXT columns all hold distinct values."""
    store_records(
        session=make_session(
            native_id,
            source_name="codex",
            title=f"title-{native_id}",
            git_branch=f"branch-{native_id}",
            git_repository_url=f"https://example.invalid/{native_id}.git",
        ),
        messages=[
            make_message(
                f"{native_id}:m{index}",
                native_id,
                role="assistant" if index % 2 else "user",
                text=f"body-{native_id}-{index}",
                blocks=[
                    {"type": "text", "text": f"marker-{native_id}-{index}"},
                    {
                        "type": "tool_use",
                        "name": f"tool-{index}",
                        "id": f"call-{native_id}-{index}",
                        "input": {"cmd": f"echo {index}"},
                    },
                ],
            )
            for index in range(count)
        ],
        attachments=[],
        conn=conn,
    )
    return _sid(native_id)


def test_session_batch_decodes_declared_fields(tmp_path: Path) -> None:
    """Every projected session field arrives at its own record name."""
    with open_connection(tmp_path / "bound-sessions.db") as conn:
        session_id = _seed(conn, "bound-a", count=1)
        batch = load_sync_batch(conn, [session_id])

    (session,) = batch.sessions
    assert str(session.session_id) == session_id
    assert session.native_id == "bound-a"
    assert session.origin.value == _ORIGIN
    # Adjacent TEXT columns: a swapped binding shows up here and nowhere else.
    assert session.title == "title-bound-a"
    assert session.git_branch == "branch-bound-a"
    assert session.git_repository_url == "https://example.invalid/bound-a.git"
    assert session.raw_id is None
    # Transform-bearing and omitted columns keep the declaration's answers.
    assert session.metadata is None
    assert session.version == 1
    assert session.content_hash
    assert session.created_at is not None
    assert session.updated_at is not None
    assert session.sort_key is None or isinstance(session.sort_key, float)
    # Not projected by the insight SELECT, so the record default must survive.
    assert session.display_name is None
    assert session.pending_drafts is None
    assert session.reported_cost_usd is None


def test_batch_message_and_block_decode_matches_the_one_row_route(tmp_path: Path) -> None:
    """The bound decoder and the one-row convenience agree field for field."""
    from polylogue.storage.sqlite.queries.mappers_archive import (
        _row_to_content_block,
        _row_to_message,
    )

    with open_connection(tmp_path / "bound-parity.db") as conn:
        session_id = _seed(conn, "bound-b", count=3)
        batch = load_sync_batch(conn, [session_id])
        markers = load_marker_blocks_sync(conn, [session_id])

        from polylogue.storage.derived.session.rebuild import (
            _SESSION_INSIGHT_BLOCK_SQL_TEMPLATE,
            _SESSION_INSIGHT_BLOCK_TEXT_PREVIEW_CHARS,
            _SESSION_INSIGHT_MARKER_BLOCK_SQL_TEMPLATE,
            _SESSION_INSIGHT_MESSAGE_SQL_TEMPLATE,
            _SESSION_INSIGHT_MESSAGE_TEXT_PREVIEW_CHARS,
        )

        expected_messages = [
            _row_to_message(row)
            for row in conn.execute(
                _SESSION_INSIGHT_MESSAGE_SQL_TEMPLATE.format(placeholders="?"),
                (_SESSION_INSIGHT_MESSAGE_TEXT_PREVIEW_CHARS, session_id),
            ).fetchall()
        ]
        expected_blocks = [
            _row_to_content_block(row)
            for row in conn.execute(
                _SESSION_INSIGHT_BLOCK_SQL_TEMPLATE.format(placeholders="?"),
                (_SESSION_INSIGHT_BLOCK_TEXT_PREVIEW_CHARS, session_id),
            ).fetchall()
        ]
        expected_markers = [
            _row_to_content_block(row)
            for row in conn.execute(
                _SESSION_INSIGHT_MARKER_BLOCK_SQL_TEMPLATE.format(placeholders="?"),
                (session_id,),
            ).fetchall()
        ]

    assert expected_messages and expected_blocks and expected_markers
    assert [record.model_dump() for record in batch.messages] == [record.model_dump() for record in expected_messages]
    assert [record.model_dump() for record in batch.blocks] == [record.model_dump() for record in expected_blocks]
    assert [record.model_dump() for record in markers[session_id]] == [
        record.model_dump() for record in expected_markers
    ]


def test_batch_loaders_plan_each_cursor_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Field planning is per cursor, not per row, on the rebuild batch path."""
    per_row: dict[str, int] = {}
    per_cursor: dict[str, int] = {}
    row_impl = TableColumnSpec.row_to_record_kwargs
    bind_impl = TableColumnSpec.bind_record_mapper

    def counted_row(self: TableColumnSpec, row: sqlite3.Row) -> dict[str, object]:
        per_row[self.table_name] = per_row.get(self.table_name, 0) + 1
        return row_impl(self, row)

    def counted_bind(self: TableColumnSpec, column_names: Sequence[str]) -> Callable[[sqlite3.Row], dict[str, object]]:
        per_cursor[self.table_name] = per_cursor.get(self.table_name, 0) + 1
        return bind_impl(self, column_names)

    monkeypatch.setattr(TableColumnSpec, "row_to_record_kwargs", counted_row)
    monkeypatch.setattr(TableColumnSpec, "bind_record_mapper", counted_bind)

    rows = 6
    with open_connection(tmp_path / "bound-plan.db") as conn:
        session_id = _seed(conn, "bound-c", count=rows)
        per_row.clear()
        per_cursor.clear()
        batch = load_sync_batch(conn, [session_id])
        markers = load_marker_blocks_sync(conn, [session_id])

    assert len(batch.messages) == rows
    assert len(batch.blocks) == rows
    assert len(markers[session_id]) == rows

    # load_sync_batch: one session, one message and one block SELECT, plus the
    # marker-block SELECT it delegates; then the explicit marker load above.
    assert per_cursor == {"sessions": 1, "messages": 1, "blocks": 3}
    # Nothing on this path re-plans a row: that is the whole point of binding.
    assert per_row == {}
