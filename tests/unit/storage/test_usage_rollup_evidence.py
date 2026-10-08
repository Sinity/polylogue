"""Model-usage rows and usage events follow the evidence the session still holds.

Each law writes synthetic sessions through the production writer
(``write_parsed_session_to_archive``) and, where the law is about
re-derivation, runs the usage-rollup stage the daemon converges
(``reconcile_session_usage_rollups``).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from polylogue.archive.message.roles import Role
from polylogue.archive.session.branch_type import BranchType
from polylogue.core.enums import BlockType, Provider
from polylogue.sources.parsers.base import (
    ParsedContentBlock,
    ParsedMessage,
    ParsedSession,
    ParsedSessionEvent,
)
from polylogue.storage.derived.session.usage_rollup import reconcile_session_usage_rollups
from polylogue.storage.io_phase_metrics import connect_measured
from polylogue.storage.sqlite.archive_tiers.write import PreparedSessionWrite
from tests.infra.archive_templates import bootstrapped_tier_path
from tests.infra.index_writer import write_fixture_index_session, write_fixture_prepared_session


def _connect(path: Path) -> sqlite3.Connection:
    conn = connect_measured(bootstrapped_tier_path(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _message(provider_id: str, text: str, *, role: Role = Role.ASSISTANT, **fields: object) -> ParsedMessage:
    return ParsedMessage(
        provider_message_id=provider_id,
        role=role,
        text=text,
        blocks=[ParsedContentBlock(type=BlockType.TEXT, text=text)],
        **fields,  # type: ignore[arg-type]
    )


def _delta_usage(
    input_tokens: int,
    output_tokens: int,
    *,
    message: str | None = None,
    model: str | None = None,
) -> ParsedSessionEvent:
    payload: dict[str, object] = {
        "type": "token_count",
        "last_token_usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
    }
    if model is not None:
        payload["model"] = model
    return ParsedSessionEvent(event_type="token_count", source_message_provider_id=message, payload=payload)


def _cumulative_usage(input_tokens: int, *, message: str, model: str) -> ParsedSessionEvent:
    return ParsedSessionEvent(
        event_type="token_count",
        source_message_provider_id=message,
        payload={
            "type": "token_count",
            "model": model,
            "total_token_usage": {"input_tokens": input_tokens, "total_tokens": input_tokens},
        },
    )


def _write(
    conn: sqlite3.Connection,
    session: ParsedSession,
    *,
    route: str = "inline",
    raw_id: str | None = None,
    merge_append: bool = False,
) -> str:
    if route == "prepared":

        def carries_union(prepared: PreparedSessionWrite) -> None:
            assert prepared.cross_acquisition_union is not None, "the prepared route must carry a union"

        return write_fixture_prepared_session(
            conn, session, inspect=carries_union, raw_id=raw_id, merge_append=merge_append
        )
    return write_fixture_index_session(conn, session, raw_id=raw_id, merge_append=merge_append)


def _usage_rows(conn: sqlite3.Connection, session_id: str) -> list[tuple[object, ...]]:
    return [
        tuple(row)
        for row in conn.execute(
            "SELECT model_name, input_tokens, output_tokens FROM session_model_usage "
            "WHERE session_id = ? ORDER BY model_name",
            (session_id,),
        )
    ]


def _usage_events(conn: sqlite3.Connection, session_id: str) -> list[tuple[object, ...]]:
    return [
        tuple(row)
        for row in conn.execute(
            "SELECT source_message_id, source_message_provider_id, source_message_resolution, "
            "last_input_tokens, last_output_tokens "
            "FROM session_provider_usage_events WHERE session_id = ? ORDER BY position",
            (session_id,),
        )
    ]


def _declared_only_session(native_id: str, **fields: object) -> ParsedSession:
    """A session whose only model evidence is the parser's declaration.

    No message names a model and the one ``token_count`` names none, so the
    event is attributable only to the session's sole (declared) model row.
    """
    return ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id=native_id,
        models_used=["declared-model"],
        messages=[
            _message("u0", "question", role=Role.USER),
            _message("a0", "answer"),
        ],
        session_events=[_delta_usage(40, 10)],
        **fields,  # type: ignore[arg-type]
    )


def test_usage_rollup_reconciliation_keeps_a_declared_model_and_its_unnamed_usage(tmp_path: Path) -> None:
    """Re-derivation keeps the declared row an unnamed event is attributed to.

    Anti-vacuity: reconcile before the declaration is honoured (deleting every
    row no message or named event supports) and the rollup ends with no row at
    all, because the unnamed event then finds no sole model row to land on.
    """
    conn = _connect(tmp_path / "index.db")
    try:
        session_id = write_fixture_index_session(conn, _declared_only_session("declared-only"))
        assert _usage_rows(conn, session_id) == [("declared-model", 40, 10)]

        reconcile_session_usage_rollups(conn, [session_id])

        assert _usage_rows(conn, session_id) == [("declared-model", 40, 10)]
    finally:
        conn.close()


def test_usage_rollup_reconciliation_still_retires_an_undeclared_renamed_model(tmp_path: Path) -> None:
    """The declaration exemption is exact: an undeclared row with no evidence goes.

    Anti-vacuity: exempt every row from deletion and ``model-before`` survives
    beside ``model-after`` after the message is renamed in place.
    """
    conn = _connect(tmp_path / "index.db")
    try:
        session = ParsedSession(
            source_name=Provider.CODEX,
            provider_session_id="renamed-model",
            messages=[_message("a0", "answer", model_name="model-before", input_tokens=12, output_tokens=3)],
        )
        session_id = write_fixture_index_session(conn, session)
        conn.execute("UPDATE messages SET model_name = 'model-after' WHERE session_id = ?", (session_id,))

        reconcile_session_usage_rollups(conn, [session_id])

        assert _usage_rows(conn, session_id) == [("model-after", 12, 3)]
    finally:
        conn.close()


def test_late_parent_reextraction_keeps_a_declared_model_and_its_unnamed_usage(tmp_path: Path) -> None:
    """A child re-sliced after its parent arrives keeps its declared usage row.

    Anti-vacuity: restore the wholesale ``DELETE FROM session_model_usage`` in
    the late-parent re-extraction (or the partial clear in
    ``_reextract_provider_usage_tail_db``) and the child ends with no model
    row, its unnamed token_count unattributed.
    """
    conn = _connect(tmp_path / "index.db")
    try:
        child = ParsedSession(
            source_name=Provider.CLAUDE_CODE,
            provider_session_id="late-parent-child",
            parent_session_provider_id="late-parent",
            branch_type=BranchType.CONTINUATION,
            updated_at="2026-01-01T00:00:02+00:00",
            models_used=["declared-model"],
            messages=[
                _message("shared", "shared prefix", role=Role.USER),
                _message("tail", "child tail"),
            ],
            session_events=[_delta_usage(40, 10)],
        )
        parent = ParsedSession(
            source_name=Provider.CLAUDE_CODE,
            provider_session_id="late-parent",
            updated_at="2026-01-01T00:00:01+00:00",
            messages=[_message("shared", "shared prefix", role=Role.USER)],
        )
        child_id = write_fixture_index_session(conn, child)
        assert _usage_rows(conn, child_id) == [("declared-model", 40, 10)]

        write_fixture_index_session(conn, parent)

        inheritance = conn.execute(
            "SELECT inheritance FROM session_links WHERE src_session_id = ?", (child_id,)
        ).fetchone()
        assert inheritance[0] == "prefix-sharing", "the parent's arrival must re-slice the child"
        assert _usage_rows(conn, child_id) == [("declared-model", 40, 10)]
    finally:
        conn.close()


@pytest.mark.parametrize("route", ["inline", "prepared"])
def test_union_drops_usage_whose_message_the_union_removed(tmp_path: Path, route: str) -> None:
    """A removed message's usage event does not survive as unanchored usage.

    The parent of a prefix-sharing child never has a message reinjected, so a
    second acquisition that omits ``p2`` removes it. Its usage event must go
    with it.

    Anti-vacuity: carry the older row forward with its message remapped to
    NULL (the previous behaviour) and a row with no message survives, adding
    ``p2``'s 40/10 to the parent's rollup.
    """
    conn = _connect(tmp_path / "index.db")
    try:
        full_parent = ParsedSession(
            source_name=Provider.CODEX,
            provider_session_id="union-parent",
            messages=[
                _message("p0", "hello", role=Role.USER),
                _message("p1", "first answer", model_name="usage-model"),
                _message("p2", "second answer", model_name="usage-model"),
            ],
            session_events=[
                _delta_usage(100, 20, message="p1", model="usage-model"),
                _delta_usage(40, 10, message="p2", model="usage-model"),
            ],
        )
        parent_id = write_fixture_index_session(conn, full_parent, raw_id="parent-acquisition-1")
        child = ParsedSession(
            source_name=Provider.CODEX,
            provider_session_id="union-child",
            parent_session_provider_id="union-parent",
            branch_type=BranchType.FORK,
            messages=[
                _message("p0", "hello", role=Role.USER),
                _message("p1", "first answer", model_name="usage-model"),
                _message("cx", "child diverges", role=Role.USER),
            ],
        )
        write_fixture_index_session(conn, child)
        assert _usage_rows(conn, parent_id) == [("usage-model", 140, 30)]

        shorter_parent = full_parent.model_copy(
            update={
                "messages": full_parent.messages[:2],
                "session_events": full_parent.session_events[:1],
            }
        )
        _write(conn, shorter_parent, route=route, raw_id="parent-acquisition-2")

        native_ids = {
            str(row[0]) for row in conn.execute("SELECT native_id FROM messages WHERE session_id = ?", (parent_id,))
        }
        assert native_ids == {"p0", "p1"}, "a prefix-sharing parent must not reinject the removed message"
        assert _usage_events(conn, parent_id) == [(f"{parent_id}:n:p1", "p1", "resolved", 100, 20)]
        assert _usage_rows(conn, parent_id) == [("usage-model", 100, 20)]
    finally:
        conn.close()


@pytest.mark.parametrize("route", ["inline", "prepared"])
def test_union_matches_unresolved_usage_to_its_resolved_form(tmp_path: Path, route: str) -> None:
    """One provider observation stays one row once its message arrives.

    Acquisition 1 records ``m1``'s usage before ``m1`` itself is present, so
    the row is unresolved. Acquisition 2 carries ``m1`` and the same event.

    Anti-vacuity: key the unresolved row apart from the resolved one (by type,
    model and ordinal instead of the provider message id) and both rows
    survive, so the rollup counts 80/20 instead of 40/10.
    """
    conn = _connect(tmp_path / "index.db")
    try:
        event = _delta_usage(40, 10, message="m1", model="usage-model")
        before = ParsedSession(
            source_name=Provider.CODEX,
            provider_session_id="late-message",
            messages=[_message("m0", "question", role=Role.USER)],
            session_events=[event],
        )
        session_id = write_fixture_index_session(conn, before, raw_id="usage-acquisition-1")
        assert _usage_events(conn, session_id) == [(None, "m1", "unresolved", 40, 10)]

        after = before.model_copy(update={"messages": [*before.messages, _message("m1", "answer")]})
        _write(conn, after, route=route, raw_id="usage-acquisition-2")

        assert _usage_events(conn, session_id) == [(f"{session_id}:n:m1", "m1", "resolved", 40, 10)]
        assert _usage_rows(conn, session_id) == [("usage-model", 40, 10)]
    finally:
        conn.close()


@pytest.mark.parametrize("route", ["inline", "prepared"])
def test_union_keeps_the_resolved_message_when_a_later_acquisition_loses_it(tmp_path: Path, route: str) -> None:
    """The reverse order: a poorer acquisition's unresolved event is the same row.

    The union reinjects ``m1``, so the older row's message is still stored and
    the merged row stays attributed to it.

    Anti-vacuity: key rows by resolved message id and the unresolved incoming
    row is kept beside the carried resolved one, doubling the tokens.
    """
    conn = _connect(tmp_path / "index.db")
    try:
        event = _delta_usage(40, 10, message="m1", model="usage-model")
        rich = ParsedSession(
            source_name=Provider.CODEX,
            provider_session_id="lost-message",
            messages=[_message("m0", "question", role=Role.USER), _message("m1", "answer")],
            session_events=[event],
        )
        session_id = write_fixture_index_session(conn, rich, raw_id="usage-acquisition-1")
        poorer = rich.model_copy(update={"messages": rich.messages[:1]})
        _write(conn, poorer, route=route, raw_id="usage-acquisition-2")

        assert _usage_events(conn, session_id) == [(f"{session_id}:n:m1", "m1", "resolved", 40, 10)]
        assert _usage_rows(conn, session_id) == [("usage-model", 40, 10)]
    finally:
        conn.close()


@pytest.mark.parametrize("model_a_message_tokens", [0, 50])
def test_append_model_switch_returns_the_old_model_to_its_message_totals(
    tmp_path: Path, model_a_message_tokens: int
) -> None:
    """A measured message count never pins a subsumed session-global cumulative.

    Model A's cumulative (300) is superseded by model B's (500). A's one
    message reports ``model_a_message_tokens`` input tokens -- an explicit
    zero, or a genuine count. A full write of the same session is the
    reference.

    Anti-vacuity: exempt a model whose messages report any counter (the
    ``IS NOT NULL`` rule) and A keeps 300, so the rollup totals 800 or 850.
    """
    conn = _connect(tmp_path / "index.db")
    try:
        first = ParsedSession(
            source_name=Provider.CODEX,
            provider_session_id="switch-append",
            models_used=["model-a", "model-b"],
            messages=[
                _message(
                    "m1",
                    "first",
                    model_name="model-a",
                    input_tokens=model_a_message_tokens,
                    output_tokens=0,
                    cache_read_tokens=0,
                    cache_write_tokens=0,
                )
            ],
            session_events=[_cumulative_usage(300, message="m1", model="model-a")],
        )
        second = first.model_copy(
            update={
                "messages": [_message("m2", "second", model_name="model-b")],
                "session_events": [_cumulative_usage(500, message="m2", model="model-b")],
            }
        )
        session_id = write_fixture_index_session(conn, first)
        write_fixture_index_session(conn, second, merge_append=True)
        appended = _usage_rows(conn, session_id)

        whole = first.model_copy(
            update={
                "provider_session_id": "switch-whole",
                "messages": [*first.messages, *second.messages],
                "session_events": [*first.session_events, *second.session_events],
            }
        )
        whole_id = write_fixture_index_session(conn, whole)

        assert appended == [("model-a", model_a_message_tokens, 0), ("model-b", 500, 0)]
        assert appended == _usage_rows(conn, whole_id)
    finally:
        conn.close()
