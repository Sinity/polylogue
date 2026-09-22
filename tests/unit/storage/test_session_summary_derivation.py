"""Laws for the sessions-table counter projection.

The oracle intentionally reads raw ``messages`` and sums values itself.  It
does not call the summary declaration, so omitted measures and a skipped
duplicate are observable through the real parsed-session writer.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

from polylogue.archive.message.roles import Role
from polylogue.archive.session.branch_type import BranchType
from polylogue.core.enums import BlockType, MaterialOrigin, Provider
from polylogue.daemon.derivation import DerivationFrame, DerivationRegistry, converge
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
from polylogue.storage.derived.session.summary import (
    SESSION_SUMMARY_DOMAIN,
    SESSION_SUMMARY_RECIPE_VERSION,
    SessionSummaryDerivation,
    inspect_session_summary,
)
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.write import write_parsed_session_to_archive
from tests.infra.sqlite_work_counter import sqlite_work_counter


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    initialize_archive_tier(conn, ArchiveTier.INDEX)
    return conn


_SUMMARY_COLUMNS = (
    "message_count",
    "word_count",
    "tool_use_count",
    "thinking_count",
    "paste_count",
    "user_message_count",
    "authored_user_message_count",
    "assistant_message_count",
    "system_message_count",
    "tool_message_count",
    "user_word_count",
    "authored_user_word_count",
    "assistant_word_count",
)


def _oracle(conn: sqlite3.Connection, session_id: str) -> dict[str, int]:
    """Independent raw-message reduction for all thirteen persisted counters."""
    totals = dict.fromkeys(_SUMMARY_COLUMNS, 0)
    rows = conn.execute(
        """
        SELECT role, material_origin, word_count, has_tool_use, has_thinking, has_paste
        FROM messages WHERE session_id = ?
        """,
        (session_id,),
    )
    for role, material_origin, word_count, has_tool_use, has_thinking, has_paste in rows:
        totals["message_count"] += 1
        totals["word_count"] += int(word_count)
        totals["tool_use_count"] += int(has_tool_use)
        totals["thinking_count"] += int(has_thinking)
        totals["paste_count"] += int(has_paste)
        if role == "user":
            totals["user_message_count"] += 1
            totals["user_word_count"] += int(word_count)
        if material_origin == "human_authored":
            totals["authored_user_message_count"] += 1
            totals["authored_user_word_count"] += int(word_count)
        if role == "assistant":
            totals["assistant_message_count"] += 1
            totals["assistant_word_count"] += int(word_count)
        if role == "system":
            totals["system_message_count"] += 1
        if role == "tool":
            totals["tool_message_count"] += 1
    return totals


def _stored(conn: sqlite3.Connection, session_id: str) -> dict[str, int]:
    columns = ", ".join(_SUMMARY_COLUMNS)
    row = conn.execute(f"SELECT {columns} FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
    assert row is not None
    return {column: int(row[column]) for column in _SUMMARY_COLUMNS}


def _adapter(index_db: Path, session_ids: tuple[str, ...]) -> SessionSummaryDerivation:
    def read_connection() -> sqlite3.Connection:
        return sqlite3.connect(f"file:{index_db}?mode=ro", uri=True)

    def write_connection() -> sqlite3.Connection:
        return sqlite3.connect(index_db)

    return SessionSummaryDerivation(
        read_connection,
        write_connection,
        session_scope=lambda _frame: session_ids,
    )


def test_session_summary_converges_append_overlap_and_late_lineage_from_messages(tmp_path: Path) -> None:
    """Every writer path uses the same persisted-message counter relation.

    Anti-vacuity: restoring the former append increment, omitting one measure,
    or skipping the late-prefix refresh leaves at least one independent oracle
    comparison wrong.
    """
    conn = _connect(tmp_path / "index.db")
    try:
        initial = ParsedSession(
            source_name=Provider.CODEX,
            provider_session_id="summary-main",
            messages=[
                ParsedMessage(
                    provider_message_id="u1",
                    role=Role.USER,
                    material_origin=MaterialOrigin.HUMAN_AUTHORED,
                    text="typed opening words",
                ),
                ParsedMessage(
                    provider_message_id="a1",
                    role=Role.ASSISTANT,
                    text="first assistant answer",
                    blocks=[
                        ParsedContentBlock(type=BlockType.TOOL_USE, tool_id="tool-1", tool_name="run"),
                        ParsedContentBlock(type=BlockType.THINKING, text="plan"),
                    ],
                ),
            ],
        )
        session_id = write_parsed_session_to_archive(conn, initial)
        assert _stored(conn, session_id) == _oracle(conn, session_id)

        # A native-id-less tail proves the authorative replacement is based on
        # what the writer stored, rather than an assumed provider-id delta.
        append = ParsedSession(
            source_name=Provider.CODEX,
            provider_session_id="summary-main",
            messages=[
                ParsedMessage(
                    provider_message_id="",
                    role=Role.USER,
                    material_origin=MaterialOrigin.GENERATED_CONTEXT_PACK,
                    text="generated context tail",
                ),
                ParsedMessage(provider_message_id="tool", role=Role.TOOL, text="tool result"),
            ],
        )
        write_parsed_session_to_archive(conn, append, merge_append=True)
        assert _stored(conn, session_id) == _oracle(conn, session_id)

        # The same native id changes role and authoredness.  Force replacement
        # must not retain the previous append arithmetic.
        corrected = initial.model_copy(
            update={
                "messages": [
                    ParsedMessage(
                        provider_message_id="u1",
                        role=Role.ASSISTANT,
                        material_origin=MaterialOrigin.ASSISTANT_AUTHORED,
                        text="corrected assistant opening",
                    ),
                    ParsedMessage(provider_message_id="system", role=Role.SYSTEM, text="system rule"),
                ]
            }
        )
        write_parsed_session_to_archive(conn, corrected, force_replace=True)
        assert _stored(conn, session_id) == _oracle(conn, session_id)

        child = ParsedSession(
            source_name=Provider.CLAUDE_CODE,
            provider_session_id="summary-child",
            parent_session_provider_id="summary-parent",
            branch_type=BranchType.CONTINUATION,
            messages=[
                ParsedMessage(provider_message_id="shared", role=Role.USER, text="shared prefix"),
                ParsedMessage(provider_message_id="tail", role=Role.ASSISTANT, text="child tail"),
            ],
        )
        child_id = write_parsed_session_to_archive(conn, child)
        parent = ParsedSession(
            source_name=Provider.CLAUDE_CODE,
            provider_session_id="summary-parent",
            messages=[ParsedMessage(provider_message_id="shared", role=Role.USER, text="shared prefix")],
        )
        write_parsed_session_to_archive(conn, parent)
        assert _stored(conn, child_id) == _oracle(conn, child_id)
    finally:
        conn.close()


def test_session_summary_inspection_repairs_corruption_then_second_pass_writes_nothing(tmp_path: Path) -> None:
    """Inspection is authoritative and an unchanged valid partition is skipped.

    Anti-vacuity: compare only session existence, omit a stored measure, or
    publish without reinspection and this corruption is incorrectly certified.
    """
    index_db = tmp_path / "index.db"
    conn = _connect(index_db)
    try:
        session_id = write_parsed_session_to_archive(
            conn,
            ParsedSession(
                source_name=Provider.CODEX,
                provider_session_id="summary-corruption",
                messages=[ParsedMessage(provider_message_id="m1", role=Role.USER, text="one two three")],
            ),
        )
        conn.execute("UPDATE sessions SET assistant_word_count = 99 WHERE session_id = ?", (session_id,))
        conn.commit()
    finally:
        conn.close()

    frame = DerivationFrame(
        archive_root=str(tmp_path),
        source_revision="summary-frame",
        recipe_versions={SESSION_SUMMARY_DOMAIN: SESSION_SUMMARY_RECIPE_VERSION},
        scope=(session_id,),
    )
    adapter = _adapter(index_db, (session_id,))
    assert adapter.inspect(frame, (session_id,))[session_id] == "stale"
    replacement = adapter.compute(frame, session_id)
    wrong_recipe = DerivationFrame(
        archive_root=str(tmp_path),
        source_revision="summary-frame",
        recipe_versions={SESSION_SUMMARY_DOMAIN: "retired"},
        scope=(session_id,),
    )
    assert adapter.inspect(wrong_recipe, (session_id,))[session_id] == "stale"
    assert adapter.publish(wrong_recipe, replacement) is False
    assert adapter.publish(frame, replacement) is True
    assert adapter.inspect(frame, (session_id,))[session_id] == "valid"

    # Compute has no writer lease.  A changed base projection between compute
    # and publish must refuse the stale replacement without a partial counter
    # update; the next ordinary pass rebuilds it from messages.
    stale_replacement = adapter.compute(frame, session_id)
    conn = sqlite3.connect(index_db)
    try:
        conn.execute("UPDATE messages SET word_count = word_count + 7 WHERE session_id = ?", (session_id,))
        conn.commit()
    finally:
        conn.close()
    assert adapter.publish(frame, stale_replacement) is False
    assert adapter.inspect(frame, (session_id,))[session_id] == "stale"

    repaired = converge(DerivationRegistry([adapter]), frame)
    assert repaired.done == 1
    report = converge(DerivationRegistry([adapter]), frame)
    assert report.wrote_nothing


def test_session_summary_census_marks_a_corrupt_counter_stale(tmp_path: Path) -> None:
    """The status census compares stored counters to messages, not stage history."""
    conn = _connect(tmp_path / "index.db")
    try:
        session_id = write_parsed_session_to_archive(
            conn,
            ParsedSession(
                source_name=Provider.CODEX,
                provider_session_id="summary-census-corruption",
                messages=[ParsedMessage(provider_message_id="m1", role=Role.USER, text="one two")],
            ),
        )
        conn.execute("UPDATE sessions SET message_count = 99 WHERE session_id = ?", (session_id,))
        inspection = inspect_session_summary(conn)
    finally:
        conn.close()

    assert inspection.state == "stale"
    assert inspection.total_sessions == 1
    assert inspection.stale_sessions == 1


def test_session_summary_census_deadline_is_unknown(tmp_path: Path) -> None:
    """A status deadline cannot certify a partially inspected counter projection."""
    conn = _connect(tmp_path / "index.db")
    try:
        inspection = inspect_session_summary(conn, deadline_s=0)
    finally:
        conn.close()

    assert inspection.state == "unknown"
    assert inspection.reason == "session-summary inspection deadline exceeded"


def test_session_summary_census_accepts_valid_empty_session(tmp_path: Path) -> None:
    """A LEFT JOIN's synthetic row cannot count as an empty session's message."""
    conn = _connect(tmp_path / "index.db")
    try:
        write_parsed_session_to_archive(
            conn,
            ParsedSession(source_name=Provider.CODEX, provider_session_id="empty-census", messages=[]),
        )
        inspection = inspect_session_summary(conn)
        assert inspection.state == "ready"
        assert inspection.total_sessions == 1
        assert inspection.stale_sessions == 0
    finally:
        conn.close()


def test_summary_inspection_refuses_retired_generation(tmp_path: Path) -> None:
    """Anti-vacuity: current counters cannot certify the previous generation."""
    from dataclasses import replace

    import pytest

    from polylogue.operations.session_profile_convergence import (
        make_session_profile_frame,
        make_session_summary_derivation,
    )

    root = tmp_path / "archive"
    root.mkdir()
    conn = _connect(root / "index.db")
    try:
        session_id = write_parsed_session_to_archive(
            conn,
            ParsedSession(source_name=Provider.CODEX, provider_session_id="frame", messages=[]),
        )
        conn.commit()
    finally:
        conn.close()
    adapter = make_session_summary_derivation(root / "index.db", archive_root=root)
    frame = make_session_profile_frame(root / "index.db", archive_root=root, scope=(session_id,))
    assert adapter.inspect(frame, (session_id,))[session_id] == "valid"
    with pytest.raises(RuntimeError, match="retired"):
        adapter.inspect(replace(frame, source_revision="index-generation:retired"), (session_id,))


def test_summary_census_preserves_operation_cancellation(tmp_path: Path) -> None:
    """Anti-vacuity: replacing the owner's progress handler hides cancellation."""
    import pytest

    conn = _connect(tmp_path / "index.db")
    conn.set_progress_handler(lambda: 1, 1)
    try:
        assert inspect_session_summary(conn).state == "unknown"
        with pytest.raises(sqlite3.OperationalError, match="interrupted"):
            conn.execute("SELECT COUNT(*) FROM sessions").fetchone()
    finally:
        conn.set_progress_handler(None, 0)
        conn.close()


def _seed_counter_archive(path: Path, *, sessions: int, messages: int) -> sqlite3.Connection:
    """Write ``sessions`` sessions of ``messages`` messages through the writer."""
    conn = _connect(path)
    for index in range(sessions):
        write_parsed_session_to_archive(
            conn,
            ParsedSession(
                source_name=Provider.CODEX,
                provider_session_id=f"scale-{index:04d}",
                messages=[
                    ParsedMessage(provider_message_id=f"m{position}", role=Role.USER, text="one two three")
                    for position in range(messages)
                ],
            ),
        )
    conn.commit()
    return conn


def _reads_message_relation(statements: list[str]) -> list[str]:
    """Statements that name the ``messages`` base relation, not a lookalike."""
    return [sql for sql in statements if re.search(r"\bmessages\b(?!_)", sql)]


def test_summary_readiness_reads_no_message_rows(tmp_path: Path) -> None:
    """Readiness consults the binding relation and never the message population.

    Anti-vacuity: restore the archive-wide
    ``sessions LEFT JOIN messages GROUP BY session_id`` census this domain used
    to run and the trace below carries it, so this assertion fails.  A timing
    number could not tell those apart on a small fixture; the executed SQL can.
    """
    conn = _seed_counter_archive(tmp_path / "index.db", sessions=4, messages=6)
    executed: list[str] = []
    try:
        conn.set_trace_callback(lambda sql: executed.append(" ".join(sql.lower().split())))
        inspection = inspect_session_summary(conn)
    finally:
        conn.set_trace_callback(None)
        conn.close()

    assert inspection.state == "ready", inspection
    assert executed, "the inspection executed no SQL at all"
    assert _reads_message_relation(executed) == []


def test_summary_cost_ignores_message_count(tmp_path: Path) -> None:
    """The same session population costs the same whatever its message volume.

    Anti-vacuity: the retired census reduced thirteen aggregates over every
    message row, so the wide archive's VM-step count was a multiple of the
    narrow one's.  Restoring it makes the ratio assertion fail.
    """
    narrow_path = tmp_path / "narrow" / "index.db"
    wide_path = tmp_path / "wide" / "index.db"
    narrow_path.parent.mkdir()
    wide_path.parent.mkdir()
    _seed_counter_archive(narrow_path, sessions=24, messages=1).close()
    _seed_counter_archive(wide_path, sessions=24, messages=40).close()

    def steps(path: Path) -> int:
        with sqlite_work_counter(step_interval=4) as counter:
            conn = sqlite3.connect(path)
            conn.row_factory = sqlite3.Row
            try:
                assert inspect_session_summary(conn).state == "ready"
            finally:
                conn.close()
        return counter.metric("vm_steps")

    narrow_steps = steps(narrow_path)
    wide_steps = steps(wide_path)
    assert narrow_steps > 0
    # 40x the messages, and the inspection is allowed no more than half again
    # the work. The census form measured roughly 20x here.
    assert wide_steps <= narrow_steps * 1.5, (narrow_steps, wide_steps)


def test_message_write_retires_the_binding(tmp_path: Path) -> None:
    """A message value change makes the domain unbound, never still-ready.

    Anti-vacuity: drop the ``session_summary_binding_messages_au`` trigger and
    the binding survives a message that no longer matches the published
    counters, so readiness reports ``ready`` and this fails.  The mutated
    column is one the counters reduce, and nothing else is touched.
    """
    conn = _seed_counter_archive(tmp_path / "index.db", sessions=2, messages=3)
    try:
        session_id = str(conn.execute("SELECT session_id FROM sessions ORDER BY session_id").fetchone()[0])
        assert inspect_session_summary(conn).state == "ready"
        conn.execute("UPDATE messages SET word_count = word_count + 7 WHERE session_id = ?", (session_id,))
        inspection = inspect_session_summary(conn)
    finally:
        conn.close()

    assert inspection.state == "stale"
    assert inspection.unbound_sessions == 1
    # The drift was never compared, so it is never reported as measured.
    assert inspection.stale_sessions == 0
    assert inspection.unmeasured_reason is not None
    assert "no session-summary binding" in inspection.unmeasured_reason


def test_unbound_partition_is_republished(tmp_path: Path) -> None:
    """Correct counters with no binding are work the kernel still owes.

    Anti-vacuity: let ``SessionSummaryDerivation.inspect`` call a partition
    valid on stored-equals-authoritative alone and this session stays unbound
    forever -- the kernel reports converged while readiness reports stale.
    """
    from polylogue.operations.session_profile_convergence import (
        make_session_profile_frame,
        make_session_summary_derivation,
    )

    root = tmp_path / "archive"
    root.mkdir()
    conn = _seed_counter_archive(root / "index.db", sessions=1, messages=2)
    try:
        session_id = str(conn.execute("SELECT session_id FROM sessions").fetchone()[0])
        conn.execute("DELETE FROM session_summary_bindings")
        conn.commit()
    finally:
        conn.close()

    adapter = make_session_summary_derivation(root / "index.db", archive_root=root)
    frame = make_session_profile_frame(root / "index.db", archive_root=root, scope=(session_id,))
    assert adapter.inspect(frame, (session_id,))[session_id] == "stale"
    assert converge(DerivationRegistry([adapter]), frame).done == 1
    assert adapter.inspect(frame, (session_id,))[session_id] == "valid"

    conn = sqlite3.connect(root / "index.db")
    try:
        assert inspect_session_summary(conn).state == "ready"
    finally:
        conn.close()
