"""The archive row declaration is executable authority, not documentation.

Each test names the mutation that turns it red, so a declaration that stopped
driving write, read or hydration cannot pass silently.
"""

from __future__ import annotations

import sqlite3
from dataclasses import replace
from operator import itemgetter
from pathlib import Path

import pytest

from polylogue.archive.session.domain_models import SessionSummary
from polylogue.core.enums import BlockType, Origin, Provider, Role
from polylogue.core.types import ContentHash, MessageId, SessionId
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
from polylogue.storage import hydrators
from polylogue.storage.hydrators import message_from_record, session_from_records, session_summary_from_record
from polylogue.storage.runtime import BlockRecord, MessageRecord, SessionRecord
from polylogue.storage.sqlite.archive_tiers import archive_tiers_specs
from polylogue.storage.sqlite.archive_tiers.archive_tiers_specs import (
    BLOCKS_SPEC,
    MESSAGES_SPEC,
    SESSION_WORKING_DIRS_SPEC,
    SESSIONS_SPEC,
    _raw_column,
)
from polylogue.storage.sqlite.archive_tiers.column_spec import ColumnSpec, TableColumnSpec
from polylogue.storage.sqlite.archive_tiers.write import (
    ARCHIVE_BLOCK_ROW_COLUMNS,
    ArchiveBlockRow,
    archive_block_row,
)
from polylogue.storage.sqlite.async_sqlite import SQLiteBackend
from polylogue.storage.sqlite.queries import message_query_reads, sessions_reads
from tests.infra.live_ingest import ingest_session


def _parsed_session(
    native_id: str,
    *,
    title: str | None = "Row plumbing",
    git_branch: str | None = "feature/rows",
) -> ParsedSession:
    return ParsedSession(
        source_name=Provider.CLAUDE_CODE,
        provider_session_id=native_id,
        title=title,
        git_branch=git_branch,
        messages=[
            ParsedMessage(
                provider_message_id="native-message",
                role=Role.ASSISTANT,
                text="finished",
                timestamp="2026-03-01T10:05:00+00:00",
                position=0,
                stop_reason="end_turn",
                is_active_path=True,
                is_active_leaf=True,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="finished")],
            )
        ],
    )


def _with_extra_column(spec: TableColumnSpec, column: ColumnSpec) -> TableColumnSpec:
    """Append one stored column, bound exactly as ``_make_table_spec`` binds it."""
    bound = replace(column, extract=itemgetter(column.name))
    return replace(
        spec,
        all_columns=(*spec.all_columns, bound),
        writable_columns=(*spec.writable_columns, bound),
    )


class _DomainSource:
    """Attribute bag standing in for a storage record during domain projection."""

    def __init__(self, **values: object) -> None:
        self.__dict__.update(values)


def test_an_added_column_reaches_write_read_and_domain_from_one_declaration() -> None:
    """A new stored column needs the declaration, not a coordinated mapper edit.

    The column is added to the messages declaration alone. Red if the rendered
    DDL, the INSERT column list, the bound-value tuple, the record projection,
    the record kwargs or the domain kwargs stops deriving from that
    declaration: each stage would then need its own hand edit before the
    column appeared.
    """
    extended = _with_extra_column(
        MESSAGES_SPEC,
        _raw_column(
            "review_state",
            "review_state TEXT",
            record_name="review_state",
            domain_name="review_state",
        ),
    )
    assert "review_state" in extended.insert_column_names

    values: dict[str, object] = dict.fromkeys((column.name for column in extended.insert_columns), None)
    values.update(
        {
            "session_id": "claude-code-session:added-column",
            "native_id": "native-message",
            "identity_source": "native",
            "position": 0,
            "role": Role.ASSISTANT.value,
            "message_type": "message",
            "material_origin": "unknown",
            "variant_index": 0,
            "is_active_path": 1,
            "is_active_leaf": 1,
            "has_tool_use": 0,
            "has_thinking": 0,
            "has_paste": 0,
            "word_count": 1,
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
            "content_hash": bytes(32),
            "occurred_at_ms": 1_772_000_000_000,
            "review_state": "accepted",
        }
    )

    # Every field the message record reads except the two a session join and a
    # literal supply; the added column travels with the rest.
    projected = tuple(
        column for column in extended.record_columns if column.record_name not in {"source_name", "version"}
    )
    projection = ",\n    ".join(column.select_sql("m") for column in projected)

    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    try:
        connection.execute(f"CREATE TABLE blocks (\n    {BLOCKS_SPEC.ddl_body}\n)")
        connection.execute(f"CREATE TABLE messages (\n    {extended.ddl_body}\n)")
        connection.execute(
            f"INSERT INTO messages ({extended.insert_column_names}) VALUES ({extended.insert_placeholder_string})",
            extended.extract_tuple(values),
        )
        row = connection.execute(f"SELECT {projection} FROM messages m").fetchone()
    finally:
        connection.close()

    kwargs = extended.row_to_record_kwargs(row)
    assert kwargs["review_state"] == "accepted"
    assert kwargs["stop_reason"] is None
    assert extended.domain_kwargs(_DomainSource(**kwargs))["review_state"] == "accepted"


def test_declared_projection_names_every_record_field_for_the_core_tables() -> None:
    """Session, message and block record models read only projected fields.

    Red when a record field has no ``record_name`` anywhere in its table's
    declaration: reads would leave it at its default with nothing to notice.
    """
    for spec, model, derived in (
        (SESSIONS_SPEC, SessionRecord, set()),
        (MESSAGES_SPEC, MessageRecord, {"blocks"}),
        (BLOCKS_SPEC, BlockRecord, set()),
    ):
        declared = {column.record_name for column in spec.record_columns}
        missing = set(model.model_fields) - declared - derived
        assert not missing, f"{spec.table_name}: record fields with no projection: {sorted(missing)}"


def test_the_declaration_owns_the_session_upsert_policy() -> None:
    """``DO UPDATE SET`` is rendered from the columns, not restated beside them.

    Red when a column's conflict policy is dropped (it silently stops moving on
    re-ingest) and when a counter column gains one (the rollup statements that
    own those counts would be overwritten by the insert's stale values).
    """
    updated = {column.name for column in SESSIONS_SPEC.conflict_update_columns}
    inserted = {column.name for column in SESSIONS_SPEC.insert_columns}

    assert updated <= inserted
    assert {"raw_id", "session_kind", "content_hash", "created_at_ms", "updated_at_ms"} <= updated
    # Identity and the message/word counters keep their stored value: the
    # counters are owned by the rollup statements that follow the upsert.
    assert not updated & {"native_id", "origin"}
    assert not {name for name in updated if name.endswith("_count")}

    rendered = SESSIONS_SPEC.conflict_update_sql()
    assert "title = COALESCE(excluded.title, sessions.title)" in rendered
    assert "pending_drafts_json = excluded.pending_drafts_json" in rendered

    # The statement's bound values are named by the columns that read them, in
    # placeholder order. Red when a conflict expression gains or loses a
    # placeholder without declaring it: the write would bind the timestamp
    # ratchet's parameters against the wrong conditions.
    assert SESSIONS_SPEC.conflict_update_param_names == (
        "producer_created",
        "force_replace",
        "producer_updated",
        "producer_created",
        "producer_created",
        "producer_updated",
        "producer_updated_or_merge_append",
    )
    assert len(SESSIONS_SPEC.conflict_update_param_names) == rendered.count("?")


def test_every_declared_conflict_policy_names_its_bound_values() -> None:
    """No table renders an upsert parameter it does not declare a name for.

    Red when a spec adds a placeholder to a conflict expression without saying
    which value fills it: the caller would have to re-derive the order by hand,
    which is the restatement the declaration replaced.
    """
    for name, spec in archive_tiers_specs.TABLE_SPECS.items():
        # Raises when a column's placeholder count and declared names disagree.
        assert len(spec.conflict_update_param_names) == spec.conflict_update_sql().count("?"), name

    # The check is not vacuous: a policy that gains a placeholder without
    # naming its value is rejected rather than silently mis-bound.
    undeclared = replace(
        SESSIONS_SPEC,
        all_columns=tuple(
            replace(column, conflict_params=()) if column.name == "created_at_ms" else column
            for column in SESSIONS_SPEC.all_columns
        ),
    )
    with pytest.raises(ValueError, match="sessions.created_at_ms"):
        _ = undeclared.conflict_update_param_names


@pytest.mark.asyncio
async def test_session_upsert_policy_holds_on_the_production_write_route(tmp_path: Path) -> None:
    """Re-ingest overwrites what the declaration overwrites and preserves the rest.

    Red when ``conflict_update`` policies are lost or inverted: an omitted title
    would erase the stored one, or a new git branch would not land.
    """
    db_path = tmp_path / "index.db"
    backend = SQLiteBackend(db_path=db_path)
    try:
        session_id = await ingest_session(_parsed_session("upsert-policy"), backend)
        await ingest_session(_parsed_session("upsert-policy", title=None, git_branch="feature/moved"), backend)
    finally:
        await backend.close()

    connection = sqlite3.connect(str(db_path))
    connection.row_factory = sqlite3.Row
    try:
        row = connection.execute(
            "SELECT title, git_branch, message_count FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
    finally:
        connection.close()

    assert row["title"] == "Row plumbing"
    assert row["git_branch"] == "feature/moved"
    assert row["message_count"] == 1


def test_every_block_read_model_shares_one_projection_and_hydrator() -> None:
    """The compact block projection is derived from the model and the declaration.

    Red when a projected name stops being a declared blocks column, or when the
    model's fields and the projection drift apart -- either way a read would
    hydrate a field from a column the table does not have.
    """
    declared = {column.name for column in BLOCKS_SPEC.all_columns}
    assert set(ARCHIVE_BLOCK_ROW_COLUMNS) <= declared
    assert len(ARCHIVE_BLOCK_ROW_COLUMNS) == len(set(ARCHIVE_BLOCK_ROW_COLUMNS))
    # ``metadata`` is carried by the read model but is not a blocks column, so
    # it must never be projected.
    assert "metadata" not in ARCHIVE_BLOCK_ROW_COLUMNS

    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    try:
        connection.execute(f"CREATE TABLE blocks (\n    {BLOCKS_SPEC.ddl_body}\n)")
        connection.execute(
            "INSERT INTO blocks (message_id, session_id, position, block_type, text, tool_name, tool_outcome) "
            "VALUES ('s:m', 's', 0, 'tool_result', 'ok', 'Bash', 'ok')"
        )
        row = connection.execute(
            f"SELECT {', '.join(ARCHIVE_BLOCK_ROW_COLUMNS)} FROM blocks",
        ).fetchone()
    finally:
        connection.close()

    hydrated = archive_block_row(row)
    assert isinstance(hydrated, ArchiveBlockRow)
    assert hydrated.block_id == "s:m:0"
    assert hydrated.block_type == "tool_result"
    # The one semantic conversion the shared hydrator owns.
    assert hydrated.tool_outcome is not None
    assert hydrated.tool_outcome.value == "ok"


@pytest.mark.asyncio
async def test_declared_semantics_round_trip_through_the_sync_and_async_routes(tmp_path: Path) -> None:
    """Generated identity, aliases, JSON decoding, enums and nullables agree.

    Red when either route stops reading through the declaration: the two would
    disagree on identity, on the block payload shape, or on an enum's wire text.
    """
    db_path = tmp_path / "index.db"
    parsed = ParsedSession(
        source_name=Provider.CLAUDE_CODE,
        provider_session_id="route-parity",
        title="Route parity",
        messages=[
            ParsedMessage(
                provider_message_id="native-message",
                role=Role.ASSISTANT,
                text="ran it",
                timestamp="2026-03-01T10:05:00+00:00",
                position=0,
                stop_reason="end_turn",
                blocks=[
                    ParsedContentBlock(
                        type=BlockType.TOOL_USE,
                        text="ran it",
                        tool_name="Bash",
                        tool_id="toolu_1",
                        tool_input={"command": "ls"},
                    )
                ],
            )
        ],
    )
    backend = SQLiteBackend(db_path=db_path)
    try:
        session_id = await ingest_session(parsed, backend)
        async with backend.connection() as conn:
            session_record = await sessions_reads.get_session(conn, session_id)
            message_records = await message_query_reads.get_messages(conn, session_id)
            blocks = await backend.get_blocks([message_records[0].message_id])
    finally:
        await backend.close()

    assert session_record is not None
    assert session_record.origin is Origin.CLAUDE_CODE_SESSION
    assert session_record.title == "Route parity"
    # Generated identity, read back through the declared projections.
    assert str(message_records[0].message_id) == f"{session_id}:n:native-message"
    block_record = blocks[str(message_records[0].message_id)][0]
    assert str(block_record.block_id) == f"{message_records[0].message_id}:0"

    hydrated = message_from_record(
        message_records[0].model_copy(update={"blocks": [block_record]}),
        [],
        origin=session_record.origin,
    )
    # Column alias, JSON decoding and enum lowering, all declared not mapped.
    assert hydrated.stop_reason == "end_turn"
    assert hydrated.blocks[0]["tool_input"] == {"command": "ls"}
    assert hydrated.blocks[0]["type"] == "tool_use"

    connection = sqlite3.connect(str(db_path))
    connection.row_factory = sqlite3.Row
    try:
        sync_row = connection.execute(
            f"SELECT {', '.join(ARCHIVE_BLOCK_ROW_COLUMNS)} FROM blocks WHERE session_id = ?", (session_id,)
        ).fetchone()
    finally:
        connection.close()
    sync_block = archive_block_row(sync_row)
    assert sync_block.block_id == str(block_record.block_id)
    assert sync_block.block_type == "tool_use"
    assert sync_block.tool_input is not None
    assert "command" in sync_block.tool_input


@pytest.mark.asyncio
async def test_a_dropped_record_name_starves_both_read_routes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Removing one column's record projection is observable on both routes.

    The controlled mutation drops ``stop_reason``'s ``record_name``. Red would
    mean a read route reconstructs the field from somewhere other than the
    declaration -- exactly the per-family mapper this bead deletes.
    """
    db_path = tmp_path / "index.db"
    backend = SQLiteBackend(db_path=db_path)
    try:
        session_id = await ingest_session(_parsed_session("record-name-mutation"), backend)
        async with backend.connection() as conn:
            before = await message_query_reads.get_messages(conn, session_id)
        assert before[0].stop_reason == "end_turn"

        mutated = replace(
            MESSAGES_SPEC,
            all_columns=tuple(
                replace(column, record_name=None, domain_name=None) if column.name == "stop_reason" else column
                for column in MESSAGES_SPEC.all_columns
            ),
        )
        monkeypatch.setattr(archive_tiers_specs, "MESSAGES_SPEC", mutated)
        monkeypatch.setattr(
            message_query_reads, "_MESSAGE_RECORD_SELECT", mutated.record_select_column_names("m"), raising=True
        )
        async with backend.connection() as conn:
            after = await message_query_reads.get_messages(conn, session_id)
    finally:
        await backend.close()

    assert after[0].stop_reason is None

    # The sync domain projection loses the field for the same reason.
    record = MessageRecord(
        message_id=before[0].message_id,
        session_id=before[0].session_id,
        content_hash=before[0].content_hash,
        stop_reason="end_turn",
    )
    assert "stop_reason" not in mutated.domain_kwargs(record)
    assert MESSAGES_SPEC.domain_kwargs(record)["stop_reason"] == "end_turn"


def test_a_dropped_enum_conversion_leaks_the_enum_into_the_domain_block() -> None:
    """Block enum lowering is declared, so removing it is immediately visible.

    Red would mean the hydrator re-derives the wire text itself, which is the
    per-family conversion the declaration replaced.
    """
    record = BlockRecord(
        block_id="s:m:0",
        message_id=MessageId("s:m"),
        session_id=SessionId("s"),
        block_index=0,
        type=BlockType.TOOL_USE,
        text="ran it",
    )
    assert BLOCKS_SPEC.domain_kwargs(record)["type"] == "tool_use"

    mutated = replace(
        BLOCKS_SPEC,
        all_columns=tuple(
            replace(column, domain_transform=None) if column.name == "block_type" else column
            for column in BLOCKS_SPEC.all_columns
        ),
    )
    assert mutated.domain_kwargs(record)["type"] is BlockType.TOOL_USE


def test_a_relaxed_nullability_changes_the_schema_both_routes_open() -> None:
    """Nullability lives in the declaration, so the rendered DDL follows it.

    Red when the DDL stops rendering from the same column objects the readers
    and writers project: the fresh schema would accept rows the declaration
    forbids.
    """
    connection = sqlite3.connect(":memory:")
    try:
        connection.execute(f"CREATE TABLE blocks (\n    {BLOCKS_SPEC.ddl_body}\n)")
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "INSERT INTO blocks (message_id, session_id, position, block_type) VALUES ('s:m', 's', 0, NULL)"
            )
    finally:
        connection.close()

    relaxed = replace(
        BLOCKS_SPEC,
        all_columns=tuple(
            replace(column, ddl_sql="block_type TEXT") if column.name == "block_type" else column
            for column in BLOCKS_SPEC.all_columns
        ),
    )
    connection = sqlite3.connect(":memory:")
    try:
        connection.execute(f"CREATE TABLE blocks (\n    {relaxed.ddl_body}\n)")
        connection.execute(
            "INSERT INTO blocks (message_id, session_id, position, block_type) VALUES ('s:m', 's', 0, NULL)"
        )
    finally:
        connection.close()


def test_a_wrong_extractor_corrupts_the_production_write(tmp_path: Path) -> None:
    """The bound-value tuple is the declaration's, not a hand-aligned order.

    The controlled mutation points one column's extractor at another column's
    value. Red would mean the write path re-derives its own tuple order and the
    declaration is documentation.
    """
    mutated = replace(
        MESSAGES_SPEC,
        writable_columns=tuple(
            replace(column, extract=itemgetter("output_tokens")) if column.name == "input_tokens" else column
            for column in MESSAGES_SPEC.writable_columns
        ),
    )
    values = dict.fromkeys((column.name for column in MESSAGES_SPEC.insert_columns), 0)
    values["input_tokens"] = 11
    values["output_tokens"] = 22

    bound = [column.name for column in MESSAGES_SPEC.insert_columns if column.extract_placeholder == "?"]
    honest = dict(zip(bound, MESSAGES_SPEC.extract_tuple(values), strict=True))
    corrupt = dict(zip(bound, mutated.extract_tuple(values), strict=True))

    assert honest["input_tokens"] == 11
    assert corrupt["input_tokens"] == 22


def _session_row_values(spec: TableColumnSpec, **overrides: object) -> dict[str, object]:
    """Every bound sessions column, defaulted, with the named overrides applied."""
    values: dict[str, object] = dict.fromkeys((column.name for column in spec.insert_columns), None)
    values.update({column.name: 0 for column in spec.insert_columns if column.name.endswith("_count")})
    values.update(
        {
            "native_id": "spec-owned-session",
            "origin": Origin.CLAUDE_CODE_SESSION.value,
            "session_kind": "standard",
            "content_hash": bytes(32),
            "created_at_ms": 1_772_000_000_000,
            "updated_at_ms": 1_772_000_100_000,
        }
    )
    values.update(overrides)
    return values


def test_an_added_session_column_reaches_write_read_and_domain_from_one_declaration() -> None:
    """A new sessions column needs the declaration, not a hydrator edit too.

    Red if the DDL, the INSERT, the bound tuple, the record projection, the
    record kwargs or the domain kwargs stops deriving from the sessions
    declaration: the column would then need a coordinated edit in the mapper
    and in both session hydrators before it appeared.
    """
    extended = _with_extra_column(
        SESSIONS_SPEC,
        _raw_column(
            "review_state",
            "review_state TEXT",
            record_name="review_state",
            domain_name="review_state",
        ),
    )
    assert "review_state" in extended.insert_column_names

    values = _session_row_values(extended, review_state="accepted", git_branch="feature/rows")
    projection = extended.record_select_column_names("sessions")

    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    try:
        connection.execute(f"CREATE TABLE sessions (\n    {extended.ddl_body}\n)")
        connection.execute(f"CREATE TABLE session_working_dirs (\n    {SESSION_WORKING_DIRS_SPEC.ddl_body}\n)")
        connection.execute(
            f"INSERT INTO sessions ({extended.insert_column_names}) VALUES ({extended.insert_placeholder_string})",
            extended.extract_tuple(values),
        )
        connection.execute(
            "INSERT INTO session_working_dirs (session_id, position, path) "
            "VALUES ('claude-code-session:spec-owned-session', 0, '/realm/project/polylogue')"
        )
        row = connection.execute(f"SELECT {projection} FROM sessions").fetchone()
    finally:
        connection.close()

    kwargs = extended.row_to_record_kwargs(row)
    assert kwargs["review_state"] == "accepted"
    assert kwargs["git_branch"] == "feature/rows"
    # The declared JSON decoding runs on the way to the record, so the record
    # model's validators receive structure rather than stored text.
    assert kwargs["metadata"] == {}

    domain = extended.domain_kwargs(_DomainSource(**kwargs))
    assert domain["review_state"] == "accepted"
    assert domain["git_branch"] == "feature/rows"
    # The declared domain transforms run too: the correlated working-directory
    # projection becomes the domain tuple, and the stored text becomes a datetime.
    assert domain["working_directories"] == ("/realm/project/polylogue",)
    assert domain["created_at"] is not None


def test_one_session_declaration_feeds_both_domain_models() -> None:
    """Session and SessionSummary are projected from one declaration.

    The summary carries a subset of the row (no reported cost); the subset is
    read off the model's own fields. Red when either hydrator restates the
    mapping, because the shared fields would stop agreeing.
    """
    record = SessionRecord(
        session_id=SessionId("claude-code-session:two-models"),
        native_id="two-models",
        origin=Origin.CLAUDE_CODE_SESSION,
        content_hash=ContentHash("deadbeef"),
        title="Two models",
        display_name="two models",
        git_branch="main",
        provider_project_ref="proj",
        reported_cost_usd=1.25,
        working_directories_json='["/realm/project/polylogue"]',
    )

    summary = session_summary_from_record(record, tags=("a",), message_count=3)
    session = session_from_records(record, [], [])

    for model in (summary, session):
        assert str(model.id) == "claude-code-session:two-models"
        assert model.title == "Two models"
        assert model.display_name == "two models"
        assert model.git_branch == "main"
        assert model.provider_project_ref == "proj"
        assert model.working_directories == ("/realm/project/polylogue",)
        assert model.metadata == {}

    assert session.reported_cost_usd == 1.25
    assert "reported_cost_usd" not in SessionSummary.model_fields
    assert summary.message_count == 3
    assert summary.tags_m2m == ("a",)


def test_a_dropped_session_domain_name_starves_both_hydrated_models(monkeypatch: pytest.MonkeyPatch) -> None:
    """Removing one column's domain projection is observable on both hydrators.

    The controlled mutation drops ``git_branch``'s ``domain_name``. Red would
    mean a hydrator re-derives the field from the record itself -- the
    per-family mapping this bead deletes.
    """
    record = SessionRecord(
        session_id=SessionId("claude-code-session:domain-mutation"),
        native_id="domain-mutation",
        origin=Origin.CLAUDE_CODE_SESSION,
        content_hash=ContentHash("deadbeef"),
        git_branch="main",
    )
    assert session_summary_from_record(record).git_branch == "main"
    assert session_from_records(record, [], []).git_branch == "main"

    mutated = replace(
        SESSIONS_SPEC,
        all_columns=tuple(
            replace(column, domain_name=None) if column.name == "git_branch" else column
            for column in SESSIONS_SPEC.all_columns
        ),
    )
    monkeypatch.setattr(hydrators, "SESSIONS_SPEC", mutated)

    assert session_summary_from_record(record).git_branch is None
    assert session_from_records(record, [], []).git_branch is None


@pytest.mark.asyncio
async def test_session_declaration_owns_the_async_route_to_the_domain_model(tmp_path: Path) -> None:
    """The production async read reaches the domain through the declaration.

    Red when the projection, the record mapping or the domain mapping stops
    coming from the sessions declaration: the hydrated session would lose a
    field the row carries.
    """
    db_path = tmp_path / "index.db"
    backend = SQLiteBackend(db_path=db_path)
    try:
        session_id = await ingest_session(_parsed_session("async-domain-route"), backend)
        async with backend.connection() as conn:
            record = await sessions_reads.get_session(conn, session_id)
    finally:
        await backend.close()

    assert record is not None
    summary = session_summary_from_record(record)
    session = session_from_records(record, [], [])

    assert str(summary.id) == session_id
    assert summary.title == "Row plumbing"
    assert summary.git_branch == "feature/rows"
    assert summary.origin is Origin.CLAUDE_CODE_SESSION
    assert summary.created_at is not None
    assert session.git_branch == "feature/rows"
    assert session.created_at == summary.created_at
    assert session.updated_at == summary.updated_at
