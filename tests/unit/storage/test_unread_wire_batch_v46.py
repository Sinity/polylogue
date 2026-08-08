"""Round-trip coverage for the polylogue-2qx.4 unread-wire batch (index v46).

Each relation/column added for the batch is exercised through the REAL
production writer (``write_parsed_session_to_archive``) and read back through
the REAL async repository/reader surface -- never a hand-rolled INSERT.
"""

from __future__ import annotations

from pathlib import Path

from polylogue.core.enums import BlockType, Provider, Role, WebConstructType
from polylogue.sources.parsers.base import (
    ParsedContentBlock,
    ParsedFileEdit,
    ParsedMessage,
    ParsedSession,
    ParsedSessionRef,
    ParsedWebConstruct,
)
from polylogue.storage.repository import SessionRepository
from polylogue.storage.sqlite.async_sqlite import SQLiteBackend
from tests.infra.identity import archive_block_id, archive_message_id
from tests.infra.live_ingest import ingest_session


async def test_stop_reason_round_trips_through_writer_and_repository(tmp_path: Path) -> None:
    """messages.stop_reason: written by the real writer, read via repository.get_messages.

    Fails if the writer stops persisting ``ParsedMessage.stop_reason`` (e.g.
    a revert of the ``messages`` INSERT column list / row-tuple builder), or
    if the read path stops selecting/mapping the column.
    """
    backend = SQLiteBackend(db_path=tmp_path / "stop-reason.db")
    repo = SessionRepository(backend=backend)
    try:
        session_id = await ingest_session(
            ParsedSession(
                source_name=Provider.CLAUDE_CODE,
                provider_session_id="stop-reason-1",
                title="Stop reason session",
                messages=[
                    ParsedMessage(
                        provider_message_id="m1",
                        role=Role.ASSISTANT,
                        text="done",
                        position=0,
                        blocks=[ParsedContentBlock(type=BlockType.TEXT, text="done")],
                        stop_reason="end_turn",
                    ),
                ],
            ),
            backend=backend,
        )
        messages = await repo.get_messages(session_id)
    finally:
        await repo.close()

    assert len(messages) == 1
    assert messages[0].stop_reason == "end_turn"


async def test_tool_result_outcome_unknown_reason_round_trips(tmp_path: Path) -> None:
    """blocks.tool_result_outcome_unknown_reason: distinguishes unknown-outcome causes.

    Fails if the writer stops persisting ``ParsedContentBlock.
    outcome_unknown_reason`` (e.g. a revert of the ``blocks`` row-tuple
    builder), or if ``get_blocks`` stops selecting the column.
    """
    backend = SQLiteBackend(db_path=tmp_path / "outcome-unknown.db")
    repo = SessionRepository(backend=backend)
    try:
        session_id = await ingest_session(
            ParsedSession(
                source_name=Provider.CLAUDE_CODE,
                provider_session_id="outcome-unknown-1",
                title="Outcome unknown session",
                messages=[
                    ParsedMessage(
                        provider_message_id="m1",
                        role=Role.ASSISTANT,
                        position=0,
                        blocks=[
                            ParsedContentBlock(
                                type=BlockType.TOOL_USE,
                                tool_name="Bash",
                                tool_id="tool-1",
                                tool_input={"command": "echo hi"},
                            ),
                        ],
                    ),
                    ParsedMessage(
                        provider_message_id="m2",
                        role=Role.USER,
                        position=1,
                        blocks=[
                            ParsedContentBlock(
                                type=BlockType.TOOL_RESULT,
                                tool_id="tool-1",
                                text="hi",
                                is_error=None,
                                outcome_unknown_reason="not_reported",
                            ),
                        ],
                    ),
                ],
            ),
            backend=backend,
        )
        messages = await repo.get_messages(session_id)
        message_ids = [str(message.message_id) for message in messages]
        blocks_by_message = await repo.queries.get_blocks(message_ids)
    finally:
        await repo.close()

    tool_result_blocks = [
        block for blocks in blocks_by_message.values() for block in blocks if block.type == BlockType.TOOL_RESULT
    ]
    assert len(tool_result_blocks) == 1
    assert tool_result_blocks[0].tool_result_is_error is None
    assert tool_result_blocks[0].tool_result_outcome_unknown_reason == "not_reported"


async def test_session_display_name_and_run_settings_round_trip(tmp_path: Path) -> None:
    """sessions.display_name / run_settings_json: written by the real writer.

    Fails if the writer stops persisting ``ParsedSession.display_name``/
    ``run_settings`` (a revert of the ``sessions`` INSERT column list), or if
    ``SessionRepository.get`` stops selecting/mapping either column.
    """
    backend = SQLiteBackend(db_path=tmp_path / "display-name.db")
    repo = SessionRepository(backend=backend)
    try:
        session_id = await ingest_session(
            ParsedSession(
                source_name=Provider.GEMINI,
                provider_session_id="display-name-1",
                title="Untitled",
                display_name="greedy-squishing-hamming",
                run_settings={"temperature": 0.7, "topP": 0.9},
                messages=[
                    ParsedMessage(
                        provider_message_id="m1",
                        role=Role.USER,
                        text="hi",
                        position=0,
                        blocks=[ParsedContentBlock(type=BlockType.TEXT, text="hi")],
                    ),
                ],
            ),
            backend=backend,
        )
        sessions = await repo.get_sessions_batch([session_id])
    finally:
        await repo.close()

    assert len(sessions) == 1
    assert sessions[0].display_name == "greedy-squishing-hamming"
    assert sessions[0].run_settings == {"temperature": 0.7, "topP": 0.9}


async def test_session_pending_drafts_round_trips_through_writer_and_repository(tmp_path: Path) -> None:
    """sessions.pending_drafts_json (v47, polylogue-o4j2): written by the real writer.

    Fails if the writer stops persisting ``ParsedSession.pending_drafts`` (a
    revert of the ``sessions`` INSERT column list), or if
    ``SessionRepository.get_sessions_batch`` stops selecting/mapping the
    column. Deliberately NOT a session_event round trip -- see
    ``ParsedSession.pending_drafts``'s docstring for why a draft must stay
    outside session_revision_projection's comparison axes.
    """
    backend = SQLiteBackend(db_path=tmp_path / "pending-drafts.db")
    repo = SessionRepository(backend=backend)
    try:
        session_id = await ingest_session(
            ParsedSession(
                source_name=Provider.GEMINI,
                provider_session_id="pending-drafts-1",
                title="Untitled",
                pending_drafts=[{"text": "unsent follow-up", "role": "user", "token_count": 3}],
                messages=[
                    ParsedMessage(
                        provider_message_id="m1",
                        role=Role.USER,
                        text="hi",
                        position=0,
                        blocks=[ParsedContentBlock(type=BlockType.TEXT, text="hi")],
                    ),
                ],
            ),
            backend=backend,
        )
        sessions = await repo.get_sessions_batch([session_id])
    finally:
        await repo.close()

    assert len(sessions) == 1
    assert sessions[0].pending_drafts == [{"text": "unsent follow-up", "role": "user", "token_count": 3}]


async def test_session_pending_drafts_empty_for_session_without_drafts(tmp_path: Path) -> None:
    """No pendingInputs on the wire must round-trip as None, not an empty list."""
    backend = SQLiteBackend(db_path=tmp_path / "pending-drafts-empty.db")
    repo = SessionRepository(backend=backend)
    try:
        session_id = await ingest_session(
            ParsedSession(
                source_name=Provider.GEMINI,
                provider_session_id="pending-drafts-empty-1",
                title="Untitled",
                messages=[
                    ParsedMessage(
                        provider_message_id="m1",
                        role=Role.USER,
                        text="hi",
                        position=0,
                        blocks=[ParsedContentBlock(type=BlockType.TEXT, text="hi")],
                    ),
                ],
            ),
            backend=backend,
        )
        sessions = await repo.get_sessions_batch([session_id])
    finally:
        await repo.close()

    assert len(sessions) == 1
    assert sessions[0].pending_drafts is None


async def test_file_edits_round_trip_keyed_by_tool_use_block(tmp_path: Path) -> None:
    """file_edits: a new relation keyed by the tool_use block id.

    Fails if the writer stops resolving/inserting ``file_edits`` rows (a
    revert of ``_write_file_edits``/``_build_file_edit_rows``), or if
    ``queries.file_edits.get_file_edits_for_session`` stops reading them back.
    """
    backend = SQLiteBackend(db_path=tmp_path / "file-edits.db")
    repo = SessionRepository(backend=backend)
    try:
        session_id = await ingest_session(
            ParsedSession(
                source_name=Provider.CLAUDE_CODE,
                provider_session_id="file-edit-1",
                title="File edit session",
                messages=[
                    ParsedMessage(
                        provider_message_id="m1",
                        role=Role.ASSISTANT,
                        position=0,
                        blocks=[
                            ParsedContentBlock(
                                type=BlockType.TOOL_USE,
                                tool_name="Edit",
                                tool_id="edit-tool-1",
                                tool_input={"file_path": "/tmp/foo.py"},
                            ),
                        ],
                    ),
                    ParsedMessage(
                        provider_message_id="m2",
                        role=Role.USER,
                        position=1,
                        blocks=[
                            ParsedContentBlock(
                                type=BlockType.TOOL_RESULT,
                                tool_id="edit-tool-1",
                                text="applied",
                                file_edit=ParsedFileEdit(
                                    file_path="/tmp/foo.py",
                                    structured_patch=[
                                        {"oldStart": 1, "oldLines": 1, "newStart": 1, "newLines": 2, "lines": ["+x"]}
                                    ],
                                    original_file="old contents\n",
                                    old_string="old",
                                    new_string="new",
                                    replace_all=False,
                                    user_modified=True,
                                ),
                            ),
                        ],
                    ),
                ],
            ),
            backend=backend,
        )
        file_edits = await repo.get_file_edits(session_id)
    finally:
        await repo.close()

    assert len(file_edits) == 1
    edit = file_edits[0]
    assert edit.session_id == session_id
    assert edit.file_path == "/tmp/foo.py"
    assert edit.original_file == "old contents\n"
    assert edit.old_string == "old"
    assert edit.new_string == "new"
    assert edit.replace_all is False
    assert edit.user_modified is True
    assert edit.structured_patch == [{"oldStart": 1, "oldLines": 1, "newStart": 1, "newLines": 2, "lines": ["+x"]}]
    # Keyed by the TOOL_USE block (message m1, position 0) even though the
    # evidence was attached to the TOOL_RESULT block reported in message m2.
    assert edit.tool_use_block_id == archive_block_id(archive_message_id(session_id, "m1", position=0), position=0)
    assert edit.message_id == archive_message_id(session_id, "m2", position=1)


async def test_file_edits_empty_for_session_without_edits(tmp_path: Path) -> None:
    backend = SQLiteBackend(db_path=tmp_path / "file-edits-empty.db")
    repo = SessionRepository(backend=backend)
    try:
        session_id = await ingest_session(
            ParsedSession(
                source_name=Provider.CLAUDE_CODE,
                provider_session_id="no-file-edit",
                messages=[
                    ParsedMessage(
                        provider_message_id="m1",
                        role=Role.USER,
                        text="hi",
                        position=0,
                        blocks=[ParsedContentBlock(type=BlockType.TEXT, text="hi")],
                    ),
                ],
            ),
            backend=backend,
        )
        file_edits = await repo.get_file_edits(session_id)
    finally:
        await repo.close()

    assert file_edits == []


async def test_session_refs_round_trip(tmp_path: Path) -> None:
    """session_refs: a new tracker-agnostic relation.

    Fails if the writer stops persisting ``ParsedSession.session_refs`` (a
    revert of ``_write_session_refs``), or if
    ``queries.session_refs.get_session_refs`` stops reading them back.
    """
    backend = SQLiteBackend(db_path=tmp_path / "session-refs.db")
    repo = SessionRepository(backend=backend)
    try:
        session_id = await ingest_session(
            ParsedSession(
                source_name=Provider.CLAUDE_CODE,
                provider_session_id="pr-link-1",
                title="PR link session",
                session_refs=[
                    ParsedSessionRef(
                        kind="pull_request", url="https://github.com/acme/repo/pull/42", repo="acme/repo", number=42
                    ),
                ],
                messages=[
                    ParsedMessage(
                        provider_message_id="m1",
                        role=Role.USER,
                        text="hi",
                        position=0,
                        blocks=[ParsedContentBlock(type=BlockType.TEXT, text="hi")],
                    ),
                ],
            ),
            backend=backend,
        )
        refs = await repo.get_session_refs(session_id)
    finally:
        await repo.close()

    assert len(refs) == 1
    ref = refs[0]
    assert ref.session_id == session_id
    assert ref.kind == "pull_request"
    assert ref.url == "https://github.com/acme/repo/pull/42"
    assert ref.repo == "acme/repo"
    assert ref.number == 42


async def test_session_refs_empty_for_session_without_refs(tmp_path: Path) -> None:
    backend = SQLiteBackend(db_path=tmp_path / "session-refs-empty.db")
    repo = SessionRepository(backend=backend)
    try:
        session_id = await ingest_session(
            ParsedSession(
                source_name=Provider.CLAUDE_CODE,
                provider_session_id="no-refs",
                messages=[
                    ParsedMessage(
                        provider_message_id="m1",
                        role=Role.USER,
                        text="hi",
                        position=0,
                        blocks=[ParsedContentBlock(type=BlockType.TEXT, text="hi")],
                    ),
                ],
            ),
            backend=backend,
        )
        refs = await repo.get_session_refs(session_id)
    finally:
        await repo.close()

    assert refs == []


async def test_web_content_constructs_round_trip_search_result(tmp_path: Path) -> None:
    """web_content_constructs (polylogue-kktg): a 155k+-row relation with no prior reader.

    Fails if the writer stops resolving/inserting ``web_content_constructs``
    rows (a revert of ``_write_web_constructs``), or if
    ``repository.get_web_content_constructs`` stops reading them back.
    """
    backend = SQLiteBackend(db_path=tmp_path / "web-content.db")
    repo = SessionRepository(backend=backend)
    try:
        session_id = await ingest_session(
            ParsedSession(
                source_name=Provider.CHATGPT,
                provider_session_id="web-content-1",
                title="Web content session",
                messages=[
                    ParsedMessage(
                        provider_message_id="m1",
                        role=Role.ASSISTANT,
                        position=0,
                        blocks=[
                            ParsedContentBlock(
                                type=BlockType.TEXT,
                                text="Searching the web",
                                web_constructs=[
                                    ParsedWebConstruct(
                                        construct_type=WebConstructType.SEARCH_QUERY,
                                        query="polylogue archive",
                                    ),
                                    ParsedWebConstruct(
                                        construct_type=WebConstructType.SEARCH_RESULT,
                                        title="Polylogue",
                                        url="https://example.com/polylogue",
                                        text="A local AI chat archiver",
                                        rank=1,
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
            backend=backend,
        )
        constructs = await repo.get_web_content_constructs(session_id)
        search_results = await repo.get_web_content_constructs(session_id, construct_type="search_result")
    finally:
        await repo.close()

    assert len(constructs) == 2
    by_type = {c.construct_type: c for c in constructs}
    query_construct = by_type["search_query"]
    assert query_construct.session_id == session_id
    assert query_construct.query == "polylogue archive"
    assert query_construct.message_id == archive_message_id(session_id, "m1", position=0)
    assert query_construct.block_id == archive_block_id(query_construct.message_id, position=0)

    result_construct = by_type["search_result"]
    assert result_construct.title == "Polylogue"
    assert result_construct.url == "https://example.com/polylogue"
    assert result_construct.text == "A local AI chat archiver"
    assert result_construct.rank == 1

    assert len(search_results) == 1
    assert search_results[0].construct_type == "search_result"


async def test_web_content_constructs_empty_for_session_without_constructs(tmp_path: Path) -> None:
    backend = SQLiteBackend(db_path=tmp_path / "web-content-empty.db")
    repo = SessionRepository(backend=backend)
    try:
        session_id = await ingest_session(
            ParsedSession(
                source_name=Provider.CHATGPT,
                provider_session_id="no-web-content",
                messages=[
                    ParsedMessage(
                        provider_message_id="m1",
                        role=Role.USER,
                        text="hi",
                        position=0,
                        blocks=[ParsedContentBlock(type=BlockType.TEXT, text="hi")],
                    ),
                ],
            ),
            backend=backend,
        )
        constructs = await repo.get_web_content_constructs(session_id)
    finally:
        await repo.close()

    assert constructs == []


async def test_session_links_parent_tool_use_block_id_resolves_via_tool_id(tmp_path: Path) -> None:
    """session_links.parent_tool_use_block_id: the real delegation join key.

    A child session declares ``parent_tool_use_provider_id`` (the wire's
    ``parentToolUseID``, a provider tool_id) rather than an ordinal position.
    Fails if the writer stops resolving it to the parent's actual
    ``blocks.block_id`` (a revert of ``_resolve_parent_tool_use_block_id``/
    ``_write_session_link``), or if
    ``queries.session_links.list_session_links_for_session`` stops reading
    the column back.
    """
    from polylogue.storage.sqlite.queries.session_links import list_session_links_for_session

    backend = SQLiteBackend(db_path=tmp_path / "parent-tool-use.db")
    repo = SessionRepository(backend=backend)
    try:
        parent_id = await ingest_session(
            ParsedSession(
                source_name=Provider.CLAUDE_CODE,
                provider_session_id="parent-1",
                title="Parent session",
                messages=[
                    ParsedMessage(
                        provider_message_id="m1",
                        role=Role.ASSISTANT,
                        position=0,
                        blocks=[
                            ParsedContentBlock(
                                type=BlockType.TOOL_USE,
                                tool_name="Task",
                                tool_id="dispatch-tool-1",
                                tool_input={"description": "spawn subagent"},
                            ),
                        ],
                    ),
                ],
            ),
            backend=backend,
        )
        await ingest_session(
            ParsedSession(
                source_name=Provider.CLAUDE_CODE,
                provider_session_id="child-1",
                title="Child session",
                parent_session_provider_id="parent-1",
                parent_tool_use_provider_id="dispatch-tool-1",
                messages=[
                    ParsedMessage(
                        provider_message_id="c1",
                        role=Role.USER,
                        text="go",
                        position=0,
                        blocks=[ParsedContentBlock(type=BlockType.TEXT, text="go")],
                    ),
                ],
            ),
            backend=backend,
        )

        async with backend.connection() as conn:
            links = await list_session_links_for_session(conn, "claude-code-session:child-1")
    finally:
        await repo.close()

    assert len(links) == 1
    link = links[0]
    assert link["parent_tool_use_block_id"] == archive_block_id(
        archive_message_id(parent_id, "m1", position=0), position=0
    )
    assert link["method"] == "parent-tool-use-id"
