"""Claude Code fragment records keep message identity without repeat charging."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from polylogue.core.enums import BlockType
from polylogue.pipeline.ids import session_content_hash
from polylogue.sources.parsers.claude import parse_code
from polylogue.storage.derived.session.usage_rollup import reconcile_session_usage_rollup
from polylogue.storage.io_phase_metrics import connect_measured
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.write import prepare_session_rows
from polylogue.storage.usage import UsageRequestModelConflictError, session_usage_reconciliation_for_connection
from tests.infra.index_writer import write_fixture_index_session


def _connect(path: Path) -> sqlite3.Connection:
    conn = connect_measured(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    initialize_archive_tier(conn, ArchiveTier.INDEX)
    return conn


def test_claude_fragments_share_one_usage_snapshot_but_keep_all_records(tmp_path: Path) -> None:
    """One request split across assistant records contributes one snapshot.

    The first three records share an API message/request and repeat the same
    usage counters while carrying distinct text/tool-use fragments. A fourth
    request reports the same numeric counters and must still count separately.
    """
    usage = {
        "input_tokens": 7,
        "output_tokens": 11,
        "cache_read_input_tokens": 13,
        "cache_creation_input_tokens": 17,
    }

    def assistant(
        record_uuid: str,
        parent_uuid: str,
        request_id: str,
        message_id: str,
        content: list[dict[str, object]],
    ) -> dict[str, object]:
        return {
            "type": "assistant",
            "uuid": record_uuid,
            "parentUuid": parent_uuid,
            "sessionId": "claude-fragment-usage",
            "requestId": request_id,
            "timestamp": "2026-01-01T00:00:00.000Z",
            "message": {
                "id": message_id,
                "role": "assistant",
                "model": "claude-opus-4",
                "content": content,
                "usage": dict(usage),
            },
        }

    parsed = parse_code(
        [
            {
                "type": "user",
                "uuid": "user-1",
                "sessionId": "claude-fragment-usage",
                "message": {"role": "user", "content": "Run the tools."},
            },
            assistant("fragment-1", "user-1", "request-1", "response-1", [{"type": "text", "text": "Starting."}]),
            assistant(
                "fragment-2",
                "fragment-1",
                "request-1",
                "response-1",
                [{"type": "tool_use", "id": "tool-1", "name": "Read", "input": {"file_path": "/tmp/a"}}],
            ),
            assistant(
                "fragment-3",
                "fragment-2",
                "request-1",
                "response-1",
                [{"type": "tool_use", "id": "tool-2", "name": "Read", "input": {"file_path": "/tmp/b"}}],
            ),
            assistant("fragment-4", "fragment-3", "request-2", "response-2", [{"type": "text", "text": "Finished."}]),
        ],
        "claude-fragment-usage",
    )

    assert len(parsed.messages) == 5
    assert [message.provider_message_id for message in parsed.messages[1:]] == [
        "fragment-1",
        "fragment-2",
        "fragment-3",
        "fragment-4",
    ]
    assert [
        block.tool_id for message in parsed.messages for block in message.blocks if block.type is BlockType.TOOL_USE
    ] == ["tool-1", "tool-2"]

    conn = _connect(tmp_path / "index.db")
    try:
        session_id = write_fixture_index_session(
            conn,
            parsed,
            content_hash=str(session_content_hash(parsed)),
            prepared_rows=prepare_session_rows(parsed),
        )
        reconcile_session_usage_rollup(conn, session_id)

        usage_events = list(
            conn.execute(
                "SELECT request_id, source_message_provider_id FROM session_provider_usage_events "
                "WHERE session_id = ? AND provider_event_type = 'message_usage' ORDER BY position",
                (session_id,),
            )
        )
        assert [(row["request_id"], row["source_message_provider_id"]) for row in usage_events] == [
            ("request-1", "fragment-1"),
            ("request-1", "fragment-2"),
            ("request-1", "fragment-3"),
            ("request-2", "fragment-4"),
        ]
        assert conn.execute("SELECT COUNT(*) FROM messages WHERE session_id = ?", (session_id,)).fetchone()[0] == 5

        reconciliation = session_usage_reconciliation_for_connection(conn, session_id=session_id)
        assert reconciliation.reconciled_tokens_evidence.value_state == "known"
        # Four assistant records carry usage, but three are fragments of one
        # response. The second request is distinct despite equal counters.
        expected_tokens = 2 * sum(usage.values())
        actual_tokens = reconciliation.reconciled_tokens_evidence.value
        assert actual_tokens == expected_tokens, f"expected {expected_tokens} tokens, got {actual_tokens}"
    finally:
        conn.close()


def test_appended_claude_fragment_rederives_request_snapshot(tmp_path: Path) -> None:
    """An append extends a request snapshot; it does not add the snapshot again."""
    session_id = "claude-fragment-append"

    def assistant(record_uuid: str, parent_uuid: str, usage: dict[str, int]) -> dict[str, object]:
        return {
            "type": "assistant",
            "uuid": record_uuid,
            "parentUuid": parent_uuid,
            "sessionId": session_id,
            "requestId": "request-1",
            "timestamp": "2026-01-01T00:00:00.000Z",
            "message": {
                "id": "response-1",
                "role": "assistant",
                "model": "claude-opus-4",
                "content": [{"type": "text", "text": "one response fragment"}],
                "usage": usage,
            },
        }

    user_record: dict[str, object] = {
        "type": "user",
        "uuid": "user-1",
        "sessionId": session_id,
        "message": {"role": "user", "content": "Continue."},
    }
    first_fragment = assistant(
        "fragment-1",
        "user-1",
        {
            "input_tokens": 7,
            "output_tokens": 4,
            "cache_read_input_tokens": 13,
            "cache_creation_input_tokens": 17,
        },
    )
    second_fragment = assistant("fragment-2", "fragment-1", {"input_tokens": 0, "output_tokens": 11})
    first = parse_code([user_record, first_fragment], session_id)
    appended = parse_code([second_fragment], session_id)
    merged = parse_code([user_record, first_fragment, second_fragment], session_id)

    conn = _connect(tmp_path / "index.db")
    try:
        stored_session_id = write_fixture_index_session(
            conn,
            first,
            content_hash=str(session_content_hash(first)),
            prepared_rows=prepare_session_rows(first),
        )
        write_fixture_index_session(
            conn,
            appended,
            merge_append=True,
            content_hash=str(session_content_hash(merged)),
            pending_input_content_hash=str(session_content_hash(appended)),
        )

        assert (
            conn.execute("SELECT COUNT(*) FROM messages WHERE session_id = ?", (stored_session_id,)).fetchone()[0] == 3
        )
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM session_provider_usage_events "
                "WHERE session_id = ? AND provider_event_type = 'message_usage'",
                (stored_session_id,),
            ).fetchone()[0]
            == 2
        )
        row = conn.execute(
            "SELECT input_tokens, output_tokens, cache_read_tokens, cache_write_tokens "
            "FROM session_model_usage WHERE session_id = ? AND model_name = 'claude-opus-4'",
            (stored_session_id,),
        ).fetchone()
        assert tuple(row) == (0, 11, 13, 17)
    finally:
        conn.close()


def test_one_claude_request_cannot_claim_multiple_models(tmp_path: Path) -> None:
    session_id = "claude-fragment-model-conflict"
    records = []
    for record_uuid, parent_uuid, model in (
        ("fragment-1", "user-1", "claude-opus-4"),
        ("fragment-2", "fragment-1", "claude-sonnet-4"),
    ):
        records.append(
            {
                "type": "assistant",
                "uuid": record_uuid,
                "parentUuid": parent_uuid,
                "sessionId": session_id,
                "requestId": "one-request",
                "message": {
                    "id": "response-1",
                    "role": "assistant",
                    "model": model,
                    "content": [{"type": "text", "text": "fragment"}],
                    "usage": {"input_tokens": 7, "output_tokens": 3},
                },
            }
        )
    parsed = parse_code(
        [
            {
                "type": "user",
                "uuid": "user-1",
                "sessionId": session_id,
                "message": {"role": "user", "content": "Continue."},
            },
            *records,
        ],
        session_id,
    )
    conn = _connect(tmp_path / "index.db")
    try:
        try:
            write_fixture_index_session(
                conn,
                parsed,
                content_hash=str(session_content_hash(parsed)),
                prepared_rows=prepare_session_rows(parsed),
            )
        except UsageRequestModelConflictError as error:
            assert error.request_id == "one-request"
        else:
            raise AssertionError("conflicting request models must be refused")
    finally:
        conn.close()
