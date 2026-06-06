"""Contracts pinned by #1252 — first-class attachment native identifiers.

These tests pin the architectural promotion of attachment native identifiers
out of `provider_meta` into typed columns on `attachments` / `attachment_refs`:

- `provider_attachment_id`, `provider_file_id`, `provider_drive_id`, and
  `upload_origin` exist as real columns on both tables.
- Native identifiers flow as typed fields from the parser through
  `ParsedAttachment` → `MaterializedAttachment` → `AttachmentRecord` and into
  the stored rows.
- The hot-path attachment identity lookup (`search_attachment_identity_evidence_hits`)
  resolves against stored columns, not `json_extract` on `provider_meta`.
- The composite `(upload_origin, session_id)` index that the #1199
  attachment library UI relies on exists in the canonical schema.

See `docs/architecture.md` for the Source vocabulary discussion;
`upload_origin` is a closed vocabulary at the storage layer
({"drive","paste","url","oauth"} or NULL).
"""

from __future__ import annotations

import sqlite3
from collections.abc import Generator, Mapping
from pathlib import Path

import pytest

from polylogue.sources.parsers.base_models import ParsedAttachment
from polylogue.storage.sqlite.queries.attachment_records import (
    search_attachment_identity_evidence_hits,
)
from polylogue.storage.sqlite.schema_ddl import SCHEMA_DDL


@pytest.fixture()
def fresh_schema_db() -> Generator[Mapping[str, sqlite3.Connection], None, None]:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA_DDL)
    try:
        yield {"conn": conn}
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Schema-shape pins for #1252
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("table", ["attachments", "attachment_refs"])
def test_native_identifier_columns_are_first_class(
    table: str,
    fresh_schema_db: Mapping[str, sqlite3.Connection],
) -> None:
    """#1252: provider_attachment_id / provider_file_id / provider_drive_id /
    upload_origin are TEXT columns on attachments and attachment_refs."""
    conn = fresh_schema_db["conn"]
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    cols = {row["name"]: row["type"] for row in rows}
    for expected in (
        "provider_attachment_id",
        "provider_file_id",
        "provider_drive_id",
        "upload_origin",
    ):
        assert expected in cols, f"#1252: {table}.{expected} must be a real column"
        assert cols[expected] == "TEXT", f"#1252: {table}.{expected} must be TEXT, got {cols[expected]}"


def test_upload_origin_index_supports_attachment_library_grouping(
    fresh_schema_db: Mapping[str, sqlite3.Connection],
) -> None:
    """#1199 needs an index over (upload_origin, session_id) so the
    attachment-library UI can group by origin without scanning."""
    conn = fresh_schema_db["conn"]
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'index' AND name = ?",
        ("idx_attachment_refs_upload_origin",),
    ).fetchone()
    assert row is not None, "#1252: idx_attachment_refs_upload_origin must be defined"
    assert "upload_origin" in row["sql"]
    assert "session_id" in row["sql"]


# ---------------------------------------------------------------------------
# Typed-field flow through the parsing/materialization surface
# ---------------------------------------------------------------------------


def test_parsed_attachment_carries_typed_native_identifiers() -> None:
    """#1252: ParsedAttachment exposes the typed native-identifier fields
    that downstream storage promotes into stored columns."""
    attachment = ParsedAttachment(
        provider_attachment_id="att-123",
        provider_file_id="file-456",
        provider_drive_id="drive-789",
        upload_origin="drive",
    )
    assert attachment.provider_attachment_id == "att-123"
    assert attachment.provider_file_id == "file-456"
    assert attachment.provider_drive_id == "drive-789"
    assert attachment.upload_origin == "drive"


# Parser parity: every supported provider's parser must emit at least the
# attachment_id and upload_origin as typed fields. This is the test parity
# requirement spelled out in #1252's acceptance criteria.


def test_parser_parity_chatgpt_attachment_has_typed_upload_origin() -> None:
    from polylogue.sources.parsers.chatgpt import extract_messages_from_mapping

    mapping: dict[str, object] = {
        "node-1": {
            "id": "node-1",
            "parent": None,
            "children": [],
            "message": {
                "id": "msg-1",
                "author": {"role": "user"},
                "create_time": 1700000000,
                "content": {"parts": ["hello"]},
                "metadata": {
                    "attachments": [
                        {"id": "file-abc", "name": "doc.pdf", "mime_type": "application/pdf"},
                    ],
                },
            },
        }
    }
    _messages, attachments = extract_messages_from_mapping(mapping)
    assert len(attachments) == 1
    assert attachments[0].provider_attachment_id == "file-abc"
    assert attachments[0].upload_origin == "oauth"


def test_parser_parity_drive_attachment_has_typed_drive_identifiers() -> None:
    from polylogue.sources.parsers.drive_support_attachments import attachment_from_doc

    attachment = attachment_from_doc(
        {
            "id": "doc-id-1",
            "fileId": "drive-file-1",
            "driveId": "drive-root-1",
            "name": "Project Plan",
            "mimeType": "application/vnd.google-apps.document",
        },
        message_id="msg-1",
    )
    assert attachment is not None
    assert attachment.provider_attachment_id == "doc-id-1"
    assert attachment.provider_file_id == "drive-file-1"
    assert attachment.provider_drive_id == "drive-root-1"
    assert attachment.upload_origin == "drive"


def test_parser_parity_claude_code_codex_attachment_has_typed_upload_origin() -> None:
    """Generic dict-based attachments from claude-code/codex flow through
    `attachment_from_meta` and inherit the oauth upload origin."""
    from polylogue.sources.parsers.base_support import attachment_from_meta

    attachment = attachment_from_meta(
        {
            "id": "att-1",
            "file_id": "claude-file-2",
            "name": "transcript.txt",
            "mime_type": "text/plain",
        },
        message_id="msg-1",
        index=0,
    )
    assert attachment is not None
    assert attachment.provider_attachment_id == "att-1"
    assert attachment.provider_file_id == "claude-file-2"
    assert attachment.upload_origin == "oauth"


# ---------------------------------------------------------------------------
# Storage-roundtrip: AttachmentRecord persists and reads back typed columns
# ---------------------------------------------------------------------------


def _insert_attachment_row(
    conn: sqlite3.Connection,
    *,
    attachment_id: str,
    provider_attachment_id: str | None,
    provider_file_id: str | None,
    provider_drive_id: str | None,
    upload_origin: str | None,
) -> None:
    conn.execute(
        """
        INSERT INTO attachments (
            attachment_id, mime_type, size_bytes, path, ref_count, provider_meta,
            provider_attachment_id, provider_file_id, provider_drive_id, upload_origin
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            attachment_id,
            None,
            None,
            None,
            0,
            None,
            provider_attachment_id,
            provider_file_id,
            provider_drive_id,
            upload_origin,
        ),
    )


def test_attachment_record_columns_persist_and_read_back(
    fresh_schema_db: Mapping[str, sqlite3.Connection],
) -> None:
    """Inserts using the canonical column shape must roundtrip every typed
    identifier — provider_attachment_id, provider_file_id, provider_drive_id,
    upload_origin."""
    conn = fresh_schema_db["conn"]
    _insert_attachment_row(
        conn,
        attachment_id="att-1",
        provider_attachment_id="pa-1",
        provider_file_id="pf-1",
        provider_drive_id="pd-1",
        upload_origin="drive",
    )
    row = conn.execute(
        "SELECT provider_attachment_id, provider_file_id, provider_drive_id, upload_origin "
        "FROM attachments WHERE attachment_id = ?",
        ("att-1",),
    ).fetchone()
    assert row["provider_attachment_id"] == "pa-1"
    assert row["provider_file_id"] == "pf-1"
    assert row["provider_drive_id"] == "pd-1"
    assert row["upload_origin"] == "drive"


# ---------------------------------------------------------------------------
# Hot-path: identity lookup uses stored columns, not json_extract
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_attachment_identity_lookup_uses_stored_columns(tmp_path: Path) -> None:
    """#1252: search_attachment_identity_evidence_hits must locate the
    attachment via stored columns even when provider_meta is empty/NULL.

    The previous implementation read identifiers through `json_extract`
    against `provider_meta`; storing the identifiers in their canonical
    columns and leaving `provider_meta = NULL` is now sufficient to find them.
    """
    import aiosqlite

    db_path = tmp_path / "ident.db"
    # Use a synchronous bootstrap because SCHEMA_DDL targets sqlite3.
    bootstrap = sqlite3.connect(db_path)
    try:
        bootstrap.executescript(SCHEMA_DDL)
        bootstrap.execute(
            "INSERT INTO sessions (session_id, source_name, provider_session_id, "
            "content_hash, version) VALUES (?, ?, ?, ?, ?)",
            ("conv-1", "gemini", "gemini-1", "deadbeef", 1),
        )
        bootstrap.execute(
            """INSERT INTO attachments (
                attachment_id, mime_type, size_bytes, path, ref_count, provider_meta,
                provider_attachment_id, provider_file_id, provider_drive_id, upload_origin
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            ("att-1", None, None, None, 1, None, "prov-att-1", "drive-file-1", "drive-root-1", "drive"),
        )
        bootstrap.execute(
            """INSERT INTO attachment_refs (
                ref_id, attachment_id, session_id, message_id, provider_meta,
                provider_attachment_id, provider_file_id, provider_drive_id, upload_origin
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            ("ref-1", "att-1", "conv-1", None, None, "prov-att-1", "drive-file-1", "drive-root-1", "drive"),
        )
        bootstrap.commit()
    finally:
        bootstrap.close()

    async with aiosqlite.connect(db_path) as conn:
        conn.row_factory = aiosqlite.Row
        for query, expected_field in (
            ("prov-att-1", "attachment.provider_attachment_id"),
            ("drive-file-1", "attachment.provider_file_id"),
            ("drive-root-1", "attachment.provider_drive_id"),
        ):
            hits = await search_attachment_identity_evidence_hits(conn, query=query, limit=10)
            assert len(hits) == 1, f"#1252: stored-column lookup for {query!r} must find one hit"
            assert hits[0].session_id == "conv-1"
            assert hits[0].match_surface == "attachment"
            assert expected_field in (hits[0].snippet or ""), (
                f"#1252: snippet must name the typed column {expected_field}, got {hits[0].snippet!r}"
            )


def test_attachment_identity_query_does_not_extract_native_ids_from_json() -> None:
    """The canonical attachment identity query must not read the moved native
    identifiers (provider_id, fileId, driveId) out of provider_meta — that was
    the fragile JSON-extract hot path #1252 retired."""
    import inspect

    from polylogue.storage.sqlite.queries import attachment_records

    source = inspect.getsource(attachment_records)
    forbidden_extracts = (
        "json_extract(attachment_meta, '$.provider_id')",
        "json_extract(attachment_meta, '$.id')",
        "json_extract(attachment_meta, '$.fileId')",
        "json_extract(attachment_meta, '$.driveId')",
        "json_extract(ref_meta, '$.provider_id')",
        "json_extract(ref_meta, '$.id')",
        "json_extract(ref_meta, '$.fileId')",
        "json_extract(ref_meta, '$.driveId')",
    )
    for needle in forbidden_extracts:
        assert needle not in source, (
            f"#1252: identity-lookup hot path must not re-read native ID via {needle!r}; use the stored typed column."
        )


# ---------------------------------------------------------------------------
# No JSON-expression indexes survive on attachments
# ---------------------------------------------------------------------------


def test_no_json_extract_indexes_on_attachment_tables(
    fresh_schema_db: Mapping[str, sqlite3.Connection],
) -> None:
    """#1240 dropped JSON-extract indexes; #1252 must not reintroduce any."""
    conn = fresh_schema_db["conn"]
    rows = conn.execute(
        "SELECT name, sql FROM sqlite_master WHERE type = 'index' AND tbl_name IN ('attachments', 'attachment_refs')"
    ).fetchall()
    for row in rows:
        sql = row["sql"] or ""
        assert "json_extract" not in sql.lower(), (
            f"#1252: index {row['name']!r} reintroduces a JSON-expression index "
            "on an attachment table; native identifiers must use stored columns"
        )
