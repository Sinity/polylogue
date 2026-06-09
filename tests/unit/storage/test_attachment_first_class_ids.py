"""Contracts pinned by #1252 — first-class attachment native identifiers.

These tests pin the architectural promotion of attachment native identifiers
out of metadata buckets into typed attachment/reference tables:

- `attachment_native_ids` stores `(ref_id, id_kind, native_id)` values.
- `upload_origin`, `source_url`, and `caption` are real columns on
  `attachment_refs`.
- Native identifiers flow as typed fields from the parser through
  `ParsedAttachment` → `MaterializedAttachment` → `AttachmentRecord` and into
  the stored rows.
- The hot-path attachment identity lookup (`search_attachment_identity_evidence_hits`)
  resolves against stored rows, not `json_extract` on metadata.
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
from polylogue.storage.sqlite.schema import SCHEMA_DDL


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


def test_native_identifier_rows_are_first_class(
    fresh_schema_db: Mapping[str, sqlite3.Connection],
) -> None:
    """#1252: attachment native IDs are stored in a typed child table."""
    conn = fresh_schema_db["conn"]
    rows = conn.execute("PRAGMA table_info(attachment_native_ids)").fetchall()
    cols = {row["name"]: row["type"] for row in rows}
    assert cols == {"ref_id": "TEXT", "id_kind": "TEXT", "native_id": "TEXT"}
    pk_cols = [row["name"] for row in rows if row["pk"]]
    assert pk_cols == ["ref_id", "id_kind", "native_id"]


def test_attachment_ref_context_columns_are_first_class(
    fresh_schema_db: Mapping[str, sqlite3.Connection],
) -> None:
    """Attachment refs carry reference context as typed columns."""
    conn = fresh_schema_db["conn"]
    rows = conn.execute("PRAGMA table_info(attachment_refs)").fetchall()
    cols = {row["name"]: row["type"] for row in rows}
    for expected in ("upload_origin", "source_url", "caption"):
        assert expected in cols, f"#1252: attachment_refs.{expected} must be a real column"
        assert cols[expected] == "TEXT", f"#1252: attachment_refs.{expected} must be TEXT, got {cols[expected]}"


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
            attachment_id, display_name, media_type, byte_count, blob_hash, ref_count
        ) VALUES (?, ?, ?, ?, zeroblob(32), ?)
        """,
        (
            attachment_id,
            None,
            None,
            0,
            1,
        ),
    )
    conn.execute(
        """
        INSERT INTO sessions (native_id, origin, content_hash)
        VALUES ('session-1', 'gemini-cli-session', zeroblob(32))
        """,
    )
    conn.execute(
        """
        INSERT INTO messages (session_id, native_id, position, role, message_type, content_hash)
        VALUES ('gemini-cli-session:session-1', 'message-1', 0, 'user', 'message', zeroblob(32))
        """,
    )
    conn.execute(
        """
        INSERT INTO attachment_refs (
            attachment_id, session_id, message_id, position, upload_origin
        ) VALUES (?, 'gemini-cli-session:session-1', 'gemini-cli-session:session-1:message-1', 0, ?)
        """,
        (attachment_id, upload_origin),
    )
    ref_id = "gemini-cli-session:session-1:message-1:attachment:0"
    native_rows = [
        ("attachment", provider_attachment_id),
        ("file", provider_file_id),
        ("drive", provider_drive_id),
    ]
    conn.executemany(
        """
        INSERT INTO attachment_native_ids (ref_id, id_kind, native_id)
        VALUES (?, ?, ?)
        """,
        [(ref_id, kind, value) for kind, value in native_rows if value is not None],
    )


def test_attachment_record_columns_persist_and_read_back(
    fresh_schema_db: Mapping[str, sqlite3.Connection],
) -> None:
    """The canonical attachment shape roundtrips native IDs and ref context."""
    conn = fresh_schema_db["conn"]
    _insert_attachment_row(
        conn,
        attachment_id="att-1",
        provider_attachment_id="pa-1",
        provider_file_id="pf-1",
        provider_drive_id="pd-1",
        upload_origin="drive",
    )
    rows = conn.execute(
        """
        SELECT ani.id_kind, ani.native_id, ar.upload_origin
        FROM attachment_refs ar
        JOIN attachment_native_ids ani ON ani.ref_id = ar.ref_id
        WHERE ar.attachment_id = ?
        ORDER BY ani.id_kind
        """,
        ("att-1",),
    ).fetchall()
    assert [(row["id_kind"], row["native_id"]) for row in rows] == [
        ("attachment", "pa-1"),
        ("drive", "pd-1"),
        ("file", "pf-1"),
    ]
    assert {row["upload_origin"] for row in rows} == {"drive"}


# ---------------------------------------------------------------------------
# Hot-path: identity lookup uses stored columns, not json_extract
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_attachment_identity_lookup_uses_stored_columns(tmp_path: Path) -> None:
    """#1252: search_attachment_identity_evidence_hits must locate the
    attachment via stored rows without any metadata bucket.
    """
    import aiosqlite

    db_path = tmp_path / "ident.db"
    # Use a synchronous bootstrap because SCHEMA_DDL targets sqlite3.
    bootstrap = sqlite3.connect(db_path)
    try:
        bootstrap.executescript(SCHEMA_DDL)
        bootstrap.execute(
            "INSERT INTO sessions (native_id, origin, content_hash) VALUES (?, ?, zeroblob(32))",
            ("gemini-1", "gemini-cli-session"),
        )
        bootstrap.execute(
            """INSERT INTO attachments (
                attachment_id, display_name, media_type, byte_count, blob_hash, ref_count
            ) VALUES (?, ?, ?, ?, zeroblob(32), ?)""",
            ("att-1", "drive doc", None, 0, 1),
        )
        bootstrap.execute(
            """
            INSERT INTO messages (session_id, native_id, position, role, message_type, content_hash)
            VALUES ('gemini-cli-session:gemini-1', 'msg-1', 0, 'user', 'message', zeroblob(32))
            """
        )
        bootstrap.execute(
            """INSERT INTO attachment_refs (
                attachment_id, session_id, message_id, position, upload_origin
            ) VALUES (?, ?, ?, ?, ?)""",
            ("att-1", "gemini-cli-session:gemini-1", "gemini-cli-session:gemini-1:msg-1", 0, "drive"),
        )
        bootstrap.executemany(
            """INSERT INTO attachment_native_ids (ref_id, id_kind, native_id)
            VALUES ('gemini-cli-session:gemini-1:msg-1:attachment:0', ?, ?)""",
            [("attachment", "prov-att-1"), ("file", "drive-file-1"), ("drive", "drive-root-1")],
        )
        bootstrap.commit()
    finally:
        bootstrap.close()

    async with aiosqlite.connect(db_path) as conn:
        conn.row_factory = aiosqlite.Row
        for query, expected_field in (
            ("prov-att-1", "native.attachment"),
            ("drive-file-1", "native.file"),
            ("drive-root-1", "native.drive"),
        ):
            hits = await search_attachment_identity_evidence_hits(conn, query=query, limit=10)
            assert len(hits) == 1, f"#1252: stored-column lookup for {query!r} must find one hit"
            assert hits[0].session_id == "gemini-cli-session:gemini-1"
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
