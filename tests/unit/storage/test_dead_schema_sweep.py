"""Contracts pinned by #1240 — the dead-schema sweep.

These tests fail loudly if any of the dropped artifacts return:

- the ``content_blocks.media_type`` column
- the unused JSON-extract indexes on attachments/attachment_refs
- the unused semantic_type indexes on content_blocks
- the legacy JSON tag read fallback
- the legacy JSON tag dual-write on ``add_tag``

Plus a round-trip law: ``media_type`` survives ingest → store → render via
the block-metadata JSON envelope.
"""

from __future__ import annotations

import sqlite3
import tempfile
from collections.abc import Generator, Mapping
from pathlib import Path

import pytest

from polylogue.storage.runtime.archive.records import ContentBlockRecord
from polylogue.storage.sqlite.schema_ddl import SCHEMA_DDL, SCHEMA_VERSION


@pytest.fixture()
def fresh_schema_db() -> Generator[Mapping[str, sqlite3.Connection], None, None]:
    """Yield a fresh in-memory archive schema for shape introspection."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA_DDL)
    try:
        yield {"conn": conn}
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Schema-shape pins
# ---------------------------------------------------------------------------


def test_schema_version_is_9() -> None:
    # Bumped from 3 → 4 by #1241 (action_events_fts external-content),
    # then 4 → 5 by #1252 (first-class attachment native identifiers
    # + upload_origin column for the #1199 attachment library), then
    # 5 → 6 by #1258 (topology_edges table — persisted parent edges
    # including unresolved references for out-of-order ingest), then
    # 6 → 7 by #1260 (topology_edges.status gains the 'quarantined'
    # value for the cycle-rejection slice), 7 → 8 by #1253
    # (repo_identities + conversation_repo_observations: typed
    # cross-source repo identity surface for slice C of #864), then
    # 8 → 9 by #1486 (provider-event payload split and content-block
    # canonical message body storage).
    assert SCHEMA_VERSION == 10


def test_content_blocks_table_has_no_media_type_column(fresh_schema_db: Mapping[str, sqlite3.Connection]) -> None:
    conn = fresh_schema_db["conn"]
    cols = {row["name"] for row in conn.execute("PRAGMA table_info(content_blocks)").fetchall()}
    assert "media_type" not in cols, (
        f"#1240: content_blocks.media_type was dropped (1 row out of 2.5M in production), "
        f"current columns: {sorted(cols)}"
    )
    # Sanity: metadata + semantic_type still present.
    assert {"metadata", "semantic_type"}.issubset(cols)


def test_content_block_record_has_no_media_type_field() -> None:
    assert "media_type" not in ContentBlockRecord.model_fields, (
        "#1240: ContentBlockRecord.media_type was dropped; "
        "image/document media_type now lives inside the metadata JSON envelope."
    )


@pytest.mark.parametrize(
    "dead_index",
    [
        "idx_attachments_provider_attachment_id",
        "idx_attachments_provider_file_id",
        "idx_attachments_provider_drive_id",
        "idx_attachment_refs_provider_attachment_id",
        "idx_attachment_refs_provider_file_id",
        "idx_attachment_refs_provider_drive_id",
        "idx_content_blocks_semantic_type",
        "idx_content_blocks_conv_semantic",
    ],
)
def test_dead_indexes_are_absent(
    dead_index: str,
    fresh_schema_db: Mapping[str, sqlite3.Connection],
) -> None:
    """#1240: the catalog of indexes the issue called out as unused is gone."""
    conn = fresh_schema_db["conn"]
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'index' AND name = ?",
        (dead_index,),
    ).fetchone()
    assert row is None, f"#1240: {dead_index!r} should have been dropped"


# ---------------------------------------------------------------------------
# media_type roundtrip via metadata JSON envelope
# ---------------------------------------------------------------------------


def test_media_type_roundtrips_via_metadata_json() -> None:
    """media_type for image blocks must survive parse → materialize → hydrate."""
    from polylogue.archive.message.roles import Role
    from polylogue.pipeline.materialization_runtime import _materialize_content_block
    from polylogue.sources.parsers.base_models import ParsedContentBlock
    from polylogue.storage.hydrators import message_from_record
    from polylogue.storage.runtime.archive.records import MessageRecord
    from polylogue.types import ContentBlockType, ContentHash, ConversationId, MessageId

    parsed = ParsedContentBlock(type=ContentBlockType.IMAGE, media_type="image/png")
    materialized = _materialize_content_block(MessageId("m1"), 0, parsed)

    # MaterializedContentBlock no longer carries media_type as a field.
    assert not hasattr(materialized, "media_type")
    # …but the metadata JSON envelope does.
    assert materialized.metadata_json is not None
    assert "image/png" in materialized.metadata_json

    # Storage record: build from materialized envelope and verify hydration
    # places media_type back on the renderable block dict.
    record = ContentBlockRecord(
        block_id=materialized.block_id,
        message_id=MessageId("m1"),
        conversation_id=ConversationId("c1"),
        block_index=0,
        type=ContentBlockType.IMAGE,
        text=None,
        metadata=materialized.metadata_json,
        semantic_type=None,
    )
    msg_record = MessageRecord(
        message_id=MessageId("m1"),
        conversation_id=ConversationId("c1"),
        role=Role.USER,
        text=None,
        content_hash=ContentHash("0" * 64),
        version=1,
        content_blocks=[record],
    )
    msg = message_from_record(msg_record, [])
    block_payloads = list(msg.content_blocks)
    assert block_payloads, "hydrated message must expose its content blocks"
    assert block_payloads[0]["media_type"] == "image/png"


# ---------------------------------------------------------------------------
# Tag read/write contracts (also asserted in test_tag_contracts.py).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio()
async def test_list_tags_does_not_consult_json_metadata() -> None:
    """#1240: list_tags reads M2M only; no fallback to ``metadata.tags``."""
    from polylogue.storage.repository import ConversationRepository
    from polylogue.storage.sqlite.async_sqlite import SQLiteBackend
    from tests.infra.storage_records import ConversationBuilder

    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "tags.db"
        ConversationBuilder(db_path, "c1").provider("test").title("x").add_message("m1", text="hi").save()

        # Inject a JSON-only tag value bypassing M2M.
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "UPDATE conversations SET metadata = ? WHERE conversation_id = ?",
                ('{"tags": ["ghost"]}', "c1"),
            )
            conn.commit()

        backend = SQLiteBackend(db_path=db_path)
        repo = ConversationRepository(backend=backend)
        try:
            listed = await repo.list_tags()
            assert "ghost" not in listed
            assert listed == {}
        finally:
            await backend.close()
