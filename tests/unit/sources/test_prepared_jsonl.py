"""Sealed worker carriers preserve private parser evidence without a tree pickle."""

from __future__ import annotations

import os
import shutil
import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest

from polylogue.core.enums import Provider, Role
from polylogue.core.message_owner import MessageOwnerCoordinate
from polylogue.pipeline.ids import session_content_hash
from polylogue.sources.parsers.base import ParsedAttachment, ParsedMessage, ParsedSession, ParsedSessionEvent
from polylogue.sources.prepared_jsonl import PreparedJsonl, _write_artifact
from polylogue.sources.prepared_message_sink import (
    _ACTIVE_PARENT_LOOKUP_SQL,
    SqliteMessageSink,
    SqliteMessageStore,
    SqliteSessionEventSink,
)
from polylogue.storage.sqlite.archive_tiers.write import prepare_session_shard


def _prepared_artifact(tmp_path: Path) -> tuple[PreparedJsonl, MessageOwnerCoordinate]:
    path = tmp_path / "prepared.db"
    store = SqliteMessageStore(path)
    messages = store.new_sink()
    events = store.new_event_sink()
    coordinate = MessageOwnerCoordinate(position=0, variant_index=2)
    messages.append(
        ParsedMessage(
            provider_message_id="message-1",
            role=Role.USER,
            text="A neutral prompt",
            variant_index=2,
            parent_message_position=0,
            owner_coordinate=coordinate,
        )
    )
    events.append(
        ParsedSessionEvent(
            event_type="turn_context",
            boundary_message_position=0,
        )
    )
    session = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="session-1",
        messages=[],
        attachments=[
            ParsedAttachment(
                provider_attachment_id="attachment-1",
                message_position=0,
                message_variant_index=2,
                owner_coordinate=coordinate,
                inline_bytes=b"\x00\xff",
                precomputed_blob=("a" * 64, 2),
            )
        ],
    ).model_copy(update={"messages": messages, "session_events": events})
    session.content_hash = session_content_hash(session)
    shard = prepare_session_shard(tmp_path, [session])
    _write_artifact(
        store,
        "b" * 64,
        [session],
        enrichment_digest="c" * 64,
        enrichment_index_path="/index.db",
    )
    store.close()

    artifact = PreparedJsonl.seal(
        "b" * 64,
        path,
        shard.path,
        enrichment_digest="c" * 64,
        enrichment_index_path="/index.db",
    )
    return artifact, coordinate


def test_prepared_active_path_parent_lookup_uses_provider_index(tmp_path: Path) -> None:
    store = SqliteMessageStore(tmp_path / "active-path.db")
    try:
        plan = store.conn.execute("EXPLAIN QUERY PLAN " + _ACTIVE_PARENT_LOOKUP_SQL, (0, "tail")).fetchall()
        assert any("prepared_message_provider" in str(row[3]) for row in plan)

        messages = store.new_sink()
        for index in range(500):
            messages.append(
                ParsedMessage(
                    provider_message_id=f"message-{index}",
                    parent_message_provider_id=f"message-{index - 1}" if index else None,
                    role=Role.USER,
                    text="A neutral prompt",
                    is_active_leaf=index == 499,
                )
            )
        messages.normalize_active_path()
        assert all(message.is_active_path for message in messages)
    finally:
        store.close()


def test_prepared_artifact_preserves_private_linkage_and_refuses_changed_seal(tmp_path: Path) -> None:
    artifact, coordinate = _prepared_artifact(tmp_path)
    artifact.verify_files(full=True)
    restored = list(artifact.iter_sessions())
    assert len(restored) == 1
    assert isinstance(restored[0].messages, SqliteMessageSink)
    assert isinstance(restored[0].session_events, SqliteSessionEventSink)
    assert restored[0].messages[0].owner_coordinate == coordinate
    assert restored[0].messages[0].parent_message_position == 0
    assert restored[0].session_events[0].boundary_message_position == 0
    assert restored[0].attachments[0].owner_coordinate == coordinate
    assert restored[0].attachments[0].inline_bytes == b"\x00\xff"
    assert restored[0].attachments[0].precomputed_blob == ("a" * 64, 2)

    assert artifact.sessions_path is not None
    os.chmod(artifact.sessions_path, 0o600)
    with sqlite3.connect(artifact.sessions_path) as conn:
        conn.execute("UPDATE artifact_seal SET enrichment_digest = ?", ("d" * 64,))
    with pytest.raises(ValueError, match="identity changed"):
        list(artifact.iter_sessions())


def test_prepared_artifact_refuses_same_count_row_change_and_file_replacement(tmp_path: Path) -> None:
    artifact, _coordinate = _prepared_artifact(tmp_path)
    assert artifact.sessions_path is not None
    assert artifact.shard_path is not None
    assert artifact.sessions_seal is not None

    # The row count and SQL seal remain unchanged; the closed-file digest
    # still catches a rewritten message if stat evidence is made to match.
    os.chmod(artifact.sessions_path, 0o600)
    with sqlite3.connect(artifact.sessions_path) as conn:
        conn.execute(
            "UPDATE prepared_message SET message_json = replace(message_json, ?, ?)",
            ("A neutral prompt", "A different text"),
        )
    changed = artifact.sessions_path.stat()
    forged_stat = replace(
        artifact.sessions_seal,
        device=changed.st_dev,
        inode=changed.st_ino,
        size=changed.st_size,
        mtime_ns=changed.st_mtime_ns,
        ctime_ns=changed.st_ctime_ns,
    )
    with pytest.raises(ValueError, match="content changed"):
        replace(artifact, sessions_seal=forged_stat).verify_files(full=True)

    replacement = tmp_path / "replacement.db"
    shutil.copyfile(artifact.shard_path, replacement)
    os.replace(replacement, artifact.shard_path)
    with pytest.raises(ValueError, match="identity changed"):
        artifact.verify_files(full=False)
