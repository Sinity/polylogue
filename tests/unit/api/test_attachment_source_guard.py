"""Metadata-only attachment targets must match retained Source semantics."""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from polylogue.api.user_state_resolver import bind_attachment_source_guard, resolve_insight_target
from polylogue.core.enums import Provider
from polylogue.operations.operation_context import open_operation_read
from polylogue.sources.parsers.claude.ai_parser import parse_ai
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from tests.infra.archive_templates import bootstrap_archive_root
from tests.infra.index_writer import fixture_index_connection, write_fixture_retained_session
from tests.infra.source_parser_cases import claude_conversation_attachment


def _archive_with_metadata_attachment(root: Path) -> tuple[str, str]:
    payload = claude_conversation_attachment(byte_relation="absent")
    payload["uuid"] = "metadata-attachment-source-guard"
    session = parse_ai(payload, "fallback")
    raw_bytes = json.dumps(payload, sort_keys=True).encode()
    bootstrap_archive_root(root)
    with ArchiveStore.open_existing(root, read_only=False) as archive:
        raw_id = archive.write_raw_payload(
            provider=Provider.CLAUDE_AI,
            payload=raw_bytes,
            source_path="metadata-attachment-source-guard.json",
            canonical_source_path="metadata-attachment-source-guard.json",
            acquired_at_ms=1,
        )
    with fixture_index_connection(root / "index.db") as index:
        write_fixture_retained_session(index, session, raw_id=raw_id)
    with closing(sqlite3.connect(root / "index.db")) as index:
        row = index.execute(
            "SELECT r.session_id,r.ref_id FROM attachment_refs r "
            "WHERE r.session_id='claude-ai-export:metadata-attachment-source-guard' "
            "ORDER BY r.ref_id LIMIT 1"
        ).fetchone()
    assert row is not None
    return str(row[0]), str(row[1])


def test_metadata_only_attachment_is_source_bound_and_guarded(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    session_id, ref_id = _archive_with_metadata_attachment(root)

    with open_operation_read(root) as snapshot:
        resolved = asyncio.run(
            resolve_insight_target(
                root,
                target_type="attachment",
                target_id=ref_id,
                session_id=session_id,
                index_connection=snapshot.archive._conn,
                source_connection=snapshot.archive.source_connection,
                snapshot=snapshot,
            )
        )
        assert resolved["target_id"] == ref_id
        current_index = sqlite3.connect(root / "index.db")
        current_source = sqlite3.connect(root / "source.db")
        try:
            guard = bind_attachment_source_guard(
                snapshot,
                session_id=session_id,
                ref_id=ref_id,
                current_index_connection=current_index,
                current_source_connection=current_source,
            )
            guard()
            supplying_raw_id = current_index.execute(
                "SELECT supplying_raw_id FROM attachment_refs WHERE session_id=? AND ref_id=?",
                (session_id, ref_id),
            ).fetchone()[0]
            current_source.execute(
                "UPDATE raw_sessions SET blob_hash=? WHERE raw_id=?",
                (bytes(32), supplying_raw_id),
            )
            current_source.commit()
            with pytest.raises(ValueError, match="changed before durable apply"):
                guard()
            assert resolved["target_id"] == ref_id
        finally:
            current_index.close()
            current_source.close()


def test_metadata_only_attachment_disagreement_with_index_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    session_id, ref_id = _archive_with_metadata_attachment(root)
    with sqlite3.connect(root / "index.db") as index:
        index.execute(
            "UPDATE attachments SET display_name='different descriptor' WHERE attachment_id=("
            "SELECT attachment_id FROM attachment_refs WHERE ref_id=?)",
            (ref_id,),
        )
        index.commit()

    with open_operation_read(root) as snapshot:
        with pytest.raises(ValueError, match="matching retained Source descriptor"):
            asyncio.run(
                resolve_insight_target(
                    root,
                    target_type="attachment",
                    target_id=ref_id,
                    session_id=session_id,
                    index_connection=snapshot.archive._conn,
                    source_connection=snapshot.archive.source_connection,
                    snapshot=snapshot,
                )
            )


def test_attachment_guard_refuses_physical_source_replacement(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    session_id, ref_id = _archive_with_metadata_attachment(root)
    with open_operation_read(root) as snapshot:
        current_index = sqlite3.connect(root / "index.db")
        current_source = sqlite3.connect(root / "source.db")
        try:
            guard = bind_attachment_source_guard(
                snapshot,
                session_id=session_id,
                ref_id=ref_id,
                current_index_connection=current_index,
                current_source_connection=current_source,
            )
            replacement = root / "source-replacement.db"
            replacement.write_bytes((root / "source.db").read_bytes())
            os.replace(replacement, root / "source.db")
            with pytest.raises(ValueError, match="archive .* changed before durable apply"):
                guard()
        finally:
            current_index.close()
            current_source.close()
