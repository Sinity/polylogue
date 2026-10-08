"""Prepared writer decisions keep unchanged acceptance distinct from refusal."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from polylogue.archive.ingest_flags import DOM_FALLBACK_INGEST_FLAG
from polylogue.archive.message.roles import Role
from polylogue.core.enums import Provider
from polylogue.sources.parsers.base import ParsedMessage, ParsedSession
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.revision_governance import (
    ArchiveRawParsedWriteResult,
    _write_parsed_precedence_result,
)
from polylogue.storage.sqlite.archive_tiers.user_write import upsert_suppression
from tests.infra.archive_templates import bootstrap_archive_root, run_archive_fixture_write
from tests.infra.index_writer import prepared_fixture_index_batch


@pytest.mark.asyncio
@pytest.mark.parametrize("decision", ["unchanged", "stale", "browser", "suppressed"])
async def test_original_prepared_writer_distinguishes_unchanged_content_from_refusal(
    tmp_path: Path, decision: str
) -> None:
    root = tmp_path / "archive"
    prior = ParsedSession(
        source_name=Provider.CHATGPT,
        provider_session_id="neutral-decisions",
        updated_at="2026-01-02T00:00:00Z",
        messages=[ParsedMessage(provider_message_id="m", role=Role.USER, text="retained neutral content")],
    )
    incoming = (
        prior
        if decision in ("unchanged", "suppressed")
        else prior.model_copy(
            update={
                "updated_at": "2026-01-01T00:00:00Z" if decision == "stale" else "2026-01-03T00:00:00Z",
                "messages": [
                    ParsedMessage(provider_message_id="dom", role=Role.USER, text="different neutral content")
                ],
                "ingest_flags": [DOM_FALLBACK_INGEST_FLAG] if decision == "browser" else [],
            }
        )
    )

    def publish() -> tuple[
        ArchiveRawParsedWriteResult, ArchiveRawParsedWriteResult, tuple[bytes, int], tuple[bytes, int]
    ]:
        bootstrap_archive_root(root)
        with ArchiveStore.open_existing(root, read_only=False) as archive:
            # These are admitted neutral ParsedSession inputs, not a claim that
            # the fixture's acquisition encoding is a provider parser output.
            raw_ids = tuple(
                archive.write_raw_payload(
                    provider=Provider.CHATGPT,
                    payload=json.dumps({"neutral_input": ordinal}).encode(),
                    source_path=f"neutral-{ordinal}.json",
                    canonical_source_path=f"neutral-{ordinal}.json",
                    acquired_at_ms=ordinal + 1,
                )
                for ordinal in range(2)
            )

            def write(session: ParsedSession, raw_id: str) -> ArchiveRawParsedWriteResult:
                with prepared_fixture_index_batch(archive._conn, (session,), archive_root=root) as (seal, carriers):
                    carrier = carriers[0]
                    with archive.index_mutation_scope(prepared_seal=seal):
                        return _write_parsed_precedence_result(
                            archive,
                            session,
                            raw_id=raw_id,
                            source_index=0,
                            stage_timings_s=None,
                            stage_timing_prefix="neutral-decision",
                            manage_transaction=False,
                            prepared_write=carrier,
                            content_hash=carrier.input_content_hash.hex(),
                        )

            first = write(prior, raw_ids[0])
            row = archive._conn.execute(
                "SELECT content_hash,message_count FROM sessions WHERE session_id=?", (first.session_id,)
            ).fetchone()
            assert row is not None
            before = (bytes(row[0]), int(row[1]))
            if decision == "suppressed":
                with closing(sqlite3.connect(root / "user.db")) as user, user:
                    user.row_factory = sqlite3.Row
                    upsert_suppression(user, session_id=first.session_id, reason="neutral suppression")
            second = write(incoming, raw_ids[1])
            row = archive._conn.execute(
                "SELECT content_hash,message_count FROM sessions WHERE session_id=?", (first.session_id,)
            ).fetchone()
            assert row is not None
            after = (bytes(row[0]), int(row[1]))
            return first, second, before, after

    first, second, before, after = await run_archive_fixture_write(root, publish)
    assert first.content_changed and not first.publication_refused
    assert not second.content_changed
    assert second.publication_refused is (decision != "unchanged")
    assert second.counts["messages"] == 0
    assert second.counts["skipped_sessions"] == 1
    assert before == after
