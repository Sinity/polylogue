"""Original retained cold completion and terminal FTS laws on current owners."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from polylogue.core.enums import Provider
from polylogue.daemon.convergence_stages import make_fts_readiness_binding_stage
from polylogue.sources.live import WatchSource
from polylogue.sources.live.cold_build import (
    ColdBuildGeneration,
    clear_cold_build_generation,
    register_cold_build_generation,
)
from polylogue.storage.fts.fts_lifecycle import fts_invariant_snapshot_sync
from polylogue.storage.sqlite import runtime_indexes, schema_bootstrap
from polylogue.storage.sqlite.archive_tiers import revision_governance as archive_revision_governance
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.reference_seal import KnownTierMutationPermit, PreparedIndexMutation
from polylogue.storage.sqlite.runtime_indexes import DEFERRED_SECONDARY_INDEX_NAMES
from tests.infra.archive_templates import bootstrap_archive_root, run_archive_fixture_write
from tests.infra.live_ingest import prepared_live_convergence_owner
from tests.infra.source_builders import ChatGPTExportBuilder


@pytest.mark.asyncio
async def test_owned_empty_generation_uses_cold_build_policy_and_finishes_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "archive"
    payload = json.dumps(
        ChatGPTExportBuilder("cold-generation").add_node("user", "hello").add_node("assistant", "world").build()
    ).encode()

    def acquire() -> str:
        bootstrap_archive_root(root)
        with ArchiveStore.open_existing(root, read_only=False) as archive:
            return archive.write_raw_payload(
                provider=Provider.CHATGPT,
                payload=payload,
                source_path="conversations.json",
                canonical_source_path="conversations.json",
                acquired_at_ms=1,
            )

    raw_id = await run_archive_fixture_write(root, acquire)

    def begin() -> ColdBuildGeneration:
        return ColdBuildGeneration.begin(
            root,
            reason="test-retained-cold-readiness",
            observed=ColdBuildGeneration.observe_source_baseline((WatchSource("fixture", root / "absent"),)),
        )

    # The candidate is materialized, and so stamped, when the generation
    # begins; recording from there covers its whole cold lifecycle.
    stamped_tiers: list[str] = []
    original_stamp = schema_bootstrap.stamp_derived_schema_identity

    def record_stamp(conn: sqlite3.Connection, tier: str) -> None:
        stamped_tiers.append(tier)
        original_stamp(conn, tier)

    monkeypatch.setattr(schema_bootstrap, "stamp_derived_schema_identity", record_stamp)
    generation = await run_archive_fixture_write(root, begin)
    dropped: list[tuple[str, ...]] = []
    original_defer = runtime_indexes.defer_secondary_indexes_sync

    def record_defer(conn: sqlite3.Connection) -> tuple[str, ...]:
        result = original_defer(conn)
        dropped.append(result)
        return result

    monkeypatch.setattr(runtime_indexes, "defer_secondary_indexes_sync", record_defer)
    register_cold_build_generation(generation)
    try:
        async with prepared_live_convergence_owner(root) as owner:
            receipts = (await owner.replay_retained_raw_ids((raw_id,))).require_complete()
            assert {key for receipt in receipts for key in receipt.written_session_ids} == {
                "chatgpt-export:cold-generation"
            }
        await run_archive_fixture_write(root, generation.prepare_promotion_candidate)
        with closing(sqlite3.connect(f"file:{generation.generation.index_path}?mode=ro", uri=True)) as conn:
            names = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='index'")}
            assert set(DEFERRED_SECONDARY_INDEX_NAMES) <= names
            assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 1
            assert conn.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0] > 0
            assert conn.execute("SELECT COUNT(*) FROM action_pairs").fetchone()[0] == 0
            # The candidate carries this runtime's Index identity after the
            # cold writes and the readiness pass.
            schema_bootstrap.assert_derived_schema_identity(conn, "index")
        # Every cold writer open asks for deferral; only the first open drops
        # the secondary indexes and later opens find nothing left to defer.
        assert [names for names in dropped if names] == [DEFERRED_SECONDARY_INDEX_NAMES], dropped
        assert all(names == () for names in dropped[1:]), dropped
        # One stamp for the whole build: only the Index candidate, once, at
        # materialization; no cold writer open re-stamps it or touches ops.
        assert stamped_tiers == ["index"]
    finally:
        clear_cold_build_generation()
        await run_archive_fixture_write(root, generation.discard)


@pytest.mark.asyncio
async def test_retained_replay_terminal_fts_verifies_nonempty_membership(tmp_path: Path) -> None:
    payload = json.dumps(
        ChatGPTExportBuilder("retained-readiness")
        .add_node("user", "retained source")
        .add_node("assistant", "retained reply")
        .build()
    ).encode()

    def acquire() -> str:
        bootstrap_archive_root(tmp_path)
        with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
            return archive.write_raw_payload(
                provider=Provider.CHATGPT,
                payload=payload,
                source_path="conversations.json",
                canonical_source_path="conversations.json",
                acquired_at_ms=1,
            )

    raw_id = await run_archive_fixture_write(tmp_path, acquire)
    async with prepared_live_convergence_owner(tmp_path) as owner:
        receipts = (await owner.replay_retained_raw_ids((raw_id,))).require_complete()
        assert {key for receipt in receipts for key in receipt.written_session_ids} == {
            "chatgpt-export:retained-readiness"
        }
    stage = make_fts_readiness_binding_stage(tmp_path / "index.db")
    assert await run_archive_fixture_write(tmp_path, lambda: stage.execute(tmp_path / "index.db")) is True
    with closing(sqlite3.connect(f"file:{tmp_path / 'index.db'}?mode=ro", uri=True)) as conn:
        messages = fts_invariant_snapshot_sync(conn).messages
    assert messages.ready
    assert messages.source_rows == messages.indexed_rows > 0
    assert (messages.missing_rows, messages.excess_rows, messages.duplicate_rows, messages.identity_mismatch_rows) == (
        0,
        0,
        0,
        0,
    )


@pytest.mark.asyncio
async def test_backfill_resumes_after_index_receipt_commits_before_source_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = (
        b'{"type":"session_meta","payload":{"id":"session-1","timestamp":"2026-06-01T00:00:00Z"}}\n'
        b'{"type":"response_item","payload":{"type":"message","id":"one","role":"user","content":'
        b'[{"type":"input_text","text":"one"}]}}\n'
    )

    def acquire() -> str:
        bootstrap_archive_root(tmp_path)
        with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
            raw_id = archive.write_raw_payload(
                provider=Provider.CODEX,
                payload=payload,
                source_path="session.jsonl",
                canonical_source_path="session.jsonl",
                acquired_at_ms=1,
            )
        return raw_id

    raw_id = await run_archive_fixture_write(tmp_path, acquire)

    # Source terminal markers are staged during preparation and published by
    # the original Source permit after the Index commit; crash at that
    # publication, once the Index receipt is durable.
    original_publish = archive_revision_governance.publish_prepared_revision_source
    reached = False

    def crash_after_index_commit(seal: PreparedIndexMutation, permit: KnownTierMutationPermit) -> None:
        nonlocal reached
        with closing(sqlite3.connect(f"file:{tmp_path / 'index.db'}?mode=ro", uri=True)) as index:
            committed = index.execute("SELECT COUNT(*) FROM raw_revision_applications").fetchone()[0]
        if committed == 1:
            reached = True
            raise RuntimeError("crash after index receipt")
        original_publish(seal, permit)

    monkeypatch.setattr(archive_revision_governance, "publish_prepared_revision_source", crash_after_index_commit)
    with pytest.raises(RuntimeError, match="crash after index receipt"):
        async with prepared_live_convergence_owner(tmp_path) as owner:
            (await owner.replay_retained_raw_ids((raw_id,))).require_complete()
    assert reached
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_revision_applications").fetchone()[0] == 1
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute("SELECT parsed_at_ms FROM raw_sessions WHERE raw_id = ?", (raw_id,)).fetchone() == (None,)

    monkeypatch.setattr(archive_revision_governance, "publish_prepared_revision_source", original_publish)
    async with prepared_live_convergence_owner(tmp_path) as owner:
        resumed = (await owner.replay_retained_raw_ids((raw_id,))).require_complete()
    assert sum(receipt.replayed_logical_sources for receipt in resumed) == 1
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute(
            "SELECT parsed_at_ms IS NOT NULL FROM raw_sessions WHERE raw_id = ?", (raw_id,)
        ).fetchone() == (1,)
