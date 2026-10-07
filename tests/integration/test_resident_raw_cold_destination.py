"""The resident Raw owner borrows the registered inactive Index destination."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from polylogue.core.enums import Provider
from polylogue.core.errors import SchemaVersionMismatchError
from polylogue.operations.daemon_ingest import IngestExecution, IngestReprepareRequiredError, SourceReceiptSpool
from polylogue.sources.live import WatchSource
from polylogue.sources.live.cold_build import (
    ColdBuildGeneration,
    clear_cold_build_generation,
    register_cold_build_generation,
)
from polylogue.storage.archive_identity import resolve_active_index_path
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.reference_seal import IndexMutationDestination
from tests.infra.archive_templates import bootstrap_archive_root, run_archive_fixture_write
from tests.infra.live_ingest import prepared_live_convergence_owner
from tests.infra.source_builders import ChatGPTExportBuilder


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["ingest", "replay"])
async def test_resident_raw_writes_registered_candidate_without_active_index_effects(
    tmp_path: Path, route: str
) -> None:
    root = tmp_path / "archive"
    payload = json.dumps(ChatGPTExportBuilder("candidate-session").add_node("user", "neutral").build()).encode()

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
    active_path = resolve_active_index_path(root)

    def begin() -> ColdBuildGeneration:
        return ColdBuildGeneration.begin(
            root,
            reason="test-resident-destination",
            observed=ColdBuildGeneration.observe_source_baseline((WatchSource("fixture", root / "absent-source"),)),
        )

    generation = await run_archive_fixture_write(root, begin)
    register_cold_build_generation(generation)
    try:
        async with prepared_live_convergence_owner(root) as owner:
            receipts = (
                (await owner.ingest_retained_raw_ids((raw_id,))).require_complete()
                if route == "ingest"
                else (await owner.replay_retained_raw_ids((raw_id,))).require_complete()
            )
            assert {key for receipt in receipts for key in receipt.written_session_ids} == {
                "chatgpt-export:candidate-session"
            }
        assert resolve_active_index_path(root) == active_path
        with closing(sqlite3.connect(f"file:{active_path}?mode=ro", uri=True)) as active:
            assert active.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0
            assert active.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 0
        with closing(sqlite3.connect(f"file:{generation.generation.index_path}?mode=ro", uri=True)) as candidate:
            assert candidate.execute("SELECT session_id FROM sessions").fetchall() == [
                ("chatgpt-export:candidate-session",)
            ]
            assert candidate.execute("SELECT text FROM blocks WHERE block_type='text'").fetchall() == [("neutral",)]
    finally:
        clear_cold_build_generation()
        await run_archive_fixture_write(root, generation.discard)


@pytest.mark.asyncio
@pytest.mark.parametrize("damage", [None, "missing-required", "changed-deferred"])
async def test_owned_cold_read_admits_only_deferred_indexes_and_restores_reader_shape(
    tmp_path: Path, damage: str | None
) -> None:
    root = tmp_path / "archive"

    def begin() -> ColdBuildGeneration:
        bootstrap_archive_root(root)
        generation = ColdBuildGeneration.begin(
            root,
            reason="test-cold-reader",
            observed=ColdBuildGeneration.observe_source_baseline((WatchSource("fixture", root / "absent"),)),
        )
        with generation.open_writer():
            pass
        return generation

    generation = await run_archive_fixture_write(root, begin)
    try:
        with pytest.raises(SchemaVersionMismatchError):
            with ArchiveStore.open_existing(root, read_only=True, index_path=Path(generation.generation.index_path)):
                pass
        with ArchiveStore.open_owned_inactive_read(generation.generation) as reader:
            assert reader._conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0
        if damage is not None:

            def damage_candidate() -> None:
                with generation.open_writer() as writer:
                    if damage == "missing-required":
                        writer._conn.execute("DROP INDEX idx_session_events_source_message")
                    else:
                        writer._conn.execute("CREATE INDEX idx_blocks_type ON blocks(position)")
                    writer._conn.commit()

            await run_archive_fixture_write(root, damage_candidate)
            with pytest.raises(SchemaVersionMismatchError):
                with ArchiveStore.open_owned_inactive_read(generation.generation):
                    pass
        else:

            def restore() -> None:
                with generation.open_writer() as writer:
                    writer.run_generation_readiness_pass()

            await run_archive_fixture_write(root, restore)
            with ArchiveStore.open_existing(
                root, read_only=True, index_path=Path(generation.generation.index_path)
            ) as reader:
                names = {row[0] for row in reader._conn.execute("SELECT name FROM sqlite_master WHERE type='index'")}
                assert "idx_messages_role" in names
                assert "idx_blocks_type" in names
    finally:
        await run_archive_fixture_write(root, generation.discard)


@pytest.mark.asyncio
async def test_ingest_refuses_wrong_and_stale_owned_cold_receipt_destinations(tmp_path: Path) -> None:
    roots = (tmp_path / "first", tmp_path / "second")

    def begin(root: Path, reason: str) -> ColdBuildGeneration:
        bootstrap_archive_root(root)
        return ColdBuildGeneration.begin(
            root,
            reason=reason,
            observed=ColdBuildGeneration.observe_source_baseline((WatchSource("fixture", root / "absent"),)),
        )

    first = await run_archive_fixture_write(roots[0], lambda: begin(roots[0], "first-candidate"))
    second = await run_archive_fixture_write(roots[1], lambda: begin(roots[1], "second-candidate"))
    first_destination = IndexMutationDestination.owned_inactive(first.generation)
    second_destination = IndexMutationDestination.owned_inactive(second.generation)
    wrong_receipt = SourceReceiptSpool(
        tmp_path / "unused-receipt.sqlite", "source-generation", 0, True, False, 0, 0, second_destination
    )
    try:
        with pytest.raises(IngestReprepareRequiredError, match="destination changed"):
            IngestExecution._require_materialization_destination(wrong_receipt, first_destination)

        await run_archive_fixture_write(roots[0], first.discard)
        with pytest.raises(IngestReprepareRequiredError, match="retry required"):
            IngestExecution._validate_index_destination(first_destination)
    finally:
        if not first.settled:
            await run_archive_fixture_write(roots[0], first.discard)
        await run_archive_fixture_write(roots[1], second.discard)
