"""Current frontier coverage comes from the real admitted inspection owner."""

from contextlib import closing
from pathlib import Path

import pytest

from polylogue.operations.raw_frontier_inspection import (
    frontier_coverage_for_archive,
    make_raw_frontier_inspection_stage,
)
from polylogue.storage.frontier_inspection import inspect_prepared_raw_authority_frontier
from polylogue.storage.sqlite.connection_profile import open_readonly_connection, owned_daemon_connection
from tests.infra.archive_templates import bootstrap_archive_root, run_archive_fixture_write
from tests.infra.live_ingest import prepared_live_convergence_owner


@pytest.mark.asyncio
@pytest.mark.timeout(0)
async def test_empty_frontier_is_measured_once_under_original_owner(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    await run_archive_fixture_write(root, lambda: bootstrap_archive_root(root))
    source_before = (root / "source.db").read_bytes()
    assert not frontier_coverage_for_archive(root)["available"]
    async with prepared_live_convergence_owner(root) as owner:
        first = await owner.run_convergence_sync(
            "fixture.frontier.inspect",
            inspect_prepared_raw_authority_frontier,
            root,
            input_demand=owner._compute_adapter.amend_current_input_demand,
        )
        second = await owner.run_convergence_sync(
            "fixture.frontier.inspect",
            inspect_prepared_raw_authority_frontier,
            root,
            input_demand=owner._compute_adapter.amend_current_input_demand,
        )
        stage = make_raw_frontier_inspection_stage(root / "index.db", compute_adapter=owner._compute_adapter)
        assert not stage.check(root)
        assert await owner.run_convergence_sync("fixture.frontier.stage", stage.execute, root)
    from polylogue.operations.daemon_status import _frontier_status
    from polylogue.storage.raw_retention import raw_frontier_integrity_projection
    from polylogue.storage.sqlite.connection_profile import attach_readonly_database, readonly_connection_context

    materialization = {"available": True, "lost_source_evidence_count": 0, "lost_source_evidence_samples": []}
    projected = raw_frontier_integrity_projection(root, materialization).to_dict()
    with (
        readonly_connection_context(root / "source.db") as source,
        readonly_connection_context(root / "index.db") as index,
        readonly_connection_context(root / "index.db") as index_with_ops,
    ):
        # The daemon's pinned Ops handle is the Index handle with Ops attached
        # as ``ops_tier`` (611e1d0e52).
        attach_readonly_database(index_with_ops, root / "ops.db", alias="ops_tier")
        assert (
            _frontier_status(source, index, index_with_ops, materialization, ops_db_path=root / "ops.db") == projected
        )
    assert projected["overall_status"] == "healthy"
    coverage = frontier_coverage_for_archive(root)
    assert coverage["current"] and coverage["healthy"]
    assert first.mode == "full" and first.healthy
    assert second.mode == "current" and second.healthy
    assert first.accepted_head_checks == first.cursor_checks == 0
    assert second.pass_id is None
    assert (root / "source.db").read_bytes() == source_before
    with closing(open_readonly_connection(root / "ops.db")) as conn:
        with closing(conn.execute("SELECT state,COUNT(*) FROM raw_frontier_inspection")) as rows:
            row = rows.fetchone()
            assert row is not None and (row[0], row[1]) == ("healthy", 1)
    # A mark binds the actual tier identities, not the copied file contents.
    from shutil import copytree

    cloned = tmp_path / "cloned"
    copytree(root, cloned)
    copied_coverage = frontier_coverage_for_archive(cloned)
    assert not copied_coverage["current"] and not copied_coverage["healthy"]


@pytest.mark.asyncio
@pytest.mark.timeout(0)
async def test_changed_cursor_without_source_is_not_a_healthy_frontier(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    await run_archive_fixture_write(root, lambda: bootstrap_archive_root(root))
    async with prepared_live_convergence_owner(root) as owner:
        first = await owner.run_convergence_sync(
            "fixture.frontier.inspect",
            inspect_prepared_raw_authority_frontier,
            root,
            input_demand=owner._compute_adapter.amend_current_input_demand,
        )
        assert first.healthy

        def add_cursor() -> None:
            with owned_daemon_connection(root / "ops.db", archive_root=root) as conn, conn:
                with closing(
                    conn.execute(
                        "INSERT INTO ingest_cursor(source_path,byte_offset,updated_at_ms) VALUES(?,?,?)",
                        ("neutral/missing.jsonl", 0, 1),
                    )
                ):
                    pass

        await run_archive_fixture_write(root, add_cursor)
        stale = frontier_coverage_for_archive(root)
        assert not stale["current"] and not stale["healthy"]
        changed = await owner.run_convergence_sync(
            "fixture.frontier.inspect",
            inspect_prepared_raw_authority_frontier,
            root,
            input_demand=owner._compute_adapter.amend_current_input_demand,
        )
    coverage = frontier_coverage_for_archive(root)
    assert coverage["current"] and not coverage["healthy"] and coverage["state"] == "blocked"
    assert changed.mode == "delta"
    assert not changed.healthy
    assert changed.cursor_checks == changed.cursor_gap_count == 1
    with closing(open_readonly_connection(root / "ops.db")) as conn:
        with closing(conn.execute("SELECT state FROM raw_frontier_inspection WHERE singleton=1")) as rows:
            assert rows.fetchone()[0] == "blocked"


@pytest.mark.asyncio
@pytest.mark.timeout(0)
async def test_deferred_cursor_findings_remain_a_proven_healthy_projection(tmp_path: Path) -> None:
    """An inspected cursor that compares against no head is not a checked cursor.

    Anti-vacuity: count every inspected cursor as ``cursor_checks`` in the
    findings and the readiness validator sees one checked cursor with zero
    comparisons, refuses the projection, and source selection blocks on a
    frontier whose every status is healthy.
    """
    from polylogue.readiness.capability import raw_frontier_integrity_is_proven_healthy
    from polylogue.storage.raw_retention import raw_frontier_integrity_projection

    root = tmp_path / "archive"
    await run_archive_fixture_write(root, lambda: bootstrap_archive_root(root))

    def add_deferred_cursor() -> None:
        with owned_daemon_connection(root / "ops.db", archive_root=root) as conn, conn:
            with closing(
                conn.execute(
                    "INSERT INTO ingest_cursor(source_path,byte_offset,deferred_end_offset,updated_at_ms) "
                    "VALUES(?,?,?,?)",
                    ("neutral/deferred.jsonl", 0, 10, 1),
                )
            ):
                pass

    await run_archive_fixture_write(root, add_deferred_cursor)
    async with prepared_live_convergence_owner(root) as owner:
        inspected = await owner.run_convergence_sync(
            "fixture.frontier.inspect",
            inspect_prepared_raw_authority_frontier,
            root,
            input_demand=owner._compute_adapter.amend_current_input_demand,
        )
    assert inspected.healthy
    assert inspected.cursor_checks == 1
    materialization = {"available": True, "lost_source_evidence_count": 0, "lost_source_evidence_samples": []}
    projected = raw_frontier_integrity_projection(root, materialization).to_dict()
    assert projected["overall_status"] == "healthy"
    assert projected["cursor_authority_deferred_count"] == 1
    assert projected["cursor_ahead_checked_count"] == projected["cursor_head_comparison_count"] == 0
    assert raw_frontier_integrity_is_proven_healthy(projected)


@pytest.mark.asyncio
@pytest.mark.timeout(0)
async def test_real_retained_head_records_blob_proof_and_current_coverage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import hashlib
    import json

    from polylogue.core.enums import Provider
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    root = tmp_path / "archive"
    payload = json.dumps(
        [
            {
                "id": "frontier-session",
                "title": "neutral",
                "create_time": 1,
                "current_node": "m",
                "mapping": {
                    "m": {
                        "id": "m",
                        "parent": None,
                        "children": [],
                        "message": {
                            "id": "m",
                            "author": {"role": "user"},
                            "create_time": 1,
                            "content": {"content_type": "text", "parts": ["neutral"]},
                        },
                    }
                },
            }
        ]
    ).encode()

    def acquire() -> str:
        bootstrap_archive_root(root)
        with ArchiveStore.open_existing(root, read_only=False) as archive:
            raw_id: str = archive.write_raw_payload(
                provider=Provider.CHATGPT,
                payload=payload,
                source_path="neutral.json",
                canonical_source_path="neutral.json",
                acquired_at_ms=1,
            )
            return raw_id

    raw_id = await run_archive_fixture_write(root, acquire)
    async with prepared_live_convergence_owner(root) as owner:
        receipts = (await owner.ingest_retained_raw_ids((raw_id,))).require_complete()
        assert receipts and sum(len(receipt.written_session_ids) for receipt in receipts) == 1
        from polylogue.storage.blob_store import BlobStore

        verify_calls: list[str] = []
        real_verify = BlobStore.verify

        def counted_verify(store: BlobStore, digest: str) -> bool:
            verify_calls.append(digest)
            return real_verify(store, digest)

        monkeypatch.setattr(BlobStore, "verify", counted_verify)
        first = await owner.run_convergence_sync(
            "fixture.frontier.nonempty",
            inspect_prepared_raw_authority_frontier,
            root,
            input_demand=owner._compute_adapter.amend_current_input_demand,
        )
        assert first.healthy and first.accepted_head_checks >= 1
        assert verify_calls == [hashlib.sha256(payload).hexdigest()]
        verify_calls.clear()
        second = await owner.run_convergence_sync(
            "fixture.frontier.nonempty",
            inspect_prepared_raw_authority_frontier,
            root,
            input_demand=owner._compute_adapter.amend_current_input_demand,
        )
        assert second.mode == "current" and second.healthy
        physical = await owner.run_convergence_sync(
            "fixture.frontier.physical",
            inspect_prepared_raw_authority_frontier,
            root,
            input_demand=owner._compute_adapter.amend_current_input_demand,
            check_physical_dependencies=True,
        )
        assert physical.healthy and physical.physical_dependencies_checked
        assert physical.accepted_head_checks >= 1 and verify_calls == []
    with closing(open_readonly_connection(root / "source.db")) as conn:
        row = conn.execute("SELECT blob_hash,blob_size FROM raw_sessions WHERE raw_id=?", (raw_id,)).fetchone()
        assert row is not None and row[0] == hashlib.sha256(payload).digest() and row[1] == len(payload)
        receipt = conn.execute("SELECT st_size FROM verified_blob_receipts WHERE blob_hash=?", (row[0],)).fetchone()
        assert receipt is not None and receipt[0] == len(payload)
    assert frontier_coverage_for_archive(root)["healthy"]
    blob = BlobStore(root / "blob").blob_path(hashlib.sha256(payload).hexdigest())
    blob.write_bytes(payload + b"neutral-corruption")
    async with prepared_live_convergence_owner(root) as owner:
        corrupted = await owner.run_convergence_sync(
            "fixture.frontier.physical",
            inspect_prepared_raw_authority_frontier,
            root,
            input_demand=owner._compute_adapter.amend_current_input_demand,
            check_physical_dependencies=True,
        )
    assert corrupted.physical_dependencies_checked and not corrupted.healthy
    assert corrupted.blocking_head_checks >= 1
    assert verify_calls == [hashlib.sha256(payload).hexdigest()]
    with closing(open_readonly_connection(root / "source.db")) as conn:
        row = conn.execute(
            "SELECT json_extract(observed_json,'$.state') FROM raw_authority_blockers WHERE resolved_at_ms IS NULL"
        ).fetchone()
        assert row is not None and row[0] == "missing_bytes_reacquire"
        reason = conn.execute("SELECT reason FROM raw_authority_blockers WHERE resolved_at_ms IS NULL").fetchone()
        assert (
            reason is not None
            and reason[0] == "accepted head raw bytes do not prove the expected content-addressed digest"
        )


@pytest.mark.timeout(0)
@pytest.mark.uses_real_clock("real UDS lifecycle and operation completion use wall-clock ownership waits")
def test_public_frontier_operation_uses_supplied_preparation_owner(tmp_path: Path) -> None:
    from tests.infra.daemon_operations import running_daemon_operations

    with running_daemon_operations(tmp_path / "archive") as stack:
        first = stack.client.operation_to_completion(
            "maintenance.raw-authority-frontier", {}, archive_root=str(stack.archive_root)
        )
        second = stack.client.operation_to_completion(
            "maintenance.raw-authority-frontier", {}, archive_root=str(stack.archive_root)
        )
        assert first is not None and second is not None
        assert first["outcome"] == second["outcome"] == "completed"
        assert first["result"]["result"]["mode"] == "full"
        assert second["result"]["result"]["mode"] == "full"
        assert second["result"]["result"]["physical_dependencies_checked"] is True
        assert first["result"]["result"]["healthy"]
        assert second["result"]["effect"] == "committed"
        coverage = frontier_coverage_for_archive(stack.archive_root)
        assert coverage["current"] and coverage["healthy"]
