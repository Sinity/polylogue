"""Production-schema proofs for the canonical current blob-owner relation."""

from __future__ import annotations

import os
import re
import sqlite3
import time
from pathlib import Path

import pytest

from polylogue.storage.blob_gc import run_blob_gc_report
from polylogue.storage.blob_liveness import (
    LivenessState,
    inspect_blob_liveness,
    project_live_blob_hashes,
    validated_blob_ref_liveness_joins,
)
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from polylogue.storage.sqlite.archive_tiers.source_items import (
    publish_source_generation,
    record_source_item_raw_member,
)


def _archive(tmp_path: Path) -> tuple[Path, bytes]:
    root = tmp_path / "archive"
    initialize_active_archive_root(root)
    return root, b"x" * 32


@pytest.mark.uses_real_clock("backdates frozen input bytes to exercise the production GC age gate")
def test_pending_frozen_input_survives_gc_before_raw_admission(tmp_path: Path) -> None:
    """Removing source_items from BLOB_OWNERS would delete accepted input bytes."""
    root, _ = _archive(tmp_path)
    store = BlobStore(root / "blob")
    blob_hash, _ = store.write_from_bytes(b"synthetic frozen input")
    old = time.time() - 3600
    os.utime(store.blob_path(blob_hash), (old, old))
    with sqlite3.connect(root / "source.db") as source:
        publish_source_generation(
            source,
            source_generation_id="pending-input",
            manifest_digest="a" * 64,
            addressing_mode="physical-file-v1",
            coordinates=("export.json",),
            input_blob_hashes={"export.json": bytes.fromhex(blob_hash)},
            enumeration_fingerprint="b" * 64,
            observed_at_ms=1,
        )
        assert source.execute("SELECT COUNT(*) FROM raw_sessions").fetchone()[0] == 0
        decision = inspect_blob_liveness(source, blob_hash)
        assert decision.state is LivenessState.LIVE
        assert "source.db.source_items" in decision.surfaces
    report = run_blob_gc_report(root / "source.db", store.root)
    assert report.blocked_reason is None
    assert report.deleted_count == 0
    assert store.exists(blob_hash)


def test_generation_projection_includes_exact_many_raw_edges_and_only_its_input(tmp_path: Path) -> None:
    root, _ = _archive(tmp_path)
    with sqlite3.connect(root / "source.db") as source:
        for generation, input_hash in (("one", b"1" * 32), ("two", b"2" * 32)):
            (item,) = publish_source_generation(
                source,
                source_generation_id=generation,
                manifest_digest="a" * 64,
                addressing_mode="physical-file-v1",
                coordinates=("export.json",),
                input_blob_hashes={"export.json": input_hash},
                enumeration_fingerprint="b" * 64,
                observed_at_ms=1,
            )
            if generation == "one":
                source.execute(
                    "INSERT INTO raw_sessions(raw_id, origin, source_path, blob_hash, blob_size, acquired_at_ms) "
                    "VALUES ('raw', 'codex-session', '/synthetic/export.json', ?, 1, 1)",
                    (b"r" * 32,),
                )
                for coordinate in ("record:0", "record:1"):
                    record_source_item_raw_member(
                        source,
                        source_generation_id=generation,
                        source_item_id=item,
                        record_coordinate=coordinate,
                        raw_id="raw",
                        raw_blob_hash=b"r" * 32,
                    )
        projection = project_live_blob_hashes(source, source_generation_id="one")
        assert not projection.blockers
        assert projection.live_hashes == frozenset({(b"1" * 32).hex(), (b"r" * 32).hex()})


def test_archives_predating_source_items_remain_inspectable(tmp_path: Path) -> None:
    root, blob_hash = _archive(tmp_path)
    with sqlite3.connect(root / "source.db") as source:
        source.execute("DROP VIEW source_item_reconciliation")
        source.execute("DROP TABLE source_item_raw_members")
        source.execute("DROP TABLE source_items")
        decision = inspect_blob_liveness(source, blob_hash.hex())
        projection = project_live_blob_hashes(source, source_generation_id="not-present")
        assert decision.state is LivenessState.UNREFERENCED
        assert not projection.blockers


@pytest.mark.parametrize(
    ("ref_type", "table", "identifier", "expected"),
    (
        ("raw_payload", "raw_sessions", "raw-live", "source.db.blob_refs"),
        ("attachment", "raw_sessions", "raw-live", "source.db.blob_refs"),
        ("hook_payload", "raw_hook_events", "hook-live", "source.db.blob_refs"),
        ("sidecar", "history_sidecars", "sidecar-live", "source.db.blob_refs"),
    ),
)
def test_joined_ledger_kinds_protect_only_live_referents(
    tmp_path: Path, ref_type: str, table: str, identifier: str, expected: str
) -> None:
    """Deleting the mapped referent is the negative twin for each ledger kind."""
    root, blob_hash = _archive(tmp_path)
    with sqlite3.connect(root / "source.db") as conn, sqlite3.connect(root / "index.db") as index:
        if table == "raw_sessions":
            conn.execute(
                """INSERT INTO raw_sessions
                (raw_id, origin, source_path, source_index, blob_hash, blob_size, acquired_at_ms)
                VALUES (?, 'codex-session', '/raw', 0, ?, 1, 1)""",
                (identifier, b"r" * 32),
            )
        elif table == "raw_hook_events":
            conn.execute(
                """INSERT INTO raw_hook_events
                (hook_event_id, origin, source_path, event_type, payload_json, observed_at_ms, blob_hash)
                VALUES (?, 'codex-session', '/hook', 'PostToolUse', '{}', 1, ?)""",
                (identifier, b"h" * 32),
            )
        else:
            conn.execute(
                """INSERT INTO history_sidecars
                (sidecar_id, origin, source_path, payload_json, observed_at_ms, content_hash)
                VALUES (?, 'codex-session', '/sidecar', '{}', 1, ?)""",
                (identifier, b"s" * 32),
            )
        conn.execute(
            """INSERT INTO blob_refs (blob_hash, ref_id, ref_type, source_path, size_bytes, acquired_at_ms)
            VALUES (?, ?, ?, '/fixture', 1, 1)""",
            (blob_hash, identifier, ref_type),
        )

        assert expected in inspect_blob_liveness(conn, blob_hash.hex(), index_conn=index, require_index=True).surfaces
        if table == "raw_sessions":
            conn.execute("DELETE FROM raw_sessions WHERE raw_id = ?", (identifier,))
        elif table == "raw_hook_events":
            conn.execute("DELETE FROM raw_hook_events WHERE hook_event_id = ?", (identifier,))
        else:
            conn.execute("DELETE FROM history_sidecars WHERE sidecar_id = ?", (identifier,))
        # A bare ledger row is not authority. Replacing the relation with a
        # membership test makes this assertion fail.
        assert (
            inspect_blob_liveness(conn, blob_hash.hex(), index_conn=index, require_index=True).state
            is LivenessState.UNREFERENCED
        )


def test_direct_owners_and_reservation_survive_missing_or_dangling_ledger(tmp_path: Path) -> None:
    """Direct ownership, duplicate ownership, and publish reservations are independent protections."""
    root, blob_hash = _archive(tmp_path)
    with sqlite3.connect(root / "source.db") as conn, sqlite3.connect(root / "index.db") as index:
        conn.execute("PRAGMA ignore_check_constraints = ON")
        conn.execute(
            """INSERT INTO raw_sessions
            (raw_id, origin, source_path, source_index, blob_hash, blob_size, acquired_at_ms)
            VALUES ('raw-direct', 'codex-session', '/raw', 0, ?, 1, 1)""",
            (blob_hash,),
        )
        index.execute(
            "INSERT INTO attachments (attachment_id, blob_hash, acquisition_status) VALUES ('att-direct', ?, 'acquired')",
            (blob_hash,),
        )
        conn.execute(
            """INSERT INTO blob_refs (blob_hash, ref_id, ref_type, source_path, size_bytes, acquired_at_ms)
            VALUES (?, 'gone', 'raw_payload', '/gone', 1, 1)""",
            (blob_hash,),
        )
        decision = inspect_blob_liveness(conn, blob_hash.hex(), index_conn=index, require_index=True)
        assert set(decision.surfaces) == {"source.db.raw_sessions", "index.db.attachments"}
        conn.execute("DELETE FROM raw_sessions")
        index.execute("DELETE FROM attachments")
        conn.execute(
            """INSERT INTO blob_publication_reservations
            (publication_id, blob_hash, size_bytes, publisher_id, reserved_at_ms)
            VALUES ('publication', ?, 1, 'publisher', 1)""",
            (blob_hash,),
        )
        assert (
            inspect_blob_liveness(conn, blob_hash.hex(), index_conn=index, require_index=True).state
            is LivenessState.UNREFERENCED
        )
        conn.execute("DELETE FROM blob_publication_reservations")
        assert (
            inspect_blob_liveness(conn, blob_hash.hex(), index_conn=index, require_index=True).state
            is LivenessState.UNREFERENCED
        )


def test_unknown_or_missing_owner_surface_is_an_observable_gc_blocker(tmp_path: Path) -> None:
    root, blob_hash = _archive(tmp_path)
    with sqlite3.connect(root / "source.db") as conn, sqlite3.connect(root / "index.db") as index:
        conn.execute("PRAGMA ignore_check_constraints = ON")
        conn.execute(
            """INSERT INTO blob_refs (blob_hash, ref_id, ref_type, source_path, size_bytes, acquired_at_ms)
            VALUES (?, 'future', 'future-kind', '/future', 1, 1)""",
            (blob_hash,),
        )
        unknown = inspect_blob_liveness(conn, blob_hash.hex(), index_conn=index, require_index=True)
        assert unknown.state is LivenessState.BLOCKED and any("unknown" in blocker for blocker in unknown.blockers)
        index.execute("DROP TABLE attachments")
        unavailable = inspect_blob_liveness(conn, blob_hash.hex(), index_conn=index, require_index=True)
        assert unavailable.state is LivenessState.BLOCKED and "index.attachments is missing" in unavailable.blockers


@pytest.mark.uses_real_clock("backdates real GC candidates")
def test_unknown_ref_kind_on_another_hash_blocks_the_entire_destructive_pass(tmp_path: Path) -> None:
    """A future ledger kind blocks unlink of an otherwise orphaned hash."""
    root, _unused = _archive(tmp_path)
    store = BlobStore(root / "blob")
    unknown_hash, _size = store.write_from_bytes(b"unknown ledger payload")
    orphan_hash, _size = store.write_from_bytes(b"orphan candidate")
    for blob_hash in (unknown_hash, orphan_hash):
        os.utime(store.blob_path(blob_hash), (time.time() - 3600, time.time() - 3600))
    with sqlite3.connect(root / "source.db") as conn:
        conn.execute("PRAGMA ignore_check_constraints = ON")
        conn.execute(
            """INSERT INTO blob_refs (blob_hash, ref_id, ref_type, source_path, size_bytes, acquired_at_ms)
            VALUES (?, 'future', 'future-kind', '/future', 1, 1)""",
            (bytes.fromhex(unknown_hash),),
        )

    report = run_blob_gc_report(root / "source.db", store.root)

    assert report.blocked_reason is not None
    assert "unknown blob_refs ref_type" in report.blocked_reason
    assert store.exists(orphan_hash)


def test_structured_decision_fails_closed_for_missing_required_source_schema(tmp_path: Path) -> None:
    root, blob_hash = _archive(tmp_path)
    with sqlite3.connect(root / "source.db") as source, sqlite3.connect(root / "index.db") as index:
        decision = inspect_blob_liveness(source, blob_hash.hex(), index_conn=index, require_index=True)
        assert decision.state is LivenessState.UNREFERENCED
        source.execute("DROP TABLE blob_refs")
        blocked = inspect_blob_liveness(source, blob_hash.hex(), index_conn=index, require_index=True)
        assert blocked.state is LivenessState.BLOCKED
        assert "source.blob_refs is missing" in blocked.blockers


def test_bulk_projection_excludes_dangling_ledger_rows_and_verification_receipts(tmp_path: Path) -> None:
    """Integrity and seal denominators count canonical owners only."""
    root, blob_hash = _archive(tmp_path)
    with sqlite3.connect(root / "source.db") as source:
        source.execute(
            """INSERT INTO blob_refs (blob_hash, ref_id, ref_type, source_path, size_bytes, acquired_at_ms)
            VALUES (?, 'gone', 'raw_payload', '/gone', 1, 1)""",
            (blob_hash,),
        )
        source.execute(
            """INSERT INTO verified_blob_receipts
            (blob_hash, st_dev, st_ino, st_size, st_mtime_ns, st_ctime_ns, verified_at_ms)
            VALUES (?, 1, 1, 1, 1, 1, 1)""",
            (blob_hash,),
        )
        projection = project_live_blob_hashes(source)

    assert projection.live_hashes == frozenset()


@pytest.mark.uses_real_clock("backdates the real blob mtime for GC's production age gate")
def test_gc_refuses_missing_current_owner_surface_before_unlink(tmp_path: Path) -> None:
    """Dropping index ownership is a controlled mutation of GC's safety gate."""
    root, _unused = _archive(tmp_path)
    store = BlobStore(root / "blob")
    blob_hash, _size = store.write_from_bytes(b"would be collected without the index owner")
    path = store.blob_path(blob_hash)
    old = time.time() - 3600
    os.utime(path, (old, old))
    with sqlite3.connect(root / "index.db") as index:
        index.execute("DROP TABLE attachments")

    report = run_blob_gc_report(root / "source.db", store.root)

    assert report.deleted_count == 0
    assert report.blocked_reason is not None
    assert "index.attachments is missing" in report.blocked_reason
    assert store.exists(blob_hash)


def test_blob_refs_check_vocabulary_and_liveness_join_map_agree(tmp_path: Path) -> None:
    """Every ref_type the source DDL admits must have a liveness referent join.

    A ref_type the schema accepts but the map cannot resolve makes liveness
    projection return BLOCKED for any archive holding such a row, stalling GC
    on data the writer was entitled to create. The two vocabularies live in
    different files, so nothing but this proof couples them.
    """

    root, _ = _archive(tmp_path)
    with sqlite3.connect(root / "source.db") as conn:
        sql = conn.execute("SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'blob_refs'").fetchone()[0]
    match = re.search(r"ref_type\s+TEXT\s+NOT NULL\s+CHECK\(ref_type IN \(([^)]*)\)\)", sql)
    assert match is not None, f"could not read the blob_refs ref_type CHECK from: {sql}"
    admitted = {literal.strip().strip("'") for literal in match.group(1).split(",")}

    mapped = {ref_type for ref_type, _table, _column in validated_blob_ref_liveness_joins()}
    assert admitted == mapped, (
        f"blob_refs CHECK admits {sorted(admitted)} but the liveness join map resolves "
        f"{sorted(mapped)}; unresolved ref_types block liveness projection"
    )


_SIDECAR_SESSION_ID = "de99ba60-ccc4-43a7-b882-1dd1f2672db7"


def _claude_code_session_tree_with_sidecar(root: Path) -> tuple[Path, Path, str]:
    """Lay out one Claude Code transcript whose tool result overflowed to a sidecar."""
    import json

    project = root / "-realm-project-x"
    sidecar_dir = project / _SIDECAR_SESSION_ID / "tool-results"
    sidecar_dir.mkdir(parents=True)
    sidecar = sidecar_dir / "bsq814i68.txt"
    sidecar_text = "persisted tool output\n" * 64
    sidecar.write_text(sidecar_text, encoding="utf-8")

    pointer = f"<persisted-output>Output too large. Full output saved to: {sidecar}</persisted-output>"
    owner = project / f"{_SIDECAR_SESSION_ID}.jsonl"
    owner.write_text(
        "\n".join(
            json.dumps(record)
            for record in (
                {
                    "type": "user",
                    "uuid": "u1",
                    "sessionId": _SIDECAR_SESSION_ID,
                    "timestamp": "2026-07-20T10:00:00Z",
                    "message": {"role": "user", "content": "run it"},
                },
                {
                    "type": "assistant",
                    "uuid": "a1",
                    "parentUuid": "u1",
                    "sessionId": _SIDECAR_SESSION_ID,
                    "timestamp": "2026-07-20T10:00:01Z",
                    "message": {
                        "role": "assistant",
                        "content": [{"type": "tool_use", "id": "toolu_abc", "name": "Bash", "input": {}}],
                    },
                },
                {
                    "type": "user",
                    "uuid": "u2",
                    "parentUuid": "a1",
                    "sessionId": _SIDECAR_SESSION_ID,
                    "timestamp": "2026-07-20T10:00:02Z",
                    "message": {
                        "role": "user",
                        "content": [{"type": "tool_result", "tool_use_id": "toolu_abc", "content": pointer}],
                    },
                },
            )
        )
        + "\n",
        encoding="utf-8",
    )
    return owner, sidecar, sidecar_text


@pytest.mark.asyncio
@pytest.mark.uses_real_clock("ages retained blob bytes past the production GC age gate")
async def test_acquired_sidecar_bytes_outlive_their_source_tree_and_the_index_tier(
    workspace_env: dict[str, Path],
) -> None:
    """A tool-output sidecar's bytes stay source-owned and re-readable (polylogue-hxrhn).

    The original defect was a sidecar whose only surviving pointer was a
    derived ``session_events`` payload: nothing owned the bytes, so GC was
    entitled to reclaim them and a reparse could only recover the transcript's
    truncated preview. This walks the production acquisition route
    (``LiveBatchProcessor.ingest_files``), then removes both things the
    original defect assumed would still be there -- the source tree the bytes
    were staged from, and the index tier -- and asserts the bytes survive a
    real ``run_blob_gc_report`` pass and still resolve through the derivation
    reader (``RetainedSidecarResolver``).

    Anti-vacuity: drop the ``BlobOwner("source", "raw_sessions", ...)`` direct
    entry *and* its ``raw_payload`` ledger entry from ``BLOB_OWNERS`` and the
    GC assertions go red (the hash has no owning surface once the index is
    rebuilt empty); make the acquisition walk skip ``tool_result_sidecar``
    paths instead of retaining them as raw artifacts and both the GC and the
    ``RetainedSidecarResolver`` assertions go red.
    """
    import shutil

    import polylogue.sources.live.watcher as live_watcher
    from polylogue import Polylogue
    from polylogue.sources.live import WatchSource
    from polylogue.sources.live.batch import LiveBatchProcessor
    from polylogue.sources.live.cursor import CursorStore
    from polylogue.sources.live.sidecar_resolution import RetainedSidecarResolver

    tree_root = workspace_env["data_root"] / "projects"
    tree_root.mkdir(parents=True)
    owner, sidecar, sidecar_text = _claude_code_session_tree_with_sidecar(tree_root)
    archive_root = workspace_env["archive_root"]

    archive = Polylogue(archive_root=archive_root, db_path=workspace_env["data_root"] / "index.db")
    cursor = CursorStore(workspace_env["data_root"] / "cursor.db")
    processor = LiveBatchProcessor(
        archive,
        (WatchSource(name="claude-code", root=tree_root, suffixes=(".jsonl",)),),
        cursor=cursor,
        parser_fingerprint=live_watcher._PARSER_FINGERPRINT,
    )
    try:
        await processor.ingest_files([sidecar, owner], emit_event=False)

        with sqlite3.connect(f"file:{archive_root / 'source.db'}?mode=ro", uri=True) as source:
            retained = source.execute(
                "SELECT hex(blob_hash) FROM raw_sessions WHERE source_path = ?", (str(sidecar),)
            ).fetchone()
            assert retained is not None, "acquisition left no source-tier row for the sidecar"
            sidecar_hash = str(retained[0]).lower()
            # The ledger ref exists beside the direct column owner, so the
            # bytes are owned twice over rather than by a derived payload.
            assert (
                source.execute(
                    "SELECT 1 FROM blob_refs WHERE blob_hash = ? AND ref_type = 'raw_payload'",
                    (bytes.fromhex(sidecar_hash),),
                ).fetchone()
                is not None
            )

        with (
            sqlite3.connect(archive_root / "source.db") as source,
            sqlite3.connect(archive_root / "index.db") as index,
        ):
            decision = inspect_blob_liveness(source, sidecar_hash, index_conn=index)
        assert decision.state is LivenessState.LIVE
        assert decision.surfaces == ("source.db.raw_sessions", "source.db.blob_refs")
    finally:
        await archive.close()

    # The transient staging is gone: the source tree the bytes came from, and
    # the whole index tier, rebuilt empty as a reindex would leave it.
    shutil.rmtree(tree_root)
    (archive_root / "index.db").unlink()
    initialize_active_archive_root(archive_root)

    store = BlobStore(archive_root / "blob")
    aged = time.time() - 3600
    for blob_file in store.root.rglob("*"):
        if blob_file.is_file():
            os.utime(blob_file, (aged, aged))

    report = run_blob_gc_report(archive_root / "source.db", store.root)
    assert report.blocked_reason is None
    assert report.deleted_count == 0
    assert store.exists(sidecar_hash)

    scope = RetainedSidecarResolver(archive_root).claude_code_scope(owner)
    assert scope.available is True
    assert [entry.filename for entry in scope.files] == [sidecar.name]
    assert scope.files[0].read_text() == sidecar_text
