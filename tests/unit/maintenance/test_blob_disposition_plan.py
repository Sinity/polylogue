"""Laws for the physical blob disposition plan.

Every test names the mutation that makes it red. The plan decides whether an
irreplaceable object is deleted, so the anti-vacuity conditions are all of the
same family: a prover that accepts material current sources do not hold, or a
classifier that converts "unknown" into "discard", must fail here.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from io import BytesIO
from pathlib import Path

import pytest

from polylogue.maintenance.blob_disposition import (
    AppendPrefixProver,
    BlobDisposition,
    BlobDispositionContext,
    BlobDispositionError,
    BlobDispositionMember,
    BlobDispositionPlan,
    ExportArchiveMemberProver,
    RawSourceCarrier,
    RawSourceFileProver,
    RestorationDestination,
    SourceProofMode,
    append_successors_by_hash,
    build_disposition_context,
    codex_state_logical_export_hashes,
    compile_disposition_plan,
    hook_event_carriers_by_hash,
    raw_source_carriers_by_hash,
    referenced_blob_hashes,
)
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier


def _hook_envelope(event_id: str = "event-1", *, text: str = "ran a tool") -> dict[str, object]:
    return {
        "event_id": event_id,
        "event_type": "PreToolUse",
        "session_id": "session-1",
        "timestamp": "2026-07-15T02:15:39Z",
        "provider": "claude-code",
        "payload": {"tool_name": "Bash", "detail": text},
    }


def _write_spool_file(root: Path, envelope: dict[str, object], *, indent: int | None = None) -> Path:
    target = root / "pending" / "2026-07-15"
    target.mkdir(parents=True, exist_ok=True)
    path = target / f"{envelope['event_id']}.json"
    path.write_text(json.dumps(envelope, ensure_ascii=False, sort_keys=True, indent=indent), encoding="utf-8")
    return path


def _publish_blob(store: BlobStore, payload: bytes) -> str:
    blob_hash, _ = store.write_from_bytes(payload)
    return blob_hash


def _stored_envelope_bytes(spool_file: Path) -> bytes:
    """Serialize the validated record the way acquisition stored it."""
    from polylogue.sources.hooks import read_hook_spool_record

    record = read_hook_spool_record(spool_file)
    return json.dumps(record, ensure_ascii=False, sort_keys=True, indent=1).encode("utf-8")


def _empty_source_db(path: Path) -> Path:
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE blob_refs (blob_hash BLOB, ref_type TEXT)")
        conn.execute(
            "CREATE TABLE raw_sessions (raw_id TEXT, origin TEXT, native_id TEXT, blob_hash BLOB, "
            "blob_size INTEGER, source_path TEXT, append_start_offset INTEGER)"
        )
    return path


def _empty_index_db(archive_root: Path) -> Path:
    """Materialize the required derived tier through its production bootstrap."""
    path = archive_root / "index.db"
    initialize_archive_database(path, ArchiveTier.INDEX)
    return path


def _reference(source_db: Path, *blob_hashes: str) -> None:
    """Name blobs in a durable relation so liveness, not silence, decides."""
    with sqlite3.connect(source_db) as conn:
        conn.executemany(
            "INSERT INTO blob_refs (blob_hash, ref_type) VALUES (?, 'raw_payload')",
            [(bytes.fromhex(blob_hash),) for blob_hash in blob_hashes],
        )


def _context(
    tmp_path: Path,
    *,
    hook_roots: tuple[tuple[str, Path], ...] = (),
    referenced: tuple[str, ...] = (),
) -> BlobDispositionContext:
    blob_root = tmp_path / "blob"
    blob_root.mkdir(exist_ok=True)
    source_db = _empty_source_db(tmp_path / "source.db")
    _empty_index_db(tmp_path)
    _reference(source_db, *referenced)
    return build_disposition_context(
        archive_root=tmp_path,
        blob_root=blob_root,
        source_db=source_db,
        hook_spool_sources=hook_roots,
        browser_capture_spool=tmp_path / "browser-capture",
    )


def test_hook_envelope_is_source_present_despite_differing_bytes(tmp_path: Path) -> None:
    """Anti-vacuity: a byte-equality prover would call this a sole copy and delete it.

    Acquisition derives ``observed_at_ms`` and both sides serialize
    independently, so the stored object never equals the spool file's bytes.
    """
    spool_root = tmp_path / "legacy-hooks"
    envelope = _hook_envelope()
    spool_file = _write_spool_file(spool_root, envelope, indent=4)
    store = BlobStore(tmp_path / "blob")
    blob_hash = _publish_blob(store, _stored_envelope_bytes(spool_file))
    assert store.blob_path(blob_hash).read_bytes() != spool_file.read_bytes()

    context = _context(tmp_path, hook_roots=(("legacy-hook-spool-0", spool_root),))
    plan = compile_disposition_plan(
        archive_root=tmp_path,
        blob_root=store.root,
        source_db=tmp_path / "source.db",
        context=context,
    )

    (member,) = plan.members
    assert member.disposition is BlobDisposition.SOURCE_PRESENT
    assert member.proof is not None
    assert member.proof.mode is SourceProofMode.SEMANTIC_EQUIVALENT
    assert member.proof.source_path == str(spool_file)
    assert plan.accepted


def test_hook_envelope_without_a_spool_file_is_restore_required(tmp_path: Path) -> None:
    """Anti-vacuity: accepting an absent source would delete the only carrier."""
    spool_root = tmp_path / "legacy-hooks"
    spool_root.mkdir()
    envelope = _hook_envelope("orphan-event")
    scratch = tmp_path / "scratch.json"
    scratch.write_text(json.dumps(envelope, sort_keys=True), encoding="utf-8")
    store = BlobStore(tmp_path / "blob")
    _publish_blob(store, _stored_envelope_bytes(scratch))

    context = _context(tmp_path, hook_roots=(("legacy-hook-spool-0", spool_root),))
    plan = compile_disposition_plan(
        archive_root=tmp_path, blob_root=store.root, source_db=tmp_path / "source.db", context=context
    )

    (member,) = plan.members
    assert member.disposition is BlobDisposition.RESTORE_REQUIRED
    assert member.restoration is not None
    assert member.restoration.destination is RestorationDestination.HOOK_EVENT_SPOOL
    assert member.restoration.logical_id == "orphan-event"


def test_an_acknowledged_only_envelope_is_not_a_source_proof(tmp_path: Path) -> None:
    """Anti-vacuity: walking the whole spool root makes this SOURCE_PRESENT and deletes it.

    ``acknowledged/`` holds the drain's commit receipts for the source.db that
    consumed each event. No drain and no watcher reads that directory, so a
    rebuilt archive re-ingests nothing from it: the bytes exist, the event is
    unreachable, and the carrier is the only copy acquisition can still use.
    """
    spool_root = tmp_path / "legacy-hooks"
    envelope = _hook_envelope("acknowledged-only")
    receipt = spool_root / "acknowledged" / "2026-07-15"
    receipt.mkdir(parents=True)
    spool_file = receipt / "acknowledged-only.json"
    spool_file.write_text(json.dumps(envelope, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    store = BlobStore(tmp_path / "blob")
    _publish_blob(store, _stored_envelope_bytes(spool_file))

    context = _context(tmp_path, hook_roots=(("legacy-hook-spool-0", spool_root),))
    plan = compile_disposition_plan(
        archive_root=tmp_path, blob_root=store.root, source_db=tmp_path / "source.db", context=context
    )

    (member,) = plan.members
    assert member.disposition is BlobDisposition.RESTORE_REQUIRED
    assert member.proof is None
    assert member.restoration is not None
    assert member.restoration.destination is RestorationDestination.HOOK_EVENT_SPOOL
    assert member.restoration.logical_id == "acknowledged-only"


def test_same_event_id_with_different_content_is_not_a_source_proof(tmp_path: Path) -> None:
    """Anti-vacuity: matching on identity alone would discard divergent material."""
    spool_root = tmp_path / "legacy-hooks"
    _write_spool_file(spool_root, _hook_envelope(text="a completely different tool call"))
    scratch = tmp_path / "scratch.json"
    scratch.write_text(json.dumps(_hook_envelope(text="the stored call"), sort_keys=True), encoding="utf-8")
    store = BlobStore(tmp_path / "blob")
    _publish_blob(store, _stored_envelope_bytes(scratch))

    context = _context(tmp_path, hook_roots=(("legacy-hook-spool-0", spool_root),))
    plan = compile_disposition_plan(
        archive_root=tmp_path, blob_root=store.root, source_db=tmp_path / "source.db", context=context
    )

    (member,) = plan.members
    assert member.disposition is BlobDisposition.RESTORE_REQUIRED


def test_unclassifiable_material_is_unresolved_and_blocks_acceptance(tmp_path: Path) -> None:
    """Anti-vacuity: routing unknown material to discard makes this green wrongly."""
    store = BlobStore(tmp_path / "blob")
    mystery = _publish_blob(store, b"%PDF-1.5\nnot a session and not an envelope\n")

    context = _context(tmp_path, referenced=(mystery,))
    plan = compile_disposition_plan(
        archive_root=tmp_path, blob_root=store.root, source_db=tmp_path / "source.db", context=context
    )

    (member,) = plan.members
    assert member.disposition is BlobDisposition.UNRESOLVED
    assert plan.unresolved_count == 1
    assert not plan.accepted


def test_source_file_proof_requires_a_fresh_hash_not_path_existence(tmp_path: Path) -> None:
    """Anti-vacuity: proving by path existence accepts a rewritten source."""
    source = tmp_path / "session.jsonl"
    source.write_text('{"a": 1}\n', encoding="utf-8")
    store = BlobStore(tmp_path / "blob")
    blob_hash = _publish_blob(store, source.read_bytes())

    prover = RawSourceFileProver({blob_hash: (RawSourceCarrier(str(source)),)})
    proof = prover.prove(blob_hash, store.blob_path(blob_hash), store.blob_path(blob_hash).stat().st_size)
    assert proof is not None and proof.mode is SourceProofMode.BYTE_IDENTICAL

    source.write_text('{"a": 2}\n', encoding="utf-8")
    assert prover.prove(blob_hash, store.blob_path(blob_hash), store.blob_path(blob_hash).stat().st_size) is None


def test_source_file_proof_accepts_an_exact_append_prefix(tmp_path: Path) -> None:
    """Anti-vacuity: requiring whole-file equality would restore every append source."""
    store = BlobStore(tmp_path / "blob")
    blob_hash = _publish_blob(store, b'{"a": 1}\n')
    source = tmp_path / "session.jsonl"
    source.write_bytes(b'{"a": 1}\n{"a": 2}\n')

    prover = RawSourceFileProver({blob_hash: (RawSourceCarrier(str(source)),)})
    proof = prover.prove(blob_hash, store.blob_path(blob_hash), 9)
    assert proof is not None and proof.mode is SourceProofMode.STRICT_PREFIX

    source.write_bytes(b'{"z": 9}\n{"a": 2}\n')
    assert prover.prove(blob_hash, store.blob_path(blob_hash), 9) is None


def test_append_prefix_only_supersedes_within_one_logical_item(tmp_path: Path) -> None:
    """Anti-vacuity: an unscoped prefix search discards unrelated carriers."""
    store = BlobStore(tmp_path / "blob")
    short = _publish_blob(store, b'{"a": 1}\n')
    long = _publish_blob(store, b'{"a": 1}\n{"a": 2}\n')

    related = AppendPrefixProver({short: (long,)}, blob_store=store)
    assert related.prove(short, store.blob_path(short), 9) is not None

    unrelated = AppendPrefixProver({}, blob_store=store)
    assert unrelated.prove(short, store.blob_path(short), 9) is None


def test_append_successors_group_by_logical_identity(tmp_path: Path) -> None:
    db = _empty_source_db(tmp_path / "source.db")
    with sqlite3.connect(db) as conn:
        conn.executemany(
            "INSERT INTO raw_sessions (raw_id, origin, native_id, blob_hash, blob_size, source_path, "
            "append_start_offset) VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                ("r1", "claude-code-session", "s1", bytes.fromhex("aa" * 32), 10, "/tmp/a", None),
                ("r2", "claude-code-session", "s1", bytes.fromhex("bb" * 32), 20, "/tmp/a", None),
                ("r3", "claude-code-session", "s2", bytes.fromhex("cc" * 32), 30, "/tmp/b", None),
            ],
        )
    successors = append_successors_by_hash(db)
    assert successors == {"aa" * 32: ("bb" * 32,)}
    assert raw_source_carriers_by_hash(db)["aa" * 32] == (RawSourceCarrier("/tmp/a", None, "claude-code-session"),)


def test_reference_union_covers_every_durable_relation(tmp_path: Path) -> None:
    """Anti-vacuity: omitting one relation reports its blobs as unreferenced."""
    db = tmp_path / "source.db"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE blob_refs (blob_hash BLOB)")
        conn.execute("CREATE TABLE raw_sessions (blob_hash BLOB)")
        conn.execute("CREATE TABLE raw_hook_events (blob_hash BLOB)")
        conn.execute("CREATE TABLE raw_artifacts (blob_hash BLOB)")
        conn.execute("CREATE TABLE blob_publication_reservations (blob_hash BLOB)")
        for index, table in enumerate(
            ("blob_refs", "raw_sessions", "raw_hook_events", "raw_artifacts", "blob_publication_reservations")
        ):
            conn.execute(f"INSERT INTO {table} (blob_hash) VALUES (?)", (bytes([index]) * 32,))

    hashes = referenced_blob_hashes(db)
    assert hashes == {bytes([index] * 32).hex() for index in range(5)}


def test_unreadable_reference_relation_fails_instead_of_reporting_zero(tmp_path: Path) -> None:
    """Anti-vacuity: swallowing the error would license deleting the namespace."""
    db = tmp_path / "source.db"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE blob_refs (blob_hash BLOB)")
        conn.execute("CREATE VIEW raw_sessions AS SELECT blob_hash FROM missing_table")

    with pytest.raises(BlobDispositionError):
        referenced_blob_hashes(db)


def test_plan_digest_binds_denominator_and_every_member(tmp_path: Path) -> None:
    """Anti-vacuity: a digest over counts alone lets a member be swapped."""
    store = BlobStore(tmp_path / "blob")
    _publish_blob(store, b"%PDF-1.5\nunexplained\n")
    context = _context(tmp_path)
    plan = compile_disposition_plan(
        archive_root=tmp_path, blob_root=store.root, source_db=tmp_path / "source.db", context=context
    )

    reloaded = BlobDispositionPlan.from_dict(json.loads(json.dumps(plan.to_dict())))
    assert reloaded.digest() == plan.digest()

    mutated = BlobDispositionPlan.from_dict(
        {
            **plan.to_dict(),
            "members": [{**plan.members[0].to_dict(), "disposition": BlobDisposition.SOURCE_PRESENT.value}],
        }
    )
    assert mutated.digest() != plan.digest()


def test_invalid_namespace_entries_block_acceptance(tmp_path: Path) -> None:
    """Anti-vacuity: ignoring stray namespace entries hides unaccounted files."""
    blob_root = tmp_path / "blob"
    (blob_root / "not-a-shard").mkdir(parents=True)
    (blob_root / "not-a-shard" / "stray").write_bytes(b"x")
    context = _context(tmp_path)
    plan = compile_disposition_plan(
        archive_root=tmp_path, blob_root=blob_root, source_db=tmp_path / "source.db", context=context
    )

    assert plan.denominator.invalid_namespace_entries
    assert not plan.accepted


def test_denominator_counts_the_complete_population(tmp_path: Path) -> None:
    """Anti-vacuity: a sampled census would not reconcile against the walk."""
    spool_root = tmp_path / "legacy-hooks"
    spool_file = _write_spool_file(spool_root, _hook_envelope("counted"))
    store = BlobStore(tmp_path / "blob")
    _publish_blob(store, _stored_envelope_bytes(spool_file))
    mystery = _publish_blob(store, b"%PDF-1.5\nunexplained\n")

    context = _context(tmp_path, hook_roots=(("legacy-hook-spool-0", spool_root),), referenced=(mystery,))
    plan = compile_disposition_plan(
        archive_root=tmp_path, blob_root=store.root, source_db=tmp_path / "source.db", context=context
    )

    assert plan.denominator.physical_file_count == 2
    assert plan.denominator.distinct_hash_count == 2
    assert sum(plan.counts.values()) == 2
    assert plan.counts[BlobDisposition.SOURCE_PRESENT.value] == 1
    assert plan.counts[BlobDisposition.UNRESOLVED.value] == 1


def test_source_file_proof_accepts_the_recorded_append_span(tmp_path: Path) -> None:
    """Anti-vacuity: without the recorded span, every increment-only row restores.

    An append-structured acquisition stores just its own increment, so the
    object is neither the file nor the file's prefix; only ``file[start:]``
    reproduces it.
    """
    store = BlobStore(tmp_path / "blob")
    increment = b'{"a": 2}\n'
    blob_hash = _publish_blob(store, increment)
    source = tmp_path / "session.jsonl"
    source.write_bytes(b'{"a": 1}\n' + increment)

    without_span = RawSourceFileProver({blob_hash: (RawSourceCarrier(str(source)),)})
    assert without_span.prove(blob_hash, store.blob_path(blob_hash), len(increment)) is None

    with_span = RawSourceFileProver({blob_hash: (RawSourceCarrier(str(source), 9),)})
    proof = with_span.prove(blob_hash, store.blob_path(blob_hash), len(increment))
    assert proof is not None and proof.mode is SourceProofMode.STRICT_PREFIX


def test_reclaimable_total_excludes_members_a_reference_pins(tmp_path: Path) -> None:
    """Anti-vacuity: totalling every source_present member overstates apply's effect.

    Apply deletes only unreferenced members, so a proven object whose hash a
    durable row still names must not be counted as reclaimable. Summing
    ``bytes_by_disposition['source_present']`` instead makes this red.
    """
    spool_root = tmp_path / "legacy-hooks"
    free = _hook_envelope("free")
    pinned = _hook_envelope("pinned", text="a much longer detail string to separate the byte totals")
    free_file = _write_spool_file(spool_root, free, indent=4)
    pinned_file = _write_spool_file(spool_root, pinned, indent=4)
    store = BlobStore(tmp_path / "blob")
    free_hash = _publish_blob(store, _stored_envelope_bytes(free_file))
    pinned_hash = _publish_blob(store, _stored_envelope_bytes(pinned_file))
    source_db = _empty_source_db(tmp_path / "source.db")
    _empty_index_db(tmp_path)
    with sqlite3.connect(source_db) as conn:
        conn.execute("INSERT INTO blob_refs (blob_hash, ref_type) VALUES (?, ?)", (bytes.fromhex(pinned_hash), "raw"))

    context = build_disposition_context(
        archive_root=tmp_path,
        blob_root=store.root,
        source_db=source_db,
        hook_spool_sources=(("legacy-hook-spool-0", spool_root),),
        browser_capture_spool=tmp_path / "browser-capture",
    )
    plan = compile_disposition_plan(archive_root=tmp_path, blob_root=store.root, source_db=source_db, context=context)

    free_size = store.blob_path(free_hash).stat().st_size
    pinned_size = store.blob_path(pinned_hash).stat().st_size
    assert plan.counts["source_present"] == 2
    assert plan.reclaimable_count == 1
    assert plan.reclaimable_bytes == free_size
    assert plan.retained_by_reference_count == 1
    assert plan.retained_by_reference_bytes == pinned_size
    assert plan.bytes_by_disposition["source_present"] == free_size + pinned_size
    payload = plan.to_dict()
    assert payload["reclaimable_bytes"] == free_size
    assert payload["retained_by_reference_bytes"] == pinned_size


def test_superseded_prefix_members_are_reported_as_retained_not_reclaimable(tmp_path: Path) -> None:
    """Anti-vacuity: a raw_sessions-keyed member is always referenced, never deletable.

    ``referenced_blob_hashes`` unions ``raw_sessions.blob_hash``, which is the
    same relation append supersession is derived from, so every
    ``superseded_prefix`` member is pinned by a durable row. Counting it as
    reclaimable promises the operator bytes apply cannot free.
    """
    store = BlobStore(tmp_path / "blob")
    short = _publish_blob(store, b'{"a": 1}\n')
    long = _publish_blob(store, b'{"a": 1}\n{"a": 2}\n')
    source_db = _empty_source_db(tmp_path / "source.db")
    _empty_index_db(tmp_path)
    with sqlite3.connect(source_db) as conn:
        conn.executemany(
            "INSERT INTO raw_sessions (raw_id, origin, native_id, blob_hash, blob_size, source_path, "
            "append_start_offset) VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                ("r1", "claude-code-session", "s1", bytes.fromhex(short), 9, None, None),
                ("r2", "claude-code-session", "s1", bytes.fromhex(long), 18, None, None),
            ],
        )

    context = build_disposition_context(
        archive_root=tmp_path,
        blob_root=store.root,
        source_db=source_db,
        hook_spool_sources=(),
        browser_capture_spool=tmp_path / "browser-capture",
    )
    plan = compile_disposition_plan(archive_root=tmp_path, blob_root=store.root, source_db=source_db, context=context)

    superseded = plan.members_for(BlobDisposition.SUPERSEDED_PREFIX)
    assert [member.blob_hash for member in superseded] == [short]
    assert all(member.referenced for member in superseded)
    assert plan.reclaimable_bytes == 0
    assert plan.retained_by_reference_bytes == 9


def _source_db_with_rows(
    path: Path,
    *,
    raw_sessions: tuple[tuple[str, str, str, str, int | None], ...] = (),
    hook_events: tuple[tuple[str, str], ...] = (),
    blob_refs: tuple[str, ...] = (),
) -> Path:
    """A source tier holding exactly the durable rows a test needs."""
    _empty_index_db(path.parent)
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE blob_refs (blob_hash BLOB, ref_type TEXT)")
        conn.execute(
            "CREATE TABLE raw_sessions (raw_id TEXT, origin TEXT, native_id TEXT, blob_hash BLOB, "
            "blob_size INTEGER, source_path TEXT, append_start_offset INTEGER)"
        )
        conn.execute("CREATE TABLE raw_hook_events (blob_hash BLOB, source_path TEXT)")
        for origin, native_id, blob_hash, source_path, size in raw_sessions:
            conn.execute(
                "INSERT INTO raw_sessions (raw_id, origin, native_id, blob_hash, blob_size, source_path) "
                "VALUES (?, ?, ?, unhex(?), ?, ?)",
                (f"raw-{native_id}", origin, native_id, blob_hash, size, source_path),
            )
        for blob_hash, source_path in hook_events:
            conn.execute(
                "INSERT INTO raw_hook_events (blob_hash, source_path) VALUES (unhex(?), ?)",
                (blob_hash, source_path),
            )
        for blob_hash in blob_refs:
            conn.execute("INSERT INTO blob_refs (blob_hash, ref_type) VALUES (unhex(?), 'raw_payload')", (blob_hash,))
    return path


def _plan_over(tmp_path: Path, *, source_db: Path, export_roots: tuple[Path, ...] = ()) -> BlobDispositionPlan:
    return compile_disposition_plan(
        archive_root=tmp_path,
        blob_root=tmp_path / "blob",
        source_db=source_db,
        hook_spool_sources=(),
        browser_capture_spool=tmp_path / "browser-capture",
        export_archive_roots=export_roots,
    )


def _member(plan: BlobDispositionPlan, blob_hash: str) -> BlobDispositionMember:
    return next(member for member in plan.members if member.blob_hash == blob_hash)


def test_an_object_no_durable_row_names_is_unreferenced_not_unresolved(tmp_path: Path) -> None:
    """A blob is published before its owning row, so a stranded object is GC's.

    Red if the classifier keeps calling an unowned object unresolved: the
    plan can then never be accepted no matter how many provers are added.
    """
    store = BlobStore(tmp_path / "blob")
    blob_hash = _publish_blob(store, b"stranded by an interrupted write")

    plan = _plan_over(tmp_path, source_db=_source_db_with_rows(tmp_path / "source.db"))

    member = _member(plan, blob_hash)
    assert member.disposition is BlobDisposition.UNREFERENCED
    assert member.referenced is False
    assert plan.accepted is True


def test_a_referenced_object_without_a_proof_still_blocks(tmp_path: Path) -> None:
    """Red if the unreferenced outcome is reached without checking liveness."""
    store = BlobStore(tmp_path / "blob")
    blob_hash = _publish_blob(store, b"material a durable row still names")

    plan = _plan_over(tmp_path, source_db=_source_db_with_rows(tmp_path / "source.db", blob_refs=(blob_hash,)))

    member = _member(plan, blob_hash)
    assert member.disposition is BlobDisposition.UNRESOLVED
    assert plan.accepted is False


def test_unreferenced_members_are_never_reclaimed_by_apply(tmp_path: Path) -> None:
    """This plan records GC's ownership; it never takes it over."""
    store = BlobStore(tmp_path / "blob")
    _publish_blob(store, b"stranded by an interrupted write")

    plan = _plan_over(tmp_path, source_db=_source_db_with_rows(tmp_path / "source.db"))

    assert plan.counts[BlobDisposition.UNREFERENCED.value] == 1
    assert plan.reclaimable_count == 0
    assert plan.reclaimable_bytes == 0


def test_a_sqlite_sidecar_beside_its_blob_is_explained(tmp_path: Path) -> None:
    """Reading a stored database in place writes a journal the namespace disowns."""
    store = BlobStore(tmp_path / "blob")
    blob_hash = _publish_blob(store, b"a stored sqlite database")
    sidecar = store.blob_path(blob_hash).with_name(store.blob_path(blob_hash).name + "-wal")
    sidecar.write_bytes(b"journal")

    plan = _plan_over(tmp_path, source_db=_source_db_with_rows(tmp_path / "source.db"))

    entries = plan.denominator.invalid_namespace_entries
    assert [entry.relative_path for entry in entries] == [f"{blob_hash[:2]}/{blob_hash[2:]}-wal"]
    assert entries[0].explained
    assert blob_hash in (entries[0].explanation or "")
    assert plan.denominator.unexplained_namespace_entries == ()
    assert plan.accepted is True


def test_a_sidecar_whose_blob_is_absent_stays_unexplained(tmp_path: Path) -> None:
    """Red if the rule explains by name alone: the base object must be present."""
    blob_root = tmp_path / "blob"
    absent = "d3" + "a" * 62
    shard = blob_root / absent[:2]
    shard.mkdir(parents=True)
    (shard / f"{absent[2:]}-wal").write_bytes(b"journal")

    plan = _plan_over(tmp_path, source_db=_source_db_with_rows(tmp_path / "source.db"))

    assert [entry.explanation for entry in plan.denominator.invalid_namespace_entries] == [None]
    assert plan.accepted is False


def test_an_arbitrary_shard_file_still_blocks_acceptance(tmp_path: Path) -> None:
    """Red if the taxonomy is widened into a blanket waiver for stray files."""
    blob_root = tmp_path / "blob"
    shard = blob_root / "ab"
    shard.mkdir(parents=True)
    (shard / "notes.txt").write_bytes(b"unexplained")

    plan = _plan_over(tmp_path, source_db=_source_db_with_rows(tmp_path / "source.db"))

    assert len(plan.denominator.unexplained_namespace_entries) == 1
    assert plan.accepted is False


_CLAUDE_CODE_FIXTURE = Path(__file__).parents[2] / "fixtures" / "claude-code" / "claude-normalization-main.jsonl"


def _codex_state_db(path: Path, *, title: str = "a thread title") -> Path:
    with sqlite3.connect(path) as conn:
        conn.execute(
            "CREATE TABLE threads (id TEXT, title TEXT, cwd TEXT, created_at_ms INTEGER, "
            "updated_at_ms INTEGER, source TEXT, model TEXT, agent_nickname TEXT, agent_role TEXT, archived INTEGER)"
        )
        conn.execute("CREATE TABLE thread_spawn_edges (parent_thread_id TEXT, child_thread_id TEXT, status TEXT)")
        conn.execute(
            "INSERT INTO threads VALUES ('thread-1', ?, '/realm/project/polylogue', 1, 2, 'cli', 'gpt-5', NULL, NULL, 0)",
            (title,),
        )
        conn.execute("INSERT INTO thread_spawn_edges VALUES ('thread-1', 'thread-2', 'completed')")
    return path


def _codex_logical_export_bytes(state_db: Path) -> bytes:
    """The canonical logical export retained for one state snapshot."""
    from polylogue.sources.sqlite_export import write_logical_export
    from polylogue.sources.sqlite_snapshot import member_export_scope

    output = BytesIO()
    write_logical_export(state_db, output, scope=member_export_scope(state_db), immutable=True)
    return output.getvalue()


def test_codex_state_logical_export_hash_matches_the_production_route(tmp_path: Path) -> None:
    """The prover and acquisition route must agree on the retained export."""
    state_db = _codex_state_db(tmp_path / "state_5.sqlite")

    written = {hashlib.sha256(_codex_logical_export_bytes(state_db)).hexdigest()}

    assert written
    assert codex_state_logical_export_hashes(state_db) == written


def test_a_state_export_is_proven_against_the_live_database(tmp_path: Path) -> None:
    """A retained logical export is proven against the current database."""
    state_db = _codex_state_db(tmp_path / "state_5.sqlite")
    store = BlobStore(tmp_path / "blob")
    hashes = [_publish_blob(store, _codex_logical_export_bytes(state_db))]
    source_db = _source_db_with_rows(
        tmp_path / "source.db",
        hook_events=tuple((blob_hash, str(state_db)) for blob_hash in hashes),
        blob_refs=tuple(hashes),
    )

    plan = _plan_over(tmp_path, source_db=source_db)

    assert len(hashes) == 1
    for blob_hash in hashes:
        member = _member(plan, blob_hash)
        assert member.disposition is BlobDisposition.SOURCE_PRESENT
        assert member.proof is not None
        assert member.proof.prover == "codex-state-evidence"
        assert member.proof.mode is SourceProofMode.SEMANTIC_EQUIVALENT
    assert plan.accepted is True


def test_a_state_export_the_database_no_longer_matches_is_not_proven(tmp_path: Path) -> None:
    """Red if the prover matches on the carrier path instead of logical content."""
    state_db = _codex_state_db(tmp_path / "state_5.sqlite", title="the title at acquisition")
    store = BlobStore(tmp_path / "blob")
    stale = _publish_blob(store, _codex_logical_export_bytes(state_db))
    state_db.unlink()
    _codex_state_db(state_db, title="the title now")
    source_db = _source_db_with_rows(tmp_path / "source.db", hook_events=((stale, str(state_db)),), blob_refs=(stale,))

    plan = _plan_over(tmp_path, source_db=source_db)

    assert _member(plan, stale).disposition is BlobDisposition.UNRESOLVED


def test_hook_event_carriers_are_read_from_the_durable_relation(tmp_path: Path) -> None:
    source_db = _source_db_with_rows(tmp_path / "source.db", hook_events=(("ab" * 32, "/a/state_5.sqlite"),))

    assert hook_event_carriers_by_hash(source_db) == {"ab" * 32: ("/a/state_5.sqlite",)}


def _export_zip(path: Path, members: dict[str, bytes]) -> Path:
    import zipfile

    with zipfile.ZipFile(path, "w") as handle:
        for name, payload in members.items():
            handle.writestr(name, payload)
    return path


def test_an_extracted_attachment_is_proven_against_its_export_member(tmp_path: Path) -> None:
    """An attachment's bytes live inside a zip, where no file walk reaches them."""
    attachment = b"an attachment extracted out of an account export"
    exports = tmp_path / "exports"
    exports.mkdir()
    _export_zip(exports / "claude-ai-data.zip", {"attachments/note.txt": attachment})
    store = BlobStore(tmp_path / "blob")
    blob_hash = _publish_blob(store, attachment)
    source_db = _source_db_with_rows(tmp_path / "source.db", blob_refs=(blob_hash,))

    plan = _plan_over(tmp_path, source_db=source_db, export_roots=(exports,))

    member = _member(plan, blob_hash)
    assert member.disposition is BlobDisposition.SOURCE_PRESENT
    assert member.proof is not None
    assert member.proof.prover == "export-archive-member"
    assert member.proof.source_path.endswith("!attachments/note.txt")


def test_an_export_member_of_the_same_size_is_not_a_proof(tmp_path: Path) -> None:
    """Red if size selection is mistaken for the proof; the hash decides."""
    exports = tmp_path / "exports"
    exports.mkdir()
    _export_zip(exports / "claude-ai-data.zip", {"attachments/other.txt": b"AAAAAAAAAAAA"})
    store = BlobStore(tmp_path / "blob")
    blob_hash = _publish_blob(store, b"BBBBBBBBBBBB")
    source_db = _source_db_with_rows(tmp_path / "source.db", blob_refs=(blob_hash,))

    plan = _plan_over(tmp_path, source_db=source_db, export_roots=(exports,))

    assert _member(plan, blob_hash).disposition is BlobDisposition.UNRESOLVED


def test_export_prover_survives_an_unreadable_archive(tmp_path: Path) -> None:
    exports = tmp_path / "exports"
    exports.mkdir()
    (exports / "truncated.zip").write_bytes(b"PK\x03\x04 not really a zip")

    assert ExportArchiveMemberProver((exports,)).prove("ab" * 32, tmp_path / "absent", 7) is None


def _rewritten_header(source: Path, destination: Path, *, cwd: str = "/a/relocated/checkout") -> Path:
    """The same material behind a rewritten leading record."""
    lines = source.read_bytes().splitlines(keepends=True)
    head = json.loads(lines[0])
    head["cwd"] = cwd
    destination.write_bytes(json.dumps(head).encode("utf-8") + b"\n" + b"".join(lines[1:]))
    return destination


def _carrier_plan(
    tmp_path: Path, *, blob_hash: str, size: int, source_path: Path, native_id: str = "s1"
) -> BlobDispositionPlan:
    source_db = _source_db_with_rows(
        tmp_path / "source.db",
        raw_sessions=(("claude-code-session", native_id, blob_hash, str(source_path), size),),
        blob_refs=(blob_hash,),
    )
    return _plan_over(tmp_path, source_db=source_db)


def test_a_rewritten_carrier_is_proven_by_the_material_not_the_bytes(tmp_path: Path) -> None:
    """A byte-only oracle calls this divergent and restores a redundant copy.

    Red if the prover is narrowed back to byte or prefix equality: the live
    source holds every stored message, only its leading record moved.
    """
    store = BlobStore(tmp_path / "blob")
    blob_hash, size = store.write_from_path(_CLAUDE_CODE_FIXTURE)
    source = _rewritten_header(_CLAUDE_CODE_FIXTURE, tmp_path / "session.jsonl")

    plan = _carrier_plan(tmp_path, blob_hash=blob_hash, size=size, source_path=source)

    member = _member(plan, blob_hash)
    assert member.disposition is BlobDisposition.SOURCE_PRESENT
    assert member.proof is not None
    assert member.proof.prover == "semantic-containment"
    assert member.proof.mode is SourceProofMode.SEMANTIC_EQUIVALENT


def test_a_source_that_lost_stored_material_proves_nothing(tmp_path: Path) -> None:
    """Red if containment is tested in the wrong direction."""
    store = BlobStore(tmp_path / "blob")
    blob_hash, size = store.write_from_path(_CLAUDE_CODE_FIXTURE)
    lines = _CLAUDE_CODE_FIXTURE.read_bytes().splitlines(keepends=True)
    truncated = tmp_path / "session.jsonl"
    truncated.write_bytes(b"".join(lines[:2]))

    plan = _carrier_plan(tmp_path, blob_hash=blob_hash, size=size, source_path=truncated)

    assert _member(plan, blob_hash).disposition is BlobDisposition.UNRESOLVED


def test_a_source_that_grew_past_the_carrier_contains_it(tmp_path: Path) -> None:
    """An append-structured carrier is contained by the file that outgrew it."""
    lines = _CLAUDE_CODE_FIXTURE.read_bytes().splitlines(keepends=True)
    later_turn = json.loads(lines[8])
    later_turn["uuid"] = "main-a3"
    later_turn["parentUuid"] = "main-a2"
    store = BlobStore(tmp_path / "blob")
    blob_hash, size = store.write_from_path(_CLAUDE_CODE_FIXTURE)
    grown = _rewritten_header(_CLAUDE_CODE_FIXTURE, tmp_path / "session.jsonl")
    grown.write_bytes(grown.read_bytes() + json.dumps(later_turn).encode("utf-8") + b"\n")

    plan = _carrier_plan(tmp_path, blob_hash=blob_hash, size=size, source_path=grown)

    member = _member(plan, blob_hash)
    assert member.disposition is BlobDisposition.SOURCE_PRESENT
    assert member.proof is not None
    assert member.proof.mode is SourceProofMode.SEMANTIC_CONTAINED


def test_a_carrier_whose_source_disappeared_is_never_proven(tmp_path: Path) -> None:
    """Red if a recorded path is trusted without reading what is there now."""
    store = BlobStore(tmp_path / "blob")
    blob_hash, size = store.write_from_path(_CLAUDE_CODE_FIXTURE)

    plan = _carrier_plan(tmp_path, blob_hash=blob_hash, size=size, source_path=tmp_path / "removed.jsonl")

    assert _member(plan, blob_hash).disposition is BlobDisposition.UNRESOLVED


def test_one_surviving_carrier_among_duplicates_proves_the_object(tmp_path: Path) -> None:
    """The same payload acquired twice is one object with two recorded sources."""
    store = BlobStore(tmp_path / "blob")
    blob_hash, size = store.write_from_path(_CLAUDE_CODE_FIXTURE)
    survivor = _rewritten_header(_CLAUDE_CODE_FIXTURE, tmp_path / "survivor.jsonl")
    source_db = _source_db_with_rows(
        tmp_path / "source.db",
        raw_sessions=(
            ("claude-code-session", "s1", blob_hash, str(tmp_path / "removed.jsonl"), size),
            ("claude-code-session", "s2", blob_hash, str(survivor), size),
        ),
        blob_refs=(blob_hash,),
    )

    plan = _plan_over(tmp_path, source_db=source_db)

    assert len([member for member in plan.members if member.blob_hash == blob_hash]) == 1
    member = _member(plan, blob_hash)
    assert member.proof is not None
    assert member.proof.source_path == str(survivor)


def test_material_the_admission_route_refuses_proves_nothing(tmp_path: Path) -> None:
    """Red if a blob that yields no session is silently 'equal' to a source.

    Two empty contributions compare equal, so a prover that skips the
    admission check would call every unparsable carrier reacquirable.
    """
    stored = tmp_path / "stored.jsonl"
    stored.write_bytes(b'{"not": "a session"}\n')
    store = BlobStore(tmp_path / "blob")
    blob_hash, size = store.write_from_path(stored)
    source = tmp_path / "session.jsonl"
    source.write_bytes(b'{"also": "not a session"}\n{"nor": "this"}\n')

    plan = _carrier_plan(tmp_path, blob_hash=blob_hash, size=size, source_path=source)

    assert _member(plan, blob_hash).disposition is BlobDisposition.UNRESOLVED
