"""Laws for the physical blob disposition plan.

Every test names the mutation that makes it red. The plan decides whether an
irreplaceable object is deleted, so the anti-vacuity conditions are all of the
same family: a prover that accepts material current sources do not hold, or a
classifier that converts "unknown" into "discard", must fail here.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from polylogue.maintenance.blob_disposition import (
    AppendPrefixProver,
    BlobDisposition,
    BlobDispositionContext,
    BlobDispositionError,
    BlobDispositionPlan,
    RawSourceCarrier,
    RawSourceFileProver,
    RestorationDestination,
    SourceProofMode,
    append_successors_by_hash,
    build_disposition_context,
    compile_disposition_plan,
    raw_source_carriers_by_hash,
    referenced_blob_hashes,
)
from polylogue.storage.blob_store import BlobStore


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


def _context(tmp_path: Path, *, hook_roots: tuple[tuple[str, Path], ...] = ()) -> BlobDispositionContext:
    blob_root = tmp_path / "blob"
    blob_root.mkdir(exist_ok=True)
    source_db = _empty_source_db(tmp_path / "source.db")
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
    _publish_blob(store, b"%PDF-1.5\nnot a session and not an envelope\n")

    context = _context(tmp_path)
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
    assert raw_source_carriers_by_hash(db)["aa" * 32] == (RawSourceCarrier("/tmp/a"),)


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
    _publish_blob(store, b"%PDF-1.5\nunexplained\n")

    context = _context(tmp_path, hook_roots=(("legacy-hook-spool-0", spool_root),))
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
