"""Laws for the physical blob disposition plan.

Every test names the mutation that makes it red. The plan decides whether an
irreplaceable object is deleted, so the anti-vacuity conditions are all of the
same family: a prover that accepts material current sources do not hold, or a
classifier that converts "unknown" into "discard", must fail here.
"""

from __future__ import annotations

import base64
import hashlib
import json
import sqlite3
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from polylogue.maintenance.blob_disposition import (
    AppendPrefixProver,
    BlobDisposition,
    BlobDispositionContext,
    BlobDispositionError,
    BlobDispositionPlan,
    BrowserCaptureAttachmentResidue,
    ByteSpanSubstringProver,
    CaptureEmbeddedPayloadProver,
    CodexStateRowProver,
    DataLakeFileProver,
    DeclaredSourceRoots,
    DriveCacheInjectedKeyProver,
    ExportBundleMemberProver,
    HookEventCarrier,
    JsonlLineContainmentProver,
    RawSourceCarrier,
    RawSourceFileProver,
    RestorationDestination,
    SidechainTranscriptResidue,
    SourceProofMode,
    SqliteRowContainmentProver,
    TestCorpusFixtureExcluder,
    TrajectoryStepPrefixProver,
    append_successors_by_hash,
    blob_candidate_sizes,
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


def _context(
    tmp_path: Path,
    *,
    hook_roots: tuple[tuple[str, Path], ...] = (),
    declared_roots: DeclaredSourceRoots | None = None,
) -> BlobDispositionContext:
    blob_root = tmp_path / "blob"
    blob_root.mkdir(exist_ok=True)
    source_db = _empty_source_db(tmp_path / "source.db")
    return build_disposition_context(
        archive_root=tmp_path,
        blob_root=blob_root,
        source_db=source_db,
        hook_spool_sources=hook_roots,
        browser_capture_spool=tmp_path / "browser-capture",
        # Declared roots are empty unless a test declares its own: a unit test
        # that fell back to the operator topology would read the machine.
        declared_roots=declared_roots if declared_roots is not None else DeclaredSourceRoots(),
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


# --------------------------------------------------------------------------
# The nine source provers. Each pair is one law and the mutation that breaks
# it: a prover that accepted the mutated input would authorize deleting the
# only carrier of the difference.
# --------------------------------------------------------------------------


def _codex_state_db(path: Path, *, title: str = "a thread title", archived: int = 0) -> Path:
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE threads (id TEXT PRIMARY KEY, title TEXT, cwd TEXT, archived INTEGER, extra TEXT)")
        conn.execute(
            "INSERT INTO threads (id, title, cwd, archived, extra) VALUES (?, ?, ?, ?, ?)",
            ("thread-1", title, "/realm/project/polylogue", archived, "added later"),
        )
    return path


def _codex_carriers(
    source: Path, payload: dict[str, object]
) -> tuple[dict[str, tuple[HookEventCarrier, ...]], str, str]:
    payload_json = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    blob_hash = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
    return {blob_hash: (HookEventCarrier(str(source), "codex_thread_title", payload_json),)}, blob_hash, payload_json


def test_codex_state_row_proves_against_the_live_row(tmp_path: Path) -> None:
    """The row is the source; the rewritten database file is not.

    Anti-vacuity: deleting the thread row, or changing its title, must refuse
    — both are exercised by the sibling test.
    """
    source = _codex_state_db(tmp_path / "state_5.sqlite")
    carriers, blob_hash, payload_json = _codex_carriers(
        source,
        {"thread_id": "thread-1", "title": "a thread title", "cwd": "/realm/project/polylogue", "archived": False},
    )
    store = BlobStore(tmp_path / "blob")
    _publish_blob(store, payload_json.encode("utf-8"))

    proof = CodexStateRowProver(carriers).prove(blob_hash, store.blob_path(blob_hash), len(payload_json))

    assert proof is not None
    assert proof.mode is SourceProofMode.SEMANTIC_EQUIVALENT
    assert proof.source_path == f"{source}::threads"


@pytest.mark.parametrize("mutation", ["deleted", "retitled"])
def test_codex_state_row_refuses_a_row_the_live_database_no_longer_holds(tmp_path: Path, mutation: str) -> None:
    """Anti-vacuity for the codex-state-row prover."""
    source = _codex_state_db(tmp_path / "state_5.sqlite")
    carriers, blob_hash, payload_json = _codex_carriers(
        source,
        {"thread_id": "thread-1", "title": "a thread title", "cwd": "/realm/project/polylogue", "archived": False},
    )
    with sqlite3.connect(source) as conn:
        if mutation == "deleted":
            conn.execute("DELETE FROM threads WHERE id = 'thread-1'")
        else:
            conn.execute("UPDATE threads SET title = 'something else' WHERE id = 'thread-1'")
    store = BlobStore(tmp_path / "blob")
    _publish_blob(store, payload_json.encode("utf-8"))

    assert CodexStateRowProver(carriers).prove(blob_hash, store.blob_path(blob_hash), len(payload_json)) is None


def _bundle(root: Path, document: object, *, name: str = "export.zip") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    bundle = root / name
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr("conversations.json", json.dumps(document, ensure_ascii=False))
    return bundle


def test_export_bundle_member_proves_an_extracted_attachment(tmp_path: Path) -> None:
    """An extracted sub-object is reproduced by re-reading the bundle."""
    extracted = "the full text of an attachment"
    root = tmp_path / "bundles"
    _bundle(root, [{"chat_messages": [{"attachments": [{"extracted_content": extracted}]}]}])
    store = BlobStore(tmp_path / "blob")
    blob_hash = _publish_blob(store, extracted.encode("utf-8"))
    sizes = blob_candidate_sizes(store.root)

    prover = ExportBundleMemberProver((root,), candidate_sizes=sizes)
    proof = prover.prove(blob_hash, store.blob_path(blob_hash), len(extracted))

    assert proof is not None
    assert proof.mode is SourceProofMode.BYTE_IDENTICAL


def test_export_bundle_member_refuses_a_truncated_value(tmp_path: Path) -> None:
    """Anti-vacuity: a bundle holding only part of the text does not hold it."""
    extracted = "the full text of an attachment"
    root = tmp_path / "bundles"
    _bundle(root, [{"chat_messages": [{"attachments": [{"extracted_content": extracted[:10]}]}]}])
    store = BlobStore(tmp_path / "blob")
    blob_hash = _publish_blob(store, extracted.encode("utf-8"))

    prover = ExportBundleMemberProver((root,), candidate_sizes=blob_candidate_sizes(store.root))

    assert prover.prove(blob_hash, store.blob_path(blob_hash), len(extracted)) is None


def test_data_lake_file_proves_only_under_a_declared_root(tmp_path: Path) -> None:
    """Anti-vacuity: identical bytes outside every declared root prove nothing.

    Undeclaring the root is exactly the mutation that must turn this red —
    otherwise the plan claims "present at its source" without naming one.
    """
    payload = b"a document that lives in the data lake" * 4
    declared = tmp_path / "declared"
    undeclared = tmp_path / "undeclared"
    for root in (declared, undeclared):
        root.mkdir()
        (root / "document.bin").write_bytes(payload)
    store = BlobStore(tmp_path / "blob")
    blob_hash = _publish_blob(store, payload)
    sizes = blob_candidate_sizes(store.root)

    proven = DataLakeFileProver((declared,), candidate_sizes=sizes).prove(
        blob_hash, store.blob_path(blob_hash), len(payload)
    )
    assert proven is not None
    assert proven.source_path == str(declared / "document.bin")

    assert (
        DataLakeFileProver((), candidate_sizes=sizes).prove(blob_hash, store.blob_path(blob_hash), len(payload)) is None
    )


def _jsonl(*records: dict[str, object]) -> bytes:
    return ("".join(json.dumps(record, sort_keys=True) + "\n" for record in records)).encode("utf-8")


_SESSION = "9629cd27-2cdf-47a9-b893-2ca5bf3b399c"


def _session_file(root: Path, payload: bytes) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{_SESSION}.jsonl"
    path.write_bytes(payload)
    return path


def test_jsonl_line_containment_allows_only_the_rewritten_leading_record(tmp_path: Path) -> None:
    """Codex rewrites line 0 in place; every other line must still be there."""
    snapshot = _jsonl(
        {"type": "session_meta", "sessionId": _SESSION, "at": "first"},
        {"type": "event", "sessionId": _SESSION, "n": 1},
        {"type": "event", "sessionId": _SESSION, "n": 2},
    )
    live = _jsonl(
        {"type": "session_meta", "sessionId": _SESSION, "at": "rewritten"},
        {"type": "event", "sessionId": _SESSION, "n": 1},
        {"type": "event", "sessionId": _SESSION, "n": 2},
        {"type": "event", "sessionId": _SESSION, "n": 3},
    )
    roots = tmp_path / "sessions"
    _session_file(roots, live)
    store = BlobStore(tmp_path / "blob")
    blob_hash = _publish_blob(store, snapshot)

    prover = JsonlLineContainmentProver({}, session_roots=(roots,))
    proof = prover.prove(blob_hash, store.blob_path(blob_hash), len(snapshot))

    assert proof is not None
    assert proof.mode is SourceProofMode.SEMANTIC_EQUIVALENT


def test_jsonl_line_containment_refuses_a_missing_interior_line(tmp_path: Path) -> None:
    """Anti-vacuity: a general subset rule would accept a carrier that dropped content.

    The live file below is missing an interior record, not the leading one.
    """
    snapshot = _jsonl(
        {"type": "session_meta", "sessionId": _SESSION, "at": "first"},
        {"type": "event", "sessionId": _SESSION, "n": 1},
        {"type": "event", "sessionId": _SESSION, "n": 2},
    )
    live = _jsonl(
        {"type": "session_meta", "sessionId": _SESSION, "at": "first"},
        {"type": "event", "sessionId": _SESSION, "n": 2},
    )
    roots = tmp_path / "sessions"
    _session_file(roots, live)
    store = BlobStore(tmp_path / "blob")
    blob_hash = _publish_blob(store, snapshot)

    prover = JsonlLineContainmentProver({}, session_roots=(roots,))

    assert prover.prove(blob_hash, store.blob_path(blob_hash), len(snapshot)) is None


def _trajectory(steps: Sequence[Mapping[str, object]], *, agent: str = "hermes") -> bytes:
    document = {
        "schema_version": 1,
        "session_id": "s-1",
        "trajectory_id": "t-1",
        "agent": agent,
        "steps": list(steps),
        "final_metrics": {"duration": len(steps)},
    }
    return json.dumps(document, sort_keys=True).encode("utf-8")


def test_trajectory_step_prefix_proves_a_truncated_step_list(tmp_path: Path) -> None:
    steps: list[Mapping[str, object]] = [{"n": index} for index in range(5)]
    source = tmp_path / "trajectory-t-1.json"
    source.write_bytes(_trajectory(steps))
    snapshot = _trajectory(steps[:2])
    store = BlobStore(tmp_path / "blob")
    blob_hash = _publish_blob(store, snapshot)

    prover = TrajectoryStepPrefixProver({blob_hash: (RawSourceCarrier(str(source)),)})
    proof = prover.prove(blob_hash, store.blob_path(blob_hash), len(snapshot))

    assert proof is not None
    assert proof.mode is SourceProofMode.STRICT_PREFIX


def test_trajectory_step_prefix_refuses_a_mutated_retained_step(tmp_path: Path) -> None:
    """Anti-vacuity: only truncation is permitted, never a changed step."""
    steps: list[Mapping[str, object]] = [{"n": index} for index in range(5)]
    source = tmp_path / "trajectory-t-1.json"
    source.write_bytes(_trajectory([{"n": 0}, {"n": 99}, *steps[2:]]))
    snapshot = _trajectory(steps[:2])
    store = BlobStore(tmp_path / "blob")
    blob_hash = _publish_blob(store, snapshot)

    prover = TrajectoryStepPrefixProver({blob_hash: (RawSourceCarrier(str(source)),)})

    assert prover.prove(blob_hash, store.blob_path(blob_hash), len(snapshot)) is None


def test_byte_span_substring_proves_a_mid_file_append_span(tmp_path: Path) -> None:
    """The span has no recorded offset, so it is located by search."""
    span = b"the appended span of a capture file"
    source = tmp_path / "capture.json"
    source.write_bytes(b"leading bytes" + span + b"trailing bytes")
    store = BlobStore(tmp_path / "blob")
    blob_hash = _publish_blob(store, span)

    prover = ByteSpanSubstringProver({blob_hash: (RawSourceCarrier(str(source)),)})
    proof = prover.prove(blob_hash, store.blob_path(blob_hash), len(span))

    assert proof is not None
    assert proof.mode is SourceProofMode.STRICT_PREFIX
    assert "offset 13" in proof.detail


def test_byte_span_substring_refuses_a_one_byte_edit_inside_the_span(tmp_path: Path) -> None:
    """Anti-vacuity: near-enough is not a contiguous reproduction."""
    span = b"the appended span of a capture file"
    source = tmp_path / "capture.json"
    source.write_bytes(b"leading bytes" + span.replace(b"capture", b"capturE") + b"trailing bytes")
    store = BlobStore(tmp_path / "blob")
    blob_hash = _publish_blob(store, span)

    prover = ByteSpanSubstringProver({blob_hash: (RawSourceCarrier(str(source)),)})

    assert prover.prove(blob_hash, store.blob_path(blob_hash), len(span)) is None


def _drive_document(*, chunk: str, injected: str | None) -> bytes:
    document: dict[str, object] = {"chunks": [{"id": "c-1", "text": chunk}]}
    if injected is not None:
        document["chunks"] = [{"id": "c-1", "text": chunk, DriveCacheInjectedKeyProver.INJECTED_KEY: injected}]
    return json.dumps(document, sort_keys=True).encode("utf-8")


def test_drive_cache_injected_key_strips_only_the_declared_key(tmp_path: Path) -> None:
    root = tmp_path / "drive-cache"
    root.mkdir()
    (root / "doc.json").write_bytes(_drive_document(chunk="a chunk", injected="aGVsbG8="))
    recorded = _drive_document(chunk="a chunk", injected=None)
    store = BlobStore(tmp_path / "blob")
    blob_hash = _publish_blob(store, recorded)
    carriers = {blob_hash: (RawSourceCarrier(str(tmp_path / "retired" / "doc.json")),)}

    proof = DriveCacheInjectedKeyProver(carriers, roots=(root,)).prove(
        blob_hash, store.blob_path(blob_hash), len(recorded)
    )

    assert proof is not None
    assert proof.mode is SourceProofMode.SEMANTIC_EQUIVALENT


def test_drive_cache_injected_key_refuses_a_changed_chunk(tmp_path: Path) -> None:
    """Anti-vacuity: stripping the injected key is the whole permitted difference."""
    root = tmp_path / "drive-cache"
    root.mkdir()
    (root / "doc.json").write_bytes(_drive_document(chunk="a different chunk", injected="aGVsbG8="))
    recorded = _drive_document(chunk="a chunk", injected=None)
    store = BlobStore(tmp_path / "blob")
    blob_hash = _publish_blob(store, recorded)
    carriers = {blob_hash: (RawSourceCarrier(str(tmp_path / "retired" / "doc.json")),)}

    assert (
        DriveCacheInjectedKeyProver(carriers, roots=(root,)).prove(blob_hash, store.blob_path(blob_hash), len(recorded))
        is None
    )


def test_capture_embedded_payload_proves_a_base64_attachment(tmp_path: Path) -> None:
    payload = b"\x89PNG\r\n\x1a\n an attachment's bytes"
    live = tmp_path / "captures"
    live.mkdir()
    (live / "capture.json").write_text(
        json.dumps(
            {"chunks": [{"driveDocument": {"_polylogue_drive_live_bytes_b64": base64.b64encode(payload).decode()}}]}
        ),
        encoding="utf-8",
    )
    store = BlobStore(tmp_path / "blob")
    blob_hash = _publish_blob(store, payload)

    prover = CaptureEmbeddedPayloadProver((live,), candidate_sizes=blob_candidate_sizes(store.root))
    proof = prover.prove(blob_hash, store.blob_path(blob_hash), len(payload))

    assert proof is not None
    assert proof.mode is SourceProofMode.BYTE_IDENTICAL


def test_capture_embedded_payload_refuses_a_retired_path_and_a_corrupted_encoding(tmp_path: Path) -> None:
    """Anti-vacuity: a payload only under a retired path must not prove.

    The same payload sits in a directory that is not declared, and a declared
    directory holds it with one base64 character changed.
    """
    payload = b"\x89PNG\r\n\x1a\n an attachment's bytes"
    encoded = base64.b64encode(payload).decode()
    retired = tmp_path / "retired"
    declared = tmp_path / "captures"
    for root in (retired, declared):
        root.mkdir()
    (retired / "capture.json").write_text(json.dumps({"data": encoded}), encoding="utf-8")
    corrupted = ("B" if encoded[0] != "B" else "C") + encoded[1:]
    (declared / "capture.json").write_text(json.dumps({"data": corrupted}), encoding="utf-8")
    store = BlobStore(tmp_path / "blob")
    blob_hash = _publish_blob(store, payload)

    prover = CaptureEmbeddedPayloadProver((declared,), candidate_sizes=blob_candidate_sizes(store.root))

    assert prover.prove(blob_hash, store.blob_path(blob_hash), len(payload)) is None


def _state_snapshot(path: Path, *, rows: list[tuple[str, str]], columns: str = "id TEXT, body TEXT") -> Path:
    with sqlite3.connect(path) as conn:
        conn.execute(f"CREATE TABLE stage1_outputs ({columns})")
        conn.executemany("INSERT INTO stage1_outputs (id, body) VALUES (?, ?)", rows)
        conn.execute("CREATE TABLE messages_fts_data (block BLOB)")
        conn.execute("INSERT INTO messages_fts_data (block) VALUES (x'00')")
    return path


def test_sqlite_row_containment_ignores_schema_drift_and_shadow_tables(tmp_path: Path) -> None:
    """Anti-vacuity: a ``SELECT *`` comparison could not pass this at all.

    The live database gained a column and its full-text shadow table differs;
    every real row is still present, which is the proposition being proved.
    """
    snapshot = _state_snapshot(tmp_path / "snapshot.db", rows=[("a", "one"), ("b", "two")])
    live = tmp_path / "live.db"
    with sqlite3.connect(live) as conn:
        conn.execute("CREATE TABLE stage1_outputs (id TEXT, body TEXT, added_later TEXT)")
        conn.executemany(
            "INSERT INTO stage1_outputs (id, body, added_later) VALUES (?, ?, ?)",
            [("a", "one", "x"), ("b", "two", "y"), ("c", "three", "z")],
        )
        conn.execute("CREATE TABLE messages_fts_data (block BLOB)")
    store = BlobStore(tmp_path / "blob")
    blob_hash = _publish_blob(store, snapshot.read_bytes())
    carriers = {blob_hash: (RawSourceCarrier(str(live)),)}

    proof = SqliteRowContainmentProver(carriers).prove(blob_hash, store.blob_path(blob_hash), snapshot.stat().st_size)

    assert proof is not None
    assert proof.mode is SourceProofMode.SEMANTIC_EQUIVALENT


def test_sqlite_row_containment_refuses_a_dropped_row(tmp_path: Path) -> None:
    """Anti-vacuity: a row only the snapshot holds is exactly what must block."""
    snapshot = _state_snapshot(tmp_path / "snapshot.db", rows=[("a", "one"), ("b", "two")])
    live = tmp_path / "live.db"
    _state_snapshot(live, rows=[("a", "one")])
    store = BlobStore(tmp_path / "blob")
    blob_hash = _publish_blob(store, snapshot.read_bytes())
    carriers = {blob_hash: (RawSourceCarrier(str(live)),)}

    assert (
        SqliteRowContainmentProver(carriers).prove(blob_hash, store.blob_path(blob_hash), snapshot.stat().st_size)
        is None
    )


def test_reading_a_blob_as_a_database_leaves_no_sidecar_in_the_namespace(tmp_path: Path) -> None:
    """Anti-vacuity: an ordinary read-write open writes ``-wal``/``-shm`` siblings.

    Those siblings are invalid namespace entries, so a prover that created
    them would invalidate the very plan it contributes to.
    """
    snapshot = _state_snapshot(tmp_path / "snapshot.db", rows=[("a", "one")])
    store = BlobStore(tmp_path / "blob")
    blob_hash = _publish_blob(store, snapshot.read_bytes())
    carriers = {blob_hash: (RawSourceCarrier(str(snapshot)),)}
    before = sorted(path.name for path in store.blob_path(blob_hash).parent.iterdir())

    SqliteRowContainmentProver(carriers).prove(blob_hash, store.blob_path(blob_hash), snapshot.stat().st_size)

    assert sorted(path.name for path in store.blob_path(blob_hash).parent.iterdir()) == before


# --------------------------------------------------------------------------
# Terminal rules: an object that needs no source proof still needs a reason.
# --------------------------------------------------------------------------


def test_acknowledged_only_hook_envelope_is_restore_required(tmp_path: Path) -> None:
    """Anti-vacuity: the whole-root walk proved this object and it was wrong.

    Nothing drains ``acknowledged``; the receipt is addressed to the source
    tier that consumed it. Proving against it would delete the only copy of
    an event a fresh archive never re-ingests.
    """
    spool_root = tmp_path / "legacy-hooks"
    envelope = _hook_envelope("acknowledged-event")
    acknowledged = spool_root / "acknowledged" / "2026-07-15"
    acknowledged.mkdir(parents=True)
    spool_file = acknowledged / f"{envelope['event_id']}.json"
    spool_file.write_text(json.dumps(envelope, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    store = BlobStore(tmp_path / "blob")
    _publish_blob(store, _stored_envelope_bytes(spool_file))

    context = _context(tmp_path, hook_roots=(("legacy-hook-spool-0", spool_root),))
    plan = compile_disposition_plan(
        archive_root=tmp_path, blob_root=store.root, source_db=tmp_path / "source.db", context=context
    )

    (member,) = plan.members
    assert member.disposition is BlobDisposition.RESTORE_REQUIRED
    assert member.restoration is not None
    assert member.restoration.destination is RestorationDestination.HOOK_EVENT_SPOOL


def test_test_corpus_fixture_is_positively_excluded_by_a_rule(tmp_path: Path) -> None:
    """A synthetic fixture carries no durable row, so the rule is its identifiers."""
    corpus = tmp_path / "tests"
    corpus.mkdir()
    (corpus / "test_thing.py").write_text('session = "journal-session"\nspeaker = "journal-user"\n', encoding="utf-8")
    store = BlobStore(tmp_path / "blob")
    fixture = _jsonl({"sessionId": "journal-session", "uuid": "journal-user", "type": "user"})
    blob_hash = _publish_blob(store, fixture)

    excluder = TestCorpusFixtureExcluder((corpus,), referenced_hashes=frozenset(), owner="polylogue-251y8")
    rule = excluder.resolve(blob_hash, store.blob_path(blob_hash), len(fixture))

    assert rule is not None
    assert rule.owner == "polylogue-251y8"
    assert rule.rule == "tracked-test-corpus-literal"


def test_test_corpus_fixture_rule_refuses_product_material(tmp_path: Path) -> None:
    """Anti-vacuity: one identifier the corpus does not declare refuses the object.

    Provider-assigned identifiers are never test constants, so without this
    the rule would launder real sessions into "not product".
    """
    corpus = tmp_path / "tests"
    corpus.mkdir()
    (corpus / "test_thing.py").write_text('session = "journal-session"\n', encoding="utf-8")
    store = BlobStore(tmp_path / "blob")
    material = _jsonl({"sessionId": "journal-session", "uuid": "9629cd27-2cdf-47a9-b893-2ca5bf3b399c"})
    blob_hash = _publish_blob(store, material)

    excluder = TestCorpusFixtureExcluder((corpus,), referenced_hashes=frozenset(), owner="polylogue-251y8")

    assert excluder.resolve(blob_hash, store.blob_path(blob_hash), len(material)) is None


def test_test_corpus_fixture_rule_refuses_a_referenced_object(tmp_path: Path) -> None:
    """Anti-vacuity: a durable row means an acquisition claimed it as product."""
    corpus = tmp_path / "tests"
    corpus.mkdir()
    (corpus / "test_thing.py").write_text('session = "journal-session"\n', encoding="utf-8")
    store = BlobStore(tmp_path / "blob")
    fixture = _jsonl({"sessionId": "journal-session"})
    blob_hash = _publish_blob(store, fixture)

    excluder = TestCorpusFixtureExcluder((corpus,), referenced_hashes=frozenset({blob_hash}), owner="polylogue-251y8")

    assert excluder.resolve(blob_hash, store.blob_path(blob_hash), len(fixture)) is None


def test_standalone_sidechain_transcript_is_owned_residue(tmp_path: Path) -> None:
    """A subagent transcript names no carrier of its own, so it has an owner."""
    store = BlobStore(tmp_path / "blob")
    payload = _jsonl(
        {"isSidechain": True, "type": "user", "message": {"role": "user", "content": "a"}},
        {"isSidechain": True, "type": "assistant", "message": {"role": "assistant", "content": "b"}},
    )
    blob_hash = _publish_blob(store, payload)

    resolver = SidechainTranscriptResidue(referenced_hashes=frozenset(), owner="polylogue-hcm7h")
    rule = resolver.resolve(blob_hash, store.blob_path(blob_hash), len(payload))

    assert rule is not None
    assert rule.owner == "polylogue-hcm7h"


def test_sidechain_residue_refuses_an_ordinary_session(tmp_path: Path) -> None:
    """Anti-vacuity: one non-sidechain record means a provider file carries this.

    Without the check every unproven JSONL object would acquire an owner and
    stop blocking, which is the laundering this vocabulary exists to prevent.
    """
    store = BlobStore(tmp_path / "blob")
    payload = _jsonl(
        {"isSidechain": True, "type": "user", "message": {"role": "user", "content": "a"}},
        {"isSidechain": False, "type": "assistant", "message": {"role": "assistant", "content": "b"}},
    )
    blob_hash = _publish_blob(store, payload)

    resolver = SidechainTranscriptResidue(referenced_hashes=frozenset(), owner="polylogue-hcm7h")

    assert resolver.resolve(blob_hash, store.blob_path(blob_hash), len(payload)) is None


def test_explained_residue_is_separated_from_unexplained_material(tmp_path: Path) -> None:
    """Residue with a named owner does not block; material without one does.

    Anti-vacuity: collapsing the two would let the plan report acceptance
    over objects nobody can account for, which is the whole thing the
    unresolved count exists to prevent.
    """
    store = BlobStore(tmp_path / "blob")
    owned = _publish_blob(store, b"an attachment payload no capture still embeds")
    unknown = _publish_blob(store, b"bytes with no provenance at all")
    context = _context(tmp_path)
    context = BlobDispositionContext(
        blob_store=context.blob_store,
        provers=(),
        referenced_hashes=context.referenced_hashes,
        residue_resolvers=(
            BrowserCaptureAttachmentResidue(
                frozenset({owned}), owner="polylogue-hcm7h", rule="browser-capture-attachment-payload"
            ),
        ),
    )

    plan = compile_disposition_plan(
        archive_root=tmp_path, blob_root=store.root, source_db=tmp_path / "source.db", context=context
    )

    by_hash = {member.blob_hash: member for member in plan.members}
    owned_member = by_hash[owned]
    assert owned_member.disposition is BlobDisposition.EXPLAINED_RESIDUE
    assert owned_member.rule is not None
    assert owned_member.rule.owner == "polylogue-hcm7h"
    assert by_hash[unknown].disposition is BlobDisposition.UNRESOLVED
    assert plan.unresolved_count == 1
    assert not plan.accepted


def test_a_terminal_rule_survives_a_plan_round_trip(tmp_path: Path) -> None:
    """Anti-vacuity: a rule dropped on reload would leave apply without the reason."""
    store = BlobStore(tmp_path / "blob")
    owned = _publish_blob(store, b"an attachment payload no capture still embeds")
    base = _context(tmp_path)
    context = BlobDispositionContext(
        blob_store=base.blob_store,
        provers=(),
        referenced_hashes=base.referenced_hashes,
        residue_resolvers=(
            BrowserCaptureAttachmentResidue(
                frozenset({owned}), owner="polylogue-hcm7h", rule="browser-capture-attachment-payload"
            ),
        ),
    )
    plan = compile_disposition_plan(
        archive_root=tmp_path, blob_root=store.root, source_db=tmp_path / "source.db", context=context
    )

    reloaded = BlobDispositionPlan.from_dict(json.loads(json.dumps(plan.to_dict())))

    assert reloaded.digest() == plan.digest()
    (member,) = reloaded.members
    assert member.rule is not None
    assert member.rule.rule == "browser-capture-attachment-payload"
