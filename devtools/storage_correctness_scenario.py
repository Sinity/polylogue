"""Archive-backed storage correctness lab scenario."""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import time
from collections.abc import Callable
from contextlib import closing
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory

from polylogue.archive.message.roles import Role
from polylogue.archive.session.branch_type import BranchType
from polylogue.core.enums import BlockType, Provider
from polylogue.core.errors import DatabaseError
from polylogue.core.outcomes import OutcomeStatus
from polylogue.core.timestamp_authority import normalize_session_timestamps
from polylogue.daemon.raw_observation_owner import RawObservationConvergenceOwner
from polylogue.daemon.write_coordinator import DaemonWriteCoordinator, DaemonWriteThreadBridge
from polylogue.operations.canonical_archive_ingest import (
    _wait_for_coordinator_idle,
    admit_one_shot_root,
    one_shot_compute_owner,
    scoped_one_shot_archive_owner,
)
from polylogue.pipeline.ids import session_content_hash
from polylogue.sources.acquisition_boundary import bound_profile_identity, bound_source_observation, open_bound_path
from polylogue.sources.dispatch import parse_payload
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
from polylogue.sources.revision_backfill import PreparedRevisionReplayResult
from polylogue.storage.blob_gc import MIN_AGE_S, read_gc_history, run_blob_gc_report
from polylogue.storage.blob_publication import ArchiveBlobPublisher
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.fts.fts_lifecycle import ensure_fts_index_sync, message_fts_readiness_sync
from polylogue.storage.index_generation import ActiveWriterLease
from polylogue.storage.search import search_messages
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.write import (
    prepare_session_write,
    read_archive_session_envelope,
    write_parsed_session_to_archive,
)
from polylogue.storage.sqlite.reference_seal import PreparedIndexMutation
from polylogue.storage.sqlite.write_lease import write_lease

STORAGE_CORRECTNESS_SCENARIO_NAME = "storage-correctness"
STORAGE_CORRECTNESS_SCOPE_ADJUDICATION = {
    "blob_gc": (
        "The old blob-lease scope was stale: no production writer populated it. "
        "This scenario instead exercises publication reservation and durable "
        "reference survival, the gc_generations age gate, and typed reclaim evidence."
    )
}


def _write_index_session(archive: ArchiveStore, session: ParsedSession) -> str:
    """Write scenario data through the shipped canonical index row writer."""
    root = archive.index_db_path.parent
    exclusion = ActiveWriterLease(root)
    exclusion.acquire()
    seal: PreparedIndexMutation | None = None
    try:
        with PreparedIndexMutation(archive.index_db_path, archive_root=root) as seal:
            with seal.original_read_snapshot():
                prepared = prepare_session_write(
                    seal.observer("index"), session, merge_append=False, before_input=seal.before_index_input
                )
            seal.retain_publication_lifetime(exclusion, prepared.close)
            with archive.index_mutation_scope(prepared_seal=seal) as scope:
                return write_parsed_session_to_archive(
                    archive._conn,
                    session,
                    content_hash=prepared.input_content_hash.hex(),
                    prepared_write=prepared,
                    mutation_scope=scope,
                    manage_transaction=False,
                )
    finally:
        if seal is None or not seal.publication_lifetime_bound:
            exclusion.close()


@dataclass(frozen=True, slots=True)
class StorageCorrectnessCheckResult:
    name: str
    passed: bool
    duration_ms: float
    details: dict[str, object]
    error: str | None = None


class StorageCorrectnessResult:
    """Result wrapper for the archive-backed storage-correctness scenario."""

    scenario_name = STORAGE_CORRECTNESS_SCENARIO_NAME

    def __init__(
        self,
        *,
        check_results: list[StorageCorrectnessCheckResult],
        report_dir: Path | None,
    ) -> None:
        self.check_results = check_results
        self.report_dir = report_dir

    @property
    def all_passed(self) -> bool:
        return not self.failed_stages()

    def stage_statuses(self) -> dict[str, OutcomeStatus]:
        return {
            result.name: OutcomeStatus.OK if result.passed else OutcomeStatus.ERROR for result in self.check_results
        }

    def failed_stages(self) -> tuple[str, ...]:
        return tuple(result.name for result in self.check_results if not result.passed)

    def extra_payload(self) -> dict[str, object]:
        return {
            "checks": [
                {
                    "name": check.name,
                    "passed": check.passed,
                    "duration_ms": round(check.duration_ms, 1),
                    "details": check.details,
                    "error": check.error,
                }
                for check in self.check_results
            ],
            "scope_adjudication": STORAGE_CORRECTNESS_SCOPE_ADJUDICATION,
        }


def storage_correctness_scenario_entry() -> dict[str, object]:
    return {
        "name": STORAGE_CORRECTNESS_SCENARIO_NAME,
        "kind": "archive-storage",
        "check_count": len(_STORAGE_CORRECTNESS_CHECKS),
        "scope_adjudication": STORAGE_CORRECTNESS_SCOPE_ADJUDICATION,
    }


def _parsed_message(provider_id: str, role: Role, text: str, position: int) -> ParsedMessage:
    return ParsedMessage(
        provider_message_id=provider_id,
        role=role,
        text=text,
        position=position,
        variant_index=0,
        is_active_path=True,
        blocks=[ParsedContentBlock(type=BlockType.TEXT, text=text)],
    )


def _parsed_session(
    native_id: str,
    messages: tuple[ParsedMessage, ...],
    *,
    title: str,
    parent_native_id: str | None = None,
    branch_type: BranchType | None = None,
) -> ParsedSession:
    return ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id=native_id,
        title=title,
        updated_at="2026-01-01T00:00:00Z",
        parent_session_provider_id=parent_native_id,
        branch_type=branch_type,
        messages=list(messages),
    )


def _storage_archive_root() -> TemporaryDirectory[str]:
    return TemporaryDirectory(prefix="polylogue-storage-correctness-")


def _row_count(conn: sqlite3.Connection, table: str, where: str = "", params: tuple[object, ...] = ()) -> int:
    clause = f" WHERE {where}" if where else ""
    row = conn.execute(f"SELECT COUNT(*) FROM {table}{clause}", params).fetchone()
    return int(row[0] if row is not None else 0)


def _storage_idempotent_reingest_check() -> dict[str, object]:
    with _storage_archive_root() as temp_root:
        root = Path(temp_root)
        acquisitions, expected_hash_hex = asyncio.run(
            _retained_lab_ingest(
                root,
                "storage-idempotent",
                (("user", "idempotent user storage token"), ("assistant", "idempotent assistant storage token")),
                (
                    ("storage-idempotent.jsonl", 1_767_000_000_000),
                    ("storage-idempotent-repeat.jsonl", 1_767_000_000_001),
                ),
            )
        )
        first, second = acquisitions
        first_counts = _retained_lab_counts(first)
        repeat_counts = _retained_lab_counts(second)
        written_ids = {session_id for receipt in first for session_id in receipt.written_session_ids}
        if len(written_ids) != 1:
            raise AssertionError(f"first ingest must write exactly one session, got {written_ids}")
        session_id = next(iter(written_ids))
        with closing(sqlite3.connect(root / "index.db")) as conn:
            conn.row_factory = sqlite3.Row
            session_row = conn.execute(
                "SELECT content_hash,raw_id FROM sessions WHERE session_id=?", (session_id,)
            ).fetchone()
            if session_row is None:
                raise AssertionError("idempotent scenario did not persist a session row")
            derived_counts = {
                "sessions": _row_count(conn, "sessions", "session_id=?", (session_id,)),
                "messages": _row_count(conn, "messages", "session_id=?", (session_id,)),
                "blocks": _row_count(conn, "blocks", "session_id=?", (session_id,)),
                "message_fts": _row_count(conn, "messages_fts"),
            }
        with closing(sqlite3.connect(root / "source.db")) as source_conn:
            raw_count = _row_count(source_conn, "raw_sessions")
    stored_hash = session_row["content_hash"]
    stored_hash_hex = stored_hash.hex() if isinstance(stored_hash, bytes) else str(stored_hash)
    if not any(receipt.writer_changed_raw_ids for receipt in first):
        raise AssertionError("first ingest should write the derived session")
    if any(receipt.writer_changed_raw_ids for receipt in second) or repeat_counts["skipped_sessions"] != 1:
        raise AssertionError(f"repeat ingest should skip unchanged content, got {repeat_counts}")
    if derived_counts != {"sessions": 1, "messages": 2, "blocks": 2, "message_fts": 2}:
        raise AssertionError(f"repeat ingest changed derived row counts: {derived_counts}")
    if raw_count != 2:
        raise AssertionError(f"raw source rows should retain both acquisitions, got {raw_count}")
    if stored_hash_hex != expected_hash_hex:
        raise AssertionError(
            f"stored content hash {stored_hash_hex} is not the canonical session hash {expected_hash_hex}"
        )
    return {
        "first_counts": first_counts,
        "repeat_counts": repeat_counts,
        "derived_counts": derived_counts,
        "raw_sessions": raw_count,
        "content_hash": stored_hash_hex,
    }


def _storage_fts_trigger_drift_check() -> dict[str, object]:
    with _storage_archive_root() as temp_root:
        root = Path(temp_root)
        acquisitions, _ = asyncio.run(
            _retained_lab_ingest(
                root,
                "storage-fts",
                (("user", "stable fts repair sentinel"),),
                (("storage-fts.jsonl", 1_767_000_000_000),),
            )
        )
        first = acquisitions[0]
        with ArchiveStore.open_existing(root, read_only=True):
            with closing(sqlite3.connect(root / "index.db")) as conn:
                conn.execute("DROP TRIGGER messages_fts_ai")
                conn.commit()
                drifted_readiness = message_fts_readiness_sync(conn)
            try:
                search_messages("sentinel", archive_root=root, db_path=root / "index.db")
            except DatabaseError as exc:
                search_failure = str(exc)
            else:
                raise AssertionError("search should fail while a canonical messages_fts trigger is missing")
            # Trigger presence belongs to canonical schema construction. FTS
            # derivation refuses incompatible schema instead of repairing it.
            with closing(sqlite3.connect(root / "index.db")) as conn:
                ensure_fts_index_sync(conn)
                conn.commit()
                after_readiness = message_fts_readiness_sync(conn)
                after_rows = _row_count(conn, "messages_fts")
            search_hits = search_messages("sentinel", archive_root=root, db_path=root / "index.db").hits
    exact_ready = {
        "exists": True,
        "indexed_rows": 1,
        "total_rows": 1,
        "ready": True,
        "triggers_present": True,
    }
    if not any(receipt.writer_changed_raw_ids for receipt in first):
        raise AssertionError("first FTS scenario ingest should write content")
    if bool(drifted_readiness["ready"]) or bool(drifted_readiness["triggers_present"]):
        raise AssertionError(f"dropped trigger did not fail exact readiness: {drifted_readiness}")
    if after_readiness != exact_ready:
        raise AssertionError(f"canonical FTS schema construction did not restore exact readiness: {after_readiness}")
    if after_rows != 1 or len(search_hits) != 1:
        raise AssertionError(
            f"FTS schema construction did not restore searchable row: after={after_rows}, hits={search_hits}"
        )
    return {
        "drifted_readiness": drifted_readiness,
        "search_failure": search_failure,
        "schema_restored": bool(after_readiness["ready"]),
        "after_readiness": after_readiness,
        "after_fts_rows": after_rows,
        "search_hits": [asdict(hit) for hit in search_hits],
    }


def _backdate_blob(store: BlobStore, blob_hash: str, *, age_s: float) -> None:
    blob_path = store.blob_path(blob_hash)
    timestamp = time.time() - age_s
    os.utime(blob_path, (timestamp, timestamp))


def _storage_blob_gc_invariant_check() -> dict[str, object]:
    """Exercise the current reservation/reference/generation GC contract."""
    reserved_payload = b"storage gc publication reservation"
    referenced_payload = b"storage gc durable raw reference"
    age_gated_payload = b"storage gc generation age gate"
    orphan_payload = b"storage gc reclaimable orphan"
    with _storage_archive_root() as temp_root:
        root = Path(temp_root)
        with ArchiveStore(root):
            pass
        publisher = ArchiveBlobPublisher(root / "source.db", root / "blob")
        reserved_hash, reserved_size = publisher.write_from_bytes(reserved_payload)
        referenced_hash, referenced_size = publisher.write_from_bytes(referenced_payload)
        with write_lease("storage.lab.blob.publication", archive_root=root):
            publisher.flush()
        referenced_receipt = publisher.receipt_id(referenced_hash)
        if referenced_receipt is None:
            raise AssertionError("published raw blob did not retain a receipt")
        with ArchiveStore(root) as archive:
            archive.write_raw_blob_ref(
                provider=Provider.CODEX,
                blob_hash_hex=referenced_hash,
                blob_size=referenced_size,
                source_path="/scenario/storage-gc-referenced.json",
                canonical_source_path="/scenario/storage-gc-referenced.json",
                acquired_at_ms=1_767_000_000_000,
                raw_id="storage-gc-referenced",
                blob_publication_receipt_id=referenced_receipt,
            )
        store = BlobStore(root / "blob")
        age_gated_hash, age_gated_size = store.write_from_bytes(age_gated_payload)
        orphan_hash, orphan_size = store.write_from_bytes(orphan_payload)
        now_s = int(time.time())
        generation_age_s = 1_000
        reclaimable_age_s = generation_age_s + 100
        _backdate_blob(store, reserved_hash, age_s=reclaimable_age_s)
        _backdate_blob(store, referenced_hash, age_s=reclaimable_age_s)
        _backdate_blob(store, age_gated_hash, age_s=MIN_AGE_S + 10)
        _backdate_blob(store, orphan_hash, age_s=reclaimable_age_s)
        completed_at_ms = (now_s - generation_age_s) * 1000
        with sqlite3.connect(root / "source.db") as conn:
            conn.execute(
                """
                INSERT INTO gc_generations
                (generation_id, started_at_ms, completed_at_ms, reclaimed_count, reclaimed_bytes)
                VALUES (?, ?, ?, 0, 0)
                """,
                ("storage-gc-age-boundary", completed_at_ms, completed_at_ms),
            )
            reservation_count = _row_count(conn, "blob_publication_reservations")
            conn.commit()
        with write_lease("storage.lab.blob.gc", archive_root=root):
            report = run_blob_gc_report(root / "index.db", root / "blob", max_batch=10)
        history = read_gc_history(root / "index.db", limit=1)
        survivors = {
            "reserved": store.exists(reserved_hash),
            "referenced": store.exists(referenced_hash),
            "generation_young": store.exists(age_gated_hash),
        }
        orphan_survives = store.exists(orphan_hash)
    if generation_age_s <= MIN_AGE_S + 10:
        raise AssertionError("GC age-gate scenario no longer distinguishes the generation boundary")
    if reservation_count != 1:
        raise AssertionError(f"expected one unconsumed publication reservation, got {reservation_count}")
    if report.deleted_count != 1 or report.reclaimed_bytes != orphan_size:
        raise AssertionError(f"GC did not reclaim exactly the orphan: {report.to_dict()}")
    if report.skipped_reserved != 1 or report.skipped_referenced != 1:
        raise AssertionError(f"GC did not preserve reservation/reference candidates: {report.to_dict()}")
    if not all(survivors.values()):
        raise AssertionError(f"GC removed a protected blob: survivors={survivors}, report={report.to_dict()}")
    if orphan_survives:
        raise AssertionError("GC did not reclaim the generation-safe orphan blob")
    if len(history) != 1 or history[0].reclaimed_count != 1 or history[0].reclaimed_bytes != orphan_size:
        raise AssertionError(f"GC did not persist typed reclaim evidence: {history}")
    return {
        "reserved_size": reserved_size,
        "referenced_size": referenced_size,
        "age_gated_size": age_gated_size,
        "orphan_size": orphan_size,
        "reservation_count": reservation_count,
        "gc_report": report.to_dict(),
        "latest_generation": {
            "reclaimed_count": history[0].reclaimed_count,
            "reclaimed_bytes": history[0].reclaimed_bytes,
        },
    }


def _storage_lineage_composition_check() -> dict[str, object]:
    parent = _parsed_session(
        "storage-parent",
        (
            _parsed_message("p0", Role.USER, "hello", 0),
            _parsed_message("p1", Role.ASSISTANT, "hi there", 1),
            _parsed_message("p2", Role.USER, "parent continues alone", 2),
        ),
        title="Storage parent",
    )
    child = _parsed_session(
        "storage-child",
        (
            _parsed_message("c0", Role.USER, "hello", 0),
            _parsed_message("c1", Role.ASSISTANT, "hi there", 1),
            _parsed_message("cx", Role.USER, "child diverges here", 2),
            _parsed_message("cy", Role.ASSISTANT, "child reply", 3),
        ),
        title="Storage child",
        parent_native_id="storage-parent",
        branch_type=BranchType.FORK,
    )
    parent_grown = _parsed_session(
        "storage-parent",
        (
            _parsed_message("p0", Role.USER, "hello", 0),
            _parsed_message("p1", Role.ASSISTANT, "hi there", 1),
            _parsed_message("p2", Role.USER, "parent continues alone", 2),
            _parsed_message("p3", Role.ASSISTANT, "parent grows later", 3),
        ),
        title="Storage parent",
    )
    with _storage_archive_root() as temp_root:
        root = Path(temp_root)
        with ArchiveStore(root) as archive:
            parent_id = _write_index_session(archive, parent)
            child_id = _write_index_session(archive, child)
            _write_index_session(archive, parent_grown)
            archive.commit()
        with sqlite3.connect(root / "index.db") as conn:
            conn.row_factory = sqlite3.Row
            stored_positions = [
                int(row["position"])
                for row in conn.execute(
                    "SELECT position FROM messages WHERE session_id = ? ORDER BY position",
                    (child_id,),
                ).fetchall()
            ]
            link = conn.execute(
                """
                SELECT resolved_dst_session_id, inheritance, branch_point_message_id
                FROM session_links
                WHERE src_session_id = ?
                """,
                (child_id,),
            ).fetchone()
            envelope = read_archive_session_envelope(conn, child_id)
            composed_texts = ["".join(block.text or "" for block in message.blocks) for message in envelope.messages]
    if parent_id != "codex-session:storage-parent":
        raise AssertionError(f"unexpected parent id: {parent_id}")
    if stored_positions != [2, 3]:
        raise AssertionError(f"child should physically store only divergent tail, got {stored_positions}")
    if link is None:
        raise AssertionError("prefix-sharing child did not persist a session_links row")
    if link["resolved_dst_session_id"] != parent_id:
        raise AssertionError(f"lineage link did not resolve to parent: {dict(link)}")
    if link["inheritance"] != "prefix-sharing" or not link["branch_point_message_id"]:
        raise AssertionError(f"lineage link did not capture prefix-sharing branch point: {dict(link)}")
    expected_texts = ["hello", "hi there", "child diverges here", "child reply"]
    if composed_texts != expected_texts:
        raise AssertionError(f"child did not compose the logical transcript: {composed_texts}")
    return {
        "parent_id": parent_id,
        "child_id": child_id,
        "stored_child_positions": stored_positions,
        "lineage": dict(link),
        "composed_texts": composed_texts,
    }


_STORAGE_CORRECTNESS_CHECKS: tuple[tuple[str, Callable[[], dict[str, object]]], ...] = (
    ("idempotent-reingest", _storage_idempotent_reingest_check),
    ("fts-trigger-drift", _storage_fts_trigger_drift_check),
    ("blob-gc-invariant", _storage_blob_gc_invariant_check),
    ("lineage-composition", _storage_lineage_composition_check),
)


def run_storage_correctness(*, report_dir: Path | None) -> StorageCorrectnessResult:
    """Run archive-backed storage correctness checks."""
    results: list[StorageCorrectnessCheckResult] = []
    for name, check in _STORAGE_CORRECTNESS_CHECKS:
        started = time.monotonic()
        try:
            details = check()
            passed = True
            error = None
        except Exception as exc:
            details = {}
            passed = False
            error = f"{type(exc).__name__}: {exc}"
        results.append(
            StorageCorrectnessCheckResult(
                name=name,
                passed=passed,
                duration_ms=(time.monotonic() - started) * 1000,
                details=details,
                error=error,
            )
        )
    result = StorageCorrectnessResult(check_results=results, report_dir=report_dir)
    _write_storage_correctness_report(result)
    return result


def _write_storage_correctness_report(result: StorageCorrectnessResult) -> None:
    if result.report_dir is None:
        return
    result.report_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "scenario": result.scenario_name,
        **result.extra_payload(),
    }
    (result.report_dir / "storage-correctness.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


__all__ = [
    "STORAGE_CORRECTNESS_SCOPE_ADJUDICATION",
    "STORAGE_CORRECTNESS_SCENARIO_NAME",
    "StorageCorrectnessCheckResult",
    "StorageCorrectnessResult",
    "run_storage_correctness",
    "storage_correctness_scenario_entry",
]


async def _retained_lab_ingest(
    root: Path,
    native_id: str,
    messages: tuple[tuple[str, str], ...],
    acquisitions: tuple[tuple[str, int], ...],
) -> tuple[tuple[tuple[PreparedRevisionReplayResult, ...], ...], str]:
    """Retain the actual writer receipts from neutral, canonical Codex inputs."""
    records = [
        {"type": "session_meta", "payload": {"id": native_id, "timestamp": "2026-01-01T00:00:00Z"}},
        *(
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "id": f"m{ordinal}",
                    "role": role,
                    "content": [{"type": "input_text" if role == "user" else "output_text", "text": text}],
                },
            }
            for ordinal, (role, text) in enumerate(messages)
        ),
    ]
    payload = b"\n".join(json.dumps(record).encode() for record in records) + b"\n"
    with scoped_one_shot_archive_owner(root):
        admit_one_shot_root(root)
        async with one_shot_compute_owner(parse_workers=1) as compute:
            coordinator = DaemonWriteCoordinator(archive_root=root)
            owner = RawObservationConvergenceOwner(
                root,
                compute_adapter=compute,
                write_bridge=DaemonWriteThreadBridge(coordinator, asyncio.get_running_loop()),
                write_coordinator=coordinator,
            )
            try:
                expected = await compute.submit(
                    partial(parse_payload, Provider.CODEX, records, native_id), estimated_bytes=len(payload)
                ).wait()
                if len(expected) != 1:
                    raise AssertionError("neutral Codex input must contain exactly one session")
                from polylogue.core.enums import TitleSource
                from polylogue.sources.assembly_codex import CodexAssemblySpec

                expected_session = CodexAssemblySpec().enrich_session(expected[0], {})
                if (
                    expected_session.title != messages[0][1]
                    or expected_session.title_source is not TitleSource.HEURISTIC
                    or expected_session.title_ref != "message:m0"
                ):
                    raise AssertionError("neutral Codex title assembly lost its original prompt evidence")
                expected_hash = str(session_content_hash(normalize_session_timestamps(expected_session)))

                def acquire(
                    original_payload: bytes,
                    path: Path,
                    canonical: str,
                    profile_key: str | None,
                    acquired_at_ms: int,
                    file_mtime_ms: int,
                ) -> str:
                    with ArchiveStore(root) as archive:
                        return archive.write_raw_payload(
                            provider=Provider.CODEX,
                            payload=original_payload,
                            source_path=str(path),
                            canonical_source_path=canonical,
                            captured_profile_key=profile_key,
                            acquired_at_ms=acquired_at_ms,
                            file_mtime_ms=file_mtime_ms,
                        )

                outcomes: list[tuple[PreparedRevisionReplayResult, ...]] = []
                for name, acquired_at_ms in acquisitions:
                    path = root / name
                    path.write_bytes(payload)
                    with open_bound_path(path, None) as original:
                        profile = bound_profile_identity(original)
                        canonical, observation = bound_source_observation(original)
                        original_payload = original.read()
                        if observation is None or canonical is None or original_payload != payload:
                            raise AssertionError("neutral lab source changed during acquisition")
                        raw_id = await coordinator.run_sync(
                            "storage.lab.acquire",
                            partial(
                                acquire,
                                original_payload,
                                path,
                                canonical,
                                profile.key if profile is not None else None,
                                acquired_at_ms,
                                observation[3] // 1_000_000,
                            ),
                        )
                    outcomes.append((await owner.ingest_retained_raw_ids((raw_id,))).require_complete())
                return tuple(outcomes), expected_hash
            finally:
                await _wait_for_coordinator_idle(coordinator)


def _retained_lab_counts(receipts: tuple[PreparedRevisionReplayResult, ...]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for receipt in receipts:
        for name, value in receipt.written_counts.items():
            counts[name] = counts.get(name, 0) + value
    return counts
