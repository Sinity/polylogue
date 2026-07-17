"""Archive-backed storage correctness lab scenario."""

from __future__ import annotations

import json
import os
import sqlite3
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from polylogue.archive.message.roles import Role
from polylogue.archive.session.branch_type import BranchType
from polylogue.core.enums import BlockType, Provider
from polylogue.core.errors import DatabaseError
from polylogue.core.outcomes import OutcomeStatus
from polylogue.daemon.convergence_stages import repair_messages_fts_surface
from polylogue.pipeline.ids import session_content_hash
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
from polylogue.storage.blob_gc import MIN_AGE_S, read_gc_history, run_blob_gc_report
from polylogue.storage.blob_publication import ArchiveBlobPublisher
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.fts.fts_lifecycle import FTS_TRIGGER_NAMES, message_fts_readiness_sync
from polylogue.storage.search import search_messages
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.write import read_archive_session_envelope

STORAGE_CORRECTNESS_SCENARIO_NAME = "storage-correctness"
STORAGE_CORRECTNESS_SCOPE_ADJUDICATION = {
    "blob_gc": (
        "The old blob-lease scope was stale: no production writer populated it. "
        "This scenario instead exercises publication reservation and durable "
        "reference survival, the gc_generations age gate, and typed reclaim evidence."
    )
}

_MESSAGE_FTS_TRIGGER_NAMES = tuple(name for name in FTS_TRIGGER_NAMES if name.startswith("messages_fts_"))


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


def _message_fts_trigger_set(conn: sqlite3.Connection) -> tuple[str, ...]:
    rows = conn.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'trigger' AND name IN (?, ?, ?)
        ORDER BY name
        """,
        _MESSAGE_FTS_TRIGGER_NAMES,
    ).fetchall()
    return tuple(str(row[0]) for row in rows)


def _storage_idempotent_reingest_check() -> dict[str, object]:
    session = _parsed_session(
        "storage-idempotent",
        (
            _parsed_message("m0", Role.USER, "idempotent user storage token", 0),
            _parsed_message("m1", Role.ASSISTANT, "idempotent assistant storage token", 1),
        ),
        title="Storage idempotent",
    )
    with _storage_archive_root() as temp_root:
        root = Path(temp_root)
        with ArchiveStore(root) as archive:
            first = archive.write_raw_and_parsed_result(
                session,
                payload=b'{"session":"storage-idempotent","version":1}',
                source_path="/scenario/storage-idempotent.json",
                acquired_at_ms=1_767_000_000_000,
            )
            second = archive.write_raw_and_parsed_result(
                session,
                payload=b'{"session":"storage-idempotent","version":1}',
                source_path="/scenario/storage-idempotent-repeat.json",
                acquired_at_ms=1_767_000_000_001,
            )
        with sqlite3.connect(root / "index.db") as conn:
            conn.row_factory = sqlite3.Row
            session_row = conn.execute(
                "SELECT content_hash, raw_id FROM sessions WHERE session_id = ?",
                (first.session_id,),
            ).fetchone()
            if session_row is None:
                raise AssertionError("idempotent scenario did not persist a session row")
            derived_counts = {
                "sessions": _row_count(conn, "sessions", "session_id = ?", (first.session_id,)),
                "messages": _row_count(conn, "messages", "session_id = ?", (first.session_id,)),
                "blocks": _row_count(conn, "blocks", "session_id = ?", (first.session_id,)),
                "message_fts": _row_count(conn, "messages_fts"),
            }
        with sqlite3.connect(root / "source.db") as source_conn:
            raw_count = _row_count(source_conn, "raw_sessions")
    expected_hash = str(session_content_hash(session))
    stored_hash = session_row["content_hash"]
    stored_hash_hex = stored_hash.hex() if isinstance(stored_hash, bytes) else str(stored_hash)
    if first.content_changed is not True:
        raise AssertionError("first ingest should write the derived session")
    if second.content_changed is not False or second.counts["skipped_sessions"] != 1:
        raise AssertionError(f"repeat ingest should skip unchanged content, got {second.counts}")
    if derived_counts != {"sessions": 1, "messages": 2, "blocks": 2, "message_fts": 2}:
        raise AssertionError(f"repeat ingest changed derived row counts: {derived_counts}")
    if raw_count != 2:
        raise AssertionError(f"raw source rows should retain both ArchiveStore writes, got {raw_count}")
    if stored_hash_hex != expected_hash:
        raise AssertionError("stored content_hash does not match session_content_hash")
    return {
        "first_counts": first.counts,
        "repeat_counts": second.counts,
        "derived_counts": derived_counts,
        "raw_sessions": raw_count,
        "content_hash": stored_hash_hex,
    }


def _storage_fts_trigger_drift_check() -> dict[str, object]:
    session = _parsed_session(
        "storage-fts",
        (_parsed_message("m0", Role.USER, "stable fts repair sentinel", 0),),
        title="Storage FTS",
    )
    with _storage_archive_root() as temp_root:
        root = Path(temp_root)
        with ArchiveStore(root) as archive:
            first = archive.write_raw_and_parsed_result(
                session,
                payload=b'{"session":"storage-fts","version":1}',
                source_path="/scenario/storage-fts.json",
                acquired_at_ms=1_767_000_000_000,
            )
            with sqlite3.connect(root / "index.db") as conn:
                before_readiness = message_fts_readiness_sync(conn)
                before_triggers = _message_fts_trigger_set(conn)
                conn.execute("DROP TRIGGER messages_fts_ai")
                conn.commit()
                drifted_readiness = message_fts_readiness_sync(conn)
                drifted_triggers = _message_fts_trigger_set(conn)
            try:
                search_messages("sentinel", archive_root=root, db_path=root / "index.db")
            except DatabaseError as exc:
                search_failure = str(exc)
            else:
                raise AssertionError("search should fail while a canonical messages_fts trigger is missing")
            repaired = repair_messages_fts_surface(root / "index.db")
            with sqlite3.connect(root / "index.db") as conn:
                after_readiness = message_fts_readiness_sync(conn)
                after_triggers = _message_fts_trigger_set(conn)
                after_rows = _row_count(conn, "messages_fts")
            search_hits = search_messages("sentinel", archive_root=root, db_path=root / "index.db").hits
    expected_triggers = tuple(sorted(_MESSAGE_FTS_TRIGGER_NAMES))
    exact_ready = {
        "exists": True,
        "indexed_rows": 1,
        "total_rows": 1,
        "ready": True,
        "triggers_present": True,
    }
    if first.content_changed is not True:
        raise AssertionError("first FTS scenario ingest should write content")
    if before_readiness != exact_ready:
        raise AssertionError(f"initial FTS readiness was not exact: {before_readiness}")
    if before_triggers != expected_triggers:
        raise AssertionError(f"initial trigger set is not canonical: {before_triggers}")
    if bool(drifted_readiness["ready"]) or bool(drifted_readiness["triggers_present"]):
        raise AssertionError(f"dropped trigger did not fail exact readiness: {drifted_readiness}")
    if "messages_fts_ai" in drifted_triggers or drifted_triggers != ("messages_fts_ad", "messages_fts_au"):
        raise AssertionError(f"dropped trigger setup failed: {drifted_triggers}")
    if repaired is not True:
        raise AssertionError("production messages_fts surface repair returned false")
    if after_readiness != exact_ready:
        raise AssertionError(f"production FTS repair did not restore exact readiness: {after_readiness}")
    if after_triggers != expected_triggers:
        raise AssertionError(f"production FTS repair did not restore all canonical triggers: {after_triggers}")
    if after_rows != 1 or len(search_hits) != 1:
        raise AssertionError(f"FTS repair did not restore searchable row: after={after_rows}, hits={search_hits}")
    return {
        "before_readiness": before_readiness,
        "before_triggers": before_triggers,
        "drifted_readiness": drifted_readiness,
        "drifted_triggers": drifted_triggers,
        "search_failure": search_failure,
        "production_repair": repaired,
        "after_readiness": after_readiness,
        "after_triggers": after_triggers,
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
            parent_id = archive.write_parsed(parent, content_hash=str(session_content_hash(parent)))
            child_id = archive.write_parsed(child, content_hash=str(session_content_hash(child)))
            archive.write_parsed(parent_grown, content_hash=str(session_content_hash(parent_grown)))
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
