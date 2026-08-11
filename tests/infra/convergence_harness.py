"""Real-SQLite fixtures and independent facts for convergence survivor tests.

This module adapts the production archive writers, daemon stages, and ops
ledger. It deliberately owns no alternate convergence state machine.

The harness starts at the production ``ParsedSession`` boundary. Its
deterministic JSON payload gives the raw writer real bytes to retain, but does
not claim provider parser-byte fidelity or inferred-package selection. Those
remain dependencies of the provider parser and corpus-inference lanes; this
property surface must not fake either with a synthetic manifest or wire
support.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, cast

import polylogue.daemon.convergence_stages as convergence_stages
from polylogue.archive.message.roles import Role
from polylogue.core.enums import BlockType, Provider
from polylogue.core.outcomes import OutcomeStatus
from polylogue.daemon.convergence import DaemonConverger, SessionState
from polylogue.daemon.convergence_stages import make_fts_stage, make_insights_stage
from polylogue.maintenance.archive_verification import ArchiveVerificationReport, verify_archive
from polylogue.pipeline.ids import session_content_hash
from polylogue.pipeline.ids import session_id as make_session_id
from polylogue.pipeline.services.ingest_batch._core import _write_session
from polylogue.pipeline.services.ingest_worker import SessionWritePayload
from polylogue.scenarios import WorkloadEnvelopeSpec, partial_convergence_canary_spec
from polylogue.sources.parsers.base import (
    ParsedAttachment,
    ParsedContentBlock,
    ParsedMessage,
    ParsedSession,
    ParsedSessionEvent,
    ParsedSessionRef,
)
from polylogue.storage.blob_publication import ArchiveBlobPublisher, consume_blob_publication_receipt
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.source_write import ArchiveSourceBlobRef, write_source_raw_session
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.write import write_parsed_session_to_archive
from polylogue.storage.sqlite.connection import open_connection
from tests.infra.archive_canonical_snapshot import (
    CanonicalArchiveSnapshot as ArchiveSnapshot,
)
from tests.infra.archive_canonical_snapshot import (
    archive_snapshot,
)
from tests.infra.archive_canonical_snapshot import (
    assert_archives_equivalent as _assert_canonical_archives_equivalent,
)
from tests.infra.pathology_composer import (
    ComposedPathology,
    compose_append_revision_chain,
    compose_fork_prefix_tail_lineage,
    compose_pathologies,
    compose_quarantined_head_arrangement,
)

if TYPE_CHECKING:
    from polylogue.maintenance.rebuild_index import RebuildIndexReceipt

SqlValue = str | int | float | bytes | None
FactRow = tuple[SqlValue, ...]


@dataclass(frozen=True, slots=True)
class PartialConvergenceArchive:
    root: Path
    index_db: Path
    source_db: Path
    ops_db: Path
    target_source: Path
    unrelated_source: Path
    target_session_id: str
    unrelated_session_id: str
    workload_spec: WorkloadEnvelopeSpec

    def make_target_quiet(self) -> None:
        truncate_sparse(self.target_source, 1_024)


@dataclass(frozen=True, slots=True)
class DebtLedgerRow:
    debt_id: str
    stage: str
    subject_type: str
    subject_id: str
    status: str
    attempts: int
    last_error: str | None
    next_retry_at: str | None
    materializer_version: str | None
    created_at_ms: int
    updated_at_ms: int


@dataclass(frozen=True, slots=True)
class SessionMaterializationFacts:
    """Stable terminal facts, excluding attempt-time materialization stamps."""

    profile: FactRow | None
    materializations: tuple[FactRow, ...]
    work_events: tuple[FactRow, ...]
    phases: tuple[FactRow, ...]
    threads: tuple[FactRow, ...]
    thread_sessions: tuple[FactRow, ...]
    table_counts: tuple[tuple[str, int], ...]


@dataclass(frozen=True, slots=True)
class DerivedReadinessSnapshot:
    """Stable production derived-model and archive-readiness projections."""

    derived_models_json: str
    archive_readiness_json: str


@dataclass(frozen=True, slots=True)
class ConvergenceArchive:
    """A temporary archive created only through the raw-and-parsed write route."""

    root: Path
    pathology: ComposedPathology
    source_paths: tuple[Path, ...]
    session_ids: tuple[str, ...]


def rich_convergence_pathology() -> ComposedPathology:
    """Return the bounded default corpus with update, lineage, and orphan semantics.

    The property loop exercises archive-write and convergence laws, not the
    whale streaming route. One-message revisions and a one-message shared
    prefix retain the distinct normalized states needed here while keeping
    each default generated case small enough for the focused managed harness.
    """
    return compose_pathologies(
        compose_append_revision_chain(revision_count=2, messages_per_revision=1),
        compose_fork_prefix_tail_lineage(shared_prefix_len=1, child_tail_len=1),
        compose_quarantined_head_arrangement(),
        name="convergence-property-rich-corpus",
    )


def build_converged_archive(
    root: Path,
    pathology: ComposedPathology,
    *,
    session_order: Sequence[int] | None = None,
    incremental: bool = False,
    append_only: bool = False,
) -> ConvergenceArchive:
    """Materialize a composed corpus through production writes, then converge it."""
    initialize_active_archive(root)
    archive = ingest_convergence_pathology(
        root,
        pathology,
        session_indexes=_complete_session_order(pathology, session_order),
        converge_after_each=incremental,
        append_only=append_only,
    )
    if not incremental:
        converge_convergence_archive(archive)
    assert_archive_verification_green(archive.root)
    return archive


def rebuild_retained_raw_index(archive: ConvergenceArchive | Path) -> RebuildIndexReceipt:
    """Run the production source.db-retained reindex route for this archive."""
    from polylogue.maintenance.rebuild_index import RebuildIndexRequest, rebuild_index_from_source_sync
    from tests.infra.rebuild_receipt import write_valid_rebuild_receipt

    root = archive.root if isinstance(archive, ConvergenceArchive) else archive
    with sqlite3.connect(root / "source.db") as conn:
        raw_session_count = int(conn.execute("SELECT COUNT(*) FROM raw_sessions").fetchone()[0])
    receipt_path = write_valid_rebuild_receipt(
        root,
        root.parent / f"{root.name}-test-schema-inference-receipt.json",
    )
    receipt = rebuild_index_from_source_sync(
        RebuildIndexRequest(
            archive_root=root,
            promote=True,
            raw_batch_size=max(1, raw_session_count),
            schema_inference_receipt_path=receipt_path,
        )
    )
    if receipt.status != "replayed" or not receipt.materialized:
        raise AssertionError(f"retained-raw production reindex did not materialize a generation: {receipt!r}")
    if receipt.selected_raw_count != receipt.raw_session_count or receipt.raw_session_count == 0:
        raise AssertionError(f"retained-raw reindex did not select every source raw row: {receipt!r}")
    if receipt.operation.get("recovery_state") != "promoted":
        raise AssertionError(f"retained-raw reindex did not record promotion recovery state: {receipt!r}")
    return receipt


def initialize_active_archive(root: Path) -> None:
    """Create all archive tiers for a temporary property-test archive."""
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    initialize_active_archive_root(root)


def ingest_convergence_pathology(
    root: Path,
    pathology: ComposedPathology,
    *,
    session_indexes: Sequence[int],
    converge_after_each: bool,
    append_only: bool = False,
) -> ConvergenceArchive:
    """Use the production raw and parsed-session writers for each selected member.

    The test harness does not emulate archive materialization or convergence.
    It writes source.db through the production raw writer and sends the parsed
    payload through ``ingest_batch._core._write_session``, exactly as the live
    ingestion layer does after a provider parser has produced a ``ParsedSession``.
    """
    selected = _validate_session_indexes(pathology, session_indexes)
    source_paths: list[Path] = []
    session_ids: list[str] = []
    for index in selected:
        session = _parsed_session(pathology.sessions[index], corpus_index=index)
        content_hash = str(session_content_hash(session))
        payload = _raw_payload(session)
        source_path = root / "sources" / f"{index:03d}-{session.provider_session_id}.json"
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_bytes(payload)
        raw_blob_publisher = ArchiveBlobPublisher(root / "source.db", root / "blob")
        raw_blob_hash, raw_blob_size = raw_blob_publisher.write_from_bytes(payload)
        preacquired_attachments: list[ParsedAttachment] = []
        attachment_blob_refs: list[ArchiveSourceBlobRef] = []
        attachment_receipts: list[tuple[str, bytes]] = []
        for attachment in session.attachments:
            if attachment.inline_bytes is None:
                preacquired_attachments.append(attachment)
                continue
            attachment_hash, attachment_size = raw_blob_publisher.write_from_bytes(attachment.inline_bytes)
            attachment_receipt = raw_blob_publisher.receipt_id(attachment_hash)
            preacquired_attachments.append(
                attachment.model_copy(
                    update={"inline_bytes": None, "precomputed_blob": (attachment_hash, attachment_size)}
                )
            )
            attachment_blob_refs.append(
                ArchiveSourceBlobRef(
                    blob_hash=bytes.fromhex(attachment_hash),
                    ref_type="attachment",
                    source_path=str(source_path),
                    size_bytes=attachment_size,
                    acquired_at_ms=_acquired_at_ms(index),
                    publication_receipt_id=attachment_receipt,
                )
            )
            if attachment_receipt is not None:
                attachment_receipts.append((attachment_receipt, bytes.fromhex(attachment_hash)))
        session = session.model_copy(update={"attachments": preacquired_attachments})
        raw_blob_publisher.flush()
        with sqlite3.connect(root / "source.db") as source_conn:
            with source_conn:
                raw_id = write_source_raw_session(
                    source_conn,
                    origin="codex-session",
                    capture_mode=Provider.CODEX,
                    source_path=str(source_path),
                    source_index=-1 if append_only else index,
                    payload=payload,
                    acquired_at_ms=_acquired_at_ms(index),
                    native_id=session.provider_session_id,
                    blob_publication_receipt_id=raw_blob_publisher.receipt_id(raw_blob_hash),
                    additional_blob_refs=tuple(attachment_blob_refs),
                    manage_transaction=False,
                )
                consume_blob_publication_receipt(
                    source_conn,
                    raw_blob_publisher.receipt_id(raw_blob_hash),
                    bytes.fromhex(raw_blob_hash),
                )
                for attachment_receipt, attachment_hash_bytes in attachment_receipts:
                    consume_blob_publication_receipt(source_conn, attachment_receipt, attachment_hash_bytes)
        if raw_blob_size != len(payload):
            raise AssertionError(f"published raw payload size drifted for {source_path}")
        payload_model = SessionWritePayload(
            session_id=str(make_session_id(session.source_name, session.provider_session_id)),
            content_hash=content_hash,
            parsed_session=session,
            message_count=len(session.messages),
            attachment_count=len(session.attachments),
            raw_id=raw_id,
            append_only=append_only,
        )
        pending_attachment_receipts: list[tuple[str, bytes]] = []
        blob_publisher = ArchiveBlobPublisher(root / "source.db", root / "blob")
        with open_connection(root / "index.db") as index_conn, sqlite3.connect(root / "source.db") as source_conn:
            changed, counts = _write_session(
                index_conn,
                payload_model,
                blob_publisher=blob_publisher,
                pending_attachment_receipts=pending_attachment_receipts,
                source_conn=source_conn,
            )
            if pending_attachment_receipts:
                source_conn.execute("BEGIN IMMEDIATE")
                for publication_id, blob_hash in pending_attachment_receipts:
                    consume_blob_publication_receipt(source_conn, publication_id, blob_hash)
                source_conn.commit()
        if not changed and counts["skipped_sessions"] == 0:
            raise AssertionError(f"production ingest writer did not account for {payload_model.session_id}")
        session_id = payload_model.session_id
        source_paths.append(source_path)
        session_ids.append(session_id)
        # Some valid provider fixtures contain no text-bearing blocks and
        # therefore have no FTS rows to corrupt. The corpus builder may skip
        # that inapplicable mutation; direct corruption tests remain strict.
        make_messages_fts_stale(root / "index.db", session_id=session_id, require_rows=False)
        archive = ConvergenceArchive(root, pathology, tuple(source_paths), tuple(dict.fromkeys(session_ids)))
        if converge_after_each:
            converge_convergence_archive(archive)

    return ConvergenceArchive(root, pathology, tuple(source_paths), tuple(dict.fromkeys(session_ids)))


def converge_convergence_archive(archive: ConvergenceArchive) -> dict[str, SessionState]:
    """Run the real FTS and insight debt stages for the materialized sessions."""
    with sqlite3.connect(archive.root / "index.db") as conn:
        persisted_session_ids = tuple(
            str(row[0]) for row in conn.execute("SELECT session_id FROM sessions ORDER BY session_id")
        )
    converger = DaemonConverger(
        (make_fts_stage(archive.root / "index.db"), make_insights_stage(archive.root / "index.db"))
    )
    states, _timings = converger.converge_sessions(persisted_session_ids)
    not_converged = {session_id: state.last_error for session_id, state in states.items() if not state.converged}
    if not_converged:
        raise AssertionError(f"production convergence left pending work: {not_converged}")
    # Insights can materialize work-event rows after the FTS stage has run.
    # Refresh the shared freshness ledger only once both real stages complete.
    from polylogue.daemon.fts_startup import record_fts_freshness_snapshot_sync

    with sqlite3.connect(archive.root / "index.db") as conn:
        record_fts_freshness_snapshot_sync(conn)
    _analyze_registry_tables(archive.root / "index.db")
    return states


def assert_archive_verification_green(root: Path) -> ArchiveVerificationReport:
    """Require every currently registered archive verification predicate to be green."""
    report = verify_archive(root)
    non_green = [
        (check.name, check.status.value, check.summary, check.details, check.breakdown)
        for check in report.checks
        if check.status is not OutcomeStatus.OK
    ]
    if non_green:
        raise AssertionError(f"archive verification registry is not green: {non_green}")
    return report


def assert_archives_equivalent(left: ConvergenceArchive, right: ConvergenceArchive) -> None:
    """Compare property archives through the shared canonical comparator."""

    _assert_canonical_archives_equivalent(left, right)


def derived_readiness_snapshot(root: Path) -> DerivedReadinessSnapshot:
    """Capture production derived status and exact archive readiness facts.

    The status/readiness helpers are read-only projections. Path-bearing
    evidence is normalized so two otherwise equivalent temporary archives are
    compared by facts rather than their unrelated temp-directory names.
    """
    from polylogue.storage.archive_readiness import archive_readiness_status
    from polylogue.storage.derived.derived_status import collect_derived_model_statuses_sync

    with sqlite3.connect(root / "index.db") as conn:
        derived_models = {name: status.to_dict() for name, status in collect_derived_model_statuses_sync(conn).items()}
    readiness = archive_readiness_status(root)
    return DerivedReadinessSnapshot(
        derived_models_json=_stable_json(derived_models, root),
        archive_readiness_json=_stable_json(readiness, root),
    )


def assert_derived_readiness_equivalent(left: Path, right: Path) -> None:
    """Require derived model snapshots and readiness projections to agree."""
    left_snapshot = derived_readiness_snapshot(left)
    right_snapshot = derived_readiness_snapshot(right)
    from polylogue.storage.archive_readiness import archive_readiness_status
    from polylogue.storage.derived.derived_status import collect_derived_model_statuses_sync

    required_insight_models = frozenset(
        {
            "session_profile_rows",
            "session_work_events",
            "session_phases",
            "threads",
            "session_tag_rollups",
        }
    )
    for root in (left, right):
        with sqlite3.connect(root / "index.db") as conn:
            derived_models = collect_derived_model_statuses_sync(conn)
        missing_models = required_insight_models.difference(derived_models)
        unready_models = sorted(
            name for name in required_insight_models if name in derived_models and not derived_models[name].ready
        )
        if missing_models or unready_models:
            raise AssertionError(
                f"primary insight readiness is incomplete for {root}: "
                f"missing={sorted(missing_models)}, unready={unready_models}"
            )
        # The status projection also reports secondary work-event FTS and
        # retrieval surfaces. They remain in the equality snapshot, as does
        # the production messages_fts status. The two-stage route owns
        # messages-FTS repair for changed sessions, while the neutral parser
        # fixture can expose archive-wide excess rows from provider-derived
        # blocks. Keep that production readiness signal in the equality law
        # instead of asserting a global repair this route does not promise.
        readiness = archive_readiness_status(root)
        if readiness.get("checked") is not True or readiness.get("blocked_surface_count") != 0:
            raise AssertionError(f"archive readiness is incomplete for {root}: {readiness!r}")
    if left_snapshot != right_snapshot:
        raise AssertionError(
            f"derived model/readiness snapshots differ:\nleft={left_snapshot!r}\nright={right_snapshot!r}"
        )


def _complete_session_order(pathology: ComposedPathology, order: Sequence[int] | None) -> tuple[int, ...]:
    expected = tuple(range(len(pathology.sessions)))
    candidate = expected if order is None else tuple(order)
    if len(candidate) != len(expected) or set(candidate) != set(expected):
        raise ValueError("session_order must be a permutation of every composed session index")
    return candidate


def rotated_session_order(pathology: ComposedPathology, shift: int) -> tuple[int, ...]:
    """Return one generated, non-identity ordering of the complete corpus."""
    session_count = len(pathology.sessions)
    if not 0 < shift < session_count:
        raise ValueError("shift must select a non-identity rotation")
    indexes = tuple(range(session_count))
    return indexes[shift:] + indexes[:shift]


def replay_convergence_archive(
    root: Path,
    pathology: ComposedPathology,
    *,
    session_indexes: Sequence[int],
    append_only: bool = False,
) -> ConvergenceArchive:
    """Build a fresh canonical archive from the exact writes seen so far."""
    initialize_active_archive(root)
    source_paths: list[Path] = []
    session_ids: list[str] = []
    for index in session_indexes:
        step = ingest_convergence_pathology(
            root,
            pathology,
            session_indexes=(index,),
            converge_after_each=False,
            append_only=append_only,
        )
        source_paths.extend(step.source_paths)
        session_ids.extend(step.session_ids)
    if not session_ids:
        raise ValueError("replay_convergence_archive requires at least one session index")
    archive = ConvergenceArchive(
        root,
        pathology,
        tuple(source_paths),
        tuple(dict.fromkeys(session_ids)),
    )
    converge_convergence_archive(archive)
    assert_archive_verification_green(root)
    return archive


def _validate_session_indexes(pathology: ComposedPathology, indexes: Sequence[int]) -> tuple[int, ...]:
    selected = tuple(indexes)
    if len(selected) != len(set(selected)) or any(index < 0 or index >= len(pathology.sessions) for index in selected):
        raise ValueError("session_indexes must be distinct valid composed-session indexes")
    return selected


def _parsed_session(session: object, *, corpus_index: int) -> ParsedSession:
    from polylogue.archive.models import Session

    if not isinstance(session, Session):
        raise TypeError(f"expected composed Session, got {type(session)!r}")
    timestamp = _corpus_timestamp(corpus_index)
    messages: list[ParsedMessage] = []
    dispatch_tool_id = f"dispatch-{session.id}"
    for position, message in enumerate(session.messages):
        blocks = [ParsedContentBlock(type=BlockType.TEXT, text=message.text)]
        if position == 0:
            blocks.append(
                ParsedContentBlock(
                    type=BlockType.TOOL_USE,
                    tool_name="Task",
                    tool_id=dispatch_tool_id,
                    tool_input={"prompt": message.text, "model": "gpt-4.1-mini"},
                )
            )
        messages.append(
            ParsedMessage(
                provider_message_id=str(message.id),
                role=Role.normalize(str(message.role)),
                text=message.text,
                position=position,
                timestamp=timestamp,
                model_name="gpt-4.1-mini",
                input_tokens=10 + corpus_index + position,
                output_tokens=5 + position,
                cache_read_tokens=2,
                blocks=blocks,
            )
        )
    attachment_message_id = messages[0].provider_message_id
    usage_total = 15 + corpus_index + len(messages)
    return ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id=str(session.id),
        title=session.title,
        created_at=timestamp,
        updated_at=timestamp,
        parent_session_provider_id=None if session.parent_id is None else str(session.parent_id),
        parent_tool_use_provider_id=None if session.parent_id is None else f"dispatch-{session.parent_id}",
        branch_type=session.branch_type,
        messages=messages,
        attachments=[
            ParsedAttachment(
                provider_attachment_id=f"attachment-{session.id}",
                message_provider_id=attachment_message_id,
                name=f"fixture-{session.id}.txt",
                mime_type="text/plain",
                size_bytes=len(f"fixture attachment bytes {session.id}"),
                upload_origin="url",
                source_url=f"https://example.test/{session.id}",
                caption=f"fixture attachment {session.id}",
                inline_bytes=f"fixture attachment bytes {session.id}".encode(),
            )
        ],
        session_events=[
            ParsedSessionEvent(
                event_type="turn_context",
                timestamp=timestamp,
                source_message_provider_id=attachment_message_id,
                payload={"fixture": corpus_index, "revision": str(session.id)},
            ),
            ParsedSessionEvent(
                event_type="token_count",
                timestamp=timestamp,
                source_message_provider_id=attachment_message_id,
                payload={
                    "model": "gpt-4.1-mini",
                    "last_token_usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
                    "total_token_usage": {
                        "input_tokens": usage_total,
                        "output_tokens": usage_total // 2,
                        "total_tokens": usage_total + usage_total // 2,
                    },
                },
            ),
        ],
        models_used=["gpt-4.1-mini"],
        working_directories=[str(Path.cwd())],
        git_branch="feature/test/convergence-property-current",
        git_repository_url="https://github.com/Sinity/polylogue.git",
        git_commit_hash="0123456789abcdef0123456789abcdef01234567",
        ingest_flags=[f"property:fixture-{corpus_index % 2}"],
        session_refs=[
            ParsedSessionRef(
                kind="pull_request",
                repo="Sinity/polylogue",
                number=3767,
                url="https://github.com/Sinity/polylogue/pull/3767",
            )
        ],
    )


def _raw_payload(session: ParsedSession) -> bytes:
    return json.dumps(session.model_dump(mode="json"), sort_keys=True, separators=(",", ":")).encode()


def _corpus_timestamp(index: int) -> str:
    return (datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=index)).isoformat()


def _acquired_at_ms(index: int) -> int:
    return 1_767_225_600_000 + index


def _analyze_registry_tables(index_db: Path) -> None:
    with open_connection(index_db) as conn:
        for table in ("blocks", "messages", "action_pairs"):
            conn.execute(f"ANALYZE {table}")
        conn.commit()


def seed_partial_convergence_archive(root: Path, *, target_hot: bool) -> PartialConvergenceArchive:
    """Seed the current partial-convergence workload through typed archive writes."""
    root.mkdir(parents=True, exist_ok=True)
    index_db = root / "index.db"
    source_db = root / "source.db"
    ops_db = root / "ops.db"
    target_source = root / "profile-growing-codex.jsonl"
    unrelated_source = root / "unrelated-codex.jsonl"
    target_size = convergence_stages._HOT_INSIGHT_SOURCE_BYTES + 1 if target_hot else 1_024
    truncate_sparse(target_source, target_size)
    truncate_sparse(unrelated_source, 1_024)

    with open_connection(index_db) as conn:
        target_session_id = _seed_raw_source_session(
            conn,
            session_id="convergence-survivor",
            source_path=target_source,
        )
        unrelated_session_id = _seed_raw_source_session(
            conn,
            session_id="convergence-unrelated",
            source_path=unrelated_source,
        )
        conn.commit()

    return PartialConvergenceArchive(
        root=root,
        index_db=index_db,
        source_db=source_db,
        ops_db=ops_db,
        target_source=target_source,
        unrelated_source=unrelated_source,
        target_session_id=target_session_id,
        unrelated_session_id=unrelated_session_id,
        workload_spec=partial_convergence_canary_spec(
            profile_id="workload-profile:testdiet-02-partial-convergence",
            archive_id="archive:testdiet-02-partial-convergence",
        ),
    )


def truncate_sparse(path: Path, size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.truncate(size)


def debt_ledger_row(
    ops_db: Path,
    *,
    stage: str,
    subject_type: str,
    subject_id: str,
) -> DebtLedgerRow | None:
    with sqlite3.connect(ops_db) as conn:
        row = conn.execute(
            """
            SELECT debt_id, stage, target_type, target_id, status, attempts,
                   last_error, next_retry_at, materializer_version,
                   created_at_ms, updated_at_ms
            FROM convergence_debt
            WHERE stage = ? AND target_type = ? AND target_id = ?
            """,
            (stage, subject_type, subject_id),
        ).fetchone()
    if row is None:
        return None
    return DebtLedgerRow(
        debt_id=str(row[0]),
        stage=str(row[1]),
        subject_type=str(row[2]),
        subject_id=str(row[3]),
        status=str(row[4]),
        attempts=int(row[5]),
        last_error=None if row[6] is None else str(row[6]),
        next_retry_at=None if row[7] is None else str(row[7]),
        materializer_version=None if row[8] is None else str(row[8]),
        created_at_ms=int(row[9]),
        updated_at_ms=int(row[10]),
    )


def set_debt_retry_at(
    ops_db: Path,
    *,
    stage: str,
    subject_type: str,
    subject_id: str,
    retry_at: str,
) -> None:
    with sqlite3.connect(ops_db) as conn:
        cursor = conn.execute(
            """
            UPDATE convergence_debt
            SET next_retry_at = ?
            WHERE stage = ? AND target_type = ? AND target_id = ?
            """,
            (retry_at, stage, subject_type, subject_id),
        )
        conn.commit()
    if cursor.rowcount != 1:
        raise AssertionError(f"expected one convergence debt row, updated {cursor.rowcount}")


def make_messages_fts_stale(index_db: Path, *, session_id: str, require_rows: bool = True) -> int:
    """Delete only this session's real FTS rows to create unrelated stage debt."""
    with open_connection(index_db) as conn:
        block_ids = tuple(
            str(row[0])
            for row in conn.execute("SELECT block_id FROM blocks WHERE session_id = ? ORDER BY rowid", (session_id,))
        )
        row_ids = (
            tuple(
                int(row[0])
                for row in conn.execute(
                    f"SELECT rowid FROM messages_fts_identity WHERE block_id IN ({','.join('?' for _ in block_ids)})",
                    block_ids,
                )
            )
            if block_ids
            else ()
        )
        conn.executemany("DELETE FROM messages_fts WHERE rowid = ?", ((row_id,) for row_id in row_ids))
        conn.executemany("DELETE FROM messages_fts_identity WHERE rowid = ?", ((row_id,) for row_id in row_ids))
        conn.commit()
    if require_rows and not row_ids:
        raise AssertionError(f"session {session_id!r} has no indexed blocks")
    return len(row_ids)


def messages_fts_match_count(index_db: Path, query: str) -> int:
    with sqlite3.connect(index_db) as conn:
        row = conn.execute(
            "SELECT COUNT(*) FROM messages_fts WHERE messages_fts MATCH ?",
            (query,),
        ).fetchone()
    return 0 if row is None else int(row[0])


def raw_authority_facts(source_db: Path, *, archive_root: Path | None = None) -> tuple[FactRow, ...]:
    with sqlite3.connect(source_db) as conn:
        rows = conn.execute(
            """
            SELECT raw_id, origin, capture_mode, native_id, source_path, source_index,
                   hex(blob_hash), blob_size, acquired_at_ms, logical_source_key,
                   revision_kind, source_revision, predecessor_source_revision,
                   predecessor_raw_id, baseline_raw_id, append_start_offset,
                   append_end_offset, acquisition_generation, revision_authority,
                   revision_authority_evidence
            FROM raw_sessions
            ORDER BY raw_id
            """
        ).fetchall()
    if archive_root is None:
        return _fact_rows(rows)
    normalized_rows = [(*row[1:4], str(Path(str(row[4])).relative_to(archive_root)), *row[5:]) for row in rows]
    normalized_rows.sort(key=lambda row: tuple("" if value is None else str(value) for value in row))
    return _fact_rows(normalized_rows)


def session_materialization_facts(index_db: Path, *, session_id: str) -> SessionMaterializationFacts:
    with sqlite3.connect(index_db) as conn:
        profile_row = conn.execute(
            """
            SELECT session_id, logical_session_id, materializer_version,
                   source_updated_at, source_sort_key, input_high_water_mark,
                   input_high_water_mark_source, input_row_count, source_name,
                   title, message_count, work_event_count, phase_count,
                   word_count, tool_use_count, thinking_count, total_cost_usd,
                   total_duration_ms, workflow_shape, terminal_state,
                   total_input_tokens, total_output_tokens,
                   evidence_payload_json, inference_payload_json,
                   enrichment_payload_json
            FROM session_profiles
            WHERE session_id = ?
            """,
            (session_id,),
        ).fetchone()
        materializations = conn.execute(
            """
            SELECT insight_type, session_id, materializer_version,
                   source_updated_at_ms, source_sort_key_ms,
                   input_high_water_mark_ms, input_high_water_mark_source,
                   input_row_count
            FROM insight_materialization
            WHERE session_id = ?
            ORDER BY insight_type
            """,
            (session_id,),
        ).fetchall()
        work_events = conn.execute(
            """
            SELECT session_id, position, work_event_type, summary, confidence,
                   start_index, end_index, started_at_ms, ended_at_ms,
                   duration_ms, file_paths_json, tools_used_json,
                   input_high_water_mark, input_high_water_mark_source,
                   evidence_json, inference_json, search_text
            FROM session_work_events
            WHERE session_id = ?
            ORDER BY position
            """,
            (session_id,),
        ).fetchall()
        phases = conn.execute(
            """
            SELECT session_id, position, start_index, end_index,
                   started_at_ms, ended_at_ms, duration_ms,
                   tool_counts_json, word_count,
                   input_high_water_mark, input_high_water_mark_source,
                   evidence_json, inference_json, search_text
            FROM session_phases
            WHERE session_id = ?
            ORDER BY position
            """,
            (session_id,),
        ).fetchall()
        threads = conn.execute(
            """
            SELECT t.thread_id, t.dominant_repo_id, t.materializer_version,
                   t.source_updated_at, t.input_high_water_mark,
                   t.input_high_water_mark_source, t.input_row_count,
                   t.start_time, t.end_time, t.dominant_repo,
                   t.session_ids_json, t.session_count, t.depth, t.branch_count,
                   t.total_messages, t.total_cost_usd, t.wall_duration_ms,
                   t.work_event_breakdown_json, t.payload_json, t.search_text
            FROM threads AS t
            JOIN thread_sessions AS ts ON ts.thread_id = t.thread_id
            WHERE ts.session_id = ?
            ORDER BY t.thread_id
            """,
            (session_id,),
        ).fetchall()
        thread_sessions = conn.execute(
            """
            SELECT thread_id, session_id, position
            FROM thread_sessions
            WHERE session_id = ?
            ORDER BY thread_id, position
            """,
            (session_id,),
        ).fetchall()
        count_tables = (
            "sessions",
            "messages",
            "blocks",
            "session_profiles",
            "insight_materialization",
            "session_work_events",
            "session_phases",
            "threads",
            "thread_sessions",
        )
        table_counts = tuple(
            (table, int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])) for table in count_tables
        )
    return SessionMaterializationFacts(
        profile=None if profile_row is None else cast(FactRow, tuple(profile_row)),
        materializations=_fact_rows(materializations),
        work_events=_fact_rows(work_events),
        phases=_fact_rows(phases),
        threads=_fact_rows(threads),
        thread_sessions=_fact_rows(thread_sessions),
        table_counts=table_counts,
    )


def _seed_raw_source_session(conn: sqlite3.Connection, *, session_id: str, source_path: Path) -> str:
    raw_id = f"raw-{session_id}"
    source_db = _main_db_path(conn).with_name("source.db")
    with sqlite3.connect(source_db) as source_conn:
        initialize_archive_tier(source_conn, ArchiveTier.SOURCE)
        source_conn.execute(
            """
            INSERT OR REPLACE INTO raw_sessions (
                raw_id, origin, native_id, source_path, blob_hash, blob_size,
                acquired_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                raw_id,
                "codex-session",
                session_id,
                str(source_path),
                hashlib.sha256(f"raw:{session_id}".encode()).digest(),
                source_path.stat().st_size,
                1_769_000_000_000,
            ),
        )
        source_conn.commit()
    return write_parsed_session_to_archive(
        conn,
        ParsedSession(
            source_name=Provider.CODEX,
            provider_session_id=session_id,
            title=session_id,
            created_at="2026-05-24T01:00:00+00:00",
            updated_at="2026-05-24T01:00:00+00:00",
            messages=[
                ParsedMessage(
                    provider_message_id="msg-1",
                    role=Role.normalize("user"),
                    text=f"Message for {session_id}",
                    position=0,
                    blocks=[
                        ParsedContentBlock(
                            type=BlockType.TEXT,
                            text=f"Message for {session_id}",
                        )
                    ],
                )
            ],
        ),
        raw_id=raw_id,
        content_hash=hashlib.sha256(f"session:{session_id}".encode()).hexdigest(),
    )


def _main_db_path(conn: sqlite3.Connection) -> Path:
    row = conn.execute("PRAGMA database_list").fetchone()
    if row is None or not row[2]:
        raise AssertionError("archive index connection has no main database path")
    return Path(str(row[2]))


def _fact_rows(rows: list[tuple[object, ...]]) -> tuple[FactRow, ...]:
    return tuple(cast(FactRow, tuple(row)) for row in rows)


def _stable_json(value: object, root: Path) -> str:
    """Serialize JSON-shaped status evidence while masking temp-root paths."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return encoded.replace(str(root), "<archive-root>")


__all__ = [
    "ArchiveSnapshot",
    "ConvergenceArchive",
    "DebtLedgerRow",
    "DerivedReadinessSnapshot",
    "PartialConvergenceArchive",
    "SessionMaterializationFacts",
    "archive_snapshot",
    "assert_archive_verification_green",
    "assert_archives_equivalent",
    "assert_derived_readiness_equivalent",
    "build_converged_archive",
    "converge_convergence_archive",
    "debt_ledger_row",
    "derived_readiness_snapshot",
    "ingest_convergence_pathology",
    "initialize_active_archive",
    "make_messages_fts_stale",
    "messages_fts_match_count",
    "raw_authority_facts",
    "rebuild_retained_raw_index",
    "rich_convergence_pathology",
    "replay_convergence_archive",
    "rotated_session_order",
    "seed_partial_convergence_archive",
    "session_materialization_facts",
    "set_debt_retry_at",
    "truncate_sparse",
]
