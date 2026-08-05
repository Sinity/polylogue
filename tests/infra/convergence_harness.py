"""Real-SQLite fixtures and independent facts for convergence survivor tests.

This module adapts the production archive writers, daemon stages, and ops
ledger. It deliberately owns no alternate convergence state machine.
"""

from __future__ import annotations

import asyncio
import os
import random
import re
import shutil
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass, fields
from functools import lru_cache
from pathlib import Path
from typing import cast

import polylogue.daemon.convergence_stages as convergence_stages
from polylogue.config import Source
from polylogue.core.outcomes import OutcomeStatus
from polylogue.core.sources import origin_from_provider
from polylogue.daemon.convergence import DaemonConverger, SessionState, StageState
from polylogue.daemon.convergence_stages import make_fts_stage, make_insights_stage
from polylogue.maintenance.archive_verification import ArchiveVerificationReport, verify_archive
from polylogue.pipeline.services.archive_ingest import parse_sources_archive
from polylogue.scenarios import CorpusSpec, WorkloadEnvelopeSpec, partial_convergence_canary_spec
from polylogue.schemas.registry import SCHEMA_DIR, SchemaRegistry
from polylogue.schemas.synthetic import SyntheticCorpus
from polylogue.schemas.synthetic.models import SyntheticArtifactFacts, SyntheticSchemaSelection, SyntheticWrittenBatch
from polylogue.schemas.synthetic.selection import select_synthetic_schema
from polylogue.schemas.synthetic.wire_formats import WireSupportReceipt, build_wire_support_receipt
from polylogue.sources.live import LiveBatchProcessor, WatchSource
from polylogue.sources.live.cursor import CursorStore
from polylogue.storage.sqlite.connection import open_connection
from tests.infra.inferred_corpus import (
    InferredCorpusManifest,
    build_inferred_corpus_convergence_handoff,
    compile_inferred_corpus_manifest,
)

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
class ArchiveSnapshot:
    """The one canonical cross-tier archive comparator.

    Rows are selected from the raw authority, normalized index, provenance,
    insight, and FTS surfaces. Volatile acquisition timestamps and absolute
    temporary paths are normalized at the boundary. FTS is represented by
    real MATCH result sets, so a missing posting cannot hide behind a
    count-only comparison.
    """

    raw: tuple[FactRow, ...]
    raw_memberships: tuple[FactRow, ...]
    sessions: tuple[FactRow, ...]
    messages: tuple[FactRow, ...]
    blocks: tuple[FactRow, ...]
    attachments: tuple[FactRow, ...]
    attachment_refs: tuple[FactRow, ...]
    session_events: tuple[FactRow, ...]
    session_links: tuple[FactRow, ...]
    insight_materialization: tuple[FactRow, ...]
    profiles: tuple[FactRow, ...]
    work_events: tuple[FactRow, ...]
    phases: tuple[FactRow, ...]
    fts_queries: tuple[tuple[str, tuple[FactRow, ...]], ...]


@dataclass(frozen=True, slots=True)
class ConvergenceArchive:
    """A temporary archive produced by real acquisition and parser routes."""

    root: Path
    corpus: ConvergenceCorpus
    source_paths: tuple[Path, ...]
    session_ids: tuple[str, ...]
    artifact_facts: tuple[SyntheticArtifactFacts, ...] = ()


@dataclass(frozen=True, slots=True)
class CorpusMember:
    provider: str
    spec: CorpusSpec
    selection: SyntheticSchemaSelection
    receipt: dict[str, object] | None
    material_path: Path | None = None


@dataclass(frozen=True, slots=True)
class ConvergenceCorpus:
    """Persisted-selection-backed provider materials used by every property."""

    members: tuple[CorpusMember, ...]
    manifest: InferredCorpusManifest | None = None

    @property
    def sessions(self) -> tuple[CorpusMember, ...]:
        """Compatibility name for the old pathology-indexed property API."""
        return self.members


def _persisted_registry_factory() -> SchemaRegistry:
    return SchemaRegistry(storage_root=SCHEMA_DIR)


@lru_cache(maxsize=1)
def _persisted_wire_support_receipt() -> WireSupportReceipt:
    """Use the registry-derived production support receipt once per process."""

    return build_wire_support_receipt(registry=SchemaRegistry(storage_root=SCHEMA_DIR))


def inferred_convergence_corpus() -> ConvergenceCorpus:
    """Load every supported persisted package selection.

    The manifest and support receipt come from the persisted registry and the
    production synthetic route authority. Every supported selection is carried
    into the real parser and ingest path; unsupported origins remain receipts.
    """
    registry = SchemaRegistry(storage_root=SCHEMA_DIR)
    support_receipt = _persisted_wire_support_receipt()
    manifest = compile_inferred_corpus_manifest(
        registry=registry,
        wire_support_receipt=support_receipt,
    )
    handoff = build_inferred_corpus_convergence_handoff(manifest)
    if not manifest.unsupported_records:
        raise AssertionError("the persisted inferred manifest lost unsupported-origin receipts")
    supported_entries = tuple(entry for entry in manifest.entries if entry.spec is not None)
    if len(handoff.selections) != len(supported_entries) or not handoff.selections:
        raise AssertionError("the persisted inferred handoff dropped all or part of the supported selection corpus")
    members: list[CorpusMember] = []
    for index, (entry, selection) in enumerate(zip(supported_entries, handoff.selections, strict=True)):
        if entry.spec is None:
            raise AssertionError("supported inferred corpus entry lost its CorpusSpec")
        pinned_ids = (
            ()
            if entry.key.provider == "codex"
            else (f"convergence-{entry.key.provider}-{entry.key.package_version}-{index}",)
        )
        spec = entry.spec.with_generation_overrides(
            style="demo-attachments",
            session_native_ids=pinned_ids,
            origin="inferred.convergence.property",
            tags=("inferred", "convergence", entry.key.provider, entry.key.package_version),
        )
        members.append(
            CorpusMember(
                provider=entry.key.provider,
                spec=spec,
                selection=selection,
                receipt=support_receipt.to_dict(),
            )
        )
    return ConvergenceCorpus(tuple(members), manifest=manifest)


def rich_convergence_pathology() -> ConvergenceCorpus:
    """Return the persisted multi-origin corpus used by the property loop."""
    return inferred_convergence_corpus()


def append_convergence_member() -> CorpusMember:
    """Compatibility helper for the first supported appendable selection."""
    return append_convergence_members()[0]


def append_convergence_members() -> tuple[CorpusMember, ...]:
    """Return persisted JSONL selections with stable append identity evidence."""
    members = tuple(
        CorpusMember(
            provider=member.provider,
            spec=member.spec.with_generation_overrides(
                style="demo-tool-heavy",
                origin="append.convergence.property",
                tags=("append", "convergence", member.provider, member.spec.package_version),
            ),
            selection=member.selection,
            receipt=member.receipt,
        )
        for member in inferred_convergence_corpus().members
        if member.selection.wire_format.encoding == "jsonl" and member.spec.session_native_ids
    )
    if not members:
        raise AssertionError("the supported inferred corpus has no appendable JSONL selection")
    return members


def append_convergence_unsupported_receipts() -> tuple[dict[str, str], ...]:
    """Return explicit receipts for JSONL selections not representable by append replay."""
    return tuple(
        {
            "provider": member.provider,
            "package_version": member.spec.package_version,
            "element_kind": member.spec.element_kind or "",
            "operation": "append_prefix",
            "status": "unsupported",
            "reason": "wire route has no stable persisted session identity for prefix replay",
        }
        for member in inferred_convergence_corpus().members
        if member.selection.wire_format.encoding == "jsonl" and not member.spec.session_native_ids
    )


def convergence_max_examples() -> int:
    """Use a small meaningful default and an explicit exhaustive lab budget."""
    profile = os.environ.get("HYPOTHESIS_PROFILE", "convergence-fast")
    return 24 if profile in {"convergence-exhaustive", "lab"} else 3


def convergence_stateful_max_examples() -> int:
    """Keep the interruption machine fast while expanding it in the lab profile."""
    return 6 if convergence_max_examples() > 3 else 1


def convergence_stateful_step_count() -> int:
    """Give the lab machine more interruption and resume boundaries to explore."""
    return 8 if convergence_max_examples() > 3 else 2


def build_converged_archive(
    root: Path,
    pathology: ConvergenceCorpus,
    *,
    session_order: Sequence[int] | None = None,
    incremental: bool = False,
) -> ConvergenceArchive:
    """Materialize generated provider bytes through the production ingest route."""
    _reset_property_archive_root(root)
    initialize_active_archive(root)
    archive = ingest_convergence_pathology(
        root,
        pathology,
        session_indexes=_complete_session_order(pathology, session_order),
        converge_after_each=incremental,
    )
    if not incremental:
        converge_convergence_archive(archive)
        assert_corpus_materialization(archive)
    return archive


def initialize_active_archive(root: Path) -> None:
    """Create all archive tiers for a temporary property-test archive."""
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    initialize_active_archive_root(root)


def ingest_convergence_pathology(
    root: Path,
    pathology: ConvergenceCorpus,
    *,
    session_indexes: Sequence[int],
    converge_after_each: bool,
) -> ConvergenceArchive:
    """Generate real provider bytes and run the production full ingest route."""
    selected = _validate_session_indexes(pathology, session_indexes)
    source_paths: list[Path] = []
    session_ids: list[str] = []
    artifact_facts: list[SyntheticArtifactFacts] = []
    for index in selected:
        member = pathology.members[index]
        source_root = root / "sources" / member.provider
        source_root.mkdir(parents=True, exist_ok=True)
        written: SyntheticWrittenBatch | None = None
        if member.material_path is not None:
            source_path = source_root / f"member-{index:02d}{member.material_path.suffix}"
            shutil.copyfile(member.material_path, source_path)
        else:
            written = SyntheticCorpus.write_selection_artifacts(
                member.selection,
                member.spec,
                source_root,
                prefix=f"member-{index:02d}",
            )
            source_path = written.files[0]
        had_durable_raw = False
        if (root / "source.db").exists():
            with sqlite3.connect(root / "source.db") as conn:
                had_durable_raw = (
                    conn.execute(
                        "SELECT 1 FROM raw_sessions WHERE source_path = ? LIMIT 1",
                        (str(source_path),),
                    ).fetchone()
                    is not None
                )
        ingest_result = asyncio.run(
            parse_sources_archive(root, [Source(name=member.provider, path=source_path)], parse_workers=1)
        )
        if not had_durable_raw and (ingest_result.counts["sessions"] < 1 or ingest_result.counts["messages"] < 1):
            raise AssertionError(
                "production parser dispatch or ingest dropped a supported inferred selection: "
                f"provider={member.provider!r}, result={ingest_result!r}"
            )
        source_paths.append(source_path)
        if written is not None:
            artifact_facts.extend(item.facts for item in written.batch.artifacts)
        with sqlite3.connect(root / "index.db") as conn:
            session_ids.extend(str(row[0]) for row in conn.execute("SELECT session_id FROM sessions"))
        archive = ConvergenceArchive(
            root,
            pathology,
            tuple(source_paths),
            tuple(sorted(set(session_ids))),
            tuple(artifact_facts),
        )
        for session_id in archive.session_ids:
            make_messages_fts_stale(root / "index.db", session_id=session_id)
        if converge_after_each:
            _quiet_test_owned_sources(archive.source_paths)
            converge_convergence_archive(archive)

    return ConvergenceArchive(
        root,
        pathology,
        tuple(source_paths),
        tuple(sorted(set(session_ids))),
        tuple(artifact_facts),
    )


def converge_convergence_archive(archive: ConvergenceArchive) -> dict[str, SessionState]:
    """Run the real FTS and insight debt stages for the materialized sessions."""
    _quiet_test_owned_sources(archive.source_paths)
    stages = (
        make_fts_stage(archive.root / "index.db"),
        make_insights_stage(archive.root / "index.db"),
    )
    converger = DaemonConverger(stages)
    required_stages = {"fts", "insights"}
    if not required_stages.issubset(converger.stage_names):
        raise AssertionError(
            "convergence harness omitted a required stage: "
            f"required={sorted(required_stages)!r}, actual={converger.stage_names!r}"
        )
    states, _timings = converger.converge_sessions(archive.session_ids)
    not_converged = {session_id: state.last_error for session_id, state in states.items() if not state.converged}
    if not_converged:
        raise AssertionError(f"production convergence left pending work: {not_converged}")
    skipped = {
        session_id: sorted(stage for stage in required_stages if state.stages.get(stage) is not StageState.DONE)
        for session_id, state in states.items()
    }
    skipped = {session_id: stages for session_id, stages in skipped.items() if stages}
    if skipped:
        raise AssertionError(f"production convergence skipped required stages: {skipped!r}")
    _analyze_registry_tables(archive.root / "index.db")
    return states


def _quiet_test_owned_sources(source_paths: Sequence[Path]) -> None:
    """Make generated test files satisfy the production quiet-window contract."""
    for source_path in source_paths:
        if source_path.exists() and source_path.stat().st_size >= convergence_stages._HOT_INSIGHT_SOURCE_BYTES:
            os.utime(source_path, (0, 0))


def assert_corpus_materialization(archive: ConvergenceArchive) -> None:
    """Require every corpus member's origin to reach all derived surfaces."""
    with sqlite3.connect(archive.root / "index.db") as conn:
        expected_by_origin: dict[str, int] = {}
        for member in archive.corpus.members:
            origin = origin_from_provider(member.provider).value
            expected_by_origin[origin] = expected_by_origin.get(origin, 0) + 1
        for origin, expected_count in expected_by_origin.items():
            sessions = conn.execute(
                """
                SELECT session_id, parent_session_id, root_session_id, message_count
                FROM sessions WHERE origin = ? ORDER BY session_id
                """,
                (origin,),
            ).fetchall()
            if len(sessions) != expected_count:
                raise AssertionError(
                    "supported inferred selections did not materialize the expected origin sessions: "
                    f"origin={origin!r}, expected={expected_count}, sessions={sessions!r}"
                )
            for session_id, parent_session_id, root_session_id, message_count in sessions:
                if not root_session_id or (parent_session_id is None and root_session_id != session_id):
                    raise AssertionError(f"lineage root was not materialized for {session_id!r}: {sessions!r}")
                if parent_session_id is not None:
                    link = conn.execute(
                        "SELECT 1 FROM session_links WHERE src_session_id = ? LIMIT 1",
                        (session_id,),
                    ).fetchone()
                    if link is None:
                        raise AssertionError(f"lineage link was not materialized for {session_id!r}")

                profile = conn.execute(
                    "SELECT message_count FROM session_profiles WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
                if profile is None or int(profile[0]) != int(message_count):
                    raise AssertionError(f"profile materialization drifted for {session_id!r}: {profile!r}")

                fts_counts = conn.execute(
                    """
                    SELECT
                        (SELECT COUNT(*) FROM blocks WHERE session_id = ? AND NULLIF(search_text, '') IS NOT NULL),
                        (SELECT COUNT(*) FROM messages_fts WHERE rowid IN (
                            SELECT rowid FROM blocks WHERE session_id = ? AND NULLIF(search_text, '') IS NOT NULL
                        ))
                    """,
                    (session_id, session_id),
                ).fetchone()
                if fts_counts is None or int(fts_counts[0]) != int(fts_counts[1]) or int(fts_counts[0]) == 0:
                    raise AssertionError(f"FTS materialization drifted for {session_id!r}: {fts_counts!r}")


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


def archive_snapshot(root: Path) -> ArchiveSnapshot:
    """Return the one canonical archive comparator shared by all properties."""
    with sqlite3.connect(root / "source.db") as source_conn, sqlite3.connect(root / "index.db") as conn:
        raw_rows = source_conn.execute(
            """
            SELECT r.raw_id, r.origin, r.native_id, r.source_index, hex(r.blob_hash), r.blob_size,
                   r.logical_source_key, r.revision_kind, r.source_revision,
                   r.predecessor_source_revision,
                   COALESCE(p.origin || ':' || p.native_id || ':' || p.source_index, r.predecessor_raw_id),
                   COALESCE(b.origin || ':' || b.native_id || ':' || b.source_index, r.baseline_raw_id),
                   r.append_start_offset, r.append_end_offset, r.acquisition_generation,
                   r.revision_authority, r.revision_authority_evidence, r.parsed_at_ms IS NOT NULL
            FROM raw_sessions AS r
            LEFT JOIN raw_sessions AS p ON p.raw_id = r.predecessor_raw_id
            LEFT JOIN raw_sessions AS b ON b.raw_id = r.baseline_raw_id
            ORDER BY r.origin, r.native_id, r.source_index
            """
        ).fetchall()
        raw = [row[1:] for row in raw_rows]
        raw_memberships = source_conn.execute(
            """
            SELECT r.origin, r.native_id, r.source_index, m.logical_source_key,
                   m.provider_session_id, m.source_revision,
                   hex(m.normalized_content_hash), m.message_count,
                   COALESCE(p.origin || ':' || p.native_id || ':' || p.source_index, m.predecessor_raw_id),
                   m.acquisition_generation, m.revision_authority, m.decision
            FROM raw_session_memberships AS m
            JOIN raw_sessions AS r ON r.raw_id = m.raw_id
            LEFT JOIN raw_sessions AS p ON p.raw_id = m.predecessor_raw_id
            ORDER BY r.origin, r.native_id, r.source_index, m.logical_source_key
            """
        ).fetchall()
        sessions = conn.execute(
            """
            SELECT session_id, origin, native_id, title, parent_session_id,
                   root_session_id, branch_type, active_leaf_message_id, message_count,
                   s.raw_id,
                   title_source, title_ref, display_name, run_settings_json,
                   pending_drafts_json, git_branch, git_repository_url, provider_project_ref,
                   commit_hash, instructions_text, reported_duration_ms, reported_cost_usd,
                   hex(s.content_hash), updated_at_ms
            FROM sessions AS s
            ORDER BY session_id
            """
        ).fetchall()
        sessions = [
            (
                *row[:9],
                f"{row[1]}:{row[2]}" if row[9] is not None else None,
                *row[10:],
            )
            for row in sessions
        ]
        messages = conn.execute(
            """
            SELECT session_id, native_id, position, variant_index, role,
                   material_origin, message_type, parent_message_id, model_name,
                   model_effort, sender_name, recipient, delivery_status, end_turn,
                   user_context_text, has_tool_use, has_thinking, has_paste,
                   paste_boundary, is_active_path, is_active_leaf, word_count,
                   input_tokens, output_tokens, cache_read_tokens, cache_write_tokens,
                   duration_ms, hex(content_hash), occurred_at_ms, stop_reason
            FROM messages ORDER BY session_id, position, variant_index
            """
        ).fetchall()
        blocks = conn.execute(
            """
            SELECT session_id, message_id, position, block_type, text, tool_id,
                   tool_name, tool_input, semantic_type, media_type, language,
                   tool_result_is_error, tool_result_exit_code,
                   tool_result_outcome_unknown_reason, signature, hex(content_hash), search_text
            FROM blocks ORDER BY session_id, message_id, position
            """
        ).fetchall()
        attachments = conn.execute(
            "SELECT attachment_id, display_name, media_type, byte_count, hex(blob_hash), acquisition_status, ref_count FROM attachments ORDER BY attachment_id"
        ).fetchall()
        attachment_refs = conn.execute(
            "SELECT ref_id, attachment_id, session_id, message_id, position, upload_origin, source_url, caption FROM attachment_refs ORDER BY ref_id"
        ).fetchall()
        session_events = conn.execute(
            "SELECT session_id, position, event_type, occurred_at_ms, source_message_id, source_message_provider_id, summary, payload_json FROM session_events ORDER BY session_id, position"
        ).fetchall()
        session_links = conn.execute(
            """
            SELECT src_session_id, dst_origin, dst_native_id, link_type,
                   resolved_dst_session_id, branch_point_message_id, inheritance, status
            FROM session_links
            ORDER BY src_session_id, dst_origin, dst_native_id, link_type
            """
        ).fetchall()
        insight_materialization = conn.execute(
            """
            SELECT insight_type, session_id, materializer_version, source_updated_at_ms,
                   source_sort_key_ms, input_high_water_mark_source, input_row_count
            FROM insight_materialization ORDER BY insight_type, session_id
            """
        ).fetchall()
        profiles = conn.execute(
            """
            SELECT session_id, logical_session_id, materializer_version, source_updated_at,
                   source_sort_key, input_high_water_mark, input_high_water_mark_source,
                   input_row_count, source_name, title, message_count, substantive_count,
                   attachment_count, work_event_count, phase_count, word_count,
                   tool_use_count, thinking_count, total_cost_usd, total_duration_ms,
                   engaged_duration_ms, tool_active_duration_ms, wall_duration_ms,
                   workflow_shape, workflow_shape_method, workflow_shape_confidence,
                   workflow_shape_features_json, terminal_state, terminal_state_method,
                   terminal_state_confidence, terminal_state_evidence_json,
                   cost_is_estimated, thinking_duration_ms, output_duration_ms,
                   tool_duration_ms, latency_percentiles_ms_json, tool_calls_per_minute,
                   timing_provenance, total_input_tokens, total_output_tokens,
                   total_cache_read_tokens, total_cache_write_tokens, total_credit_cost,
                   cost_provenance, per_model_cost_json, evidence_payload_json,
                   inference_payload_json, enrichment_payload_json, evidence_search_text,
                   inference_search_text, enrichment_search_text, enrichment_version,
                   enrichment_family, inference_version, inference_family, search_text,
                   duration_ms, cost_credits, cost_usd, priced_with, NULL AS priced_at_ms,
                   primary_model_name, primary_model_family
            FROM session_profiles ORDER BY session_id
            """
        ).fetchall()
        work_events = conn.execute(
            "SELECT session_id, position, work_event_type, summary, confidence, start_index, end_index, duration_ms, evidence_json, inference_json, search_text FROM session_work_events ORDER BY session_id, position"
        ).fetchall()
        phases = conn.execute(
            "SELECT session_id, position, start_index, end_index, duration_ms, tool_counts_json, word_count, evidence_json, inference_json, search_text FROM session_phases ORDER BY session_id, position"
        ).fetchall()
        fts_terms: list[str] = []
        for (search_text,) in conn.execute(
            "SELECT search_text FROM blocks WHERE NULLIF(trim(search_text), '') IS NOT NULL ORDER BY block_id"
        ):
            terms = re.findall(r"\w{2,}", str(search_text).casefold())
            if terms:
                fts_terms.append(terms[0])
        fts_queries = tuple(
            (
                term,
                _fact_rows(
                    conn.execute(
                        "SELECT b.block_id, b.message_id, b.session_id FROM messages_fts AS f JOIN blocks AS b ON b.rowid = f.rowid WHERE messages_fts MATCH ? ORDER BY b.block_id",
                        (f'"{term.replace(chr(34), chr(34) * 2)}"',),
                    ).fetchall()
                ),
            )
            for term in dict.fromkeys(fts_terms)
        )
    return ArchiveSnapshot(
        raw=_fact_rows(raw),
        raw_memberships=_fact_rows(raw_memberships),
        sessions=_fact_rows(sessions),
        messages=_fact_rows(messages),
        blocks=_fact_rows(blocks),
        attachments=_fact_rows(attachments),
        attachment_refs=_fact_rows(attachment_refs),
        session_events=_fact_rows(session_events),
        session_links=_fact_rows(session_links),
        insight_materialization=_fact_rows(insight_materialization),
        profiles=_fact_rows(profiles),
        work_events=_fact_rows(work_events),
        phases=_fact_rows(phases),
        fts_queries=fts_queries,
    )


def assert_archives_equivalent(
    left: ConvergenceArchive,
    right: ConvergenceArchive,
    *,
    compare_acquisition_route: bool = True,
) -> None:
    """Compare archives through the one canonical cross-tier snapshot."""
    left_snapshot = archive_snapshot(left.root)
    right_snapshot = archive_snapshot(right.root)
    if not compare_acquisition_route:
        left_snapshot = ArchiveSnapshot(
            raw=(),
            raw_memberships=(),
            sessions=left_snapshot.sessions,
            messages=left_snapshot.messages,
            blocks=left_snapshot.blocks,
            attachments=left_snapshot.attachments,
            attachment_refs=left_snapshot.attachment_refs,
            session_events=left_snapshot.session_events,
            session_links=left_snapshot.session_links,
            insight_materialization=left_snapshot.insight_materialization,
            profiles=left_snapshot.profiles,
            work_events=left_snapshot.work_events,
            phases=left_snapshot.phases,
            fts_queries=left_snapshot.fts_queries,
        )
        right_snapshot = ArchiveSnapshot(
            raw=(),
            raw_memberships=(),
            sessions=right_snapshot.sessions,
            messages=right_snapshot.messages,
            blocks=right_snapshot.blocks,
            attachments=right_snapshot.attachments,
            attachment_refs=right_snapshot.attachment_refs,
            session_events=right_snapshot.session_events,
            session_links=right_snapshot.session_links,
            insight_materialization=right_snapshot.insight_materialization,
            profiles=right_snapshot.profiles,
            work_events=right_snapshot.work_events,
            phases=right_snapshot.phases,
            fts_queries=right_snapshot.fts_queries,
        )
    if left_snapshot != right_snapshot:
        differences = {
            field.name: (getattr(left_snapshot, field.name), getattr(right_snapshot, field.name))
            for field in fields(ArchiveSnapshot)
            if getattr(left_snapshot, field.name) != getattr(right_snapshot, field.name)
        }
        raise AssertionError(f"canonical archive snapshots differ in {tuple(differences)}: {differences!r}")


def _complete_session_order(pathology: ConvergenceCorpus, order: Sequence[int] | None) -> tuple[int, ...]:
    expected = tuple(range(len(pathology.sessions)))
    candidate = expected if order is None else tuple(order)
    if len(candidate) != len(expected) or set(candidate) != set(expected):
        raise ValueError("session_order must be a permutation of every composed session index")
    return candidate


def permuted_session_order(pathology: ConvergenceCorpus, seed: int) -> tuple[int, ...]:
    """Return a deterministic non-identity permutation, including swaps."""
    indexes = list(range(len(pathology.sessions)))
    random.Random(seed).shuffle(indexes)
    if tuple(indexes) == tuple(range(len(indexes))):
        indexes.reverse()
    return tuple(indexes)


def rotated_session_order(pathology: ConvergenceCorpus, shift: int) -> tuple[int, ...]:
    """Compatibility wrapper retained for callers outside this batch."""
    return permuted_session_order(pathology, shift)


def build_append_prefix_archive(
    root: Path,
    member: CorpusMember,
    *,
    split_line: int,
) -> ConvergenceArchive:
    """Ingest one real JSONL prefix, then its literal delta through live append."""
    _reset_property_archive_root(root)
    initialize_active_archive(root)
    batch = SyntheticCorpus.generate_batch_for_selection(member.selection, member.spec)
    artifact = batch.artifacts[0]
    lines = artifact.raw_bytes.splitlines(keepends=True)
    if len(lines) < 3:
        raise AssertionError("append corpus artifact needs metadata and at least two complete records")
    split = max(2, min(split_line, len(lines) - 1))
    prefix = b"".join(lines[:split])
    delta = b"".join(lines[split:])
    source_root = root / "live" / member.provider
    source_root.mkdir(parents=True, exist_ok=True)
    source_path = source_root / "capture-proof.jsonl"
    source_path.write_bytes(prefix)

    async def ingest() -> None:
        from polylogue.api import Polylogue

        archive = Polylogue(archive_root=root, db_path=root / "index.db")
        try:
            processor = LiveBatchProcessor(
                archive,
                (WatchSource(name=member.provider, root=source_root),),
                cursor=CursorStore(root / "index.db"),
                parser_fingerprint="convergence-property-live-v1",
            )
            first = await processor.ingest_files([source_path], emit_event=False)
            if first.succeeded_file_count != 1:
                raise AssertionError(f"live prefix ingest did not succeed: {first!r}")
            with source_path.open("ab") as handle:
                handle.write(delta)
            second = await processor.ingest_files([source_path], emit_event=False)
            if second.append_file_count != 1 or second.succeeded_file_count != 1:
                raise AssertionError(f"live append ingest did not take append route: {second!r}")
        finally:
            await archive.close()

    asyncio.run(ingest())
    with sqlite3.connect(root / "index.db") as conn:
        session_ids = tuple(sorted(str(row[0]) for row in conn.execute("SELECT session_id FROM sessions")))
    if not session_ids:
        raise AssertionError("live append route produced no indexed session")
    for session_id in session_ids:
        make_messages_fts_stale(root / "index.db", session_id=session_id)
    archive = ConvergenceArchive(
        root,
        ConvergenceCorpus((member,)),
        (source_path,),
        session_ids,
        (artifact.facts,),
    )
    converge_convergence_archive(archive)
    assert_corpus_materialization(archive)
    return archive


def build_full_live_archive(root: Path, member: CorpusMember) -> ConvergenceArchive:
    """Replay the complete generated artifact through the live full route."""
    _reset_property_archive_root(root)
    initialize_active_archive(root)
    artifact = SyntheticCorpus.generate_batch_for_selection(member.selection, member.spec).artifacts[0]
    source_root = root / "live" / member.provider
    source_root.mkdir(parents=True, exist_ok=True)
    source_path = source_root / "capture-proof.jsonl"
    source_path.write_bytes(artifact.raw_bytes)

    async def ingest() -> None:
        from polylogue.api import Polylogue

        archive = Polylogue(archive_root=root, db_path=root / "index.db")
        try:
            processor = LiveBatchProcessor(
                archive,
                (WatchSource(name=member.provider, root=source_root),),
                cursor=CursorStore(root / "index.db"),
                parser_fingerprint="convergence-property-live-v1",
            )
            metrics = await processor.ingest_files([source_path], emit_event=False)
            if metrics.full_file_count != 1 or metrics.succeeded_file_count != 1:
                raise AssertionError(f"live full ingest did not succeed: {metrics!r}")
        finally:
            await archive.close()

    asyncio.run(ingest())
    with sqlite3.connect(root / "index.db") as conn:
        session_ids = tuple(sorted(str(row[0]) for row in conn.execute("SELECT session_id FROM sessions")))
    if not session_ids:
        raise AssertionError("live full route produced no indexed session")
    for session_id in session_ids:
        make_messages_fts_stale(root / "index.db", session_id=session_id)
    archive = ConvergenceArchive(root, ConvergenceCorpus((member,)), (source_path,), session_ids, (artifact.facts,))
    converge_convergence_archive(archive)
    assert_corpus_materialization(archive)
    return archive


def assert_append_provenance(root: Path) -> None:
    """Require a byte-ranged revision chain with production lineage metadata."""
    with sqlite3.connect(root / "source.db") as conn:
        rows = conn.execute(
            """
            SELECT raw_id, revision_kind, blob_size, source_revision,
                   predecessor_raw_id, baseline_raw_id, append_start_offset,
                   append_end_offset, revision_authority
            FROM raw_sessions
            ORDER BY CASE revision_kind WHEN 'full' THEN 0 WHEN 'append' THEN 1 ELSE 2 END, raw_id
            """
        ).fetchall()
    if len(rows) != 2:
        raise AssertionError(f"append raw revision provenance is incomplete: {rows!r}")
    full, append = rows
    if full[1] != "full" or append[1] != "append":
        raise AssertionError(f"append route revision kinds are not full then append: {rows!r}")
    if not full[2] or not append[2] or not full[3] or not append[3] or full[3] == append[3]:
        raise AssertionError(f"append route did not preserve distinct raw content revisions: {rows!r}")
    if append[4] != full[0] or append[5] != full[0]:
        raise AssertionError(f"append route did not attach predecessor and baseline: {rows!r}")
    if append[6] is None or append[7] is None or int(append[7]) <= int(append[6]):
        raise AssertionError(f"append route did not persist a non-empty byte range: {rows!r}")
    if append[8] != "byte_proven":
        raise AssertionError(f"append route was not byte-proven: {rows!r}")


def drop_one_insight_row(root: Path) -> None:
    with sqlite3.connect(root / "index.db") as conn:
        conn.execute("DELETE FROM session_profiles WHERE rowid IN (SELECT rowid FROM session_profiles LIMIT 1)")
        conn.commit()


def drop_one_fts_posting(root: Path) -> None:
    with sqlite3.connect(root / "index.db") as conn:
        row = conn.execute("SELECT rowid FROM messages_fts LIMIT 1").fetchone()
        if row is None:
            raise AssertionError("cannot construct FTS red twin without a posting")
        conn.execute("DELETE FROM messages_fts WHERE rowid = ?", (int(row[0]),))
        conn.commit()


def _validate_session_indexes(pathology: ConvergenceCorpus, indexes: Sequence[int]) -> tuple[int, ...]:
    selected = tuple(indexes)
    if len(selected) != len(set(selected)) or any(index < 0 or index >= len(pathology.sessions) for index in selected):
        raise ValueError("session_indexes must be distinct valid composed-session indexes")
    return selected


def _reset_property_archive_root(root: Path) -> None:
    """Reset only a caller-owned temporary property archive between examples."""
    if root.exists():
        shutil.rmtree(root)


def _analyze_registry_tables(index_db: Path) -> None:
    with sqlite3.connect(index_db) as conn:
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
    truncate_sparse(target_source, target_size)
    truncate_sparse(unrelated_source, 1_024)

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


def make_messages_fts_stale(index_db: Path, *, session_id: str) -> int:
    """Delete only this session's real FTS rows to create unrelated stage debt."""
    with sqlite3.connect(index_db) as conn:
        row_ids = [
            int(row[0])
            for row in conn.execute(
                "SELECT rowid FROM blocks WHERE session_id = ? ORDER BY rowid",
                (session_id,),
            ).fetchall()
        ]
        conn.executemany("DELETE FROM messages_fts WHERE rowid = ?", ((row_id,) for row_id in row_ids))
        conn.commit()
    if not row_ids:
        raise AssertionError(f"session {session_id!r} has no indexed blocks")
    return len(row_ids)


def messages_fts_match_count(index_db: Path, query: str) -> int:
    with sqlite3.connect(index_db) as conn:
        row = conn.execute(
            "SELECT COUNT(*) FROM messages_fts WHERE messages_fts MATCH ?",
            (query,),
        ).fetchone()
    return 0 if row is None else int(row[0])


def raw_authority_facts(source_db: Path) -> tuple[FactRow, ...]:
    with sqlite3.connect(source_db) as conn:
        rows = conn.execute(
            """
            SELECT raw_id, origin, native_id, source_path, hex(blob_hash),
                   blob_size, acquired_at_ms
            FROM raw_sessions
            ORDER BY raw_id
            """
        ).fetchall()
    return _fact_rows(rows)


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
    archive_root = _main_db_path(conn).parent
    selection = select_synthetic_schema(
        "codex",
        version="v1",
        element_kind="session_record_stream",
        registry_factory=_persisted_registry_factory,
    )
    spec = CorpusSpec.for_provider(
        "codex",
        package_version=selection.package_version,
        element_kind=selection.element_kind,
        count=1,
        messages_min=1,
        messages_max=1,
        seed=sum(ord(character) for character in session_id),
        style="demo-tool-heavy",
        session_native_ids=(),
        origin="partial.convergence.fixture",
    )
    artifact = SyntheticCorpus.generate_batch_for_selection(selection, spec).artifacts[0]
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_bytes(artifact.raw_bytes)
    result = asyncio.run(parse_sources_archive(archive_root, [Source(name="codex", path=source_path)]))
    if result.counts["sessions"] != 1 or result.counts["messages"] < 1:
        raise AssertionError(f"partial convergence fixture did not use the real Codex ingest route: {result!r}")
    row = conn.execute(
        "SELECT session_id FROM sessions WHERE origin = 'codex-session' AND native_id = ?",
        (source_path.stem,),
    ).fetchone()
    if row is None:
        raise AssertionError(f"real Codex ingest did not write session {session_id!r}")
    return str(row[0])


def _main_db_path(conn: sqlite3.Connection) -> Path:
    row = conn.execute("PRAGMA database_list").fetchone()
    if row is None or not row[2]:
        raise AssertionError("archive index connection has no main database path")
    return Path(str(row[2]))


def _fact_rows(rows: list[tuple[object, ...]]) -> tuple[FactRow, ...]:
    return tuple(cast(FactRow, tuple(row)) for row in rows)


__all__ = [
    "ArchiveSnapshot",
    "CorpusMember",
    "ConvergenceCorpus",
    "ConvergenceArchive",
    "DebtLedgerRow",
    "PartialConvergenceArchive",
    "SessionMaterializationFacts",
    "archive_snapshot",
    "assert_archive_verification_green",
    "assert_archives_equivalent",
    "assert_corpus_materialization",
    "append_convergence_member",
    "append_convergence_members",
    "append_convergence_unsupported_receipts",
    "build_append_prefix_archive",
    "build_converged_archive",
    "build_full_live_archive",
    "converge_convergence_archive",
    "debt_ledger_row",
    "drop_one_fts_posting",
    "drop_one_insight_row",
    "inferred_convergence_corpus",
    "ingest_convergence_pathology",
    "initialize_active_archive",
    "make_messages_fts_stale",
    "messages_fts_match_count",
    "make_fts_stage",
    "make_insights_stage",
    "raw_authority_facts",
    "rich_convergence_pathology",
    "convergence_max_examples",
    "convergence_stateful_max_examples",
    "convergence_stateful_step_count",
    "rotated_session_order",
    "permuted_session_order",
    "seed_partial_convergence_archive",
    "session_materialization_facts",
    "set_debt_retry_at",
    "truncate_sparse",
]
