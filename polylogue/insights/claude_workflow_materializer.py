"""Production materialization for admitted Claude Code Workflow artifacts.

``source.db`` remains the authority: mutable files retain every raw revision and
``raw_artifacts`` points at the currently observed revision.  This module reads
that authority plus already-indexed Claude sessions/events, then atomically
replaces only ``claude-workflow:*`` graphs in the generic work-evidence tables.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

from polylogue.core.enums import Origin, Provider
from polylogue.core.refs import EvidenceRef, ObjectRef
from polylogue.insights.claude_workflow_evidence import (
    ClaudeWorkflowCoordinatorInvocation,
    ClaudeWorkflowPromptEvidence,
    ClaudeWorkflowSessionEvidence,
    project_claude_workflow_evidence,
)
from polylogue.insights.work_evidence import WorkEvidenceGraph
from polylogue.logging import get_logger
from polylogue.sources.origin_specs import artifact_rule_for_path
from polylogue.sources.parsers.base_models import ParsedSessionEvent
from polylogue.sources.parsers.claude.orchestration import (
    ClaudeOrchestrationArtifact,
    ClaudeOrchestrationFact,
    parse_claude_orchestration_artifact,
)
from polylogue.storage.artifacts.inspection import artifact_observation_id
from polylogue.storage.blob_store import BlobStore

logger = get_logger(__name__)

CLAUDE_WORKFLOW_MATERIALIZER_VERSION = 1
_GRAPH_PREFIX = "claude-workflow:"
_FACT_KINDS = frozenset(
    {
        "workflow_run_snapshot",
        "workflow_journal",
        "agent_sidecar_meta",
        "adopt_manifest",
    }
)
_SESSION_KINDS = frozenset({"agent_transcript", "coordinator_session_stream"})


@dataclass(frozen=True, slots=True)
class ClaudeWorkflowMaterializationSummary:
    """Quantified result of one source-to-derived materialization pass."""

    semantic_version: int
    corpus_snapshot_ref: str
    current_artifact_count: int
    retained_raw_revision_count: int
    artifact_counts: dict[str, int]
    graph_count: int
    run_count: int
    coordinator_invocation_count: int
    call_count: int
    attempt_count: int
    linked_session_count: int
    metadata_sidecar_count: int
    journal_result_count: int
    final_result_count: int
    unresolved_call_count: int
    excluded_session_count: int
    generated_prompt_count: int
    human_prompt_count: int
    gaps: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ClaudeWorkflowReparsePlan:
    """Bounded semantic-reparse implications without reading private live data."""

    semantic_version: int
    current_artifacts: int
    retained_raw_revisions_preserved: int
    projector_only_raw_artifact_reads: int
    coordinator_event_rows_reused: int
    indexed_session_bindings_reused: int
    session_parser_raw_reads_if_authorship_or_event_semantics_change: int
    graphs_atomically_replaced: int
    stale: bool

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class _RawArtifact:
    artifact_id: str
    raw_id: str
    source_path: str
    source_index: int
    artifact_kind: str
    blob_hash: str
    acquired_at_ms: int

    @property
    def evidence_ref(self) -> ObjectRef:
        return ObjectRef(kind="artifact", object_id=f"raw:{self.raw_id}")


@dataclass(frozen=True, slots=True)
class _PreparedInputs:
    raw_artifacts: tuple[_RawArtifact, ...]
    parsed_artifacts: tuple[ClaudeOrchestrationArtifact, ...]
    artifact_evidence: dict[str, ObjectRef]
    coordinator_invocations: tuple[ClaudeWorkflowCoordinatorInvocation, ...]
    session_evidence: tuple[ClaudeWorkflowSessionEvidence, ...]
    coordinator_prompts: tuple[ClaudeWorkflowPromptEvidence, ...]
    run_ids: tuple[str, ...]
    corpus_snapshot_ref: ObjectRef
    retained_raw_revision_count: int
    existing_graph_refs: dict[str, str]


def materialize_claude_workflow_archive(archive_root: Path) -> ClaudeWorkflowMaterializationSummary:
    """Rebuild all Claude Workflow graphs from current retained authority."""

    archive_root = Path(archive_root)
    prepared = _prepare_inputs(archive_root)
    graphs: list[WorkEvidenceGraph] = []
    for run_id in prepared.run_ids:
        run_artifacts = _artifacts_for_run(run_id, prepared.parsed_artifacts)
        graph = project_claude_workflow_evidence(
            graph_id=f"{_GRAPH_PREFIX}{run_id}",
            run_id=run_id,
            corpus_snapshot_ref=prepared.corpus_snapshot_ref,
            artifacts=run_artifacts,
            artifact_evidence=prepared.artifact_evidence,
            coordinator_invocations=prepared.coordinator_invocations,
            session_evidence=prepared.session_evidence,
            coordinator_prompt_evidence=prepared.coordinator_prompts,
        )
        graphs.append(graph)

    _replace_graph_family(archive_root / "index.db", graphs)
    return _summarize(prepared, graphs)


def claude_workflow_materialization_needed(archive_root: Path) -> bool:
    """Return whether current source/index evidence differs from stored graphs."""

    prepared = _prepare_inputs(Path(archive_root))
    expected = {f"{_GRAPH_PREFIX}{run_id}": prepared.corpus_snapshot_ref.format() for run_id in prepared.run_ids}
    return expected != prepared.existing_graph_refs


def claude_workflow_reparse_plan(archive_root: Path) -> ClaudeWorkflowReparsePlan:
    """Quantify projector-only and parser-semantic rebuild scopes."""

    prepared = _prepare_inputs(Path(archive_root))
    counts = Counter(item.artifact_kind for item in prepared.raw_artifacts)
    expected = {f"{_GRAPH_PREFIX}{run_id}": prepared.corpus_snapshot_ref.format() for run_id in prepared.run_ids}
    return ClaudeWorkflowReparsePlan(
        semantic_version=CLAUDE_WORKFLOW_MATERIALIZER_VERSION,
        current_artifacts=len(prepared.raw_artifacts),
        retained_raw_revisions_preserved=prepared.retained_raw_revision_count,
        projector_only_raw_artifact_reads=sum(counts[kind] for kind in _FACT_KINDS),
        coordinator_event_rows_reused=len(prepared.coordinator_invocations),
        indexed_session_bindings_reused=sum(item.session_id is not None for item in prepared.session_evidence),
        session_parser_raw_reads_if_authorship_or_event_semantics_change=sum(counts[kind] for kind in _SESSION_KINDS),
        graphs_atomically_replaced=len(prepared.run_ids),
        stale=expected != prepared.existing_graph_refs,
    )


def _prepare_inputs(archive_root: Path) -> _PreparedInputs:
    source_db = archive_root / "source.db"
    index_db = archive_root / "index.db"
    if not source_db.exists() or not index_db.exists():
        raise FileNotFoundError("Claude Workflow materialization requires source.db and index.db")

    with sqlite3.connect(source_db) as source_conn:
        source_conn.row_factory = sqlite3.Row
        source_conn.execute("PRAGMA foreign_keys = ON")
        _ensure_current_artifact_inventory(source_conn)
        source_conn.commit()
        raw_artifacts = _load_current_artifacts(source_conn)
        retained_revisions = _count_retained_revisions(source_conn)

    blob_store = BlobStore(archive_root / "blob")
    parsed: list[ClaudeOrchestrationArtifact] = []
    artifact_evidence: dict[str, ObjectRef] = {}
    for raw in raw_artifacts:
        artifact_evidence[_normalize_path(raw.source_path)] = raw.evidence_ref
        if raw.artifact_kind not in _FACT_KINDS:
            continue
        try:
            payload = blob_store.read_all(raw.blob_hash)
            value = parse_claude_orchestration_artifact(raw.source_path, payload)
            if value is None:
                value = ClaudeOrchestrationArtifact(
                    kind=raw.artifact_kind,
                    source_path=raw.source_path,
                    parse_policy="fact",
                    facts=(),
                    parse_error="OriginSpec rule was unavailable during materialization",
                )
        except Exception as exc:
            logger.warning(
                "Claude Workflow artifact parse degraded to path-identity fact: %s (%s)",
                raw.source_path,
                exc,
            )
            fallback_fact = _path_identity_fact(raw)
            value = ClaudeOrchestrationArtifact(
                kind=raw.artifact_kind,
                source_path=raw.source_path,
                parse_policy="fact",
                facts=(fallback_fact,) if fallback_fact is not None else (),
                parse_error=f"{type(exc).__name__}: {exc}",
            )
        parsed.append(value)

    with sqlite3.connect(index_db) as index_conn:
        index_conn.row_factory = sqlite3.Row
        coordinator_invocations = _load_coordinator_invocations(index_conn, raw_artifacts)
        sessions = _load_session_evidence(index_conn, raw_artifacts)
        coordinator_prompts = _load_coordinator_prompts(index_conn, coordinator_invocations)
        existing_graph_refs = _load_existing_graph_refs(index_conn)

    run_ids = _run_ids(parsed, coordinator_invocations)
    corpus_snapshot_ref = _corpus_snapshot(
        raw_artifacts,
        coordinator_invocations,
        sessions,
        coordinator_prompts,
    )
    return _PreparedInputs(
        raw_artifacts=raw_artifacts,
        parsed_artifacts=tuple(parsed),
        artifact_evidence=artifact_evidence,
        coordinator_invocations=coordinator_invocations,
        session_evidence=sessions,
        coordinator_prompts=coordinator_prompts,
        run_ids=run_ids,
        corpus_snapshot_ref=corpus_snapshot_ref,
        retained_raw_revision_count=retained_revisions,
        existing_graph_refs=existing_graph_refs,
    )


def _ensure_current_artifact_inventory(conn: sqlite3.Connection) -> None:
    """Refresh current pointers for OriginSpec-declared Claude artifacts.

    Canonical configured acquisition already writes these rows.  The same
    source-tier invariant is repaired here for daemon/direct raw writers so all
    production routes converge on one inventory rather than a Workflow-only
    registry.
    """

    rows = conn.execute(
        """
        WITH ranked AS (
            SELECT rowid AS raw_rowid, *,
                   ROW_NUMBER() OVER (
                       PARTITION BY origin, source_path, source_index
                       ORDER BY acquired_at_ms DESC, rowid DESC
                   ) AS revision_rank
            FROM raw_sessions
            WHERE origin = ?
        )
        SELECT * FROM ranked WHERE revision_rank = 1
        ORDER BY source_path, source_index
        """,
        (Origin.CLAUDE_CODE_SESSION.value,),
    ).fetchall()
    for row in rows:
        rule = artifact_rule_for_path(Provider.CLAUDE_CODE, str(row["source_path"]))
        if rule is None:
            continue
        existing = conn.execute(
            """
            SELECT artifact_id, first_observed_at_ms
            FROM raw_artifacts
            WHERE origin = ? AND source_path = ? AND source_index = ?
            """,
            (row["origin"], row["source_path"], row["source_index"]),
        ).fetchone()
        source_name = str(row["capture_mode"] or Provider.CLAUDE_CODE.value)
        observation_id = (
            str(existing["artifact_id"])
            if existing is not None
            else artifact_observation_id(
                source_name=source_name,
                source_path=str(row["source_path"]),
                source_index=int(row["source_index"]),
            )
        )
        first_observed = int(existing["first_observed_at_ms"]) if existing is not None else int(row["acquired_at_ms"])
        support_status = "supported_parseable" if rule.parse_policy == "session" else "recognized_unparsed"
        conn.execute(
            """
            INSERT INTO raw_artifacts(
                artifact_id, raw_id, origin, source_path, source_index,
                artifact_kind, support_status, classification_reason,
                parse_as_session, schema_eligible, malformed_jsonl_lines,
                decode_error, cohort_id, link_group_key, sidecar_agent_type,
                first_observed_at_ms, last_observed_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, NULL, NULL, ?, NULL, ?, ?)
            ON CONFLICT(artifact_id) DO UPDATE SET
                raw_id = excluded.raw_id,
                artifact_kind = excluded.artifact_kind,
                support_status = excluded.support_status,
                classification_reason = excluded.classification_reason,
                parse_as_session = excluded.parse_as_session,
                schema_eligible = excluded.schema_eligible,
                link_group_key = excluded.link_group_key,
                last_observed_at_ms = excluded.last_observed_at_ms
            """,
            (
                observation_id,
                row["raw_id"],
                row["origin"],
                row["source_path"],
                row["source_index"],
                rule.kind,
                support_status,
                f"OriginSpec Claude artifact rule: {rule.coverage_role}",
                int(rule.parse_policy == "session"),
                int(rule.parse_policy == "session"),
                _agent_link_group(str(row["source_path"])),
                first_observed,
                int(row["acquired_at_ms"]),
            ),
        )


def _load_current_artifacts(conn: sqlite3.Connection) -> tuple[_RawArtifact, ...]:
    rows = conn.execute(
        """
        SELECT a.artifact_id, a.raw_id, a.source_path, a.source_index,
               a.artifact_kind, lower(hex(r.blob_hash)) AS blob_hash,
               r.acquired_at_ms
        FROM raw_artifacts AS a
        JOIN raw_sessions AS r ON r.raw_id = a.raw_id
        WHERE a.origin = ?
          AND a.artifact_kind IN (
              'workflow_run_snapshot', 'workflow_journal', 'agent_transcript',
              'agent_sidecar_meta', 'adopt_manifest', 'coordinator_session_stream'
          )
        ORDER BY a.source_path, a.source_index
        """,
        (Origin.CLAUDE_CODE_SESSION.value,),
    ).fetchall()
    return tuple(
        _RawArtifact(
            artifact_id=str(row["artifact_id"]),
            raw_id=str(row["raw_id"]),
            source_path=str(row["source_path"]),
            source_index=int(row["source_index"]),
            artifact_kind=str(row["artifact_kind"]),
            blob_hash=str(row["blob_hash"]),
            acquired_at_ms=int(row["acquired_at_ms"]),
        )
        for row in rows
    )


def _count_retained_revisions(conn: sqlite3.Connection) -> int:
    rows = conn.execute(
        """
        SELECT source_path
        FROM raw_sessions
        WHERE origin = ?
        """,
        (Origin.CLAUDE_CODE_SESSION.value,),
    ).fetchall()
    return sum(artifact_rule_for_path(Provider.CLAUDE_CODE, str(row["source_path"])) is not None for row in rows)


def _load_session_evidence(
    conn: sqlite3.Connection,
    raw_artifacts: Sequence[_RawArtifact],
) -> tuple[ClaudeWorkflowSessionEvidence, ...]:
    evidence: list[ClaudeWorkflowSessionEvidence] = []
    for raw in raw_artifacts:
        if raw.artifact_kind != "agent_transcript":
            continue
        rows = conn.execute(
            """
            SELECT session_id
            FROM sessions
            WHERE raw_id = ? AND origin = ?
            ORDER BY session_id
            """,
            (raw.raw_id, Origin.CLAUDE_CODE_SESSION.value),
        ).fetchall()
        if len(rows) != 1:
            state: Literal["unresolved", "ambiguous"] = "unresolved" if not rows else "ambiguous"
            detail = (
                "transcript has no indexed session" if not rows else f"transcript maps to {len(rows)} indexed sessions"
            )
            evidence.append(
                ClaudeWorkflowSessionEvidence(
                    source_path=raw.source_path,
                    raw_artifact_ref=raw.evidence_ref,
                    session_id=None,
                    session_evidence_ref=None,
                    association_state=state,
                    detail=detail,
                )
            )
            continue
        session_id = str(rows[0]["session_id"])
        prompt = conn.execute(
            """
            SELECT message_id, material_origin
            FROM messages
            WHERE session_id = ? AND role = 'user'
            ORDER BY position, variant_index
            LIMIT 1
            """,
            (session_id,),
        ).fetchone()
        evidence.append(
            ClaudeWorkflowSessionEvidence(
                source_path=raw.source_path,
                raw_artifact_ref=raw.evidence_ref,
                session_id=session_id,
                session_evidence_ref=EvidenceRef(session_id=session_id),
                prompt_evidence_ref=(
                    EvidenceRef(session_id=session_id, message_id=str(prompt["message_id"]))
                    if prompt is not None
                    else None
                ),
                prompt_material_origin=str(prompt["material_origin"]) if prompt is not None else None,
            )
        )
    return tuple(evidence)


def _load_coordinator_invocations(
    conn: sqlite3.Connection,
    raw_artifacts: Sequence[_RawArtifact],
) -> tuple[ClaudeWorkflowCoordinatorInvocation, ...]:
    raw_ids = tuple(raw.raw_id for raw in raw_artifacts if raw.artifact_kind == "coordinator_session_stream")
    if not raw_ids:
        return ()
    placeholders = ",".join("?" for _ in raw_ids)
    rows = conn.execute(
        f"""
        SELECT e.session_id, e.source_message_id, e.source_message_provider_id,
               e.payload_json, e.occurred_at_ms
        FROM session_events AS e
        JOIN sessions AS s ON s.session_id = e.session_id
        WHERE s.raw_id IN ({placeholders})
          AND e.event_type = 'claude_workflow_invocation'
        ORDER BY e.occurred_at_ms, e.session_id, e.position
        """,
        raw_ids,
    ).fetchall()
    values: list[ClaudeWorkflowCoordinatorInvocation] = []
    for row in rows:
        payload = json.loads(str(row["payload_json"] or "{}"))
        if not isinstance(payload, dict):
            continue
        session_id = str(row["session_id"])
        message_id = row["source_message_id"]
        evidence_ref = EvidenceRef(
            session_id=session_id,
            message_id=str(message_id) if message_id is not None else None,
        )
        values.append(
            ClaudeWorkflowCoordinatorInvocation(
                session_id=session_id,
                event=ParsedSessionEvent(
                    event_type="claude_workflow_invocation",
                    payload=payload,
                    source_message_provider_id=(
                        str(row["source_message_provider_id"])
                        if row["source_message_provider_id"] is not None
                        else None
                    ),
                ),
                evidence_ref=evidence_ref,
                occurred_at_ms=int(row["occurred_at_ms"]) if row["occurred_at_ms"] is not None else None,
            )
        )
    return tuple(values)


def _load_coordinator_prompts(
    conn: sqlite3.Connection,
    invocations: Sequence[ClaudeWorkflowCoordinatorInvocation],
) -> tuple[ClaudeWorkflowPromptEvidence, ...]:
    values: list[ClaudeWorkflowPromptEvidence] = []
    for session_id in sorted({item.session_id for item in invocations}):
        row = conn.execute(
            """
            SELECT message_id, material_origin
            FROM messages
            WHERE session_id = ? AND role = 'user'
            ORDER BY position, variant_index
            LIMIT 1
            """,
            (session_id,),
        ).fetchone()
        if row is None:
            continue
        values.append(
            ClaudeWorkflowPromptEvidence(
                session_id=session_id,
                evidence_ref=EvidenceRef(session_id=session_id, message_id=str(row["message_id"])),
                material_origin=str(row["material_origin"]),
            )
        )
    return tuple(values)


def _load_existing_graph_refs(conn: sqlite3.Connection) -> dict[str, str]:
    try:
        rows = conn.execute(
            "SELECT graph_id, corpus_snapshot_ref FROM work_evidence_graphs WHERE graph_id LIKE ? ORDER BY graph_id",
            (f"{_GRAPH_PREFIX}%",),
        ).fetchall()
    except sqlite3.OperationalError:
        return {}
    return {str(row["graph_id"]): str(row["corpus_snapshot_ref"]) for row in rows}


def _run_ids(
    artifacts: Iterable[ClaudeOrchestrationArtifact],
    invocations: Iterable[ClaudeWorkflowCoordinatorInvocation],
) -> tuple[str, ...]:
    values = {fact.run_id for artifact in artifacts for fact in artifact.facts if fact.run_id is not None}
    values.update(
        run_id
        for item in invocations
        if (run_id := _string(item.event.payload, "runId", "run_id", "workflowRunId", "workflow_run_id"))
    )
    return tuple(sorted(values))


def _artifacts_for_run(
    run_id: str,
    artifacts: Sequence[ClaudeOrchestrationArtifact],
) -> tuple[ClaudeOrchestrationArtifact, ...]:
    directly_scoped = [artifact for artifact in artifacts if any(fact.run_id == run_id for fact in artifact.facts)]
    journal_facts = [
        fact for artifact in directly_scoped for fact in artifact.facts if fact.kind == "workflow_journal_entry"
    ]
    keys = {value for fact in journal_facts for value in (fact.attempt_id, fact.agent_id) if value is not None}
    meta_paths = {_normalize_path(path) for fact in journal_facts if (path := fact.meta_path)}
    selected = list(directly_scoped)
    for artifact in artifacts:
        if artifact in selected or artifact.kind != "agent_sidecar_meta":
            continue
        normalized = _normalize_path(artifact.source_path)
        facts = artifact.facts
        if normalized in meta_paths or any(
            value in keys for fact in facts for value in (fact.attempt_id, fact.agent_id) if value is not None
        ):
            selected.append(artifact)
    return tuple(sorted(selected, key=lambda item: (_normalize_path(item.source_path), item.kind)))


def _corpus_snapshot(
    raw_artifacts: Sequence[_RawArtifact],
    invocations: Sequence[ClaudeWorkflowCoordinatorInvocation],
    sessions: Sequence[ClaudeWorkflowSessionEvidence],
    prompts: Sequence[ClaudeWorkflowPromptEvidence],
) -> ObjectRef:
    payload = {
        "semantic_version": CLAUDE_WORKFLOW_MATERIALIZER_VERSION,
        "artifacts": [
            (item.artifact_id, item.raw_id, item.artifact_kind, _normalize_path(item.source_path), item.blob_hash)
            for item in raw_artifacts
        ],
        "invocations": [
            (
                item.session_id,
                item.evidence_ref.format(),
                item.event.source_message_provider_id,
                item.event.payload,
            )
            for item in invocations
        ],
        "sessions": [
            (
                _normalize_path(item.source_path),
                item.raw_artifact_ref.format(),
                item.session_id,
                item.session_evidence_ref.format() if item.session_evidence_ref else None,
                item.prompt_evidence_ref.format() if item.prompt_evidence_ref else None,
                item.prompt_material_origin,
                item.association_state,
            )
            for item in sessions
        ],
        "prompts": [(item.session_id, item.evidence_ref.format(), item.material_origin) for item in prompts],
    }
    canonical = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode()).hexdigest()
    return ObjectRef(
        kind="context-snapshot",
        object_id=f"claude-workflow-v{CLAUDE_WORKFLOW_MATERIALIZER_VERSION}:{digest}",
    )


def _replace_graph_family(index_db: Path, graphs: Sequence[WorkEvidenceGraph]) -> None:
    conn = sqlite3.connect(index_db)
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("DELETE FROM work_evidence_graphs WHERE graph_id LIKE ?", (f"{_GRAPH_PREFIX}%",))
        for graph in graphs:
            conn.execute(
                "INSERT INTO work_evidence_graphs(graph_id, corpus_snapshot_ref) VALUES (?, ?)",
                (graph.graph_id, graph.corpus_snapshot_ref.format()),
            )
            conn.executemany(
                """
                INSERT INTO work_evidence_nodes(
                    graph_id, node_ref, node_kind, label, evidence_refs_json, corpus_snapshot_ref,
                    authority, confidence, occurred_at_ms, actor_ref, execution_context_id,
                    execution_context_known_json, execution_context_unknown_json,
                    execution_context_addressed, association_state, claim_text
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        graph.graph_id,
                        node.ref.format(),
                        node.kind,
                        node.label,
                        json.dumps([ref.format() for ref in node.evidence_refs], separators=(",", ":")),
                        node.corpus_snapshot_ref.format(),
                        node.authority,
                        node.confidence,
                        node.occurred_at_ms,
                        node.actor_ref.format() if node.actor_ref else None,
                        node.execution_context_ref.context_id if node.execution_context_ref else None,
                        (
                            json.dumps(list(node.execution_context_ref.known_fields))
                            if node.execution_context_ref
                            else "[]"
                        ),
                        (
                            json.dumps(list(node.execution_context_ref.unknown_fields))
                            if node.execution_context_ref
                            else "[]"
                        ),
                        int(node.execution_context_ref.content_addressed) if node.execution_context_ref else None,
                        node.association_state,
                        node.claim_text,
                    )
                    for node in graph.nodes
                ],
            )
            conn.executemany(
                """
                INSERT INTO work_evidence_edges(
                    graph_id, edge_ref, edge_kind, source_ref, target_ref, evidence_refs_json,
                    corpus_snapshot_ref, authority, confidence, occurred_at_ms, association_state
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        graph.graph_id,
                        edge.ref.format(),
                        edge.kind,
                        edge.source_ref.format(),
                        edge.target_ref.format(),
                        json.dumps([ref.format() for ref in edge.evidence_refs], separators=(",", ":")),
                        edge.corpus_snapshot_ref.format(),
                        edge.authority,
                        edge.confidence,
                        edge.occurred_at_ms,
                        edge.association_state,
                    )
                    for edge in graph.edges
                ],
            )
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()


def _summarize(
    prepared: _PreparedInputs,
    graphs: Sequence[WorkEvidenceGraph],
) -> ClaudeWorkflowMaterializationSummary:
    nodes = [node for graph in graphs for node in graph.nodes]
    kind_counts = Counter(node.kind for node in nodes)
    current_counts = Counter(item.artifact_kind for item in prepared.raw_artifacts)
    relevant_sidecars = {
        _normalize_path(artifact.source_path)
        for run_id in prepared.run_ids
        for artifact in _artifacts_for_run(run_id, prepared.parsed_artifacts)
        if artifact.kind == "agent_sidecar_meta"
    }
    journal_results = sum(
        fact.is_result
        for artifact in prepared.parsed_artifacts
        if artifact.kind == "workflow_journal"
        for fact in artifact.facts
    )
    final_results = sum(
        any(field in fact.payload for field in ("finalResult", "final_result"))
        for artifact in prepared.parsed_artifacts
        if artifact.kind == "workflow_run_snapshot"
        for fact in artifact.facts
    )
    linked_session_ids = {
        node.ref.object_id.rsplit(":session:", 1)[-1]
        for node in nodes
        if node.kind == "session-segment" and ":session:" in node.ref.object_id
    }
    indexed_agent_sessions = {item.session_id for item in prepared.session_evidence if item.session_id is not None}
    gaps = tuple(
        sorted(
            {
                node.claim_text or node.label
                for node in nodes
                if node.kind == "claim" and node.association_state in {"unresolved", "ambiguous", "contradicted"}
            }
        )
    )
    return ClaudeWorkflowMaterializationSummary(
        semantic_version=CLAUDE_WORKFLOW_MATERIALIZER_VERSION,
        corpus_snapshot_ref=prepared.corpus_snapshot_ref.format(),
        current_artifact_count=len(prepared.raw_artifacts),
        retained_raw_revision_count=prepared.retained_raw_revision_count,
        artifact_counts=dict(sorted(current_counts.items())),
        graph_count=len(graphs),
        run_count=kind_counts["run"],
        coordinator_invocation_count=kind_counts["invocation"],
        call_count=kind_counts["call"],
        attempt_count=kind_counts["attempt"],
        linked_session_count=kind_counts["session-segment"],
        metadata_sidecar_count=len(relevant_sidecars),
        journal_result_count=journal_results,
        final_result_count=final_results,
        unresolved_call_count=sum(node.kind == "call" and node.association_state != "resolved" for node in nodes),
        excluded_session_count=len(indexed_agent_sessions - linked_session_ids),
        generated_prompt_count=sum(
            node.kind == "claim"
            and node.claim_text is not None
            and "worker prompt material origin" in node.claim_text
            and "generated_context_pack" in node.claim_text
            for node in nodes
        ),
        human_prompt_count=sum(prompt.material_origin == "human_authored" for prompt in prepared.coordinator_prompts),
        gaps=gaps,
    )


def _path_identity_fact(raw: _RawArtifact) -> ClaudeOrchestrationFact | None:
    normalized = _normalize_path(raw.source_path)
    if raw.artifact_kind == "workflow_run_snapshot":
        run_id = PurePosixPath(normalized).stem
    elif raw.artifact_kind == "workflow_journal":
        run_id = PurePosixPath(normalized).parent.name
    else:
        return None
    return ClaudeOrchestrationFact(
        kind=raw.artifact_kind,
        source_path=raw.source_path,
        source_line=None,
        run_id=run_id,
        agent_id=None,
        content_key=None,
        payload={},
    )


def _agent_link_group(source_path: str) -> str | None:
    normalized = _normalize_path(source_path).lower()
    for suffix in (".meta.json", ".jsonl", ".ndjson"):
        if normalized.endswith(suffix) and PurePosixPath(normalized).name.startswith("agent-"):
            return normalized[: -len(suffix)]
    return None


def _normalize_path(value: str) -> str:
    normalized = value.replace("\\", "/")
    while "//" in normalized:
        normalized = normalized.replace("//", "/")
    return normalized


def _string(payload: Mapping[str, object], *keys: str) -> str | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    return None


__all__ = [
    "CLAUDE_WORKFLOW_MATERIALIZER_VERSION",
    "ClaudeWorkflowMaterializationSummary",
    "ClaudeWorkflowReparsePlan",
    "claude_workflow_materialization_needed",
    "claude_workflow_reparse_plan",
    "materialize_claude_workflow_archive",
]
