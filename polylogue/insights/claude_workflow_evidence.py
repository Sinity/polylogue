"""Project admitted Claude Workflow evidence into the generic work graph.

The adapter is deliberately topology-blind.  A Claude Code child session joins a
Workflow run only when an admitted journal/sidecar fact explicitly identifies
its transcript.  Raw artifact revisions are cited with ``artifact:raw:<raw_id>``
ObjectRefs; message/session evidence continues to use ``EvidenceRef``.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath

from polylogue.core.refs import EvidenceRef, ObjectRef
from polylogue.insights.work_evidence import (
    WorkEvidenceAssociationState,
    WorkEvidenceEdge,
    WorkEvidenceGraph,
    WorkEvidenceNode,
    WorkEvidenceSourceRef,
)
from polylogue.sources.parsers.base_models import ParsedSessionEvent
from polylogue.sources.parsers.claude.orchestration import ClaudeOrchestrationArtifact, ClaudeOrchestrationFact


@dataclass(frozen=True, slots=True)
class ClaudeWorkflowCoordinatorInvocation:
    """One parsed Workflow tool-use event and its exact message evidence."""

    session_id: str
    event: ParsedSessionEvent
    evidence_ref: EvidenceRef
    occurred_at_ms: int | None = None


@dataclass(frozen=True, slots=True)
class ClaudeWorkflowSessionEvidence:
    """A current transcript artifact and its optional indexed-session binding."""

    source_path: str
    raw_artifact_ref: ObjectRef
    session_id: str | None
    session_evidence_ref: EvidenceRef | None
    prompt_evidence_ref: EvidenceRef | None = None
    prompt_material_origin: str | None = None
    association_state: WorkEvidenceAssociationState = "resolved"
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class ClaudeWorkflowPromptEvidence:
    """Authorship/material-origin evidence for a coordinator prompt."""

    session_id: str
    evidence_ref: EvidenceRef
    material_origin: str


_STATE_RANK: Mapping[WorkEvidenceAssociationState, int] = {
    "resolved": 0,
    "superseded": 1,
    "unresolved": 2,
    "ambiguous": 3,
    "contradicted": 4,
}

_PROVENANCE_FIELDS = (
    "taskId",
    "task_id",
    "resumeFromRunId",
    "resume_from_run_id",
    "workflow",
    "workflowName",
    "workflow_name",
    "name",
    "scriptPath",
    "script_path",
    "scriptHash",
    "script_hash",
    "phases",
    "labels",
    "agentId",
    "agent_id",
    "attemptId",
    "attempt_id",
    "contentKey",
    "content_key",
    "phase",
    "progress",
    "model",
    "status",
    "timing",
    "startedAt",
    "started_at",
    "completedAt",
    "completed_at",
    "tokens",
    "tools",
    "transcriptPath",
    "transcript_path",
    "metaPath",
    "meta_path",
    "result",
    "results",
    "structuredResult",
    "structured_result",
    "finalResult",
    "final_result",
    "entrypoint",
    "adoptedSessionId",
    "adopted_session_id",
    "error",
)


def project_claude_workflow_evidence(
    *,
    graph_id: str,
    run_id: str,
    corpus_snapshot_ref: ObjectRef,
    artifacts: Iterable[ClaudeOrchestrationArtifact],
    artifact_evidence: Mapping[str, ObjectRef],
    coordinator_invocations: Iterable[ClaudeWorkflowCoordinatorInvocation] = (),
    session_evidence: Iterable[ClaudeWorkflowSessionEvidence] = (),
    coordinator_prompt_evidence: Iterable[ClaudeWorkflowPromptEvidence] = (),
) -> WorkEvidenceGraph:
    """Build one replaceable graph for a single provider-reported Workflow run.

    Membership is admitted only from Workflow invocation payloads, journal
    records, sidecar metadata, and explicit transcript paths.  Parent/child
    session topology is intentionally not an input to this function.
    """

    artifacts = tuple(artifacts)
    invocations = tuple(
        item
        for item in coordinator_invocations
        if item.event.event_type == "claude_workflow_invocation"
        and _string(item.event.payload, "runId", "run_id", "workflowRunId", "workflow_run_id") == run_id
    )
    sessions = tuple(session_evidence)
    invocation_session_ids = {item.session_id for item in invocations}
    prompts = tuple(prompt for prompt in coordinator_prompt_evidence if prompt.session_id in invocation_session_ids)

    nodes: dict[str, WorkEvidenceNode] = {}
    edges: dict[str, WorkEvidenceEdge] = {}

    def merge_state(
        left: WorkEvidenceAssociationState,
        right: WorkEvidenceAssociationState,
    ) -> WorkEvidenceAssociationState:
        return left if _STATE_RANK[left] >= _STATE_RANK[right] else right

    def merge_refs(
        left: tuple[WorkEvidenceSourceRef, ...],
        right: Iterable[WorkEvidenceSourceRef],
    ) -> tuple[WorkEvidenceSourceRef, ...]:
        by_format = {ref.format(): ref for ref in left}
        for ref in right:
            by_format.setdefault(ref.format(), ref)
        return tuple(by_format[key] for key in sorted(by_format))

    def add_node(
        *,
        ref: ObjectRef,
        kind: str,
        label: str,
        evidence_refs: Iterable[WorkEvidenceSourceRef],
        state: WorkEvidenceAssociationState = "resolved",
        occurred_at_ms: int | None = None,
        claim_text: str | None = None,
    ) -> ObjectRef:
        refs = tuple(evidence_refs)
        if not refs:
            raise ValueError(f"{ref.format()} has no source evidence")
        key = ref.format()
        existing = nodes.get(key)
        if existing is None:
            nodes[key] = WorkEvidenceNode(
                ref=ref,
                kind=kind,  # type: ignore[arg-type]
                label=label,
                evidence_refs=refs,
                corpus_snapshot_ref=corpus_snapshot_ref,
                authority="provider",
                confidence=1.0,
                occurred_at_ms=occurred_at_ms,
                association_state=state,
                claim_text=claim_text,
            )
        else:
            nodes[key] = existing.model_copy(
                update={
                    "evidence_refs": merge_refs(existing.evidence_refs, refs),
                    "association_state": merge_state(existing.association_state, state),
                    "occurred_at_ms": (
                        existing.occurred_at_ms if existing.occurred_at_ms is not None else occurred_at_ms
                    ),
                }
            )
        return ref

    def add_edge(
        *,
        kind: str,
        source_ref: ObjectRef,
        target_ref: ObjectRef,
        evidence_refs: Iterable[WorkEvidenceSourceRef],
        state: WorkEvidenceAssociationState = "resolved",
        occurred_at_ms: int | None = None,
    ) -> None:
        refs = tuple(evidence_refs)
        if not refs:
            raise ValueError(f"edge {source_ref.format()} -> {target_ref.format()} has no source evidence")
        seed = "\0".join((kind, source_ref.format(), target_ref.format()))
        edge_ref = ObjectRef(kind="work-edge", object_id=f"claude:{hashlib.sha256(seed.encode()).hexdigest()[:32]}")
        key = edge_ref.format()
        existing = edges.get(key)
        if existing is None:
            edges[key] = WorkEvidenceEdge(
                ref=edge_ref,
                kind=kind,  # type: ignore[arg-type]
                source_ref=source_ref,
                target_ref=target_ref,
                evidence_refs=refs,
                corpus_snapshot_ref=corpus_snapshot_ref,
                authority="provider",
                confidence=1.0,
                occurred_at_ms=occurred_at_ms,
                association_state=state,
            )
        else:
            edges[key] = existing.model_copy(
                update={
                    "evidence_refs": merge_refs(existing.evidence_refs, refs),
                    "association_state": merge_state(existing.association_state, state),
                    "occurred_at_ms": (
                        existing.occurred_at_ms if existing.occurred_at_ms is not None else occurred_at_ms
                    ),
                }
            )

    def set_node_state(ref: ObjectRef, state: WorkEvidenceAssociationState) -> None:
        existing = nodes[ref.format()]
        nodes[ref.format()] = existing.model_copy(
            update={"association_state": merge_state(existing.association_state, state)}
        )

    def add_claim(
        *,
        subject_ref: ObjectRef,
        field: str,
        value: object,
        evidence_refs: Iterable[WorkEvidenceSourceRef],
        state: WorkEvidenceAssociationState = "resolved",
        occurred_at_ms: int | None = None,
    ) -> ObjectRef:
        refs = tuple(evidence_refs)
        rendered = _render(value)
        text = f"Claude Workflow {field}: {rendered}"
        seed = "\0".join((subject_ref.format(), field, rendered, *(ref.format() for ref in refs)))
        claim_ref = ObjectRef(kind="work-claim", object_id=f"claude:{hashlib.sha256(seed.encode()).hexdigest()[:32]}")
        add_node(
            ref=claim_ref,
            kind="claim",
            label=field,
            evidence_refs=refs,
            state=state,
            occurred_at_ms=occurred_at_ms,
            claim_text=text,
        )
        add_edge(
            kind="claimed" if state == "resolved" else "unresolved",
            source_ref=subject_ref,
            target_ref=claim_ref,
            evidence_refs=refs,
            state=state,
            occurred_at_ms=occurred_at_ms,
        )
        return claim_ref

    artifact_refs: dict[str, ObjectRef] = {}
    for artifact in artifacts:
        normalized = _normalize_path(artifact.source_path)
        raw_ref = artifact_evidence.get(normalized) or artifact_evidence.get(artifact.source_path)
        if raw_ref is None:
            raise ValueError(f"missing raw-artifact evidence for {artifact.source_path}")
        if raw_ref.kind != "artifact":
            raise ValueError("Claude Workflow artifact evidence must use artifact ObjectRefs")
        state: WorkEvidenceAssociationState = "unresolved" if artifact.parse_error else "resolved"
        artifact_refs[normalized] = add_node(
            ref=raw_ref,
            kind="artifact",
            label=f"Claude {artifact.kind}: {PurePosixPath(normalized).name}",
            evidence_refs=(raw_ref,),
            state=state,
        )
        if artifact.parse_error:
            add_claim(
                subject_ref=raw_ref,
                field="artifact parse gap",
                value=artifact.parse_error,
                evidence_refs=(raw_ref,),
                state="unresolved",
            )

    run_evidence: list[WorkEvidenceSourceRef] = []
    for artifact in artifacts:
        raw_ref = artifact_refs[_normalize_path(artifact.source_path)]
        if any(fact.run_id == run_id for fact in artifact.facts):
            run_evidence.append(raw_ref)
    run_evidence.extend(item.evidence_ref for item in invocations)
    if not run_evidence:
        raise ValueError(f"run {run_id!r} has no admitted source evidence")

    run_ref = add_node(
        ref=ObjectRef(kind="run", object_id=f"claude-workflow:{run_id}"),
        kind="run",
        label=f"Claude Workflow {run_id}",
        evidence_refs=run_evidence,
    )

    run_snapshot_seen = False
    journal_seen = False
    journal_facts: list[tuple[ClaudeOrchestrationFact, ObjectRef]] = []
    sidecar_facts: list[tuple[ClaudeOrchestrationFact, ObjectRef]] = []

    for artifact in artifacts:
        normalized = _normalize_path(artifact.source_path)
        artifact_ref = artifact_refs[normalized]
        relevant_facts = tuple(fact for fact in artifact.facts if fact.run_id == run_id)
        if not relevant_facts:
            continue
        add_edge(kind="mentioned", source_ref=run_ref, target_ref=artifact_ref, evidence_refs=(artifact_ref,))
        if artifact.kind == "workflow_run_snapshot":
            run_snapshot_seen = True
        elif artifact.kind == "workflow_journal":
            journal_seen = True
        for fact in relevant_facts:
            if fact.kind == "workflow_journal_entry":
                journal_facts.append((fact, artifact_ref))
            elif fact.kind == "agent_sidecar_meta":
                sidecar_facts.append((fact, artifact_ref))
            target = run_ref
            for field in _PROVENANCE_FIELDS:
                if field in fact.payload:
                    add_claim(
                        subject_ref=target,
                        field=field,
                        value=fact.payload[field],
                        evidence_refs=(artifact_ref,),
                    )

    if not run_snapshot_seen:
        add_claim(
            subject_ref=run_ref,
            field="coverage gap",
            value="missing workflow run snapshot",
            evidence_refs=run_evidence,
            state="unresolved",
        )
    if not journal_seen:
        add_claim(
            subject_ref=run_ref,
            field="coverage gap",
            value="missing workflow journal",
            evidence_refs=run_evidence,
            state="unresolved",
        )

    previous_invocation: ObjectRef | None = None
    invocation_refs: list[ObjectRef] = []
    for ordinal, invocation in enumerate(invocations, start=1):
        provider_id = invocation.event.source_message_provider_id or f"event-{ordinal}"
        invocation_ref = add_node(
            ref=ObjectRef(
                kind="work-invocation",
                object_id=f"claude-workflow:{run_id}:invocation:{invocation.session_id}:{provider_id}",
            ),
            kind="invocation",
            label=f"Workflow invocation {ordinal}",
            evidence_refs=(invocation.evidence_ref,),
            occurred_at_ms=invocation.occurred_at_ms,
        )
        invocation_refs.append(invocation_ref)
        add_edge(
            kind="invoked",
            source_ref=run_ref,
            target_ref=invocation_ref,
            evidence_refs=(invocation.evidence_ref,),
            occurred_at_ms=invocation.occurred_at_ms,
        )
        resumed = _string(invocation.event.payload, "resumeFromRunId", "resume_from_run_id")
        if resumed and previous_invocation is not None:
            add_edge(
                kind="resumed",
                source_ref=invocation_ref,
                target_ref=previous_invocation,
                evidence_refs=(invocation.evidence_ref,),
                occurred_at_ms=invocation.occurred_at_ms,
            )
        for field in _PROVENANCE_FIELDS:
            if field in invocation.event.payload:
                add_claim(
                    subject_ref=invocation_ref,
                    field=field,
                    value=invocation.event.payload[field],
                    evidence_refs=(invocation.evidence_ref,),
                    occurred_at_ms=invocation.occurred_at_ms,
                )
        previous_invocation = invocation_ref

    for prompt in prompts:
        subject = next(
            (
                ref
                for ref, invocation in zip(invocation_refs, invocations, strict=False)
                if invocation.session_id == prompt.session_id
            ),
            run_ref,
        )
        add_claim(
            subject_ref=subject,
            field="coordinator prompt material origin",
            value=prompt.material_origin,
            evidence_refs=(prompt.evidence_ref,),
            state="resolved" if prompt.material_origin == "human_authored" else "contradicted",
        )

    calls: dict[str, ObjectRef] = {}
    call_evidence: dict[str, list[WorkEvidenceSourceRef]] = defaultdict(list)
    call_has_result: set[str] = set()
    call_unresolved: set[str] = set()
    attempts: dict[str, ObjectRef] = {}
    attempt_facts: dict[str, list[tuple[ClaudeOrchestrationFact, ObjectRef]]] = defaultdict(list)
    call_attempt_order: dict[str, list[str]] = defaultdict(list)

    for fact, raw_ref in journal_facts:
        if fact.content_key is None:
            add_claim(
                subject_ref=run_ref,
                field="journal coverage gap",
                value=f"line {fact.source_line or 0} has no content key",
                evidence_refs=(raw_ref,),
                state="unresolved",
            )
            continue
        content_key = fact.content_key
        call_evidence[content_key].append(raw_ref)
        call_ref = calls.get(content_key)
        if call_ref is None:
            call_ref = add_node(
                ref=ObjectRef(kind="work-call", object_id=f"claude-workflow:{run_id}:call:{content_key}"),
                kind="call",
                label=content_key,
                evidence_refs=(raw_ref,),
            )
            calls[content_key] = call_ref
            add_edge(kind="invoked", source_ref=run_ref, target_ref=call_ref, evidence_refs=(raw_ref,))
        else:
            add_node(ref=call_ref, kind="call", label=content_key, evidence_refs=(raw_ref,))

        if bool(fact.payload.get("unresolved")):
            call_unresolved.add(content_key)
        attempt_id = fact.attempt_id
        attempt_ref: ObjectRef | None = None
        if attempt_id is not None:
            attempt_ref = attempts.get(attempt_id)
            if attempt_ref is None:
                attempt_ref = add_node(
                    ref=ObjectRef(kind="work-attempt", object_id=f"claude-workflow:{run_id}:attempt:{attempt_id}"),
                    kind="attempt",
                    label=attempt_id,
                    evidence_refs=(raw_ref,),
                )
                attempts[attempt_id] = attempt_ref
            else:
                add_node(ref=attempt_ref, kind="attempt", label=attempt_id, evidence_refs=(raw_ref,))
            attempt_facts[attempt_id].append((fact, raw_ref))
            if attempt_id not in call_attempt_order[content_key]:
                prior = call_attempt_order[content_key][-1] if call_attempt_order[content_key] else None
                call_attempt_order[content_key].append(attempt_id)
                add_edge(kind="invoked", source_ref=call_ref, target_ref=attempt_ref, evidence_refs=(raw_ref,))
                if prior is not None:
                    add_edge(
                        kind="retried",
                        source_ref=attempt_ref,
                        target_ref=attempts[prior],
                        evidence_refs=(raw_ref,),
                    )
        if fact.is_result:
            call_has_result.add(content_key)
            result_seed = f"{content_key}:{fact.source_line or 0}:{_render(fact.payload)}"
            result_ref = add_node(
                ref=ObjectRef(
                    kind="work-result",
                    object_id=(
                        f"claude-workflow:{run_id}:result:{hashlib.sha256(result_seed.encode()).hexdigest()[:24]}"
                    ),
                ),
                kind="structured-result",
                label=f"Workflow result for {content_key}",
                evidence_refs=(raw_ref,),
            )
            add_edge(
                kind="produced",
                source_ref=attempt_ref or call_ref,
                target_ref=result_ref,
                evidence_refs=(raw_ref,),
            )
        provenance_target = attempt_ref or call_ref
        for field in _PROVENANCE_FIELDS:
            if field in fact.payload:
                add_claim(
                    subject_ref=provenance_target,
                    field=field,
                    value=fact.payload[field],
                    evidence_refs=(raw_ref,),
                )

    # Final structured result is distinct from the 65 journal result records.
    for artifact in artifacts:
        if artifact.kind != "workflow_run_snapshot":
            continue
        raw_ref = artifact_refs[_normalize_path(artifact.source_path)]
        for fact in artifact.facts:
            if fact.run_id != run_id:
                continue
            for field in ("finalResult", "final_result"):
                if field not in fact.payload:
                    continue
                final_seed = _render(fact.payload[field])
                final_ref = add_node(
                    ref=ObjectRef(
                        kind="work-result",
                        object_id=(
                            f"claude-workflow:{run_id}:final:{hashlib.sha256(final_seed.encode()).hexdigest()[:24]}"
                        ),
                    ),
                    kind="structured-result",
                    label="Workflow final result",
                    evidence_refs=(raw_ref,),
                )
                add_edge(kind="produced", source_ref=run_ref, target_ref=final_ref, evidence_refs=(raw_ref,))

    for content_key, call_ref in calls.items():
        if content_key not in call_has_result or content_key in call_unresolved:
            set_node_state(call_ref, "unresolved")
            refs = call_evidence[content_key]
            add_claim(
                subject_ref=call_ref,
                field="coverage gap",
                value="content key has no completed result",
                evidence_refs=refs,
                state="unresolved",
            )

    sidecars_by_key: dict[str, list[tuple[ClaudeOrchestrationFact, ObjectRef]]] = defaultdict(list)
    sidecars_by_path: dict[str, tuple[ClaudeOrchestrationFact, ObjectRef]] = {}
    for fact, raw_ref in sidecar_facts:
        for key in (fact.attempt_id, fact.agent_id):
            if key:
                sidecars_by_key[key].append((fact, raw_ref))
        sidecars_by_path[_normalize_path(fact.source_path)] = (fact, raw_ref)

    session_by_path: dict[str, list[ClaudeWorkflowSessionEvidence]] = defaultdict(list)
    for session in sessions:
        session_by_path[_normalize_path(session.source_path)].append(session)

    linked_session_ids: set[str] = set()
    for attempt_id, attempt_ref in attempts.items():
        facts = attempt_facts[attempt_id]
        source_refs: list[WorkEvidenceSourceRef] = [raw_ref for _fact, raw_ref in facts]
        explicit_meta = next((fact.meta_path for fact, _raw_ref in facts if fact.meta_path), None)
        sidecar_candidates: list[tuple[ClaudeOrchestrationFact, ObjectRef]] = []
        if explicit_meta:
            sidecar_candidates.extend(_path_fact_candidates(explicit_meta, sidecars_by_path))
        if not sidecar_candidates:
            sidecar_candidates.extend(sidecars_by_key.get(attempt_id, ()))
        if not sidecar_candidates:
            for fact, _raw_ref in facts:
                if fact.agent_id:
                    sidecar_candidates.extend(sidecars_by_key.get(fact.agent_id, ()))
        sidecar_candidates = _dedupe_fact_pairs(sidecar_candidates)
        if not sidecar_candidates:
            add_claim(
                subject_ref=attempt_ref,
                field="coverage gap",
                value="missing paired agent metadata sidecar",
                evidence_refs=source_refs,
                state="unresolved",
            )
            set_node_state(attempt_ref, "unresolved")
            continue
        if len(sidecar_candidates) > 1:
            side_refs = [raw_ref for _fact, raw_ref in sidecar_candidates]
            add_claim(
                subject_ref=attempt_ref,
                field="coverage gap",
                value="ambiguous paired agent metadata sidecar",
                evidence_refs=(*source_refs, *side_refs),
                state="ambiguous",
            )
            set_node_state(attempt_ref, "ambiguous")
            continue
        sidecar_fact, sidecar_ref = sidecar_candidates[0]
        source_refs.append(sidecar_ref)
        for field in _PROVENANCE_FIELDS:
            if field in sidecar_fact.payload:
                add_claim(
                    subject_ref=attempt_ref,
                    field=field,
                    value=sidecar_fact.payload[field],
                    evidence_refs=(sidecar_ref,),
                )

        transcript_reference = next((fact.transcript_path for fact, _raw_ref in facts if fact.transcript_path), None)
        transcript_reference = transcript_reference or sidecar_fact.transcript_path
        if not transcript_reference:
            add_claim(
                subject_ref=attempt_ref,
                field="coverage gap",
                value="metadata has no transcript reference",
                evidence_refs=source_refs,
                state="unresolved",
            )
            set_node_state(attempt_ref, "unresolved")
            continue
        transcript_candidates = _path_session_candidates(transcript_reference, session_by_path)
        if not transcript_candidates:
            add_claim(
                subject_ref=attempt_ref,
                field="coverage gap",
                value=f"missing transcript {transcript_reference}",
                evidence_refs=source_refs,
                state="unresolved",
            )
            set_node_state(attempt_ref, "unresolved")
            continue
        if len(transcript_candidates) > 1:
            refs = [candidate.raw_artifact_ref for candidate in transcript_candidates]
            add_claim(
                subject_ref=attempt_ref,
                field="coverage gap",
                value=f"ambiguous transcript reference {transcript_reference}",
                evidence_refs=(*source_refs, *refs),
                state="ambiguous",
            )
            set_node_state(attempt_ref, "ambiguous")
            continue
        transcript = transcript_candidates[0]
        transcript_refs: list[WorkEvidenceSourceRef] = [transcript.raw_artifact_ref]
        if transcript.session_evidence_ref is not None:
            transcript_refs.append(transcript.session_evidence_ref)
        if transcript.prompt_evidence_ref is not None:
            transcript_refs.append(transcript.prompt_evidence_ref)
        state = transcript.association_state
        if transcript.session_id is None or transcript.session_evidence_ref is None:
            state = merge_state(state, "unresolved")
        if state != "resolved":
            add_claim(
                subject_ref=attempt_ref,
                field="coverage gap",
                value=transcript.detail or f"transcript {transcript_reference} is not uniquely indexed",
                evidence_refs=(*source_refs, *transcript_refs),
                state=state,
            )
            set_node_state(attempt_ref, state)
            continue
        assert transcript.session_id is not None
        segment_ref = add_node(
            ref=ObjectRef(
                kind="work-session-segment",
                object_id=f"claude-workflow:{run_id}:attempt:{attempt_id}:session:{transcript.session_id}",
            ),
            kind="session-segment",
            label=f"Attempt transcript {attempt_id}",
            evidence_refs=transcript_refs,
        )
        linked_session_ids.add(transcript.session_id)
        add_edge(
            kind="represented_by",
            source_ref=attempt_ref,
            target_ref=segment_ref,
            evidence_refs=(*source_refs, *transcript_refs),
        )
        if transcript.prompt_evidence_ref is not None and transcript.prompt_material_origin is not None:
            state = "resolved" if transcript.prompt_material_origin == "generated_context_pack" else "contradicted"
            add_claim(
                subject_ref=segment_ref,
                field="worker prompt material origin",
                value=transcript.prompt_material_origin,
                evidence_refs=(transcript.prompt_evidence_ref,),
                state=state,
            )

    return WorkEvidenceGraph(
        graph_id=graph_id,
        corpus_snapshot_ref=corpus_snapshot_ref,
        nodes=tuple(nodes[key] for key in sorted(nodes)),
        edges=tuple(edges[key] for key in sorted(edges)),
    )


def _normalize_path(value: str) -> str:
    normalized = value.replace("\\", "/")
    while "//" in normalized:
        normalized = normalized.replace("//", "/")
    return normalized


def _path_fact_candidates(
    reference: str,
    candidates: Mapping[str, tuple[ClaudeOrchestrationFact, ObjectRef]],
) -> list[tuple[ClaudeOrchestrationFact, ObjectRef]]:
    normalized = _normalize_path(reference)
    exact = candidates.get(normalized)
    if exact is not None:
        return [exact]
    suffix = "/" + normalized.lstrip("/")
    return [value for path, value in candidates.items() if path.endswith(suffix)]


def _path_session_candidates(
    reference: str,
    candidates: Mapping[str, list[ClaudeWorkflowSessionEvidence]],
) -> list[ClaudeWorkflowSessionEvidence]:
    normalized = _normalize_path(reference)
    exact = candidates.get(normalized)
    if exact:
        return list(exact)
    suffix = "/" + normalized.lstrip("/")
    resolved = [item for path, values in candidates.items() if path.endswith(suffix) for item in values]
    if resolved:
        return resolved
    # Provider metadata sometimes records only ``agent-*.jsonl``.  A basename
    # match is accepted only when unique; duplicate basenames remain ambiguous.
    if "/" not in normalized:
        return [
            item for path, values in candidates.items() if PurePosixPath(path).name == normalized for item in values
        ]
    return []


def _dedupe_fact_pairs(
    values: Iterable[tuple[ClaudeOrchestrationFact, ObjectRef]],
) -> list[tuple[ClaudeOrchestrationFact, ObjectRef]]:
    deduped: dict[str, tuple[ClaudeOrchestrationFact, ObjectRef]] = {}
    for fact, ref in values:
        deduped.setdefault(ref.format(), (fact, ref))
    return [deduped[key] for key in sorted(deduped)]


def _string(payload: Mapping[str, object], *keys: str) -> str | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _render(value: object) -> str:
    try:
        return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    except TypeError:
        return repr(value)


__all__ = [
    "ClaudeWorkflowCoordinatorInvocation",
    "ClaudeWorkflowPromptEvidence",
    "ClaudeWorkflowSessionEvidence",
    "project_claude_workflow_evidence",
]
