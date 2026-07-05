"""Typed payloads for agent coordination projections."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from polylogue.surfaces.payloads import SurfacePayloadModel

CoordinationView = Literal["status", "self", "work-item", "conflicts", "handoff"]


class CoordinationProvenancePayload(SurfacePayloadModel):
    source: str
    command: tuple[str, ...] = ()
    path: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    freshness: str
    note: str | None = None


class CoordinationRepoPayload(SurfacePayloadModel):
    cwd: str
    root: str | None = None
    branch: str | None = None
    head: str | None = None
    dirty: bool = False
    changed_paths: tuple[str, ...] = ()
    provenance: CoordinationProvenancePayload


class CoordinationSelfPayload(SurfacePayloadModel):
    agent_kind: str
    pid: int
    cwd: str
    branch: str | None = None
    session_ref: str | None = None
    provenance: CoordinationProvenancePayload


class CoordinationWorkItemPayload(SurfacePayloadModel):
    source: Literal["beads", "git", "inferred", "none"]
    ref: str | None = None
    title: str | None = None
    status: str | None = None
    priority: int | None = None
    assignee: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    provenance: CoordinationProvenancePayload
    fields: dict[str, object] = Field(default_factory=dict)


class CoordinationPeerPayload(SurfacePayloadModel):
    pid: int
    kind: str
    command: str
    cwd: str | None = None
    provenance: CoordinationProvenancePayload


class CoordinationResourceEpisodePayload(SurfacePayloadModel):
    pid: int
    kind: str
    command: str
    status: str
    scope: str | None = None
    provenance: CoordinationProvenancePayload


class CoordinationOverlapPayload(SurfacePayloadModel):
    kind: str
    severity: Literal["info", "warning", "critical"] = "info"
    blocking: bool = False
    summary: str
    refs: tuple[str, ...] = ()
    provenance: CoordinationProvenancePayload


class CoordinationHandoffPayload(SurfacePayloadModel):
    path: str
    kind: str
    exists: bool
    updated_at: str | None = None
    bytes: int | None = None
    provenance: CoordinationProvenancePayload


class CoordinationArchivePayload(SurfacePayloadModel):
    archive_root: str
    index_db: str
    index_exists: bool
    index_user_version: int | None = None
    source_user_version: int | None = None
    user_user_version: int | None = None
    daemon_processes: tuple[CoordinationResourceEpisodePayload, ...] = ()
    provenance: CoordinationProvenancePayload


class CoordinationSessionTreeNodePayload(SurfacePayloadModel):
    session_id: str
    source_name: str | None = None
    title: str | None = None
    depth: int = 0
    is_target: bool = False


class CoordinationSessionTreeEdgePayload(SurfacePayloadModel):
    child_id: str
    parent_id: str | None = None
    parent_native_id: str | None = None
    kind: str
    resolved: bool = True


class CoordinationSessionTreePayload(SurfacePayloadModel):
    target_session_id: str
    root_session_id: str
    nodes: tuple[CoordinationSessionTreeNodePayload, ...] = ()
    edges: tuple[CoordinationSessionTreeEdgePayload, ...] = ()
    cycle_detected: bool = False
    provenance: CoordinationProvenancePayload


class CoordinationActivityEpisodePayload(SurfacePayloadModel):
    ref: str
    session_id: str
    run_ref: str | None = None
    kind: str
    status: str | None = None
    summary: str | None = None
    occurred_at: str | None = None
    refs: tuple[str, ...] = ()
    provenance: CoordinationProvenancePayload


class CoordinationSubagentExchangePayload(SurfacePayloadModel):
    ref: str
    session_id: str
    run_ref: str
    agent_ref: str | None = None
    dispatch_prompt: str | None = None
    returned_final_message: str | None = None
    status: str | None = None
    child_session_id: str | None = None
    context_snapshot_ref: str | None = None
    evidence_refs: tuple[str, ...] = ()
    provenance: CoordinationProvenancePayload


class CoordinationProofRefPayload(SurfacePayloadModel):
    ref: str
    session_id: str
    kind: str
    status: str | None = None
    summary: str | None = None
    evidence_refs: tuple[str, ...] = ()
    provenance: CoordinationProvenancePayload


class CoordinationContextFlowRefPayload(SurfacePayloadModel):
    ref: str
    session_id: str
    run_ref: str | None = None
    boundary: str
    inheritance_mode: str | None = None
    segment_refs: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()
    provenance: CoordinationProvenancePayload


class CoordinationBeadsHookPayload(SurfacePayloadModel):
    name: str
    installed: bool
    version: str | None = None
    is_shim: bool | None = None
    outdated: bool | None = None


class CoordinationBeadsGatePayload(SurfacePayloadModel):
    id: str | None = None
    title: str | None = None
    status: str | None = None
    gate_type: str | None = None
    await_id: str | None = None


class CoordinationBeadsMergeSlotPayload(SurfacePayloadModel):
    id: str | None = None
    available: bool | None = None
    status: str | None = None
    holder: str | None = None
    waiters: tuple[str, ...] = ()
    error: str | None = None


class CoordinationBeadsPayload(SurfacePayloadModel):
    root: str
    hooks: tuple[CoordinationBeadsHookPayload, ...] = ()
    hooks_all_installed: bool | None = None
    hooks_outdated_count: int | None = None
    gates: tuple[CoordinationBeadsGatePayload, ...] = ()
    open_gate_count: int | None = None
    merge_slot: CoordinationBeadsMergeSlotPayload | None = None
    provenance: CoordinationProvenancePayload


class CoordinationLimitsPayload(SurfacePayloadModel):
    peer_limit: int
    resource_limit: int
    changed_path_limit: int
    command_chars: int


class AgentCoordinationPayload(SurfacePayloadModel):
    view: CoordinationView
    generated_at: str
    repo: CoordinationRepoPayload
    self: CoordinationSelfPayload
    work_item: CoordinationWorkItemPayload
    peers: tuple[CoordinationPeerPayload, ...] = ()
    resource_episodes: tuple[CoordinationResourceEpisodePayload, ...] = ()
    overlaps: tuple[CoordinationOverlapPayload, ...] = ()
    handoff: tuple[CoordinationHandoffPayload, ...] = ()
    archive: CoordinationArchivePayload | None = None
    session_trees: tuple[CoordinationSessionTreePayload, ...] = ()
    activity_episodes: tuple[CoordinationActivityEpisodePayload, ...] = ()
    subagent_exchanges: tuple[CoordinationSubagentExchangePayload, ...] = ()
    proof_refs: tuple[CoordinationProofRefPayload, ...] = ()
    context_flow_refs: tuple[CoordinationContextFlowRefPayload, ...] = ()
    beads: CoordinationBeadsPayload | None = None
    advisories: tuple[str, ...] = ()
    limits: CoordinationLimitsPayload
    provenance: tuple[CoordinationProvenancePayload, ...] = ()
