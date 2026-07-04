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
    advisories: tuple[str, ...] = ()
    limits: CoordinationLimitsPayload
    provenance: tuple[CoordinationProvenancePayload, ...] = ()
