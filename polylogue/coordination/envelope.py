"""Build bounded coordination envelopes from local repo and archive evidence."""

from __future__ import annotations

import json
import os
import re
import shlex
import sqlite3
import subprocess
from collections.abc import Callable, MutableMapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import TypeVar, cast

from polylogue.coordination.payloads import (
    AgentCoordinationPayload,
    CoordinationActivityEpisodePayload,
    CoordinationArchivePayload,
    CoordinationBeadsGatePayload,
    CoordinationBeadsHookPayload,
    CoordinationBeadsMergeSlotPayload,
    CoordinationBeadsPayload,
    CoordinationContextFlowRefPayload,
    CoordinationHandoffPayload,
    CoordinationLimitsPayload,
    CoordinationOverlapPayload,
    CoordinationPeerPayload,
    CoordinationProjectionPayload,
    CoordinationProofRefPayload,
    CoordinationProvenancePayload,
    CoordinationRepoPayload,
    CoordinationResourceEpisodePayload,
    CoordinationSelfPayload,
    CoordinationSessionTreeEdgePayload,
    CoordinationSessionTreeNodePayload,
    CoordinationSessionTreePayload,
    CoordinationSubagentExchangePayload,
    CoordinationView,
    CoordinationWorkItemPayload,
)
from polylogue.logging import get_logger
from polylogue.paths import active_index_db_path, archive_root
from polylogue.storage.sqlite.run_projection_relations import (
    context_snapshot_relation_sql,
    observed_event_relation_sql,
    run_relation_sql,
)

logger = get_logger(__name__)

CommandRunner = Callable[[Sequence[str], Path | None], "CommandResult"]
_StageResult = TypeVar("_StageResult")

_COMMAND_CHARS = 220
_CHANGED_PATH_LIMIT = 40
# Leave headroom for the MCP brief's fixed instructions while keeping the
# complete default response below the bead's 8 KiB transport ceiling.
_COMPACT_BYTE_BUDGET = 7_600
_PROCESS_COMMAND = ("ps", "ww", "-eo", "pid=,ppid=,comm=,cgroup:200=,args=")
_BEADS_PROBE_TIMEOUT_SECONDS = 0.35
_AGENT_NAMES = ("codex", "claude", "gemini")
_SYSTEM_RESOURCE_NAMES = frozenset(
    {
        "below",
        "dbus-daemon",
        "earlyoom",
        "nix-daemon",
        "systemd-oomd",
        "systemd-resolved",
        "systemd-timesyncd",
        "systemd-udevd",
    }
)
_WORK_SCOPE_RE = re.compile(r"(?:sinnix-(?:background|build|nix-build)|polylogue[^/]*(?:rebuild|verify|test))")


@dataclass(frozen=True, slots=True)
class CommandResult:
    args: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str


@dataclass(frozen=True, slots=True)
class _ProcessRow:
    pid: int
    ppid: int
    comm: str
    cgroup: str
    command: str


def build_coordination_envelope(
    *,
    view: CoordinationView = "status",
    cwd: Path | None = None,
    limit: int = 10,
    detail: bool = False,
    runner: CommandRunner | None = None,
    stage_timings_ms: MutableMapping[str, float] | None = None,
) -> AgentCoordinationPayload:
    """Return a bounded, JSON-first coordination envelope for agents.

    ``stage_timings_ms`` is an optional caller-owned diagnostic sink.  It is
    intentionally not projected into the agent payload: timing collection is
    for the benchmark harness, while the compact payload remains bounded and
    semantically stable.
    """

    now = datetime.now(UTC).isoformat()
    root_cwd = (cwd or Path.cwd()).resolve()
    command_runner = runner or _run_command
    peer_limit = max(1, min(limit, 50))
    resource_limit = max(1, min(limit, 50))

    repo = _timed_stage(stage_timings_ms, "repo", lambda: _repo_payload(root_cwd, command_runner))
    process_result, process_rows = _timed_stage(stage_timings_ms, "process", lambda: _process_snapshot(command_runner))
    self_payload = _timed_stage(
        stage_timings_ms,
        "self",
        lambda: _self_payload(root_cwd, repo, process_rows, process_result),
    )
    work_item = _timed_stage(
        stage_timings_ms,
        "work_item",
        lambda: _work_item_payload(root_cwd, repo, command_runner),
    )
    beads = _timed_stage(stage_timings_ms, "beads", lambda: _beads_payload(root_cwd, repo, command_runner))
    all_peers = _timed_stage(
        stage_timings_ms,
        "peers",
        lambda: _logical_peer_payloads(
            process_rows,
            process_result,
            owner_pid=self_payload.owner_pid,
            invocation_pid=self_payload.invocation_pid,
        ),
    )
    all_resources = _timed_stage(
        stage_timings_ms,
        "resources",
        lambda: _resource_scope_payloads(
            process_rows,
            process_result,
            root_cwd,
            archive_resource=_configured_archive_resource(),
        ),
    )
    peers = all_peers[:peer_limit]
    resources = all_resources[:resource_limit]
    archive = _timed_stage(stage_timings_ms, "archive", lambda: _archive_payload(resources))
    handoff = _timed_stage(
        stage_timings_ms,
        "handoff",
        lambda: _handoff_payloads(repo.root or str(root_cwd), archive=archive, limit=peer_limit),
    )
    (
        session_trees,
        activity_episodes,
        subagent_exchanges,
        proof_refs,
        context_flow_refs,
        archive_evidence_degraded_reason,
    ) = _timed_stage(
        stage_timings_ms,
        "archive_evidence",
        lambda: _archive_evidence_payloads(
            repo,
            self_payload,
            archive,
            limit=peer_limit,
        ),
    )
    overlaps = _timed_stage(stage_timings_ms, "overlaps", lambda: _overlap_payloads(repo, work_item, peers, resources))
    advisories = _timed_stage(
        stage_timings_ms,
        "advisories",
        lambda: _advisories(repo, work_item, overlaps, archive, archive_evidence_degraded_reason),
    )
    provenance = (repo.provenance, work_item.provenance, self_payload.provenance)
    payload = AgentCoordinationPayload(
        view=view,
        generated_at=now,
        repo=repo,
        self=self_payload,
        work_item=work_item,
        peers=peers,
        resource_episodes=resources,
        overlaps=overlaps,
        handoff=handoff,
        archive=archive,
        session_trees=session_trees,
        activity_episodes=activity_episodes,
        subagent_exchanges=subagent_exchanges,
        proof_refs=proof_refs,
        context_flow_refs=context_flow_refs,
        beads=beads,
        advisories=advisories,
        limits=CoordinationLimitsPayload(
            peer_limit=peer_limit,
            resource_limit=resource_limit,
            changed_path_limit=_CHANGED_PATH_LIMIT,
            command_chars=_COMMAND_CHARS,
        ),
        projection=CoordinationProjectionPayload(detail=detail),
        provenance=provenance,
    )
    projected = project_coordination_envelope(payload, view)
    total_counts = _coordination_counts(projected)
    total_counts["peers"] = len(all_peers)
    total_counts["resource_episodes"] = len(all_resources)
    total_counts["peer_components"] = sum(peer.component_count for peer in all_peers)
    total_counts["resource_components"] = sum(episode.component_count for episode in all_resources)
    total_counts["resource_refs"] = sum(episode.resource_count for episode in all_resources)
    if detail:
        return _timed_stage(
            stage_timings_ms,
            "projection",
            lambda: _finalize_projection(projected, total_counts=total_counts, detail=True),
        )
    return _timed_stage(
        stage_timings_ms,
        "projection",
        lambda: _compact_coordination_payload(projected, total_counts=total_counts),
    )


def _timed_stage(
    timings: MutableMapping[str, float] | None,
    name: str,
    call: Callable[[], _StageResult],
) -> _StageResult:
    """Execute one envelope stage and record its wall time when requested."""

    started = perf_counter()
    try:
        return call()
    finally:
        if timings is not None:
            timings[name] = round((perf_counter() - started) * 1_000, 3)


def project_coordination_envelope(
    payload: AgentCoordinationPayload,
    view: CoordinationView,
) -> AgentCoordinationPayload:
    """Return the same typed envelope with non-relevant arrays bounded by view."""

    if view == "status":
        return payload.model_copy(update={"view": view})
    if view == "self":
        return payload.model_copy(
            update={
                "view": view,
                "peers": (),
                "resource_episodes": (),
                "overlaps": (),
                "handoff": (),
                "beads": None,
            }
        )
    if view == "work-item":
        return payload.model_copy(
            update={
                "view": view,
                "peers": (),
                "resource_episodes": (),
                "overlaps": (),
                "handoff": (),
                "beads": payload.beads,
            }
        )
    if view == "conflicts":
        return payload.model_copy(update={"view": view, "handoff": ()})
    if view == "handoff":
        return payload.model_copy(
            update={
                "view": view,
                "peers": (),
                "resource_episodes": (),
                "overlaps": (),
                "beads": None,
            }
        )
    return payload


_COUNTED_FAMILIES = (
    "peers",
    "resource_episodes",
    "overlaps",
    "handoff",
    "session_trees",
    "activity_episodes",
    "subagent_exchanges",
    "proof_refs",
    "context_flow_refs",
    "advisories",
    "provenance",
)


def _coordination_counts(payload: AgentCoordinationPayload) -> dict[str, int]:
    counts = {name: len(cast(Sequence[object], getattr(payload, name))) for name in _COUNTED_FAMILIES}
    counts["changed_paths"] = len(payload.repo.changed_paths)
    counts["work_item_fields"] = len(payload.work_item.fields)
    if payload.beads is not None:
        counts["beads_hooks"] = len(payload.beads.hooks)
        counts["beads_gates"] = len(payload.beads.gates)
        counts["beads_merge_waiters"] = len(payload.beads.merge_slot.waiters) if payload.beads.merge_slot else 0
        counts["beads_gate_text_chars"] = sum(
            _text_chars(gate.id, gate.title, gate.status, gate.gate_type, gate.await_id) for gate in payload.beads.gates
        )
        merge_slot = payload.beads.merge_slot
        counts["beads_merge_text_chars"] = (
            _text_chars(
                merge_slot.id,
                merge_slot.status,
                merge_slot.holder,
                merge_slot.error,
                *merge_slot.waiters,
            )
            if merge_slot is not None
            else 0
        )
    if payload.archive is not None:
        counts["archive_hook_states"] = len(payload.archive.hook_flow_states)
        counts["archive_hook_gaps"] = len(payload.archive.hook_flow_gaps)
        counts["archive_daemon_processes"] = len(payload.archive.daemon_processes)
    counts["session_tree_nodes"] = sum(len(tree.nodes) for tree in payload.session_trees)
    counts["session_tree_edges"] = sum(len(tree.edges) for tree in payload.session_trees)
    counts["peer_components"] = sum(len(peer.component_pids) for peer in payload.peers)
    counts["resource_components"] = sum(len(episode.component_pids) for episode in payload.resource_episodes)
    counts["resource_refs"] = sum(len(episode.resources) for episode in payload.resource_episodes)
    counts["activity_refs"] = sum(len(episode.refs) for episode in payload.activity_episodes)
    counts["proof_evidence_refs"] = sum(len(proof.evidence_refs) for proof in payload.proof_refs)
    counts["context_segment_refs"] = sum(len(ref.segment_refs) for ref in payload.context_flow_refs)
    counts["context_evidence_refs"] = sum(len(ref.evidence_refs) for ref in payload.context_flow_refs)
    return counts


def _compact_coordination_payload(
    payload: AgentCoordinationPayload,
    *,
    total_counts: dict[str, int],
) -> AgentCoordinationPayload:
    repo = payload.repo.model_copy(
        update={
            "changed_paths": payload.repo.changed_paths[:8],
            "provenance": _compact_provenance(payload.repo.provenance),
        }
    )
    self_payload = payload.self.model_copy(update={"provenance": _compact_provenance(payload.self.provenance)})
    work_item = payload.work_item.model_copy(
        update={"fields": {}, "provenance": _compact_provenance(payload.work_item.provenance, keep_note=True)}
    )
    peers = tuple(
        peer.model_copy(
            update={
                "command": _short_to(peer.command, 120),
                "component_pids": peer.component_pids[:4],
                "provenance": _compact_provenance(peer.provenance),
            }
        )
        for peer in payload.peers[:4]
    )
    resources = tuple(_compact_resource_episode(episode) for episode in payload.resource_episodes[:4])
    overlaps = tuple(
        overlap.model_copy(update={"refs": overlap.refs[:4], "provenance": _compact_provenance(overlap.provenance)})
        for overlap in payload.overlaps[:4]
    )
    handoff = tuple(
        item.model_copy(update={"provenance": _compact_provenance(item.provenance)}) for item in payload.handoff[:3]
    )
    session_trees = tuple(
        tree.model_copy(
            update={
                "nodes": tuple(
                    node.model_copy(update={"title": _short_to(node.title, 100) if node.title else None})
                    for node in tree.nodes[:3]
                ),
                "edges": tree.edges[:3],
                "provenance": _compact_provenance(tree.provenance),
            }
        )
        for tree in payload.session_trees[:1]
    )
    activity = tuple(
        item.model_copy(
            update={
                "summary": _short_to(item.summary, 140) if item.summary else None,
                "refs": item.refs[:2],
                "provenance": _compact_provenance(item.provenance),
            }
        )
        for item in payload.activity_episodes[:2]
    )
    proofs = tuple(
        item.model_copy(
            update={
                "summary": _short_to(item.summary, 140) if item.summary else None,
                "evidence_refs": item.evidence_refs[:2],
                "provenance": _compact_provenance(item.provenance),
            }
        )
        for item in payload.proof_refs[:2]
    )
    context_refs = tuple(
        item.model_copy(
            update={
                "segment_refs": item.segment_refs[:2],
                "evidence_refs": item.evidence_refs[:2],
                "provenance": _compact_provenance(item.provenance),
            }
        )
        for item in payload.context_flow_refs[:2]
    )
    archive = payload.archive
    if archive is not None:
        archive = archive.model_copy(
            update={
                "hook_flow_states": dict(tuple(sorted(archive.hook_flow_states.items()))[:6]),
                "hook_flow_gaps": archive.hook_flow_gaps[:4],
                "daemon_processes": tuple(
                    _compact_daemon_reference(episode) for episode in archive.daemon_processes[:2]
                ),
                "provenance": _compact_provenance(archive.provenance),
            }
        )
    beads = payload.beads
    if beads is not None:
        beads = beads.model_copy(
            update={
                "root": _short_to(beads.root, 200),
                "hooks": beads.hooks[:3],
                "gates": tuple(_compact_beads_gate(gate) for gate in beads.gates[:2]),
                "merge_slot": _compact_beads_merge_slot(beads.merge_slot),
                "provenance": _compact_provenance(beads.provenance),
            }
        )
    compact = payload.model_copy(
        update={
            "repo": repo,
            "self": self_payload,
            "work_item": work_item,
            "peers": peers,
            "resource_episodes": resources,
            "overlaps": overlaps,
            "handoff": handoff,
            "archive": archive,
            "session_trees": session_trees,
            "activity_episodes": activity,
            "subagent_exchanges": (),
            "proof_refs": proofs,
            "context_flow_refs": context_refs,
            "beads": beads,
            "advisories": tuple(_short_to(item, 180) for item in payload.advisories[:3]),
            "provenance": (),
        }
    )
    compact = _finalize_projection(compact, total_counts=total_counts, detail=False)
    for family in ("context_flow_refs", "activity_episodes", "proof_refs", "session_trees", "handoff"):
        if _serialized_size(compact) <= _COMPACT_BYTE_BUDGET:
            break
        compact = compact.model_copy(update={family: ()})
        compact = _finalize_projection(compact, total_counts=total_counts, detail=False)
    if _serialized_size(compact) > _COMPACT_BYTE_BUDGET and compact.beads is not None:
        compact = compact.model_copy(update={"beads": compact.beads.model_copy(update={"hooks": ()})})
        compact = _finalize_projection(compact, total_counts=total_counts, detail=False)
    if _serialized_size(compact) > _COMPACT_BYTE_BUDGET and compact.archive is not None:
        compact = compact.model_copy(
            update={
                "archive": compact.archive.model_copy(update={"hook_flow_states": {}, "hook_flow_gaps": ()}),
                "overlaps": (),
                "advisories": (),
            }
        )
        compact = _finalize_projection(compact, total_counts=total_counts, detail=False)
    if _serialized_size(compact) > _COMPACT_BYTE_BUDGET:
        archive_writers = tuple(episode for episode in compact.resource_episodes if episode.kind == "daemon")[:2]
        compact = compact.model_copy(
            update={
                "peers": compact.peers[:1],
                "resource_episodes": archive_writers or compact.resource_episodes[:1],
            }
        )
        compact = _finalize_projection(compact, total_counts=total_counts, detail=False)
    if _serialized_size(compact) > _COMPACT_BYTE_BUDGET and compact.repo.changed_paths:
        compact = compact.model_copy(update={"repo": compact.repo.model_copy(update={"changed_paths": ()})})
        compact = _finalize_projection(compact, total_counts=total_counts, detail=False)
    if _serialized_size(compact) > _COMPACT_BYTE_BUDGET:
        compact = _terminal_compact_payload(compact, total_counts=total_counts)
    return compact


def _compact_resource_episode(
    episode: CoordinationResourceEpisodePayload,
) -> CoordinationResourceEpisodePayload:
    return episode.model_copy(
        update={
            "command": _short_to(episode.command, 100),
            "scope": None if episode.unit else episode.scope,
            "resources": episode.resources[:2],
            "component_pids": episode.component_pids[:4],
            "provenance": _compact_provenance(episode.provenance),
        }
    )


def _compact_daemon_reference(
    episode: CoordinationResourceEpisodePayload,
) -> CoordinationResourceEpisodePayload:
    resources = episode.resources[:1]
    return episode.model_copy(
        update={
            "command": "",
            "scope": None,
            "cwd": None,
            "resource_count": len(resources),
            "resources": resources,
            "component_count": 0,
            "component_pids": (),
            "provenance": _compact_provenance(episode.provenance),
        }
    )


def _compact_beads_gate(gate: CoordinationBeadsGatePayload, *, limit: int = 96) -> CoordinationBeadsGatePayload:
    return gate.model_copy(
        update={
            "id": _short_optional(gate.id, limit),
            "title": _short_optional(gate.title, limit),
            "status": _short_optional(gate.status, 32),
            "gate_type": _short_optional(gate.gate_type, 32),
            "await_id": _short_optional(gate.await_id, limit),
        }
    )


def _compact_beads_merge_slot(
    merge_slot: CoordinationBeadsMergeSlotPayload | None,
    *,
    limit: int = 96,
) -> CoordinationBeadsMergeSlotPayload | None:
    if merge_slot is None:
        return None
    return merge_slot.model_copy(
        update={
            "id": _short_optional(merge_slot.id, limit),
            "status": _short_optional(merge_slot.status, 32),
            "holder": _short_optional(merge_slot.holder, limit),
            "waiters": tuple(_short_to(waiter, limit) for waiter in merge_slot.waiters[:3]),
            "error": _short_optional(merge_slot.error, 160),
        }
    )


def _terminal_compact_payload(
    payload: AgentCoordinationPayload,
    *,
    total_counts: dict[str, int],
) -> AgentCoordinationPayload:
    writers = tuple(
        _compact_daemon_reference(episode) for episode in payload.resource_episodes if episode.kind == "daemon"
    )[:2]
    if not writers and payload.resource_episodes:
        writers = (_compact_resource_episode(payload.resource_episodes[0]),)
    archive = payload.archive
    if archive is not None:
        archive = archive.model_copy(
            update={
                "archive_root": _short_to(archive.archive_root, 160),
                "index_db": _short_to(archive.index_db, 180),
                "hook_flow_states": {},
                "hook_flow_gaps": (),
                "daemon_processes": tuple(
                    _compact_daemon_reference(episode) for episode in archive.daemon_processes[:2]
                ),
                "provenance": _compact_provenance(archive.provenance),
            }
        )
    beads = payload.beads
    if beads is not None:
        beads = beads.model_copy(
            update={
                "root": _short_to(beads.root, 120),
                "hooks": (),
                "gates": tuple(_compact_beads_gate(gate, limit=64) for gate in beads.gates[:1]),
                "merge_slot": _compact_beads_merge_slot(beads.merge_slot, limit=64),
                "provenance": _compact_provenance(beads.provenance),
            }
        )
    minimal = payload.model_copy(
        update={
            "repo": payload.repo.model_copy(
                update={
                    "cwd": _short_to(payload.repo.cwd, 120),
                    "root": _short_optional(payload.repo.root, 120),
                    "branch": _short_optional(payload.repo.branch, 80),
                    "changed_paths": (),
                    "provenance": _compact_provenance(payload.repo.provenance),
                }
            ),
            "self": payload.self.model_copy(
                update={
                    "logical_id": _short_optional(payload.self.logical_id, 120),
                    "cwd": _short_to(payload.self.cwd, 120),
                    "branch": _short_optional(payload.self.branch, 80),
                    "session_ref": _short_optional(payload.self.session_ref, 120),
                    "provenance": _compact_provenance(payload.self.provenance),
                }
            ),
            "work_item": payload.work_item.model_copy(
                update={
                    "ref": _short_optional(payload.work_item.ref, 80),
                    "title": _short_optional(payload.work_item.title, 120),
                    "status": _short_optional(payload.work_item.status, 32),
                    "assignee": _short_optional(payload.work_item.assignee, 80),
                    "fields": {},
                    "provenance": _compact_provenance(payload.work_item.provenance),
                }
            ),
            "peers": (),
            "resource_episodes": writers,
            "overlaps": (),
            "handoff": (),
            "archive": archive,
            "session_trees": (),
            "activity_episodes": (),
            "subagent_exchanges": (),
            "proof_refs": (),
            "context_flow_refs": (),
            "beads": beads,
            "advisories": (),
            "provenance": (),
        }
    )
    minimal = _finalize_projection(minimal, total_counts=total_counts, detail=False)
    if _serialized_size(minimal) <= _COMPACT_BYTE_BUDGET:
        return minimal
    archive = minimal.archive
    if archive is not None:
        archive = archive.model_copy(update={"daemon_processes": ()})
    beads = minimal.beads
    if beads is not None:
        beads = beads.model_copy(update={"merge_slot": None})
    irreducible = minimal.model_copy(
        update={
            "resource_episodes": minimal.resource_episodes[:1],
            "archive": archive,
            "beads": beads,
        }
    )
    irreducible = _finalize_projection(irreducible, total_counts=total_counts, detail=False)
    if _serialized_size(irreducible) > _COMPACT_BYTE_BUDGET:
        raise RuntimeError("compact coordination payload exceeds its fixed-field byte budget")
    return irreducible


def _compact_provenance(
    provenance: CoordinationProvenancePayload,
    *,
    keep_note: bool = False,
) -> CoordinationProvenancePayload:
    return provenance.model_copy(
        update={
            "command": (),
            "path": _short_optional(provenance.path, 200),
            "note": _short_optional(provenance.note, 160) if keep_note else None,
        }
    )


def _short_optional(value: str | None, limit: int) -> str | None:
    return _short_to(value, limit) if value is not None else None


def _text_chars(*values: str | None) -> int:
    return sum(len(value) for value in values if value is not None)


def _short_to(value: str, limit: int) -> str:
    normalized = " ".join(value.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 1] + "…"


def _finalize_projection(
    payload: AgentCoordinationPayload,
    *,
    total_counts: dict[str, int],
    detail: bool,
) -> AgentCoordinationPayload:
    visible = _coordination_counts(payload)
    omitted = {
        name: total - visible.get(name, 0)
        for name, total in sorted(total_counts.items())
        if total > visible.get(name, 0)
    }
    projection = CoordinationProjectionPayload(
        detail=detail,
        byte_budget=None if detail else _COMPACT_BYTE_BUDGET,
        serialized_bytes=0,
        total_counts=dict(sorted(total_counts.items())),
        omitted_counts=omitted,
        detail_hint=None if detail else "CLI: --detail; MCP: detail=true",
    )
    finalized = payload.model_copy(update={"projection": projection})
    for _ in range(3):
        size = _serialized_size(finalized)
        if finalized.projection.serialized_bytes == size:
            break
        finalized = finalized.model_copy(
            update={"projection": finalized.projection.model_copy(update={"serialized_bytes": size})}
        )
    return finalized


def _serialized_size(payload: AgentCoordinationPayload) -> int:
    return len(payload.to_json(exclude_none=True).encode("utf-8"))


def _run_command(args: Sequence[str], cwd: Path | None, *, timeout_seconds: float = 2.0) -> CommandResult:
    try:
        completed = subprocess.run(
            [str(arg) for arg in args],
            cwd=str(cwd) if cwd is not None else None,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return CommandResult(tuple(str(arg) for arg in args), 124, "", str(exc))
    return CommandResult(
        tuple(str(arg) for arg in args),
        completed.returncode,
        completed.stdout,
        completed.stderr,
    )


def _prov(
    source: str,
    *,
    command: Sequence[str] = (),
    path: str | None = None,
    confidence: float,
    freshness: str = "live",
    note: str | None = None,
) -> CoordinationProvenancePayload:
    return CoordinationProvenancePayload(
        source=source,
        command=tuple(str(part) for part in command),
        path=path,
        confidence=confidence,
        freshness=freshness,
        note=note,
    )


def _repo_payload(cwd: Path, runner: CommandRunner) -> CoordinationRepoPayload:
    top = runner(("git", "-C", str(cwd), "rev-parse", "--show-toplevel"), None)
    if top.returncode != 0:
        return CoordinationRepoPayload(
            cwd=str(cwd),
            provenance=_prov("cwd", confidence=0.45, freshness="live", note="not inside a git worktree"),
        )
    root = Path(top.stdout.strip()).resolve()
    branch = _git_one(root, runner, "branch", "--show-current")
    head = _git_one(root, runner, "rev-parse", "--short=12", "HEAD")
    status = runner(("git", "-C", str(root), "status", "--porcelain=v1"), None)
    changed = tuple(line[3:] for line in status.stdout.splitlines() if len(line) >= 4)[:_CHANGED_PATH_LIMIT]
    return CoordinationRepoPayload(
        cwd=str(cwd),
        root=str(root),
        branch=branch or None,
        head=head or None,
        dirty=bool(changed),
        changed_paths=changed,
        provenance=_prov("git", command=top.args, path=str(root), confidence=0.95, freshness="live"),
    )


def _git_one(root: Path, runner: CommandRunner, *args: str) -> str | None:
    result = runner(("git", "-C", str(root), *args), None)
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _self_payload(
    cwd: Path,
    repo: CoordinationRepoPayload,
    rows: tuple[_ProcessRow, ...],
    process_result: CommandResult,
) -> CoordinationSelfPayload:
    invocation_pid = os.getpid()
    explicit = _explicit_session_identity()
    if explicit is not None:
        env_name, declared_kind, session_ref = explicit
        owner = _nearest_logical_agent_owner(rows, invocation_pid, expected_kind=declared_kind)
        agent_kind = declared_kind or (owner[1] if owner is not None else "unknown")
        logical_id = _logical_agent_id(agent_kind, session_ref)
        return CoordinationSelfPayload(
            identity_status="resolved",
            agent_kind=agent_kind,
            logical_id=logical_id,
            owner_pid=owner[0].pid if owner is not None else None,
            invocation_pid=invocation_pid,
            cwd=str(cwd),
            branch=repo.branch,
            session_ref=session_ref,
            provenance=_prov(
                "caller-environment",
                command=process_result.args,
                confidence=0.98,
                freshness="live",
                note=f"stable session identity from {env_name}; invocation PID retained separately",
            ),
        )
    owner = _nearest_logical_agent_owner(rows, invocation_pid)
    if owner is not None:
        owner_row, agent_kind = owner
        ancestor_session_ref = _session_ref(owner_row.command)
        logical_id = _logical_agent_id(agent_kind, ancestor_session_ref, fallback_pid=owner_row.pid)
        return CoordinationSelfPayload(
            identity_status="resolved",
            agent_kind=agent_kind,
            logical_id=logical_id,
            owner_pid=owner_row.pid,
            invocation_pid=invocation_pid,
            cwd=str(cwd),
            branch=repo.branch,
            session_ref=ancestor_session_ref,
            provenance=_prov(
                "process-tree",
                command=process_result.args,
                confidence=0.8 if ancestor_session_ref else 0.7,
                freshness="live",
                note="nearest logical agent ancestor owns the inspection subprocess",
            ),
        )
    return CoordinationSelfPayload(
        identity_status="unknown",
        agent_kind="unknown",
        invocation_pid=invocation_pid,
        cwd=str(cwd),
        branch=repo.branch,
        provenance=_prov(
            "caller-identity",
            command=process_result.args,
            confidence=0.0,
            freshness="live",
            note="no stable session metadata or logical agent ancestor; invocation process is not treated as an agent",
        ),
    )


def _explicit_session_identity() -> tuple[str, str | None, str] | None:
    specs = (
        ("POLYLOGUE_SESSION_REF", None),
        ("CODEX_THREAD_ID", "codex"),
        ("CODEX_SESSION_ID", "codex"),
        ("CLAUDE_SESSION_ID", "claude"),
        ("GEMINI_SESSION_ID", "gemini"),
    )
    for env_name, declared_kind in specs:
        raw = os.environ.get(env_name)
        if raw is None or not raw.strip():
            continue
        session_ref = raw.strip()
        return env_name, declared_kind or _agent_kind_from_session_ref(session_ref), session_ref
    return None


def _agent_kind_from_session_ref(session_ref: str) -> str | None:
    prefixes = {
        "codex-session:": "codex",
        "claude-code-session:": "claude",
        "gemini-cli-session:": "gemini",
    }
    return next((kind for prefix, kind in prefixes.items() if session_ref.startswith(prefix)), None)


def _nearest_logical_agent_owner(
    rows: tuple[_ProcessRow, ...],
    invocation_pid: int,
    *,
    expected_kind: str | None = None,
) -> tuple[_ProcessRow, str] | None:
    by_pid = {row.pid: row for row in rows}
    current = by_pid.get(invocation_pid)
    if current is None:
        return None
    parent_pid = current.ppid
    seen = {invocation_pid}
    while parent_pid in by_pid and parent_pid not in seen:
        seen.add(parent_pid)
        row = by_pid[parent_pid]
        kind = _agent_kind_for_process(row)
        if kind is not None and not _is_agent_component(row) and (expected_kind is None or kind == expected_kind):
            return row, kind
        parent_pid = row.ppid
    return None


def _logical_agent_id(agent_kind: str, session_ref: str | None, *, fallback_pid: int | None = None) -> str:
    if session_ref is not None:
        return f"{agent_kind}:{_logical_session_token(session_ref)}"
    return f"{agent_kind}:pid:{fallback_pid}" if fallback_pid is not None else f"{agent_kind}:unknown"


def _logical_session_token(session_ref: str) -> str:
    for prefix in ("codex-session:", "claude-code-session:", "gemini-cli-session:"):
        if session_ref.startswith(prefix):
            return session_ref.removeprefix(prefix)
    return session_ref


def _work_item_payload(cwd: Path, repo: CoordinationRepoPayload, runner: CommandRunner) -> CoordinationWorkItemPayload:
    beads_root = Path(repo.root or cwd)
    if (beads_root / ".beads").exists():
        beads = runner(("bd", "list", "--status=in_progress", "--json"), beads_root)
        if beads.returncode == 0:
            items = _json_list(beads.stdout)
            if items:
                item = _choose_bead(items)
                return CoordinationWorkItemPayload(
                    source="beads",
                    ref=_str_or_none(item.get("id")),
                    title=_str_or_none(item.get("title")),
                    status=_str_or_none(item.get("status")),
                    priority=_int_or_none(item.get("priority")),
                    assignee=_str_or_none(item.get("assignee")),
                    confidence=0.95,
                    provenance=_prov("beads", command=beads.args, path=str(beads_root / ".beads"), confidence=0.95),
                    fields={
                        "labels": tuple(str(label) for label in cast(list[object], item.get("labels") or []))[:20],
                        "updated_at": _str_or_none(item.get("updated_at")),
                    },
                )
            return CoordinationWorkItemPayload(
                source="beads",
                confidence=0.75,
                provenance=_prov("beads", command=beads.args, path=str(beads_root / ".beads"), confidence=0.75),
                fields={"status": "no in-progress Beads work item"},
            )
        return CoordinationWorkItemPayload(
            source="beads",
            confidence=0.45,
            provenance=_prov("beads", command=beads.args, path=str(beads_root / ".beads"), confidence=0.45),
            fields={"error": beads.stderr.strip()[:200] or "bd command failed"},
        )
    if repo.branch:
        return CoordinationWorkItemPayload(
            source="git",
            ref=repo.branch,
            title=f"Branch {repo.branch}",
            status="inferred",
            confidence=0.35,
            provenance=_prov("git", path=repo.root, confidence=0.35, note="no .beads workspace found"),
        )
    return CoordinationWorkItemPayload(
        source="none",
        confidence=0.15,
        provenance=_prov("cwd", confidence=0.15, note="no git branch or Beads work item detected"),
    )


def _choose_bead(items: list[dict[str, object]]) -> dict[str, object]:
    return sorted(items, key=lambda item: str(item.get("updated_at") or ""), reverse=True)[0]


def _json_list(raw: str) -> list[dict[str, object]]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(value, list):
        return []
    return [cast(dict[str, object], item) for item in value if isinstance(item, dict)]


def _json_document(raw: str) -> dict[str, object] | None:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return cast(dict[str, object], value) if isinstance(value, dict) else None


def _json_list_or_empty(raw: str) -> list[dict[str, object]]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if value is None:
        return []
    if isinstance(value, list):
        return [cast(dict[str, object], item) for item in value if isinstance(item, dict)]
    return []


def _beads_payload(cwd: Path, repo: CoordinationRepoPayload, runner: CommandRunner) -> CoordinationBeadsPayload | None:
    beads_root = Path(repo.root or cwd)
    beads_dir = beads_root / ".beads"
    if not beads_dir.exists():
        return None
    # These three Beads read probes have no data dependency.  They used to
    # serialize three CLI startups on every compact status request, which was
    # the dominant live-route latency.  Preserve all three result contracts
    # while letting their subprocess waits overlap.
    with ThreadPoolExecutor(max_workers=3, thread_name_prefix="coordination-beads") as executor:
        hooks_future = executor.submit(_run_beads_probe, runner, ("bd", "hooks", "list", "--json"), beads_root)
        gates_future = executor.submit(_run_beads_probe, runner, ("bd", "gate", "list", "--json"), beads_root)
        merge_future = executor.submit(_run_beads_probe, runner, ("bd", "merge-slot", "check", "--json"), beads_root)
        hooks_result = hooks_future.result()
        gates_result = gates_future.result()
        merge_result = merge_future.result()
    hooks = _beads_hooks(hooks_result)
    gates = _beads_gates(gates_result)
    merge_slot = _beads_merge_slot(merge_result)
    hooks_all_installed = all(hook.installed for hook in hooks) if hooks else None
    hooks_outdated_count = sum(1 for hook in hooks if hook.outdated) if hooks else None
    return CoordinationBeadsPayload(
        root=str(beads_dir),
        hooks=hooks,
        hooks_all_installed=hooks_all_installed,
        hooks_outdated_count=hooks_outdated_count,
        gates=gates,
        open_gate_count=len(gates),
        merge_slot=merge_slot,
        provenance=_prov(
            "beads",
            command=hooks_result.args,
            path=str(beads_dir),
            confidence=0.8 if hooks_result.returncode == 0 else 0.45,
            note=None
            if hooks_result.returncode == 0
            else (hooks_result.stderr.strip()[:200] or "bd hooks list failed"),
        ),
    )


def _run_beads_probe(runner: CommandRunner, args: Sequence[str], cwd: Path) -> CommandResult:
    """Bound live Beads probes so an unavailable optional source stays honest.

    A timeout returns the existing error-shaped ``CommandResult`` rather than
    stale merge state.  Test runners retain their injected deterministic
    behavior, while production status stays interactive when a Beads command
    blocks on its own daemon or lock.
    """

    if runner is _run_command:
        return _run_command(args, cwd, timeout_seconds=_BEADS_PROBE_TIMEOUT_SECONDS)
    return runner(args, cwd)


def _beads_hooks(result: CommandResult) -> tuple[CoordinationBeadsHookPayload, ...]:
    if result.returncode != 0:
        return ()
    document = _json_document(result.stdout)
    rows = document.get("hooks") if document else None
    if not isinstance(rows, list):
        return ()
    hooks: list[CoordinationBeadsHookPayload] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        hooks.append(
            CoordinationBeadsHookPayload(
                name=str(row.get("Name") or row.get("name") or "unknown"),
                installed=bool(row.get("Installed") if "Installed" in row else row.get("installed")),
                version=_str_or_none(row.get("Version") or row.get("version")),
                is_shim=_bool_or_none(row.get("IsShim") if "IsShim" in row else row.get("is_shim")),
                outdated=_bool_or_none(row.get("Outdated") if "Outdated" in row else row.get("outdated")),
            )
        )
    return tuple(hooks)


def _beads_gates(result: CommandResult) -> tuple[CoordinationBeadsGatePayload, ...]:
    if result.returncode != 0:
        return ()
    rows = _json_list_or_empty(result.stdout)
    gates: list[CoordinationBeadsGatePayload] = []
    for row in rows[:20]:
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        metadata = cast(dict[str, object], metadata)
        gates.append(
            CoordinationBeadsGatePayload(
                id=_str_or_none(row.get("id")),
                title=_str_or_none(row.get("title")),
                status=_str_or_none(row.get("status")),
                gate_type=_str_or_none(row.get("gate_type") or metadata.get("gate_type") or metadata.get("type")),
                await_id=_str_or_none(row.get("await_id") or metadata.get("await_id")),
            )
        )
    return tuple(gates)


def _beads_merge_slot(result: CommandResult) -> CoordinationBeadsMergeSlotPayload | None:
    if result.returncode != 0 and not result.stdout.strip():
        return None
    document = _json_document(result.stdout)
    if document is None:
        return CoordinationBeadsMergeSlotPayload(error=(result.stderr.strip() or result.stdout.strip())[:200])
    waiters_raw = document.get("waiters") or document.get("Waiters") or ()
    waiters = tuple(str(waiter) for waiter in waiters_raw) if isinstance(waiters_raw, (list, tuple)) else ()
    return CoordinationBeadsMergeSlotPayload(
        id=_str_or_none(document.get("id")),
        available=_bool_or_none(document.get("available")),
        status=_str_or_none(document.get("status")),
        holder=_str_or_none(document.get("holder")),
        waiters=waiters,
        error=_str_or_none(document.get("error")),
    )


def _str_or_none(value: object) -> str | None:
    return str(value) if value is not None else None


def _int_or_none(value: object) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _bool_or_none(value: object) -> bool | None:
    return value if isinstance(value, bool) else None


def _process_snapshot(runner: CommandRunner) -> tuple[CommandResult, tuple[_ProcessRow, ...]]:
    result = runner(_PROCESS_COMMAND, None)
    if result.returncode != 0:
        return result, ()
    rows = tuple(parsed for line in result.stdout.splitlines() if (parsed := _parse_ps_row(line)) is not None)
    return result, rows


def _configured_archive_resource() -> str | None:
    try:
        return str(archive_root().resolve())
    except Exception as exc:
        logger.warning("coordination archive-resource resolution failed: %s", exc, exc_info=True)
        return None


def _parse_ps_row(row: str) -> _ProcessRow | None:
    parts = row.strip().split(None, 4)
    if len(parts) < 5:
        return None
    try:
        pid = int(parts[0])
        ppid = int(parts[1])
    except ValueError:
        return None
    return _ProcessRow(pid=pid, ppid=ppid, comm=parts[2], cgroup=parts[3], command=parts[4])


def _logical_peer_payloads(
    rows: tuple[_ProcessRow, ...],
    result: CommandResult,
    *,
    owner_pid: int | None,
    invocation_pid: int,
) -> tuple[CoordinationPeerPayload, ...]:
    by_pid = {row.pid: row for row in rows}
    self_tree_pids: set[int] = set()
    if owner_pid is not None and owner_pid in by_pid:
        self_tree_pids.add(owner_pid)
        self_tree_pids.update(row.pid for row in rows if _descends_from(row, owner_pid, by_pid))
        owner_kind = _agent_kind_for_process(by_pid[owner_pid])
        parent_pid = by_pid[owner_pid].ppid
        # Sessionless same-provider ancestors are launch wrappers. A
        # session-bearing ancestor is a distinct nested agent and stays visible.
        while parent_pid in by_pid:
            parent = by_pid[parent_pid]
            if _agent_kind_for_process(parent) != owner_kind or _session_ref(parent.command) is not None:
                break
            self_tree_pids.add(parent_pid)
            parent_pid = parent.ppid
    candidate_kinds = {row.pid: kind for row in rows if (kind := _agent_kind_for_process(row)) is not None}
    logical_kinds = {pid: kind for pid, kind in candidate_kinds.items() if not _is_agent_component(by_pid[pid])}
    roots: dict[int, list[_ProcessRow]] = {}
    component_ancestors: dict[int, set[int]] = {}
    for row in rows:
        if row.pid not in logical_kinds or row.pid == invocation_pid or row.pid in self_tree_pids:
            continue
        root_pid = row.pid
        parent_pid = row.ppid
        seen = {row.pid}
        ancestors: set[int] = set()
        while parent_pid in by_pid and parent_pid not in seen:
            seen.add(parent_pid)
            if parent_pid in logical_kinds:
                root_pid = parent_pid
            elif parent_pid in candidate_kinds:
                ancestors.add(parent_pid)
            parent_pid = by_pid[parent_pid].ppid
        roots.setdefault(root_pid, []).append(row)
        component_ancestors.setdefault(root_pid, set()).update(ancestors)

    peers: list[CoordinationPeerPayload] = []
    for root_pid, agent_rows in sorted(roots.items()):
        root = by_pid[root_pid]
        if _is_agent_component(root) or root_pid == owner_pid:
            continue
        descendants = tuple(
            row.pid
            for row in rows
            if row.pid != root_pid and row.pid not in self_tree_pids and _descends_from(row, root_pid, by_pid)
        )
        kind = logical_kinds[root_pid]
        session_ref = _session_ref(root.command)
        component_pids = tuple(
            sorted(
                {
                    *(row.pid for row in agent_rows if row.pid != root_pid),
                    *descendants,
                    *component_ancestors.get(root_pid, set()),
                }
            )
        )
        peers.append(
            CoordinationPeerPayload(
                pid=root_pid,
                kind=kind,
                logical_id=_logical_agent_id(kind, session_ref, fallback_pid=root_pid),
                session_ref=session_ref,
                command=_short(root.command),
                cwd=_proc_cwd(root_pid),
                component_count=len(component_pids),
                component_pids=component_pids[:50],
                provenance=_prov(
                    "process-tree",
                    command=result.args,
                    confidence=0.8,
                    note="launcher/host/MCP/spare descendants collapsed into one logical peer",
                ),
            )
        )
    return tuple(peers)


def _agent_kind_for_process(row: _ProcessRow) -> str | None:
    comm = Path(row.comm).name.lower()
    executable = _executable_name(row.command)
    for name in _AGENT_NAMES:
        if comm == name or comm.startswith(f"{name}-") or executable == name or executable.startswith(f"{name}-"):
            return name
    command = row.command.lower()
    if "@anthropic-ai/claude-code" in command:
        return "claude"
    if "@openai/codex" in command:
        return "codex"
    return None


def _is_agent_component(row: _ProcessRow) -> bool:
    text = f"{row.comm} {row.command}".lower()
    executable = _executable_name(row.command)
    wrapper = any(executable.endswith(suffix) for suffix in ("-browser", "-deepseek", "-full", "-lean", "-local"))
    return wrapper or any(
        marker in text for marker in ("mcp-server", "--spare-daemon", "code-mode-host", "claude-code-acp")
    )


def _descends_from(row: _ProcessRow, ancestor_pid: int, by_pid: dict[int, _ProcessRow]) -> bool:
    parent_pid = row.ppid
    seen = {row.pid}
    while parent_pid in by_pid and parent_pid not in seen:
        if parent_pid == ancestor_pid:
            return True
        seen.add(parent_pid)
        parent_pid = by_pid[parent_pid].ppid
    return False


def _resource_scope_payloads(
    rows: tuple[_ProcessRow, ...],
    result: CommandResult,
    repo_cwd: Path,
    *,
    archive_resource: str | None,
) -> tuple[CoordinationResourceEpisodePayload, ...]:
    by_unit: dict[str, list[_ProcessRow]] = {}
    direct: list[_ProcessRow] = []
    for row in rows:
        kind = _classify_resource(row)
        if kind is None:
            continue
        unit = _systemd_unit(row.cgroup)
        if unit and _WORK_SCOPE_RE.search(unit.lower()):
            by_unit.setdefault(unit, []).append(row)
        else:
            direct.append(row)

    episodes: list[CoordinationResourceEpisodePayload] = []
    for unit, unit_rows in sorted(by_unit.items()):
        representative = min(unit_rows, key=lambda row: row.pid)
        kind = _classify_resource(representative) or "work"
        episodes.append(
            _resource_payload(
                representative,
                unit_rows,
                kind,
                result,
                repo_cwd,
                unit=unit,
                archive_resource=archive_resource,
            )
        )
    scoped_pids = {row.pid for unit_rows in by_unit.values() for row in unit_rows}
    for row in direct:
        if row.pid in scoped_pids:
            continue
        kind = _classify_resource(row)
        if kind is not None:
            episodes.append(
                _resource_payload(
                    row,
                    [row],
                    kind,
                    result,
                    repo_cwd,
                    unit=_systemd_unit(row.cgroup),
                    archive_resource=archive_resource,
                )
            )
    return tuple(sorted(episodes, key=_resource_priority))


def _resource_priority(episode: CoordinationResourceEpisodePayload) -> tuple[int, str, int]:
    text = f"{episode.unit or ''} {episode.command}".lower()
    archive_writer = episode.kind == "daemon" or "polylogue-index-rebuild" in text or "rebuild-index" in text
    return (0 if archive_writer else 1, episode.unit or "", episode.pid)


def _resource_payload(
    representative: _ProcessRow,
    rows: Sequence[_ProcessRow],
    kind: str,
    result: CommandResult,
    repo_cwd: Path,
    *,
    unit: str | None,
    archive_resource: str | None,
) -> CoordinationResourceEpisodePayload:
    process_cwd = _proc_cwd(representative.pid)
    resources = _command_resource_refs(representative.command, process_cwd, repo_cwd)
    if archive_resource and _resource_owns_archive(representative, unit):
        resources = tuple(dict.fromkeys((archive_resource, *resources)))
    return CoordinationResourceEpisodePayload(
        pid=representative.pid,
        kind=kind,
        command=_short(representative.command),
        status="running",
        scope=representative.cgroup,
        unit=unit,
        cwd=process_cwd,
        resource_count=len(resources),
        resources=resources[:50],
        component_count=max(0, len(rows) - 1),
        component_pids=tuple(sorted(row.pid for row in rows if row.pid != representative.pid))[:50],
        provenance=_prov("process-cgroup", command=result.args, confidence=0.85 if unit else 0.75),
    )


def _resource_owns_archive(row: _ProcessRow, unit: str | None) -> bool:
    text = f"{unit or ''} {row.comm} {row.command}".lower()
    return "polylogued" in text or "polylogue-index-rebuild" in text or "rebuild-index" in text


def _classify_resource(row: _ProcessRow) -> str | None:
    comm = Path(row.comm).name.lower()
    executable = _executable_name(row.command)
    unit = (_systemd_unit(row.cgroup) or "").lower()
    if comm in _SYSTEM_RESOURCE_NAMES or executable in _SYSTEM_RESOURCE_NAMES or "/system.slice/" in row.cgroup:
        return None
    if _WORK_SCOPE_RE.search(unit):
        if "rebuild" in unit or "nix-build" in unit or "build" in unit:
            return "build"
        if "test" in unit or "verify" in unit:
            return "test"
        return "work"
    if executable == "polylogued" or comm == "polylogued":
        return "daemon"
    if executable == "pytest" or comm == "pytest" or _python_module(row.command) == "pytest":
        return "test"
    if executable in {"nix", "cargo", "rustc"} or comm in {"cargo", "rustc"}:
        return "build"
    if executable == "uv" and any(token in row.command for token in ("pytest", "devtools verify", "devtools test")):
        return "test"
    return None


def _executable_name(command: str) -> str:
    try:
        tokens = shlex.split(command)
    except ValueError:
        tokens = command.split()
    if not tokens:
        return ""
    return Path(tokens[0]).name.lower()


def _python_module(command: str) -> str | None:
    try:
        tokens = shlex.split(command)
    except ValueError:
        return None
    for index, token in enumerate(tokens[:-1]):
        if token == "-m":
            return tokens[index + 1].lower()
    return None


def _systemd_unit(cgroup: str) -> str | None:
    for segment in reversed(cgroup.split("/")):
        if segment.endswith((".service", ".scope")):
            return segment
    return None


def _session_ref(command: str) -> str | None:
    try:
        tokens = shlex.split(command)
    except ValueError:
        tokens = command.split()
    names = ("--session-id", "--resume", "--thread-id")
    for index, token in enumerate(tokens):
        for name in names:
            prefix = f"{name}="
            if token.startswith(prefix):
                value = token.removeprefix(prefix)
                return value if value and not value.startswith("-") else None
            if token == name:
                if index + 1 >= len(tokens):
                    return None
                value = tokens[index + 1]
                return value if value and not value.startswith("-") else None
    return None


def _command_resource_refs(command: str, process_cwd: str | None, repo_cwd: Path) -> tuple[str, ...]:
    refs: list[str] = []
    if process_cwd:
        refs.append(process_cwd)
    try:
        tokens = shlex.split(command)
    except ValueError:
        tokens = command.split()
    for token in tokens:
        candidate = token.split("=", 1)[-1]
        if candidate.startswith("/") and (
            candidate.startswith(str(repo_cwd))
            or "/polylogue" in candidate
            or candidate.startswith("/realm/tmp/worktrees/")
        ):
            refs.append(candidate)
    return tuple(dict.fromkeys(refs))


def _short(command: str) -> str:
    command = " ".join(command.split())
    if len(command) <= _COMMAND_CHARS:
        return command
    return command[: _COMMAND_CHARS - 1] + "…"


def _proc_cwd(pid: int) -> str | None:
    try:
        return str(Path(f"/proc/{pid}/cwd").resolve())
    except OSError:
        return None


def _handoff_payloads(
    root: str,
    *,
    archive: CoordinationArchivePayload | None,
    limit: int,
) -> tuple[CoordinationHandoffPayload, ...]:
    repo_root = Path(root)
    payloads: list[CoordinationHandoffPayload] = []
    scratch = repo_root / ".agent" / "scratch"
    paths = sorted(
        (
            path
            for pattern in ("*handoff*.md", "*coordination-message*.md")
            for path in scratch.rglob(pattern)
            if path.is_file()
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in paths[: max(1, limit)]:
        stat = path.stat()
        payloads.append(
            CoordinationHandoffPayload(
                ref=str(path),
                path=str(path),
                kind="scratch-handoff",
                exists=True,
                updated_at=datetime.fromtimestamp(stat.st_mtime, UTC).isoformat(),
                bytes=stat.st_size,
                provenance=_prov("scratch", path=str(path), confidence=0.8),
            )
        )
    remaining = max(0, limit - len(payloads))
    if remaining and archive is not None:
        payloads.extend(
            _assertion_handoff_payloads(
                Path(archive.index_db).with_name("user.db"),
                repo_root=repo_root,
                limit=remaining,
            )
        )
    return tuple(payloads[: max(1, limit)])


def _assertion_handoff_payloads(
    user_db: Path,
    *,
    repo_root: Path,
    limit: int,
) -> tuple[CoordinationHandoffPayload, ...]:
    if not user_db.exists() or limit <= 0:
        return ()
    try:
        with closing(sqlite3.connect(f"file:{user_db}?mode=ro", uri=True, timeout=0.2)) as conn:
            tables = {str(row[0]) for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if "assertions" not in tables:
                return ()
            rows = conn.execute(
                """
                SELECT assertion_id, kind, body_text, scope_ref, target_ref, updated_at_ms
                FROM assertions
                WHERE status = 'active'
                  AND (kind = 'handoff' OR (kind = 'note' AND body_text LIKE '[handoff]%'))
                ORDER BY updated_at_ms DESC
                LIMIT ?
                """,
                (limit * 4,),
            ).fetchall()
    except sqlite3.Error as exc:
        logger.warning("coordination assertion-handoff query failed: %s", exc, exc_info=True)
        return ()
    repo_tokens = {str(repo_root), repo_root.name}
    scoped_rows = [
        row
        for row in rows
        if _handoff_matches_repo(
            body=str(row[2] or ""),
            scope_ref=str(row[3] or ""),
            target_ref=str(row[4] or ""),
            repo_tokens=repo_tokens,
        )
    ][:limit]
    return tuple(
        CoordinationHandoffPayload(
            ref=f"assertion:{row[0]}",
            path=f"user.db:assertions:{row[0]}",
            kind="assertion-handoff",
            exists=True,
            updated_at=datetime.fromtimestamp(int(row[5]) / 1000, UTC).isoformat(),
            bytes=len(str(row[2] or "").encode()),
            provenance=_prov(
                "user-assertion",
                path=f"user.db:assertions:{row[0]}",
                confidence=0.9,
            ),
        )
        for row in scoped_rows
    )


def _handoff_matches_repo(*, body: str, scope_ref: str, target_ref: str, repo_tokens: set[str]) -> bool:
    body_scope = next(
        (line.removeprefix("scope_repo: ").strip() for line in body.splitlines() if line.startswith("scope_repo: ")),
        None,
    )
    if body_scope is not None:
        return body_scope in repo_tokens
    explicit_refs = {value for value in (scope_ref, target_ref) if value}
    if not explicit_refs:
        return True
    return bool(explicit_refs & repo_tokens) or any(token in ref for ref in explicit_refs for token in repo_tokens)


def _archive_payload(resources: tuple[CoordinationResourceEpisodePayload, ...]) -> CoordinationArchivePayload | None:
    try:
        archive = archive_root().resolve()
        index = active_index_db_path().resolve()
    except Exception as exc:
        # archive_root()/active_index_db_path() raise both for the ordinary
        # "no archive configured" case and for genuine config-resolution
        # bugs; a bare None return makes the two indistinguishable to the
        # caller (no archive field, no advisory). Log loudly so the failure
        # is visible even though the payload shape can't carry a reason here
        # (polylogue-cpf.4).
        logger.warning("coordination archive-root resolution failed: %s", exc, exc_info=True)
        return None
    hook_flow_states: dict[str, str] = {}
    hook_flow_healthy: bool | None = None
    hook_flow_gaps: tuple[str, ...] = ()
    try:
        from polylogue.hooks import hook_statuses

        hook_status_rows = hook_statuses(coverage=True, archive_root_path=archive)
        hook_flow_states = {status.harness: status.flow_state for status in hook_status_rows}
        configured = [status for status in hook_status_rows if status.wired_events]
        if configured:
            hook_flow_healthy = all(status.flow_healthy is True for status in configured)
        hook_flow_gaps = tuple(
            f"{status.harness}:{status.flow_state}" for status in configured if status.flow_healthy is not True
        )
    except Exception as exc:
        # hook_flow_healthy stays None ("unknown"), not True, so this does
        # not misreport as healthy — but it looks identical to "no hooks
        # configured" without a log line.
        logger.warning("coordination hook-status query failed: %s", exc, exc_info=True)
        hook_flow_states = {}
    return CoordinationArchivePayload(
        archive_root=str(archive),
        index_db=str(index),
        index_exists=index.exists(),
        index_user_version=_sqlite_user_version(index),
        source_user_version=_sqlite_user_version(index.with_name("source.db")),
        user_user_version=_sqlite_user_version(index.with_name("user.db")),
        hook_flow_states=hook_flow_states,
        hook_flow_healthy=hook_flow_healthy,
        hook_flow_gaps=hook_flow_gaps,
        daemon_processes=tuple(resource for resource in resources if resource.kind == "daemon"),
        provenance=_prov("archive-paths", path=str(archive), confidence=0.75),
    )


def _sqlite_user_version(path: Path) -> int | None:
    if not path.exists():
        return None
    uri = f"file:{path}?mode=ro"
    try:
        with closing(sqlite3.connect(uri, uri=True, timeout=0.2)) as conn:
            row = conn.execute("PRAGMA user_version").fetchone()
    except sqlite3.Error as exc:
        logger.warning("coordination user_version probe failed for %s: %s", path, exc, exc_info=True)
        return None
    return int(row[0]) if row else None


def _archive_evidence_payloads(
    repo: CoordinationRepoPayload,
    self_payload: CoordinationSelfPayload,
    archive: CoordinationArchivePayload | None,
    *,
    limit: int,
) -> tuple[
    tuple[CoordinationSessionTreePayload, ...],
    tuple[CoordinationActivityEpisodePayload, ...],
    tuple[CoordinationSubagentExchangePayload, ...],
    tuple[CoordinationProofRefPayload, ...],
    tuple[CoordinationContextFlowRefPayload, ...],
    str | None,
]:
    """Return bounded archive-evidence tuples plus a degradation reason.

    Empty tuples are ambiguous on their own: they mean "no archive
    configured", "archive schema not ready yet", and "the read-only SQLite
    query hit the 0.2s timeout/lock/corruption" alike. The trailing
    ``str | None`` distinguishes the last case (a genuine query failure,
    logged here too) from ordinary absence of evidence, per the
    degrade-loudly doctrine (polylogue-cpf.4).
    """
    empty: tuple[
        tuple[CoordinationSessionTreePayload, ...],
        tuple[CoordinationActivityEpisodePayload, ...],
        tuple[CoordinationSubagentExchangePayload, ...],
        tuple[CoordinationProofRefPayload, ...],
        tuple[CoordinationContextFlowRefPayload, ...],
    ] = ((), (), (), (), ())
    if archive is None or not archive.index_exists or archive.index_user_version is None:
        return (*empty, None)
    index = Path(archive.index_db)
    try:
        conn = sqlite3.connect(f"file:{index}?mode=ro", uri=True, timeout=0.2)
        conn.row_factory = sqlite3.Row
    except sqlite3.Error as exc:
        logger.warning("coordination archive-evidence connect failed: %s", exc, exc_info=True)
        return (*empty, f"archive-evidence connect failed: {exc}")
    try:
        # polylogue-dab/itvd: session_runs/session_observed_events/
        # session_context_snapshots are source-derived CTE relations
        # (run_projection_relations.py), not tables -- they can never appear
        # in sqlite_master, so requiring their presence here always
        # short-circuited to "no evidence", silently degrading every
        # coordination archive-evidence query. Only `sessions`/`session_links`
        # need to exist; the run/event/snapshot relations are always
        # computable once `sessions` does.
        if not _archive_tables_present(conn, ("sessions", "session_links")):
            return (*empty, None)
        target_session_id = _resolve_coordination_session(conn, repo, self_payload)
        session_tree: tuple[CoordinationSessionTreePayload, ...] = ()
        if target_session_id is not None:
            tree = _session_tree_payload(conn, target_session_id, limit=limit)
            session_tree = (tree,) if tree is not None else ()
        activity = _archive_activity_payloads(conn, target_session_id, repo, limit=limit)
        subagent_exchanges = _archive_subagent_exchange_payloads(conn, target_session_id, repo, limit=limit)
        proof_refs = _archive_proof_payloads(conn, target_session_id, repo, limit=limit)
        context_refs = _archive_context_flow_payloads(conn, target_session_id, repo, limit=limit)
        return session_tree, activity, subagent_exchanges, proof_refs, context_refs, None
    except sqlite3.Error as exc:
        logger.warning("coordination archive-evidence query failed: %s", exc, exc_info=True)
        return (*empty, f"archive-evidence query failed: {exc}")
    finally:
        conn.close()


def _archive_tables_present(conn: sqlite3.Connection, names: tuple[str, ...]) -> bool:
    placeholders = ",".join("?" for _ in names)
    rows = conn.execute(
        f"SELECT name FROM sqlite_master WHERE type = 'table' AND name IN ({placeholders})",
        names,
    ).fetchall()
    return {str(row["name"]) for row in rows} >= set(names)


def _combined_relation_sql(*fragments: str) -> str:
    """Merge multiple ``run_projection_relations.py`` ``WITH`` fragments into one clause.

    Each ``*_relation_sql()`` helper returns its own standalone ``WITH ...``
    string (one or two independently-named CTEs). SQLite allows only one
    ``WITH`` keyword per statement, so a query that needs relations from more
    than one helper (e.g. runs LEFT JOIN observed_events) must merge their
    CTE bodies under a single ``WITH``. The CTE names declared by the run,
    observed-event, and context-snapshot fragments are disjoint by
    construction, so concatenation is safe.
    """
    bodies = []
    for fragment in fragments:
        stripped = fragment.strip()
        if not stripped.upper().startswith("WITH "):
            raise ValueError(f"expected a WITH-clause fragment, got: {stripped[:40]!r}")
        bodies.append(stripped[len("WITH ") :])
    return "WITH " + ",\n".join(bodies) + "\n"


def _resolve_coordination_session(
    conn: sqlite3.Connection,
    repo: CoordinationRepoPayload,
    self_payload: CoordinationSelfPayload,
) -> str | None:
    tokens = _candidate_session_tokens(self_payload.session_ref)
    for token in tokens:
        row = conn.execute(
            """
            SELECT session_id
            FROM sessions
            WHERE session_id = ?
               OR native_id = ?
            ORDER BY sort_key_ms DESC
            LIMIT 1
            """,
            (token, token),
        ).fetchone()
        if row is not None:
            return str(row["session_id"])
    if repo.branch:
        row = conn.execute(
            """
            SELECT session_id
            FROM sessions
            WHERE git_branch = ?
            ORDER BY sort_key_ms DESC
            LIMIT 1
            """,
            (repo.branch,),
        ).fetchone()
        if row is not None:
            return str(row["session_id"])
    root_name = Path(repo.root).name if repo.root else None
    if root_name:
        row = conn.execute(
            """
            SELECT session_id
            FROM sessions
            WHERE title LIKE ?
            ORDER BY sort_key_ms DESC
            LIMIT 1
            """,
            (f"%{root_name}%",),
        ).fetchone()
        if row is not None:
            return str(row["session_id"])
    return None


def _candidate_session_tokens(session_ref: str | None) -> tuple[str, ...]:
    env_tokens = (
        session_ref,
        os.environ.get("POLYLOGUE_SESSION_REF"),
        os.environ.get("CODEX_THREAD_ID"),
        os.environ.get("CODEX_SESSION_ID"),
        os.environ.get("CLAUDE_SESSION_ID"),
    )
    tokens: list[str] = []
    for raw in env_tokens:
        if not raw:
            continue
        raw = raw.strip()
        if not raw:
            continue
        tokens.append(raw)
        if ":" not in raw:
            tokens.extend(
                [
                    f"codex-session:{raw}",
                    f"claude-code-session:{raw}",
                    f"gemini-cli-session:{raw}",
                    f"antigravity-session:{raw}",
                ]
            )
    deduped: list[str] = []
    for token in tokens:
        if token not in deduped:
            deduped.append(token)
    return tuple(deduped)


def _session_tree_payload(
    conn: sqlite3.Connection,
    target_session_id: str,
    *,
    limit: int,
) -> CoordinationSessionTreePayload | None:
    target = conn.execute(
        """
        SELECT session_id, COALESCE(root_session_id, session_id) AS root_session_id
        FROM sessions
        WHERE session_id = ?
        """,
        (target_session_id,),
    ).fetchone()
    if target is None:
        return None
    root_id = str(target["root_session_id"])
    rows = conn.execute(
        """
        SELECT session_id, origin, title, parent_session_id, branch_type,
               CASE WHEN session_id = ? THEN 1 ELSE 0 END AS is_target
        FROM sessions
        WHERE session_id = ?
           OR root_session_id = ?
           OR parent_session_id = ?
        ORDER BY
            CASE WHEN session_id = ? THEN 0 WHEN session_id = ? THEN 1 ELSE 2 END,
            sort_key_ms DESC
        LIMIT ?
        """,
        (target_session_id, target_session_id, root_id, root_id, root_id, target_session_id, max(1, limit)),
    ).fetchall()
    if not rows:
        return None
    depth_by_id = _depths_from_rows(rows, root_id)
    row_ids = tuple(str(row["session_id"]) for row in rows)
    nodes = tuple(
        CoordinationSessionTreeNodePayload(
            session_id=str(row["session_id"]),
            source_name=_str_or_none(row["origin"]),
            title=(_short_to(str(row["title"]), 500) if row["title"] is not None else None),
            depth=depth_by_id.get(str(row["session_id"]), 0),
            is_target=bool(row["is_target"]),
        )
        for row in rows
    )
    row_id_set = set(row_ids)
    edges: list[CoordinationSessionTreeEdgePayload] = []
    for row in rows:
        parent_id = _str_or_none(row["parent_session_id"])
        if parent_id and parent_id in row_id_set:
            edges.append(
                CoordinationSessionTreeEdgePayload(
                    child_id=str(row["session_id"]),
                    parent_id=parent_id,
                    kind=_str_or_none(row["branch_type"]) or "unknown",
                    resolved=True,
                )
            )
    if row_ids:
        placeholders = ",".join("?" for _ in row_ids)
        unresolved = conn.execute(
            f"""
            SELECT src_session_id, dst_native_id, link_type
            FROM session_links
            WHERE resolved_dst_session_id IS NULL
              AND src_session_id IN ({placeholders})
            ORDER BY observed_at_ms IS NULL, observed_at_ms, dst_native_id, link_type
            LIMIT ?
            """,
            (*row_ids, max(1, limit)),
        ).fetchall()
        for row in unresolved:
            edges.append(
                CoordinationSessionTreeEdgePayload(
                    child_id=str(row["src_session_id"]),
                    parent_native_id=_str_or_none(row["dst_native_id"]),
                    kind=_str_or_none(row["link_type"]) or "unresolved_native",
                    resolved=False,
                )
            )
    return CoordinationSessionTreePayload(
        target_session_id=target_session_id,
        root_session_id=root_id,
        nodes=nodes,
        edges=tuple(edges[: max(1, limit)]),
        cycle_detected=False,
        provenance=_prov(
            "archive-session-topology",
            path="index.db:sessions,session_links",
            confidence=0.8,
            note="bounded topology projection; full graph may contain additional descendants",
        ),
    )


def _depths_from_rows(rows: Sequence[sqlite3.Row], root_id: str) -> dict[str, int]:
    parent_by_id = {str(row["session_id"]): _str_or_none(row["parent_session_id"]) for row in rows}
    depths: dict[str, int] = {}
    for session_id in parent_by_id:
        depth = 0
        current = session_id
        seen: set[str] = set()
        while current != root_id and parent_by_id.get(current) and current not in seen:
            seen.add(current)
            parent = parent_by_id[current]
            if parent is None:
                break
            depth += 1
            current = parent
        depths[session_id] = depth
    return depths


def _archive_activity_payloads(
    conn: sqlite3.Connection,
    target_session_id: str | None,
    repo: CoordinationRepoPayload,
    *,
    limit: int,
) -> tuple[CoordinationActivityEpisodePayload, ...]:
    rows = _archive_activity_rows(conn, target_session_id, repo, limit=limit)
    if not rows and target_session_id is not None and repo.branch:
        rows = _archive_activity_rows(conn, None, repo, limit=limit)
    rows.sort(key=lambda row: str(row["occurred_at"] or ""), reverse=True)
    return tuple(_activity_payload_from_row(row) for row in rows[: max(1, limit)])


def _archive_activity_rows(
    conn: sqlite3.Connection,
    target_session_id: str | None,
    repo: CoordinationRepoPayload,
    *,
    limit: int,
) -> list[sqlite3.Row]:
    params: list[object] = []
    where = _archive_scope_where(target_session_id, repo, params, alias="r")
    run_rows = conn.execute(
        f"""
        {run_relation_sql()}
        SELECT r.run_ref AS ref, r.session_id, r.run_ref, 'run' AS kind, r.status,
               COALESCE(NULLIF(r.title, ''), r.search_text) AS summary,
               r.source_updated_at AS occurred_at,
               r.evidence_refs_json AS refs_json
        FROM runs r
        {where}
        ORDER BY COALESCE(r.source_updated_at, r.materialized_at) DESC, r.position
        LIMIT ?
        """,
        (*params, max(1, limit)),
    ).fetchall()
    params = []
    where = _archive_scope_where(target_session_id, repo, params, alias="e")
    event_rows = conn.execute(
        f"""
        {observed_event_relation_sql(source_where="1")}
        SELECT e.event_ref AS ref, e.session_id, e.run_ref, e.kind, NULL AS status,
               e.summary, e.source_updated_at AS occurred_at,
               e.evidence_refs_json AS refs_json
        FROM observed_events e
        {where}
        ORDER BY COALESCE(e.source_updated_at, e.materialized_at) DESC, e.position
        LIMIT ?
        """,
        (*params, max(1, limit)),
    ).fetchall()
    return list(run_rows) + list(event_rows)


def _archive_proof_payloads(
    conn: sqlite3.Connection,
    target_session_id: str | None,
    repo: CoordinationRepoPayload,
    *,
    limit: int,
) -> tuple[CoordinationProofRefPayload, ...]:
    rows = _archive_proof_rows(conn, target_session_id, repo, limit=limit)
    if not rows and target_session_id is not None and repo.branch:
        rows = _archive_proof_rows(conn, None, repo, limit=limit)
    return tuple(_proof_payload_from_row(row) for row in rows[: max(1, limit)])


def _archive_subagent_exchange_payloads(
    conn: sqlite3.Connection,
    target_session_id: str | None,
    repo: CoordinationRepoPayload,
    *,
    limit: int,
) -> tuple[CoordinationSubagentExchangePayload, ...]:
    rows = _archive_subagent_exchange_rows(conn, target_session_id, repo, limit=limit)
    if not rows and target_session_id is not None and repo.branch:
        rows = _archive_subagent_exchange_rows(conn, None, repo, limit=limit)
    return tuple(_subagent_exchange_payload_from_row(row) for row in rows[: max(1, limit)])


def _archive_subagent_exchange_rows(
    conn: sqlite3.Connection,
    target_session_id: str | None,
    repo: CoordinationRepoPayload,
    *,
    limit: int,
) -> list[sqlite3.Row]:
    params: list[object] = []
    where = _archive_scope_where(target_session_id, repo, params, alias="r")
    if where:
        where += " AND "
    else:
        where = "WHERE "
    where += "r.role = 'subagent'"
    # polylogue-dab/itvd: the pre-dab materialized writer synthesized a
    # 'subagent_finished' marker event carrying the child's final report
    # text; the source-derived CTE (run_projection_relations.py) has no
    # equivalent event kind, so `finished` never matches and
    # finished_event_ref/returned_final_message are always NULL now. The
    # subagent run row itself (status/title/context_snapshot_ref) is still
    # accurate -- only the "what did it report back" enrichment is gone.
    rows = conn.execute(
        f"""
        {_combined_relation_sql(run_relation_sql(), observed_event_relation_sql(source_where="1"))}
        SELECT
            r.run_ref,
            r.session_id,
            r.native_session_id,
            r.agent_ref,
            r.context_snapshot_ref,
            r.status,
            r.title AS dispatch_prompt,
            r.evidence_refs_json,
            finished.event_ref AS finished_event_ref,
            finished.summary AS returned_final_message,
            finished.evidence_refs_json AS finished_evidence_refs_json
        FROM runs r
        LEFT JOIN observed_events finished
          ON finished.run_ref = r.run_ref
         AND finished.kind = 'subagent_finished'
        {where}
        ORDER BY COALESCE(finished.source_updated_at, r.source_updated_at, r.materialized_at) DESC, r.position
        LIMIT ?
        """,
        (*params, max(1, limit)),
    ).fetchall()
    return list(rows)


def _archive_proof_rows(
    conn: sqlite3.Connection,
    target_session_id: str | None,
    repo: CoordinationRepoPayload,
    *,
    limit: int,
) -> list[sqlite3.Row]:
    params: list[object] = []
    where = _archive_scope_where(target_session_id, repo, params, alias="e")
    if where:
        where += " AND "
    else:
        where = "WHERE "
    # polylogue-dab/itvd: the source-derived observed-event CTE only ever
    # emits kind in ('session_started', 'tool_finished') -- the pre-dab
    # writer's richer command/test/session-finished marker kinds have no
    # equivalent, so this now only ever matches plain tool_finished events.
    where += "e.kind IN ('tool_finished', 'command_finished', 'test_finished', 'session_finished')"
    rows = conn.execute(
        f"""
        {observed_event_relation_sql(source_where="1")}
        SELECT e.event_ref, e.session_id, e.kind, e.summary, e.evidence_refs_json, e.payload_json
        FROM observed_events e
        {where}
        ORDER BY COALESCE(e.source_updated_at, e.materialized_at) DESC, e.position
        LIMIT ?
        """,
        (*params, max(1, limit)),
    ).fetchall()
    return list(rows)


def _archive_context_flow_payloads(
    conn: sqlite3.Connection,
    target_session_id: str | None,
    repo: CoordinationRepoPayload,
    *,
    limit: int,
) -> tuple[CoordinationContextFlowRefPayload, ...]:
    rows = _archive_context_flow_rows(conn, target_session_id, repo, limit=limit)
    if not rows and target_session_id is not None and repo.branch:
        rows = _archive_context_flow_rows(conn, None, repo, limit=limit)
    return tuple(_context_flow_payload_from_row(row, limit=limit) for row in rows[: max(1, limit)])


def _archive_context_flow_rows(
    conn: sqlite3.Connection,
    target_session_id: str | None,
    repo: CoordinationRepoPayload,
    *,
    limit: int,
) -> list[sqlite3.Row]:
    params: list[object] = []
    where = _archive_scope_where(target_session_id, repo, params, alias="c")
    rows = conn.execute(
        f"""
        {context_snapshot_relation_sql()}
        SELECT c.snapshot_ref, c.session_id, c.run_ref, c.boundary, c.inheritance_mode,
               c.segment_refs_json, c.evidence_refs_json
        FROM context_snapshots c
        {where}
        ORDER BY COALESCE(c.source_updated_at, c.materialized_at) DESC, c.position
        LIMIT ?
        """,
        (*params, max(1, limit)),
    ).fetchall()
    return list(rows)


def _context_flow_payload_from_row(row: sqlite3.Row, *, limit: int) -> CoordinationContextFlowRefPayload:
    return CoordinationContextFlowRefPayload(
        ref=str(row["snapshot_ref"]),
        session_id=str(row["session_id"]),
        run_ref=_str_or_none(row["run_ref"]),
        boundary=str(row["boundary"]),
        inheritance_mode=_str_or_none(row["inheritance_mode"]),
        segment_refs=_json_str_tuple(row["segment_refs_json"], limit=limit),
        evidence_refs=_json_str_tuple(row["evidence_refs_json"], limit=limit),
        provenance=_prov(
            "archive-context-flow",
            path="index.db:session_context_snapshots",
            confidence=0.7,
            note="exact session first; branch fallback when exact session has no context refs",
        ),
    )


def _archive_scope_where(
    target_session_id: str | None,
    repo: CoordinationRepoPayload,
    params: list[object],
    *,
    alias: str,
) -> str:
    if target_session_id is not None:
        params.append(target_session_id)
        return f"WHERE {alias}.session_id = ?"
    if repo.branch:
        params.append(repo.branch)
        return f"WHERE EXISTS (SELECT 1 FROM sessions s WHERE s.session_id = {alias}.session_id AND s.git_branch = ?)"
    return ""


def _activity_payload_from_row(row: sqlite3.Row) -> CoordinationActivityEpisodePayload:
    return CoordinationActivityEpisodePayload(
        ref=str(row["ref"]),
        session_id=str(row["session_id"]),
        run_ref=_str_or_none(row["run_ref"]),
        kind=str(row["kind"]),
        status=_str_or_none(row["status"]),
        summary=(_short_to(str(row["summary"]), 1_000) if row["summary"] is not None else None),
        occurred_at=_str_or_none(row["occurred_at"]),
        refs=_json_str_tuple(row["refs_json"], limit=10),
        provenance=_prov(
            "archive-run-projection", path="index.db:session_runs,session_observed_events", confidence=0.75
        ),
    )


def _proof_payload_from_row(row: sqlite3.Row) -> CoordinationProofRefPayload:
    payload = _json_dict(row["payload_json"])
    status = _str_or_none(payload.get("status") or payload.get("outcome") or payload.get("exit_code"))
    return CoordinationProofRefPayload(
        ref=str(row["event_ref"]),
        session_id=str(row["session_id"]),
        kind=str(row["kind"]),
        status=status,
        summary=(_short_to(str(row["summary"]), 1_000) if row["summary"] is not None else None),
        evidence_refs=_json_str_tuple(row["evidence_refs_json"], limit=10),
        provenance=_prov("archive-proof-outcome", path="index.db:session_observed_events", confidence=0.7),
    )


def _subagent_exchange_payload_from_row(row: sqlite3.Row) -> CoordinationSubagentExchangePayload:
    run_refs = _json_str_tuple(row["evidence_refs_json"], limit=10)
    final_refs = _json_str_tuple(row["finished_evidence_refs_json"], limit=10)
    evidence_refs = tuple(dict.fromkeys((*run_refs, *final_refs)))
    return CoordinationSubagentExchangePayload(
        ref=_str_or_none(row["finished_event_ref"]) or str(row["run_ref"]),
        session_id=str(row["session_id"]),
        run_ref=str(row["run_ref"]),
        agent_ref=_str_or_none(row["agent_ref"]),
        dispatch_prompt=(_short_to(str(row["dispatch_prompt"]), 1_000) if row["dispatch_prompt"] is not None else None),
        returned_final_message=(
            _short_to(str(row["returned_final_message"]), 1_000) if row["returned_final_message"] is not None else None
        ),
        status=_str_or_none(row["status"]),
        child_session_id=_str_or_none(row["native_session_id"]),
        context_snapshot_ref=_str_or_none(row["context_snapshot_ref"]),
        evidence_refs=evidence_refs,
        provenance=_prov(
            "archive-subagent-exchange",
            path="index.db:session_runs,session_observed_events",
            confidence=0.75,
            note="subagent dispatch comes from the projected subagent run title; final return comes from subagent_finished",
        ),
    )


def _json_str_tuple(raw: object, *, limit: int) -> tuple[str, ...]:
    value = _json_value(raw)
    if not isinstance(value, list):
        return ()
    return tuple(str(item) for item in value[: max(0, limit)] if item is not None)


def _json_dict(raw: object) -> dict[str, object]:
    value = _json_value(raw)
    return cast(dict[str, object], value) if isinstance(value, dict) else {}


def _json_value(raw: object) -> object:
    if not isinstance(raw, str) or not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def _overlap_payloads(
    repo: CoordinationRepoPayload,
    work_item: CoordinationWorkItemPayload,
    peers: tuple[CoordinationPeerPayload, ...],
    resources: tuple[CoordinationResourceEpisodePayload, ...],
) -> tuple[CoordinationOverlapPayload, ...]:
    overlaps: list[CoordinationOverlapPayload] = []
    same_repo_peers = [peer for peer in peers if peer.cwd and repo.root and peer.cwd.startswith(repo.root)]
    if same_repo_peers:
        overlaps.append(
            CoordinationOverlapPayload(
                kind="same-repo-agent",
                summary=f"{len(same_repo_peers)} other agent process(es) appear to be using this repo.",
                refs=tuple(str(peer.pid) for peer in same_repo_peers),
                provenance=_prov("process-table", confidence=0.55),
            )
        )
    heavy = [episode for episode in resources if episode.kind in {"build", "test", "daemon"}]
    if heavy:
        overlaps.append(
            CoordinationOverlapPayload(
                kind="resource-episode",
                severity="warning" if len(heavy) > 2 else "info",
                blocking=False,
                summary=f"{len(heavy)} build/test/daemon episode(s) are currently visible.",
                refs=tuple(str(episode.pid) for episode in heavy[:10]),
                provenance=_prov("process-table", confidence=0.6),
            )
        )
    if repo.dirty and work_item.source == "none":
        overlaps.append(
            CoordinationOverlapPayload(
                kind="dirty-unclaimed-work",
                severity="warning",
                blocking=False,
                summary="Repository has local changes but no work item was inferred.",
                refs=repo.changed_paths[:10],
                provenance=_prov("git", confidence=0.65),
            )
        )
    return tuple(overlaps)


def _advisories(
    repo: CoordinationRepoPayload,
    work_item: CoordinationWorkItemPayload,
    overlaps: tuple[CoordinationOverlapPayload, ...],
    archive: CoordinationArchivePayload | None,
    archive_evidence_degraded_reason: str | None = None,
) -> tuple[str, ...]:
    advisories: list[str] = []
    if repo.dirty:
        advisories.append(f"{len(repo.changed_paths)} changed path(s) in current repo projection.")
    if work_item.confidence < 0.5:
        advisories.append("Current work item is inferred with low confidence.")
    if any(overlap.severity != "info" for overlap in overlaps):
        advisories.append("One or more overlap/resource signals deserve review before heavy work.")
    if archive is not None and not archive.index_exists:
        advisories.append("Active index.db is not present for the resolved archive root.")
    if archive_evidence_degraded_reason is not None:
        # Distinguishes "the archive genuinely has no matching evidence" from
        # "the bounded archive-evidence query failed" — session_trees/
        # activity_episodes/subagent_exchanges/proof_refs/context_flow_refs
        # are empty in both cases without this advisory (polylogue-cpf.4).
        advisories.append(f"Archive-evidence lookup degraded: {archive_evidence_degraded_reason}")
    return tuple(advisories)
