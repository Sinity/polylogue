"""Build bounded coordination envelopes from local repo and archive evidence."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from polylogue.coordination.payloads import (
    AgentCoordinationPayload,
    CoordinationArchivePayload,
    CoordinationBeadsGatePayload,
    CoordinationBeadsHookPayload,
    CoordinationBeadsMergeSlotPayload,
    CoordinationBeadsPayload,
    CoordinationHandoffPayload,
    CoordinationLimitsPayload,
    CoordinationOverlapPayload,
    CoordinationPeerPayload,
    CoordinationProvenancePayload,
    CoordinationRepoPayload,
    CoordinationResourceEpisodePayload,
    CoordinationSelfPayload,
    CoordinationView,
    CoordinationWorkItemPayload,
)
from polylogue.paths import active_index_db_path, archive_root

CommandRunner = Callable[[Sequence[str], Path | None], "CommandResult"]

_COMMAND_CHARS = 220
_CHANGED_PATH_LIMIT = 40
_PROCESS_COMMAND = ("ps", "-eo", "pid,ppid,comm,args", "--no-headers")
_AGENT_NAMES = ("codex", "claude", "gemini")
_RESOURCE_NAMES = ("polylogued", "pytest", "python", "uv", "nix", "cargo", "rustc", "node")


@dataclass(frozen=True, slots=True)
class CommandResult:
    args: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str


def build_coordination_envelope(
    *,
    view: CoordinationView = "status",
    cwd: Path | None = None,
    limit: int = 10,
    runner: CommandRunner | None = None,
) -> AgentCoordinationPayload:
    """Return a bounded, JSON-first coordination envelope for agents."""

    now = datetime.now(UTC).isoformat()
    root_cwd = (cwd or Path.cwd()).resolve()
    command_runner = runner or _run_command
    peer_limit = max(1, min(limit, 50))
    resource_limit = max(1, min(limit, 50))

    repo = _repo_payload(root_cwd, command_runner)
    self_payload = _self_payload(root_cwd, repo)
    work_item = _work_item_payload(root_cwd, repo, command_runner)
    beads = _beads_payload(root_cwd, repo, command_runner)
    peers, resources = _process_payloads(command_runner, root_cwd, peer_limit=peer_limit, resource_limit=resource_limit)
    handoff = _handoff_payloads(repo.root or str(root_cwd))
    archive = _archive_payload(resources)
    overlaps = _overlap_payloads(repo, work_item, peers, resources)
    advisories = _advisories(repo, work_item, overlaps, archive)
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
        beads=beads,
        advisories=advisories,
        limits=CoordinationLimitsPayload(
            peer_limit=peer_limit,
            resource_limit=resource_limit,
            changed_path_limit=_CHANGED_PATH_LIMIT,
            command_chars=_COMMAND_CHARS,
        ),
        provenance=provenance,
    )
    return project_coordination_envelope(payload, view)


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
                "archive": None,
                "beads": payload.beads,
            }
        )
    if view == "conflicts":
        return payload.model_copy(update={"view": view, "handoff": (), "archive": None})
    if view == "handoff":
        return payload.model_copy(
            update={
                "view": view,
                "peers": (),
                "resource_episodes": (),
                "overlaps": (),
                "archive": None,
                "beads": None,
            }
        )
    return payload


def _run_command(args: Sequence[str], cwd: Path | None) -> CommandResult:
    try:
        completed = subprocess.run(
            [str(arg) for arg in args],
            cwd=str(cwd) if cwd is not None else None,
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
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


def _self_payload(cwd: Path, repo: CoordinationRepoPayload) -> CoordinationSelfPayload:
    agent_kind = _infer_agent_kind()
    return CoordinationSelfPayload(
        agent_kind=agent_kind,
        pid=os.getpid(),
        cwd=str(cwd),
        branch=repo.branch,
        session_ref=os.environ.get("POLYLOGUE_SESSION_REF") or os.environ.get("CODEX_SESSION_ID"),
        provenance=_prov("process", confidence=0.8, freshness="live", note="derived from current process environment"),
    )


def _infer_agent_kind() -> str:
    argv = " ".join(sys.argv).lower()
    env = " ".join(f"{key}={value}" for key, value in os.environ.items() if key.startswith(("CODEX", "CLAUDE")))
    needle = f"{argv} {env.lower()}"
    for name in _AGENT_NAMES:
        if name in needle:
            return name
    return "agent"


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
    hooks_result = runner(("bd", "hooks", "list", "--json"), beads_root)
    hooks = _beads_hooks(hooks_result)
    gates_result = runner(("bd", "gate", "list", "--json"), beads_root)
    gates = _beads_gates(gates_result)
    merge_result = runner(("bd", "merge-slot", "check", "--json"), beads_root)
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


def _process_payloads(
    runner: CommandRunner,
    cwd: Path,
    *,
    peer_limit: int,
    resource_limit: int,
) -> tuple[tuple[CoordinationPeerPayload, ...], tuple[CoordinationResourceEpisodePayload, ...]]:
    result = runner(_PROCESS_COMMAND, None)
    if result.returncode != 0:
        return (), ()
    peers: list[CoordinationPeerPayload] = []
    resources: list[CoordinationResourceEpisodePayload] = []
    for row in result.stdout.splitlines():
        parsed = _parse_ps_row(row)
        if parsed is None:
            continue
        pid, comm, command = parsed
        comm_lower = comm.lower()
        command_lower = command.lower()
        if (
            pid != os.getpid()
            and any(name in comm_lower or name in command_lower for name in _AGENT_NAMES)
            and len(peers) < peer_limit
        ):
            peers.append(
                CoordinationPeerPayload(
                    pid=pid,
                    kind=_classify_agent(comm_lower, command_lower),
                    command=_short(command),
                    cwd=_proc_cwd(pid),
                    provenance=_prov("process-table", command=result.args, confidence=0.65),
                )
            )
        if (
            any(name in comm_lower or name in command_lower for name in _RESOURCE_NAMES)
            and len(resources) < resource_limit
        ):
            resources.append(
                CoordinationResourceEpisodePayload(
                    pid=pid,
                    kind=_classify_resource(comm_lower, command_lower),
                    command=_short(command),
                    status="running",
                    scope=str(cwd),
                    provenance=_prov("process-table", command=result.args, confidence=0.6),
                )
            )
    return tuple(peers), tuple(resources)


def _parse_ps_row(row: str) -> tuple[int, str, str] | None:
    parts = row.strip().split(None, 3)
    if len(parts) < 4:
        return None
    try:
        pid = int(parts[0])
    except ValueError:
        return None
    return pid, parts[2], parts[3]


def _classify_agent(comm: str, command: str) -> str:
    for name in _AGENT_NAMES:
        if name in comm or name in command:
            return name
    return "agent"


def _classify_resource(comm: str, command: str) -> str:
    text = f"{comm} {command}"
    if "polylogued" in text:
        return "daemon"
    if "pytest" in text:
        return "test"
    if "nix" in text:
        return "build"
    if "cargo" in text or "rustc" in text:
        return "build"
    if "uv" in text or "python" in text:
        return "python"
    return "process"


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


def _handoff_payloads(root: str) -> tuple[CoordinationHandoffPayload, ...]:
    repo_root = Path(root)
    refs = (
        (repo_root / ".agent" / "conductor-devloop" / "ACTIVE-LOOP.md", "active-loop"),
        (repo_root / ".agent" / "conductor-devloop" / "HANDOFF-LATEST.md", "handoff"),
        (repo_root / ".agent" / "conductor-devloop" / "OPERATING-LOG.md", "operating-log"),
    )
    payloads: list[CoordinationHandoffPayload] = []
    for path, kind in refs:
        stat = path.stat() if path.exists() else None
        payloads.append(
            CoordinationHandoffPayload(
                path=str(path),
                kind=kind,
                exists=stat is not None,
                updated_at=(datetime.fromtimestamp(stat.st_mtime, UTC).isoformat() if stat else None),
                bytes=(stat.st_size if stat else None),
                provenance=_prov("filesystem", path=str(path), confidence=0.8 if stat else 0.5),
            )
        )
    return tuple(payloads)


def _archive_payload(resources: tuple[CoordinationResourceEpisodePayload, ...]) -> CoordinationArchivePayload | None:
    try:
        archive = archive_root().resolve()
        index = active_index_db_path().resolve()
    except Exception:
        return None
    return CoordinationArchivePayload(
        archive_root=str(archive),
        index_db=str(index),
        index_exists=index.exists(),
        index_user_version=_sqlite_user_version(index),
        source_user_version=_sqlite_user_version(index.with_name("source.db")),
        user_user_version=_sqlite_user_version(index.with_name("user.db")),
        daemon_processes=tuple(resource for resource in resources if resource.kind == "daemon"),
        provenance=_prov("archive-paths", path=str(archive), confidence=0.75),
    )


def _sqlite_user_version(path: Path) -> int | None:
    if not path.exists():
        return None
    uri = f"file:{path}?mode=ro"
    try:
        with sqlite3.connect(uri, uri=True, timeout=0.2) as conn:
            row = conn.execute("PRAGMA user_version").fetchone()
    except sqlite3.Error:
        return None
    return int(row[0]) if row else None


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
    return tuple(advisories)
