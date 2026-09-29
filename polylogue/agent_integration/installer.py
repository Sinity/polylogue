"""Ownership-aware installation for native AI agent clients.

The installer edits only one named MCP entry and explicitly marked/owned guidance
artifacts. Every mutation is recorded in a self-digested ownership-state file so
uninstall can remove only content Polylogue actually created. Drift is retained
rather than overwritten or deleted.
"""

from __future__ import annotations

import contextlib
import copy
import fcntl
import hashlib
import json
import os
import shlex
import shutil
import stat
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import tomllib
import yaml

from polylogue.agent_integration.assets import agent_asset_digest, read_agent_asset
from polylogue.agent_integration.spec import (
    ASSET_VERSION,
    CLIENTS,
    GUIDANCE_MODES,
    AgentClient,
    GuidanceMode,
)
from polylogue.core.digest import REFERENCE, canonical_bytes
from polylogue.core.durable_fs import atomic_replace
from polylogue.mcp.declarations import MCPCapabilities

#: v2 (polylogue-800m): per-client record carries "capabilities"
#: (write/judge/maintenance independent booleans) instead of the retired
#: "role" field. Older v1 state fails closed via _decode_state's schema-
#: version gate rather than silently exposing a missing/renamed field.
STATE_SCHEMA_VERSION = 2
STATE_FILE_NAME = "agent-integrations.json"
MCP_SERVER_NAME = "polylogue"
_MARKER_PREFIX = "polylogue agent integration"

JSONScalar = str | int | float | bool | None
JSONValue = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]
OperationKind = Literal["json_value", "yaml_value", "marked_block", "owned_file"]


class AgentIntegrationError(RuntimeError):
    """Base error for native client integration."""


class StateIntegrityError(AgentIntegrationError):
    """Raised when installer ownership state cannot be trusted."""


class NativeConfigConflict(AgentIntegrationError):  # noqa: N818 - public beads-06 API
    """Raised when an operator-owned value conflicts with desired configuration."""


@dataclass(frozen=True, slots=True)
class InstallOptions:
    """Resolved installation request."""

    clients: tuple[AgentClient, ...]
    capabilities: MCPCapabilities = MCPCapabilities()
    guidance: GuidanceMode = "full"
    include_reference: bool = True
    install_mcp: bool = True
    archive_root: Path | None = None
    config_path: Path | None = None
    server_command: str = "polylogue-mcp"
    polylogue_command: str = "polylogue"
    replace_clients: bool = False

    def __post_init__(self) -> None:
        if not self.clients:
            raise ValueError("at least one client is required")
        unknown = sorted(set(self.clients).difference(CLIENTS))
        if unknown:
            raise ValueError(f"unknown clients: {', '.join(unknown)}")
        if self.guidance not in GUIDANCE_MODES:
            raise ValueError(f"unknown guidance mode: {self.guidance}")


@dataclass(frozen=True, slots=True)
class ResolvedPaths:
    """Native paths resolved for one client profile."""

    root: Path
    mcp_config: Path | None
    guidance: Path | None
    reference: Path | None


@dataclass(frozen=True, slots=True)
class OperationStatus:
    """Observed state of one recorded native mutation."""

    identity: str
    path: str
    kind: str
    ownership: str
    state: str
    detail: str

    def to_dict(self) -> dict[str, str]:
        return {
            "identity": self.identity,
            "path": self.path,
            "kind": self.kind,
            "ownership": self.ownership,
            "state": self.state,
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class FileSnapshot:
    """Transaction snapshot for a path that may be rewritten."""

    path: Path
    existed: bool
    content: bytes
    mode: int | None


@dataclass(slots=True)
class _Transaction:
    snapshots: dict[Path, FileSnapshot]

    def __init__(self) -> None:
        self.snapshots = {}

    def capture(self, path: Path) -> None:
        if path in self.snapshots:
            return
        _refuse_symlink(path)
        if path.exists():
            self.snapshots[path] = FileSnapshot(
                path=path,
                existed=True,
                content=path.read_bytes(),
                mode=stat.S_IMODE(path.stat().st_mode),
            )
        else:
            self.snapshots[path] = FileSnapshot(path=path, existed=False, content=b"", mode=None)

    def rollback(self) -> None:
        for snapshot in reversed(tuple(self.snapshots.values())):
            with contextlib.suppress(OSError):
                if snapshot.existed:
                    snapshot.path.parent.mkdir(parents=True, exist_ok=True)
                    _atomic_write(snapshot.path, snapshot.content, mode=snapshot.mode)
                elif snapshot.path.exists() and not snapshot.path.is_dir():
                    snapshot.path.unlink()


def _canonical_json(value: object) -> bytes:
    return canonical_bytes(value, REFERENCE)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _state_integrity(payload: Mapping[str, object]) -> str:
    unsigned = {key: value for key, value in payload.items() if key != "integrity"}
    return _sha256_bytes(_canonical_json(unsigned))


def _sign_state(payload: dict[str, object]) -> dict[str, object]:
    signed = copy.deepcopy(payload)
    signed["integrity"] = _state_integrity(signed)
    return signed


def _decode_state(raw: bytes, *, path: Path) -> dict[str, object]:
    try:
        decoded = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StateIntegrityError(f"invalid installer state at {path}: {exc}") from exc
    if not isinstance(decoded, dict):
        raise StateIntegrityError(f"invalid installer state at {path}: expected a JSON object")
    payload = cast(dict[str, object], decoded)
    if payload.get("schema_version") != STATE_SCHEMA_VERSION:
        raise StateIntegrityError(f"unsupported installer state schema at {path}")
    integrity = payload.get("integrity")
    if not isinstance(integrity, str) or integrity != _state_integrity(payload):
        raise StateIntegrityError(f"installer state integrity check failed at {path}")
    clients = payload.get("clients")
    if not isinstance(clients, dict):
        raise StateIntegrityError(f"invalid installer state at {path}: missing clients object")
    return payload


def _empty_state() -> dict[str, object]:
    return {
        "schema_version": STATE_SCHEMA_VERSION,
        "content_version": ASSET_VERSION,
        "asset_digest": agent_asset_digest(),
        "clients": {},
        "created_directories": [],
    }


def _atomic_write(path: Path, content: bytes, *, mode: int | None = None) -> None:
    _refuse_symlink(path)
    atomic_replace(path, content, mode=mode if mode is not None else 0o600)


def _refuse_symlink(path: Path) -> None:
    if path.is_symlink():
        raise NativeConfigConflict(f"refusing to manage symlinked path: {path}")


def _ensure_parent(path: Path, created_directories: set[Path]) -> None:
    missing: list[Path] = []
    current = path.parent
    while not current.exists():
        missing.append(current)
        if current.parent == current:
            break
        current = current.parent
    for directory in reversed(missing):
        # Owner-only: these are the operator's private state and agent-client
        # configuration directories. Creating them with the ambient umask
        # (commonly 0o755) publishes their listing, and a group- or
        # world-writable ancestor would let another local account pre-create
        # the state or lock file this installer then opens.
        directory.mkdir(mode=0o700)
        created_directories.add(directory)


def _plan_parent(path: Path, planned_directories: set[Path]) -> None:
    """Record the ancestors ``_ensure_parent`` would create, without creating them."""
    current = path.parent
    while not current.exists():
        planned_directories.add(current)
        if current.parent == current:
            break
        current = current.parent


def _remove_empty_directories(paths: Sequence[Path]) -> None:
    for path in sorted(set(paths), key=lambda value: len(value.parts), reverse=True):
        with contextlib.suppress(OSError):
            path.rmdir()


def _read_text(path: Path) -> str:
    _refuse_symlink(path)
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise NativeConfigConflict(f"native configuration is not UTF-8: {path}") from exc


def _json_load(path: Path) -> dict[str, JSONValue]:
    if not path.exists():
        return {}
    text = _read_text(path)
    if not text.strip():
        return {}
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise NativeConfigConflict(f"invalid JSON native configuration at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise NativeConfigConflict(f"native configuration must be a JSON object: {path}")
    return cast(dict[str, JSONValue], value)


def _yaml_load(path: Path) -> dict[str, JSONValue]:
    if not path.exists():
        return {}
    text = _read_text(path)
    if not text.strip():
        return {}
    try:
        value = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise NativeConfigConflict(f"invalid YAML native configuration at {path}: {exc}") from exc
    if value is None:
        return {}
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise NativeConfigConflict(f"native configuration must be a string-keyed YAML mapping: {path}")
    return cast(dict[str, JSONValue], value)


def _json_dump(value: Mapping[str, JSONValue]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def _yaml_dump(value: Mapping[str, JSONValue]) -> bytes:
    return yaml.safe_dump(dict(value), allow_unicode=True, default_flow_style=False, sort_keys=False).encode()


def _path_get(root: Mapping[str, JSONValue], keys: Sequence[str]) -> tuple[bool, JSONValue | None]:
    current: JSONValue = cast(JSONValue, root)
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return False, None
        current = current[key]
    return True, current


def _path_set(root: dict[str, JSONValue], keys: Sequence[str], value: JSONValue) -> None:
    if not keys:
        raise ValueError("empty native configuration key path")
    current = root
    for key in keys[:-1]:
        child = current.get(key)
        if child is None:
            nested: dict[str, JSONValue] = {}
            current[key] = nested
            current = nested
        elif isinstance(child, dict):
            current = child
        else:
            raise NativeConfigConflict(f"native configuration key {key!r} is not a mapping")
    current[keys[-1]] = copy.deepcopy(value)


def _path_delete(root: dict[str, JSONValue], keys: Sequence[str]) -> None:
    stack: list[tuple[dict[str, JSONValue], str]] = []
    current = root
    for key in keys[:-1]:
        child = current.get(key)
        if not isinstance(child, dict):
            return
        stack.append((current, key))
        current = child
    current.pop(keys[-1], None)
    for parent, key in reversed(stack):
        child = parent.get(key)
        if isinstance(child, dict) and not child:
            parent.pop(key, None)
        else:
            break


def _operation_identity(kind: OperationKind, path: Path, token: str) -> str:
    return f"{kind}:{path}:{token}"


def _as_operation_map(raw: object) -> dict[str, dict[str, object]]:
    if not isinstance(raw, list):
        return {}
    operations: dict[str, dict[str, object]] = {}
    for item in raw:
        if isinstance(item, dict) and isinstance(item.get("identity"), str):
            operations[cast(str, item["identity"])] = cast(dict[str, object], item)
    return operations


def _managed_block(marker: str, body: str, *, comment: str = "#") -> str:
    normalized = body.rstrip() + "\n"
    return f"{comment} >>> {_MARKER_PREFIX}:{marker} >>>\n{normalized}{comment} <<< {_MARKER_PREFIX}:{marker} <<<\n"


def _extract_marked_block(text: str, marker: str, *, comment: str = "#") -> tuple[int, int, str] | None:
    start_line = f"{comment} >>> {_MARKER_PREFIX}:{marker} >>>"
    end_line = f"{comment} <<< {_MARKER_PREFIX}:{marker} <<<"
    start = text.find(start_line)
    if start < 0:
        return None
    end_marker = text.find(end_line, start + len(start_line))
    if end_marker < 0:
        raise NativeConfigConflict(f"unterminated managed block {marker!r}")
    end = end_marker + len(end_line)
    if end < len(text) and text[end] == "\n":
        end += 1
    return start, end, text[start:end]


def _replace_or_append_block(text: str, marker: str, desired: str) -> str:
    existing = _extract_marked_block(text, marker)
    if existing is None:
        separator = "" if not text or text.endswith("\n\n") else ("\n" if text.endswith("\n") else "\n\n")
        return text + separator + desired
    start, end, _ = existing
    return text[:start] + desired + text[end:]


def _remove_block(text: str, marker: str) -> str:
    existing = _extract_marked_block(text, marker)
    if existing is None:
        return text
    start, end, _ = existing
    result = text[:start] + text[end:]
    if not result.strip():
        return ""
    return result


def _toml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _mcp_entry(options: InstallOptions) -> dict[str, JSONValue]:
    """Build the generated MCP client entry.

    polylogue-800m: there is no ``--role`` launch flag. Write/judge/
    maintenance capability is a server-side config opt-in only; when the
    operator requests one at install time, it is expressed as an env var in
    the generated entry (mirroring ``polylogue.toml``'s ``[mcp]`` keys), not
    a CLI argument.
    """
    environment: dict[str, JSONValue] = {}
    if options.archive_root is not None:
        environment["POLYLOGUE_ARCHIVE_ROOT"] = str(options.archive_root.expanduser().resolve())
    if options.config_path is not None:
        environment["POLYLOGUE_CONFIG"] = str(options.config_path.expanduser().resolve())
    # Always emit an explicit value, including "0" for a disabled capability:
    # an absent env var lets a lower config layer (including a discovered
    # ``polylogue.toml`` in whatever directory the client happens to launch
    # from) supply the capability instead.  Defence in depth -- config.py
    # rejects capability keys from a discovered user layer -- but the
    # generated entry should still be unambiguous about the boundary.
    environment["POLYLOGUE_MCP_WRITE_ENABLED"] = "1" if options.capabilities.write else "0"
    environment["POLYLOGUE_MCP_JUDGE_ENABLED"] = "1" if options.capabilities.judge else "0"
    environment["POLYLOGUE_MCP_MAINTENANCE_ENABLED"] = "1" if options.capabilities.maintenance else "0"
    return {
        "command": options.server_command,
        "args": [],
        "env": environment,
    }


def _codex_mcp_block(options: InstallOptions) -> str:
    entry = _mcp_entry(options)
    command = cast(str, entry["command"])
    args = cast(list[JSONValue], entry["args"])
    environment = cast(dict[str, JSONValue], entry["env"])
    lines = [
        "[mcp_servers.polylogue]",
        f"command = {_toml_string(command)}",
        "args = [" + ", ".join(_toml_string(cast(str, item)) for item in args) + "]",
    ]
    if environment:
        lines.append("")
        lines.append("[mcp_servers.polylogue.env]")
        lines.extend(f"{key} = {_toml_string(cast(str, value))}" for key, value in sorted(environment.items()))
    return _managed_block("codex-mcp", "\n".join(lines))


def _guidance_block(client: AgentClient) -> str:
    manual = read_agent_asset("standing-manual.md")
    return _managed_block(f"{client}-guidance", manual)


def _hermes_skill() -> str:
    manual = read_agent_asset("standing-manual.md")
    return (
        "---\n"
        "name: polylogue\n"
        "description: Use the Polylogue evidence archive to recover prior AI work, verify claims, and diagnose continuity.\n"
        "---\n\n"
        f"{manual.rstrip()}\n"
    )


def _reference_text() -> str:
    return read_agent_asset("deep-reference.md")


def _profile_paths(client: AgentClient, *, home: Path, environment: Mapping[str, str]) -> ResolvedPaths:
    if client == "claude-code":
        root = Path(environment.get("CLAUDE_CONFIG_DIR", str(home / ".claude"))).expanduser()
        mcp_path = root / ".claude.json" if "CLAUDE_CONFIG_DIR" in environment else home / ".claude.json"
        return ResolvedPaths(root, mcp_path, root / "settings.json", root / "polylogue-reference.md")
    if client == "codex":
        root = Path(environment.get("CODEX_HOME", str(home / ".codex"))).expanduser()
        override = root / "AGENTS.override.md"
        guidance = override if override.exists() and _read_text(override).strip() else root / "AGENTS.md"
        return ResolvedPaths(root, root / "config.toml", guidance, root / "polylogue-reference.md")
    if client == "gemini":
        containing_root = Path(environment.get("GEMINI_CLI_HOME", str(home))).expanduser()
        root = containing_root / ".gemini"
        return ResolvedPaths(root, root / "settings.json", root / "GEMINI.md", root / "polylogue-reference.md")
    root = Path(environment.get("HERMES_HOME", str(home / ".hermes"))).expanduser()
    skill_root = root / "skills" / "productivity" / "polylogue"
    return ResolvedPaths(
        root, root / "config.yaml", skill_root / "SKILL.md", skill_root / "references" / "reference.md"
    )


def _claude_hook_entry(options: InstallOptions) -> dict[str, JSONValue]:
    command = f"{shlex.quote(options.polylogue_command)} agent session-start --client claude-code"
    return {
        "matcher": "",
        "hooks": [
            {
                "type": "command",
                "command": command,
                "timeout": 10,
            }
        ],
    }


def _find_claude_hook_index(value: JSONValue | None) -> int | None:
    if not isinstance(value, list):
        return None
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            continue
        hooks = item.get("hooks")
        if not isinstance(hooks, list):
            continue
        for hook in hooks:
            if (
                isinstance(hook, dict)
                and isinstance(hook.get("command"), str)
                and " agent session-start --client claude-code" in cast(str, hook["command"])
            ):
                return index
    return None


def _client_desired_operations(
    client: AgentClient,
    options: InstallOptions,
    *,
    home: Path,
    environment: Mapping[str, str],
) -> list[dict[str, object]]:
    paths = _profile_paths(client, home=home, environment=environment)
    operations: list[dict[str, object]] = []
    if options.install_mcp:
        if client in {"claude-code", "gemini"}:
            assert paths.mcp_config is not None
            operations.append(
                {
                    "kind": "json_value",
                    "path": str(paths.mcp_config),
                    "keys": ["mcpServers", MCP_SERVER_NAME],
                    "desired": _mcp_entry(options),
                    "token": "mcpServers.polylogue",
                }
            )
        elif client == "codex":
            assert paths.mcp_config is not None
            operations.append(
                {
                    "kind": "marked_block",
                    "path": str(paths.mcp_config),
                    "marker": "codex-mcp",
                    "desired": _codex_mcp_block(options),
                    "token": "codex-mcp",
                }
            )
        else:
            assert paths.mcp_config is not None
            operations.append(
                {
                    "kind": "yaml_value",
                    "path": str(paths.mcp_config),
                    "keys": ["mcp_servers", MCP_SERVER_NAME],
                    "desired": _mcp_entry(options),
                    "token": "mcp_servers.polylogue",
                }
            )
    if options.guidance == "full":
        assert paths.guidance is not None
        if client == "claude-code":
            operations.append(
                {
                    "kind": "json_value",
                    "path": str(paths.guidance),
                    "keys": ["hooks", "SessionStart"],
                    "desired": [_claude_hook_entry(options)],
                    "token": "hooks.SessionStart.polylogue",
                    "merge": "claude_hook",
                }
            )
        elif client in {"codex", "gemini"}:
            operations.append(
                {
                    "kind": "marked_block",
                    "path": str(paths.guidance),
                    "marker": f"{client}-guidance",
                    "desired": _guidance_block(client),
                    "token": f"{client}-guidance",
                }
            )
        else:
            operations.append(
                {
                    "kind": "owned_file",
                    "path": str(paths.guidance),
                    "desired": _hermes_skill(),
                    "token": "hermes-skill",
                }
            )
    if options.include_reference and options.guidance == "full":
        assert paths.reference is not None
        operations.append(
            {
                "kind": "owned_file",
                "path": str(paths.reference),
                "desired": _reference_text(),
                "token": "deep-reference",
            }
        )
    for operation in operations:
        kind = cast(OperationKind, operation["kind"])
        path = Path(cast(str, operation["path"]))
        operation["identity"] = _operation_identity(kind, path, cast(str, operation["token"]))
    return operations


def _write_structured(
    path: Path, data: Mapping[str, JSONValue], kind: OperationKind, transaction: _Transaction
) -> None:
    transaction.capture(path)
    content = _json_dump(data) if kind == "json_value" else _yaml_dump(data)
    existing_mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else 0o600
    _atomic_write(path, content, mode=existing_mode)


def _apply_structured_operation(
    desired: dict[str, object],
    previous: dict[str, object] | None,
    *,
    transaction: _Transaction,
    created_directories: set[Path],
    dry_run: bool = False,
) -> dict[str, object]:
    kind = cast(OperationKind, desired["kind"])
    path = Path(cast(str, desired["path"]))
    keys = tuple(cast(list[str], desired["keys"]))
    wanted = cast(JSONValue, desired["desired"])
    file_existed = path.exists()
    if dry_run:
        _plan_parent(path, created_directories)
    else:
        _ensure_parent(path, created_directories)
    data = _json_load(path) if kind == "json_value" else _yaml_load(path)
    present, current = _path_get(data, keys)

    if cast(str | None, desired.get("merge")) == "claude_hook":
        hooks_present, hooks_value = _path_get(data, ("hooks", "SessionStart"))
        if hooks_present and not isinstance(hooks_value, list):
            raise NativeConfigConflict(f"Claude hooks.SessionStart must be a list at {path}")
        hook_list = copy.deepcopy(hooks_value) if isinstance(hooks_value, list) else []
        index = _find_claude_hook_index(cast(JSONValue, hook_list))
        desired_hook = cast(list[JSONValue], wanted)[0]
        previous_owned = previous is not None and previous.get("owned") is True
        previous_desired = previous.get("desired") if previous is not None else None
        previous_hook = None
        if isinstance(previous_desired, list) and previous_desired:
            previous_hook = previous_desired[0]
        if index is None:
            hook_list.append(copy.deepcopy(desired_hook))
            owned = True
        elif hook_list[index] == desired_hook:
            owned = previous_owned
        elif previous_owned and hook_list[index] == previous_hook:
            hook_list[index] = copy.deepcopy(desired_hook)
            owned = True
        else:
            raise NativeConfigConflict(f"operator-owned Claude SessionStart hook conflicts at {path}")
        _path_set(data, ("hooks", "SessionStart"), cast(JSONValue, hook_list))
        if not dry_run and (not path.exists() or current != hook_list):
            _write_structured(path, data, kind, transaction)
        record = copy.deepcopy(desired)
        record.update(
            {
                "owned": owned,
                "before_present": previous.get("before_present", False) if previous else False,
                "before_value": previous.get("before_value") if previous else None,
                "desired": [desired_hook],
                "created_file": previous.get("created_file", not file_existed) if previous else not file_existed,
                "digest": _sha256_bytes(_canonical_json(desired_hook)),
            }
        )
        return record

    previous_owned = previous is not None and previous.get("owned") is True
    if previous is not None and previous_owned:
        old_wanted = cast(JSONValue, previous.get("desired"))
        if not present or current != old_wanted:
            raise NativeConfigConflict(f"managed native value drifted at {path}:{'.'.join(keys)}")
        before_present = bool(previous.get("before_present"))
        before_value = cast(JSONValue, previous.get("before_value"))
        owned = True
    elif present and current == wanted:
        before_present = True
        before_value = copy.deepcopy(current)
        owned = False
    elif present:
        raise NativeConfigConflict(f"operator-owned native value conflicts at {path}:{'.'.join(keys)}")
    else:
        before_present = False
        before_value = None
        owned = True

    if current != wanted and not dry_run:
        _path_set(data, keys, wanted)
        _write_structured(path, data, kind, transaction)
    record = copy.deepcopy(desired)
    record.update(
        {
            "owned": owned,
            "before_present": before_present,
            "before_value": before_value,
            "desired": copy.deepcopy(wanted),
            "created_file": previous.get("created_file", not file_existed) if previous else not file_existed,
            "digest": _sha256_bytes(_canonical_json(wanted)),
        }
    )
    return record


def _apply_marked_block(
    desired: dict[str, object],
    previous: dict[str, object] | None,
    *,
    transaction: _Transaction,
    created_directories: set[Path],
    dry_run: bool = False,
) -> dict[str, object]:
    path = Path(cast(str, desired["path"]))
    marker = cast(str, desired["marker"])
    wanted = cast(str, desired["desired"])
    if dry_run:
        _plan_parent(path, created_directories)
    else:
        _ensure_parent(path, created_directories)
    text = _read_text(path) if path.exists() else ""
    existing = _extract_marked_block(text, marker)
    if marker == "codex-mcp":
        unmanaged_text = text if existing is None else text[: existing[0]] + text[existing[1] :]
        if existing is None and "[mcp_servers.polylogue]" in unmanaged_text:
            raise NativeConfigConflict(f"operator-owned Codex MCP table conflicts at {path}")
        try:
            tomllib.loads(text) if text.strip() else {}
        except tomllib.TOMLDecodeError as exc:
            raise NativeConfigConflict(f"invalid TOML native configuration at {path}: {exc}") from exc
    previous_owned = previous is not None and previous.get("owned") is True
    if previous is not None and previous_owned:
        old_wanted = cast(str, previous.get("desired"))
        if existing is None or existing[2] != old_wanted:
            raise NativeConfigConflict(f"managed block drifted at {path}:{marker}")
        owned = True
        created_file = bool(previous.get("created_file"))
    elif existing is not None and existing[2] == wanted:
        owned = False
        created_file = False
    elif existing is not None:
        raise NativeConfigConflict(f"operator-owned marked block conflicts at {path}:{marker}")
    else:
        owned = True
        created_file = not path.exists()
    updated = _replace_or_append_block(text, marker, wanted)
    if marker == "codex-mcp":
        try:
            tomllib.loads(updated)
        except tomllib.TOMLDecodeError as exc:
            raise NativeConfigConflict(f"generated Codex TOML is invalid at {path}: {exc}") from exc
    if updated != text and not dry_run:
        transaction.capture(path)
        mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else 0o600
        _atomic_write(path, updated.encode(), mode=mode)
    record = copy.deepcopy(desired)
    record.update(
        {
            "owned": owned,
            "created_file": created_file,
            "desired": wanted,
            "digest": _sha256_bytes(wanted.encode()),
        }
    )
    return record


def _apply_owned_file(
    desired: dict[str, object],
    previous: dict[str, object] | None,
    *,
    transaction: _Transaction,
    created_directories: set[Path],
    dry_run: bool = False,
) -> dict[str, object]:
    path = Path(cast(str, desired["path"]))
    wanted = cast(str, desired["desired"])
    if dry_run:
        _plan_parent(path, created_directories)
    else:
        _ensure_parent(path, created_directories)
    current = _read_text(path) if path.exists() else None
    previous_owned = previous is not None and previous.get("owned") is True
    if previous is not None and previous_owned:
        old_wanted = cast(str, previous.get("desired"))
        if current != old_wanted:
            raise NativeConfigConflict(f"managed file drifted at {path}")
        owned = True
        created_file = bool(previous.get("created_file"))
    elif current == wanted:
        owned = False
        created_file = False
    elif current is not None:
        raise NativeConfigConflict(f"operator-owned file conflicts at {path}")
    else:
        owned = True
        created_file = True
    if current != wanted and not dry_run:
        transaction.capture(path)
        mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else 0o600
        _atomic_write(path, wanted.encode(), mode=mode)
    record = copy.deepcopy(desired)
    record.update(
        {
            "owned": owned,
            "created_file": created_file,
            "desired": wanted,
            "digest": _sha256_bytes(wanted.encode()),
        }
    )
    return record


def _apply_operation(
    desired: dict[str, object],
    previous: dict[str, object] | None,
    *,
    transaction: _Transaction,
    created_directories: set[Path],
    dry_run: bool = False,
) -> dict[str, object]:
    """Apply one desired native mutation, or with ``dry_run`` only plan its record.

    The planned record carries the same ownership and pre-state an apply
    would record, and a conflict raises before anything is written.
    """
    kind = cast(OperationKind, desired["kind"])
    if kind in {"json_value", "yaml_value"}:
        return _apply_structured_operation(
            desired,
            previous,
            transaction=transaction,
            created_directories=created_directories,
            dry_run=dry_run,
        )
    if kind == "marked_block":
        return _apply_marked_block(
            desired,
            previous,
            transaction=transaction,
            created_directories=created_directories,
            dry_run=dry_run,
        )
    return _apply_owned_file(
        desired,
        previous,
        transaction=transaction,
        created_directories=created_directories,
        dry_run=dry_run,
    )


_EFFECT_PRESENT_STATES = frozenset({"ok", "satisfied-unowned"})


def _prepared_operations(client_record: Mapping[str, object]) -> dict[str, dict[str, object]]:
    return _as_operation_map(client_record.get("prepared_operations"))


def _prepared_removals(client_record: Mapping[str, object]) -> frozenset[str]:
    raw = client_record.get("prepared_removals")
    return frozenset(item for item in raw if isinstance(item, str)) if isinstance(raw, list) else frozenset()


def _interrupted(client_record: Mapping[str, object]) -> bool:
    """Whether an install journaled work for this client and never committed it."""
    return "prepared_operations" in client_record or "prepared_removals" in client_record


def _reconciled_operations(client_record: Mapping[str, object]) -> list[tuple[dict[str, object], bool]]:
    """Return each recorded operation once, as ``(record, unconfirmed)``.

    An interrupted install leaves a prepared record beside (or instead of) the
    committed one. When the prepared effect is present it is what is on disk --
    including an upgrade that reuses the committed identity -- so it wins;
    otherwise the committed record stands. A prepared record with no committed
    counterpart and no present effect is reported ``unconfirmed`` so its
    removal can treat absence as "never applied" rather than drift. So is a
    committed record the install journaled for removal whose effect is no
    longer present: the interrupted install may already have removed it.
    """
    committed = _as_operation_map(client_record.get("operations"))
    prepared = _prepared_operations(client_record)
    removals = _prepared_removals(client_record)
    reconciled: list[tuple[dict[str, object], bool]] = []
    for identity in dict.fromkeys([*committed, *prepared]):
        candidate = prepared.get(identity)
        if candidate is not None and _observe_operation(candidate).state in _EFFECT_PRESENT_STATES:
            reconciled.append((candidate, False))
        elif identity in committed:
            record = committed[identity]
            removal_unconfirmed = (
                identity in removals and _observe_operation(record).state not in _EFFECT_PRESENT_STATES
            )
            reconciled.append((record, removal_unconfirmed))
        elif candidate is not None:
            reconciled.append((candidate, True))
    return reconciled


def _remove_recorded_operation(
    operation: dict[str, object], unconfirmed: bool, *, transaction: _Transaction
) -> tuple[bool, str]:
    if unconfirmed:
        return _remove_prepared_operation(operation, transaction=transaction)
    return _remove_operation(operation, transaction=transaction)


def _remove_prepared_operation(operation: dict[str, object], *, transaction: _Transaction) -> tuple[bool, str]:
    """Undo one operation an interrupted install may or may not have carried out.

    A prepared operation (or a journaled removal) is written to the state file
    before its native effect, so after a crash its effect is either present
    (remove it exactly as a committed operation) or absent (never applied, or
    already removed; nothing to remove). Only content that is neither the
    recorded value nor its pre-state is drift.
    """
    if operation.get("owned") is not True:
        return True, "prepared operation did not own its value"
    observed = _observe_operation(operation)
    if observed.state in _EFFECT_PRESENT_STATES:
        return _remove_operation(operation, transaction=transaction)
    if observed.state == "missing":
        return True, "prepared operation was never applied"
    if operation.get("kind") in {"json_value", "yaml_value"} and operation.get("merge") != "claude_hook":
        path = Path(cast(str, operation["path"]))
        kind = cast(OperationKind, operation["kind"])
        data = _json_load(path) if kind == "json_value" else _yaml_load(path)
        present, current = _path_get(data, cast(list[str], operation["keys"]))
        before_present = operation.get("before_present") is True
        if not present and not before_present:
            return True, "prepared operation was never applied"
        if present and before_present and current == cast(JSONValue, operation.get("before_value")):
            return True, "prepared operation was never applied"
        return False, observed.detail
    if operation.get("kind") == "marked_block":
        path = Path(cast(str, operation["path"]))
        if _extract_marked_block(_read_text(path), cast(str, operation["marker"])) is None:
            return True, "prepared operation was never applied"
    if operation.get("merge") == "claude_hook":
        path = Path(cast(str, operation["path"]))
        _, hooks_value = _path_get(_json_load(path), ("hooks", "SessionStart"))
        if _find_claude_hook_index(hooks_value) is None:
            return True, "prepared operation was never applied"
    return False, observed.detail


def _remove_structured_operation(
    operation: dict[str, object],
    *,
    transaction: _Transaction,
) -> tuple[bool, str]:
    if operation.get("owned") is not True:
        return True, "pre-existing equal value was not owned"
    kind = cast(OperationKind, operation["kind"])
    path = Path(cast(str, operation["path"]))
    if not path.exists():
        return False, "managed file is missing"
    keys = tuple(cast(list[str], operation["keys"]))
    data = _json_load(path) if kind == "json_value" else _yaml_load(path)

    if cast(str | None, operation.get("merge")) == "claude_hook":
        present, hooks_value = _path_get(data, ("hooks", "SessionStart"))
        if not present or not isinstance(hooks_value, list):
            return False, "managed SessionStart hook is missing"
        desired_list = operation.get("desired")
        desired_hook = desired_list[0] if isinstance(desired_list, list) and desired_list else None
        index = _find_claude_hook_index(cast(JSONValue, hooks_value))
        if index is None or hooks_value[index] != desired_hook:
            return False, "managed SessionStart hook drifted"
        hooks_value.pop(index)
        if hooks_value:
            _path_set(data, ("hooks", "SessionStart"), cast(JSONValue, hooks_value))
        else:
            _path_delete(data, ("hooks", "SessionStart"))
    else:
        present, current = _path_get(data, keys)
        if not present or current != cast(JSONValue, operation.get("desired")):
            return False, "managed native value drifted"
        if operation.get("before_present") is True:
            _path_set(data, keys, cast(JSONValue, operation.get("before_value")))
        else:
            _path_delete(data, keys)

    transaction.capture(path)
    current_mode = stat.S_IMODE(path.stat().st_mode)
    if data:
        _atomic_write(path, _json_dump(data) if kind == "json_value" else _yaml_dump(data), mode=current_mode)
    elif operation.get("created_file") is True:
        path.unlink()
    else:
        empty_content = b"{}\n" if kind == "json_value" else b"{}\n"
        _atomic_write(path, empty_content, mode=current_mode)
    return True, "removed exact managed native value"


def _remove_marked_block(operation: dict[str, object], *, transaction: _Transaction) -> tuple[bool, str]:
    if operation.get("owned") is not True:
        return True, "pre-existing equal block was not owned"
    path = Path(cast(str, operation["path"]))
    if not path.exists():
        return False, "managed file is missing"
    marker = cast(str, operation["marker"])
    text = _read_text(path)
    existing = _extract_marked_block(text, marker)
    if existing is None or existing[2] != cast(str, operation.get("desired")):
        return False, "managed marked block drifted"
    updated = _remove_block(text, marker)
    transaction.capture(path)
    if not updated and operation.get("created_file") is True:
        path.unlink()
    else:
        _atomic_write(path, updated.encode(), mode=stat.S_IMODE(path.stat().st_mode))
    return True, "removed exact managed block"


def _remove_owned_file(operation: dict[str, object], *, transaction: _Transaction) -> tuple[bool, str]:
    if operation.get("owned") is not True:
        return True, "pre-existing equal file was not owned"
    path = Path(cast(str, operation["path"]))
    if not path.exists():
        return False, "managed file is missing"
    if _read_text(path) != cast(str, operation.get("desired")):
        return False, "managed file drifted"
    transaction.capture(path)
    path.unlink()
    return True, "removed exact managed file"


def _remove_operation(operation: dict[str, object], *, transaction: _Transaction) -> tuple[bool, str]:
    kind = cast(OperationKind, operation["kind"])
    if kind in {"json_value", "yaml_value"}:
        return _remove_structured_operation(operation, transaction=transaction)
    if kind == "marked_block":
        return _remove_marked_block(operation, transaction=transaction)
    return _remove_owned_file(operation, transaction=transaction)


def _observe_operation(operation: dict[str, object]) -> OperationStatus:
    identity = cast(str, operation.get("identity", "unknown"))
    path = Path(cast(str, operation["path"]))
    kind = cast(str, operation["kind"])
    owned = operation.get("owned") is True
    ownership = "owned" if owned else "pre-existing-equal"
    if not path.exists():
        return OperationStatus(identity, str(path), kind, ownership, "missing", "managed path is missing")
    try:
        if kind in {"json_value", "yaml_value"}:
            data = _json_load(path) if kind == "json_value" else _yaml_load(path)
            if operation.get("merge") == "claude_hook":
                _, hooks_value = _path_get(data, ("hooks", "SessionStart"))
                index = _find_claude_hook_index(hooks_value)
                desired_list = operation.get("desired")
                desired_hook = desired_list[0] if isinstance(desired_list, list) and desired_list else None
                matches = isinstance(hooks_value, list) and index is not None and hooks_value[index] == desired_hook
            else:
                present, current = _path_get(data, cast(list[str], operation["keys"]))
                matches = present and current == cast(JSONValue, operation.get("desired"))
        elif kind == "marked_block":
            text = _read_text(path)
            marker = cast(str, operation["marker"])
            if marker == "codex-mcp":
                try:
                    tomllib.loads(text)
                except tomllib.TOMLDecodeError as exc:
                    return OperationStatus(identity, str(path), kind, ownership, "invalid", f"invalid TOML: {exc}")
            existing = _extract_marked_block(text, marker)
            matches = existing is not None and existing[2] == cast(str, operation.get("desired"))
        else:
            matches = _read_text(path) == cast(str, operation.get("desired"))
    except AgentIntegrationError as exc:
        return OperationStatus(identity, str(path), kind, ownership, "invalid", str(exc))
    if matches:
        state = "ok" if owned else "satisfied-unowned"
        return OperationStatus(identity, str(path), kind, ownership, state, "native content matches recorded value")
    return OperationStatus(
        identity, str(path), kind, ownership, "drifted", "native content differs from recorded value"
    )


def _executable_status(command: str, environment: Mapping[str, str]) -> dict[str, object]:
    candidate = Path(command).expanduser()
    if candidate.parent != Path(".") or os.sep in command:
        resolved = candidate.resolve(strict=False)
        ok = resolved.is_file() and os.access(resolved, os.X_OK)
        return {"command": command, "resolved": str(resolved), "ok": ok}
    found = shutil.which(command, path=environment.get("PATH"))
    return {"command": command, "resolved": found, "ok": found is not None}


class AgentIntegrationManager:
    """Install, inspect, diagnose, and remove native client integration."""

    def __init__(
        self,
        *,
        home: Path | None = None,
        environment: Mapping[str, str] | None = None,
        state_path: Path | None = None,
    ) -> None:
        self.home = (home or Path.home()).expanduser().resolve()
        self.environment = dict(os.environ if environment is None else environment)
        self.state_path = (state_path or self._default_state_path()).expanduser().resolve()
        self.lock_path = self.state_path.with_suffix(self.state_path.suffix + ".lock")
        self._lock_created_directories: set[Path] = set()
        self._post_lock_cleanup_directories: set[Path] = set()

    def _default_state_path(self) -> Path:
        state_home = self.environment.get("XDG_STATE_HOME")
        root = Path(state_home).expanduser() if state_home else self.home / ".local" / "state"
        return root / "polylogue" / STATE_FILE_NAME

    @contextlib.contextmanager
    def _lock(self) -> Iterator[None]:
        _ensure_parent(self.lock_path, self._lock_created_directories)
        _refuse_symlink(self.lock_path)
        with self.lock_path.open("a+b") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                if not self.state_path.exists():
                    with contextlib.suppress(OSError):
                        self.lock_path.unlink()
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        if not self.state_path.exists():
            _remove_empty_directories(
                tuple(self._post_lock_cleanup_directories)
                + tuple(self._lock_created_directories)
                + (self.lock_path.parent, self.lock_path.parent.parent)
            )
            self._post_lock_cleanup_directories.clear()

    def _load_state(self) -> dict[str, object]:
        _refuse_symlink(self.state_path)
        if not self.state_path.exists():
            return _empty_state()
        return _decode_state(self.state_path.read_bytes(), path=self.state_path)

    def _write_state(self, state: dict[str, object], transaction: _Transaction) -> None:
        state["content_version"] = ASSET_VERSION
        state["asset_digest"] = agent_asset_digest()
        signed = _sign_state(state)
        content = _json_dump(cast(dict[str, JSONValue], signed))
        if self.state_path.exists() and self.state_path.read_bytes() == content:
            return
        transaction.capture(self.state_path)
        _ensure_parent(self.state_path, set())
        _atomic_write(self.state_path, content, mode=0o600)

    def install(self, options: InstallOptions) -> dict[str, object]:
        """Reconcile selected clients and return a machine-readable receipt."""
        with self._lock():
            state = self._load_state()
            transaction = _Transaction()
            created_directories = {
                Path(path) for path in cast(list[object], state.get("created_directories", [])) if isinstance(path, str)
            }
            created_directories.update(self._lock_created_directories)
            clients_state_raw = state.get("clients")
            clients_state = cast(dict[str, object], clients_state_raw) if isinstance(clients_state_raw, dict) else {}
            selected = tuple(dict.fromkeys(options.clients))
            removed_clients: list[str] = []
            retained_drift: list[dict[str, str]] = []
            try:
                if options.replace_clients:
                    # Journal every unselected client's removals before any
                    # native effect is removed, as the selected-client path
                    # does: a kill mid-removal then leaves each removed
                    # effect recorded as a journaled removal, not as drift.
                    journaled = False
                    for client in tuple(clients_state):
                        raw_client = clients_state.get(client)
                        if client in selected or not isinstance(raw_client, dict):
                            continue
                        removals = sorted(
                            cast(str, operation["identity"])
                            for operation, _unconfirmed in _reconciled_operations(raw_client)
                        )
                        if removals and raw_client.get("prepared_removals") != removals:
                            raw_client["prepared_removals"] = removals
                            journaled = True
                    if journaled:
                        state["clients"] = clients_state
                        self._write_state(state, transaction)
                    for client in tuple(clients_state):
                        if client in selected:
                            continue
                        raw_client = clients_state.get(client)
                        if not isinstance(raw_client, dict):
                            continue
                        remaining: list[dict[str, object]] = []
                        for operation, unconfirmed in _reconciled_operations(raw_client):
                            removed, detail = _remove_recorded_operation(
                                operation, unconfirmed, transaction=transaction
                            )
                            if not removed:
                                remaining.append(operation)
                                retained_drift.append(
                                    {"client": client, "identity": cast(str, operation["identity"]), "detail": detail}
                                )
                        raw_client.pop("prepared_operations", None)
                        raw_client.pop("prepared_removals", None)
                        if remaining:
                            raw_client["operations"] = remaining
                        else:
                            clients_state.pop(client, None)
                            removed_clients.append(client)

                # Two-phase: record every intended operation in the state file
                # BEFORE any native file is touched. A crash between a native
                # write and the final record then still leaves uninstall a
                # record of work install may have done (polylogue-dzk48).
                plans: dict[
                    str,
                    tuple[dict[str, dict[str, object]], dict[str, dict[str, object]], list[dict[str, object]]],
                ] = {}
                planned_directories: set[Path] = set()
                needs_prepared_record = False
                for client in selected:
                    raw_previous = clients_state.get(client)
                    previous_client = cast(dict[str, object], raw_previous) if isinstance(raw_previous, dict) else {}
                    committed_identities = set(_as_operation_map(previous_client.get("operations")))
                    previous_operations: dict[str, dict[str, object]] = {}
                    # Records whose effect is unconfirmed: a committed record
                    # whose journaled removal may already have happened, or a
                    # prepared-only operation an interrupted install may have
                    # written. Either is removed only if still present, and a
                    # drifted value is retained rather than dropped untracked.
                    unconfirmed_removals: dict[str, dict[str, object]] = {}
                    for operation, unconfirmed in _reconciled_operations(previous_client):
                        identity = cast(str, operation["identity"])
                        if not unconfirmed:
                            previous_operations[identity] = operation
                        else:
                            unconfirmed_removals[identity] = operation
                    desired_operations = _client_desired_operations(
                        client,
                        options,
                        home=self.home,
                        environment=self.environment,
                    )
                    planned = [
                        _apply_operation(
                            desired,
                            previous_operations.get(cast(str, desired["identity"])),
                            transaction=transaction,
                            created_directories=planned_directories,
                            dry_run=True,
                        )
                        for desired in desired_operations
                    ]
                    plans[client] = (previous_operations, unconfirmed_removals, desired_operations)
                    desired_identities = {cast(str, operation["identity"]) for operation in desired_operations}
                    pending_removals = sorted(
                        identity
                        for identity in (*previous_operations, *unconfirmed_removals)
                        if identity not in desired_identities
                    )
                    already_committed = (
                        not pending_removals
                        and not _interrupted(previous_client)
                        and all(previous_operations.get(cast(str, item["identity"])) == item for item in planned)
                        and len(planned) == len(committed_identities)
                    )
                    if already_committed:
                        continue
                    prepared_client = dict(previous_client)
                    prepared_client.setdefault("client", client)
                    prepared_client["prepared_operations"] = planned
                    # Removals are journaled too: a kill after an obsolete
                    # effect is deleted but before the final record must not
                    # leave the committed record claiming it as drift.
                    prepared_client["prepared_removals"] = pending_removals
                    clients_state[client] = prepared_client
                    needs_prepared_record = True
                if needs_prepared_record:
                    state["clients"] = clients_state
                    state["created_directories"] = [
                        str(path) for path in sorted(created_directories | planned_directories)
                    ]
                    self._write_state(state, transaction)

                receipts: list[dict[str, object]] = []
                for client in selected:
                    previous_operations, unconfirmed_removals, desired_operations = plans[client]
                    desired_identities = {cast(str, operation["identity"]) for operation in desired_operations}
                    remaining_operations: list[dict[str, object]] = []
                    for identity, old_operation in (*previous_operations.items(), *unconfirmed_removals.items()):
                        if identity in desired_identities:
                            continue
                        removed, detail = _remove_recorded_operation(
                            old_operation, identity in unconfirmed_removals, transaction=transaction
                        )
                        if not removed:
                            remaining_operations.append(old_operation)
                            retained_drift.append({"client": client, "identity": identity, "detail": detail})

                    applied: list[dict[str, object]] = []
                    for desired in desired_operations:
                        identity = cast(str, desired["identity"])
                        applied.append(
                            _apply_operation(
                                desired,
                                previous_operations.get(identity),
                                transaction=transaction,
                                created_directories=created_directories,
                            )
                        )
                    operations = [*remaining_operations, *applied]
                    client_record: dict[str, object] = {
                        "client": client,
                        "content_version": ASSET_VERSION,
                        "asset_digest": agent_asset_digest(),
                        "capabilities": {
                            "write": options.capabilities.write,
                            "judge": options.capabilities.judge,
                            "maintenance": options.capabilities.maintenance,
                        },
                        "guidance": options.guidance,
                        "include_reference": options.include_reference,
                        "install_mcp": options.install_mcp,
                        "archive_root": str(options.archive_root.expanduser().resolve())
                        if options.archive_root
                        else None,
                        "config_path": str(options.config_path.expanduser().resolve()) if options.config_path else None,
                        "server_command": options.server_command,
                        "polylogue_command": options.polylogue_command,
                        "operations": operations,
                    }
                    clients_state[client] = client_record
                    receipts.append(
                        {
                            "client": client,
                            "operations": len(applied),
                            "retained_drift": sum(1 for item in retained_drift if item["client"] == client),
                        }
                    )

                state["clients"] = clients_state
                state["created_directories"] = [str(path) for path in sorted(created_directories)]
                self._write_state(state, transaction)
            except Exception:
                transaction.rollback()
                _remove_empty_directories(tuple(created_directories))
                raise

            return {
                "ok": not retained_drift,
                "action": "install",
                "state_path": str(self.state_path),
                "clients": receipts,
                "removed_clients": removed_clients,
                "retained_drift": retained_drift,
                "asset_digest": agent_asset_digest(),
                "content_version": ASSET_VERSION,
            }

    def uninstall(self, clients: Sequence[AgentClient] | None = None) -> dict[str, object]:
        """Remove exact owned operations for selected clients; retain drift."""
        with self._lock():
            state = self._load_state()
            transaction = _Transaction()
            clients_raw = state.get("clients")
            clients_state = cast(dict[str, object], clients_raw) if isinstance(clients_raw, dict) else {}
            selected = set(clients or cast(Sequence[AgentClient], tuple(clients_state)))
            receipts: list[dict[str, object]] = []
            try:
                # Journal every selected client's removals before the first
                # effect is touched, as install does: a kill mid-removal then
                # leaves each removed effect recorded as a journaled removal,
                # not as drift that keeps the client from converging.
                journaled = False
                for client in tuple(clients_state):
                    raw_client = clients_state.get(client)
                    if client not in selected or not isinstance(raw_client, dict):
                        continue
                    removals = sorted(
                        cast(str, operation["identity"])
                        for operation, _unconfirmed in _reconciled_operations(raw_client)
                    )
                    if removals and raw_client.get("prepared_removals") != removals:
                        raw_client["prepared_removals"] = removals
                        journaled = True
                if journaled:
                    state["clients"] = clients_state
                    self._write_state(state, transaction)
                for client in tuple(clients_state):
                    if client not in selected:
                        continue
                    raw_client = clients_state.get(client)
                    if not isinstance(raw_client, dict):
                        continue
                    remaining: list[dict[str, object]] = []
                    removed_count = 0
                    drifted: list[dict[str, str]] = []
                    for operation, unconfirmed in _reconciled_operations(raw_client):
                        removed, detail = _remove_recorded_operation(operation, unconfirmed, transaction=transaction)
                        if removed:
                            removed_count += 1 if detail != "prepared operation was never applied" else 0
                        else:
                            remaining.append(operation)
                            drifted.append({"identity": cast(str, operation["identity"]), "detail": detail})
                    raw_client.pop("prepared_operations", None)
                    raw_client.pop("prepared_removals", None)
                    if remaining:
                        raw_client["operations"] = remaining
                    else:
                        clients_state.pop(client, None)
                    receipts.append(
                        {
                            "client": client,
                            "removed_operations": removed_count,
                            "retained_drift": drifted,
                        }
                    )

                created = [
                    Path(path)
                    for path in cast(list[object], state.get("created_directories", []))
                    if isinstance(path, str)
                ]
                _remove_empty_directories(created)
                remaining_dirs = [path for path in created if path.exists()]
                state["clients"] = clients_state
                state["created_directories"] = [str(path) for path in remaining_dirs]
                if clients_state:
                    self._write_state(state, transaction)
                elif self.state_path.exists():
                    self._post_lock_cleanup_directories.update(created)
                    transaction.capture(self.state_path)
                    self.state_path.unlink()
            except Exception:
                transaction.rollback()
                raise

            retained = [item for receipt in receipts for item in cast(list[dict[str, str]], receipt["retained_drift"])]
            return {
                "ok": not retained,
                "action": "uninstall",
                "state_path": str(self.state_path),
                "clients": receipts,
            }

    def status(self) -> dict[str, object]:
        """Observe installation state without changing native files."""
        try:
            state = self._load_state()
        except StateIntegrityError as exc:
            return {
                "ok": False,
                "blocking": True,
                "state_path": str(self.state_path),
                "state_integrity": "failed",
                "problems": [str(exc)],
                "clients": [],
            }
        clients_raw = state.get("clients")
        clients_state = cast(dict[str, object], clients_raw) if isinstance(clients_raw, dict) else {}
        clients_payload: list[dict[str, object]] = []
        current_asset_digest = agent_asset_digest()
        asset_current = (
            state.get("asset_digest") == current_asset_digest and state.get("content_version") == ASSET_VERSION
        )
        blocking = not asset_current
        problems: list[str] = (
            [] if asset_current else ["installed guidance assets are stale; reinstall to reconcile them"]
        )
        for client, raw_client in sorted(clients_state.items()):
            if not isinstance(raw_client, dict):
                continue
            operation_statuses = [
                _observe_operation(operation).to_dict()
                for operation in _as_operation_map(raw_client.get("operations")).values()
            ]
            # A kill mid-install leaves journaled work that status cannot
            # attribute: its committed operations may look intact while the
            # prepared ones are absent or half-applied, so it blocks until an
            # install or uninstall resolves it.
            interrupted = _interrupted(raw_client)
            operation_blocking = interrupted or any(
                item["state"] in {"missing", "drifted", "invalid"} for item in operation_statuses
            )
            if interrupted:
                problems.append(f"{client}: an install was interrupted; rerun install or uninstall to resolve it")
            blocking = blocking or operation_blocking
            server_command = cast(str, raw_client.get("server_command", "polylogue-mcp"))
            polylogue_command = cast(str, raw_client.get("polylogue_command", "polylogue"))
            executables = {
                "mcp_server": _executable_status(server_command, self.environment),
                "polylogue": _executable_status(polylogue_command, self.environment),
            }
            clients_payload.append(
                {
                    "client": client,
                    "capabilities": raw_client.get("capabilities"),
                    "guidance": raw_client.get("guidance"),
                    "include_reference": raw_client.get("include_reference"),
                    "install_mcp": raw_client.get("install_mcp"),
                    "archive_root": raw_client.get("archive_root"),
                    "config_path": raw_client.get("config_path"),
                    "content_version": raw_client.get("content_version"),
                    "asset_digest": raw_client.get("asset_digest"),
                    "operations": operation_statuses,
                    "executables": executables,
                    "interrupted": interrupted,
                    "native_ok": not operation_blocking,
                }
            )
        return {
            "ok": not blocking,
            "blocking": blocking,
            "state_path": str(self.state_path),
            "state_integrity": "ok",
            "installed": bool(clients_payload),
            "content_version": state.get("content_version"),
            "asset_digest": state.get("asset_digest"),
            "current_asset_digest": current_asset_digest,
            "asset_current": asset_current,
            "clients": clients_payload,
            "problems": problems,
        }

    def doctor(self) -> dict[str, object]:
        """Return blocking diagnostics, including executable and identity checks."""
        payload = self.status()
        problems = cast(list[str], payload.setdefault("problems", []))
        if payload.get("state_integrity") != "ok":
            payload["ok"] = False
            payload["blocking"] = True
            return payload
        clients = cast(list[dict[str, object]], payload.get("clients", []))
        if not clients:
            problems.append("no agent clients are installed")
        for client in clients:
            name = cast(str, client["client"])
            if client.get("install_mcp") is True:
                executable = cast(dict[str, object], cast(dict[str, object], client["executables"])["mcp_server"])
                if executable.get("ok") is not True:
                    problems.append(f"{name}: MCP server executable is not resolvable: {executable.get('command')}")
            if client.get("guidance") == "full" and name == "claude-code":
                executable = cast(dict[str, object], cast(dict[str, object], client["executables"])["polylogue"])
                if executable.get("ok") is not True:
                    problems.append(f"{name}: SessionStart executable is not resolvable: {executable.get('command')}")
            if client.get("guidance") in {"mcp-only", "off"}:
                problems.append(
                    f"{name}: native standing guidance is explicitly opted down to {client.get('guidance')}"
                )
            if client.get("install_mcp") is not True:
                problems.append(f"{name}: MCP installation is explicitly disabled")
            if client.get("native_ok") is not True:
                problems.append(f"{name}: one or more managed native operations are missing, invalid, or drifted")
            if name == "codex":
                paths = _profile_paths("codex", home=self.home, environment=self.environment)
                override = paths.root / "AGENTS.override.md"
                guidance_path = next(
                    (
                        Path(cast(str, operation["path"]))
                        for operation in cast(list[dict[str, object]], client["operations"])
                        if operation["kind"] == "marked_block"
                        and cast(str, operation["identity"]).endswith(":codex-guidance")
                    ),
                    None,
                )
                if (
                    guidance_path is not None
                    and guidance_path.name == "AGENTS.md"
                    and override.exists()
                    and _read_text(override).strip()
                ):
                    problems.append(
                        "codex: AGENTS.override.md now shadows the managed AGENTS.md guidance; reinstall to relocate it"
                    )
        blocking = bool(problems) or payload.get("blocking") is True
        payload["blocking"] = blocking
        payload["ok"] = not blocking
        return payload


def claude_session_start_payload(*, reboot_from: str | None = None) -> dict[str, object]:
    """Return the supported Claude Code SessionStart JSON output."""
    reboot = reboot_from or os.environ.get("POLYLOGUE_REBOOT_FROM")
    context = read_agent_asset("standing-manual.md")
    if reboot:
        context += (
            "\n\n## Reboot handoff\n"
            f"Compile the ref-bearing continuation context for `{reboot}` before proceeding. "
            "Expand any `<ref:action:...>` marker with `resolve_ref`; markers are evidence pointers, not transcript text.\n"
        )
    return {
        "hookSpecificOutput": {
            "hookEventName": "SessionStart",
            "additionalContext": context,
        }
    }


__all__ = [
    "AgentIntegrationError",
    "AgentIntegrationManager",
    "InstallOptions",
    "NativeConfigConflict",
    "OperationStatus",
    "ResolvedPaths",
    "StateIntegrityError",
    "claude_session_start_payload",
]
