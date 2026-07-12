"""Harness hook wiring, coverage, and liveness projections.

The settings adapters in this module deliberately own only Polylogue command
handlers.  They preserve every unrelated matcher group and handler byte-for-
byte at the document level, while normalizing the JSON file after a real
change.  Claude Code uses ``settings.json``; Codex uses its officially
supported ``hooks.json`` layer so ``config.toml`` does not need a lossy TOML
rewrite.
"""

from __future__ import annotations

import difflib
import json
import os
import shutil
import sqlite3
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

import tomllib

from polylogue.storage.sqlite.connection_profile import open_readonly_connection

HookHarness = Literal["claude-code", "codex"]
HookChangeAction = Literal["install", "uninstall"]

RECOMMENDED_EVENTS: tuple[str, ...] = (
    "SessionStart",
    "UserPromptSubmit",
    "PreToolUse",
    "PostToolUse",
    "Stop",
)

CLAUDE_CODE_EVENTS: tuple[str, ...] = (
    "SessionStart",
    "Setup",
    "InstructionsLoaded",
    "UserPromptSubmit",
    "UserPromptExpansion",
    "MessageDisplay",
    "PreToolUse",
    "PermissionRequest",
    "PostToolUse",
    "PostToolUseFailure",
    "PostToolBatch",
    "PermissionDenied",
    "Notification",
    "SubagentStart",
    "SubagentStop",
    "TaskCreated",
    "TaskCompleted",
    "Stop",
    "StopFailure",
    "TeammateIdle",
    "ConfigChange",
    "CwdChanged",
    "FileChanged",
    "WorktreeCreate",
    "WorktreeRemove",
    "PreCompact",
    "PostCompact",
    "Elicitation",
    "ElicitationResult",
    "SessionEnd",
)

CODEX_EVENTS: tuple[str, ...] = (
    "SessionStart",
    "UserPromptSubmit",
    "PreToolUse",
    "PermissionRequest",
    "PostToolUse",
    "PreCompact",
    "PostCompact",
    "SubagentStart",
    "SubagentStop",
    "Stop",
)

EVENTS_BY_HARNESS: dict[HookHarness, tuple[str, ...]] = {
    "claude-code": CLAUDE_CODE_EVENTS,
    "codex": CODEX_EVENTS,
}

ORIGIN_BY_HARNESS: dict[HookHarness, str] = {
    "claude-code": "claude-code-session",
    "codex": "codex-session",
}

EVENT_ENRICHMENT: dict[str, str] = {
    "SessionStart": "session identity, cwd, model, and permission posture",
    "Setup": "one-time setup and maintenance preparation",
    "InstructionsLoaded": "instruction-file loads and load reasons",
    "UserPromptSubmit": "prompt and paste ground truth before expansion",
    "UserPromptExpansion": "command expansion before it reaches the model",
    "MessageDisplay": "assistant message display evidence",
    "PreToolUse": "structured tool input before execution",
    "PermissionRequest": "requested action before operator approval",
    "PostToolUse": "structured successful tool output",
    "PostToolUseFailure": "structured failure and interrupt details",
    "PostToolBatch": "resolved parallel-tool batch boundary",
    "PermissionDenied": "structured denial evidence",
    "Notification": "operator-facing harness notifications",
    "SubagentStart": "subagent dispatch and lineage evidence",
    "SubagentStop": "subagent completion and outcome evidence",
    "TaskCreated": "task creation evidence",
    "TaskCompleted": "task completion evidence",
    "Stop": "turn/session stop evidence",
    "StopFailure": "turn termination caused by a harness or API failure",
    "TeammateIdle": "agent-team idle transition evidence",
    "ConfigChange": "mid-session harness configuration changes",
    "Elicitation": "interactive prompt and option evidence",
    "ElicitationResult": "interactive response evidence",
    "CwdChanged": "mid-session working-directory changes",
    "FileChanged": "per-tool file-change evidence",
    "WorktreeCreate": "tool-driven worktree creation",
    "WorktreeRemove": "tool-driven worktree removal",
    "PreCompact": "pre-compaction lifecycle evidence",
    "PostCompact": "post-compaction lifecycle evidence",
    "SessionEnd": "session termination reason and lifecycle evidence",
}

_FLOW_STATES: tuple[str, ...] = (
    "not-wired",
    "disabled",
    "incomplete",
    "inactive",
    "healthy",
    "partial",
    "gap",
    "unknown",
)
_WINDOW_DAYS = 7
_WINDOW_MS = _WINDOW_DAYS * 24 * 60 * 60 * 1000


class HookSettingsError(ValueError):
    """A harness settings document cannot be safely inspected or changed."""


@dataclass(frozen=True, slots=True)
class HookSettingsSource:
    path: str
    exists: bool
    writable: bool
    format: str
    contains_polylogue_hooks: bool


@dataclass(frozen=True, slots=True)
class HookEventCoverage:
    event: str
    wired: bool
    recommended: bool
    observed_session_count: int
    eligible_session_count: int
    expected_session_count: int | None
    missing_expected_count: int | None
    observed_rate: float | None
    enrichment: str


@dataclass(frozen=True, slots=True)
class HookHarnessStatus:
    harness: HookHarness
    settings_path: str
    settings_sources: tuple[HookSettingsSource, ...]
    feature_enabled: bool
    executable_available: bool
    supported_events: tuple[str, ...]
    recommended_events: tuple[str, ...]
    wired_events: tuple[str, ...]
    missing_recommended_events: tuple[str, ...]
    observed_last_7d: tuple[str, ...]
    eligible_session_count: int
    sessions_with_hook_events: int
    sessions_without_hook_events: int
    flow_state: str
    coverage_checked: bool
    coverage: tuple[HookEventCoverage, ...]
    evidence_note: str | None = None

    @property
    def flow_healthy(self) -> bool | None:
        if self.flow_state in {"healthy", "inactive"}:
            return True
        if self.flow_state in {"disabled", "incomplete", "partial", "gap"}:
            return False
        return None

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["flow_healthy"] = self.flow_healthy
        return cast(dict[str, object], payload)


@dataclass(frozen=True, slots=True)
class HookChangePlan:
    action: HookChangeAction
    harness: HookHarness
    settings_path: str
    selected_events: tuple[str, ...]
    changed_events: tuple[str, ...]
    changed: bool
    written: bool
    before: str
    after: str
    diff: str

    def to_dict(self) -> dict[str, object]:
        return cast(dict[str, object], asdict(self))


@dataclass(frozen=True, slots=True)
class _SessionOpportunity:
    native_id: str
    authored_prompts: bool
    tool_uses: bool


def normalize_harness(value: str) -> HookHarness:
    normalized = value.strip().lower()
    if normalized == "claude-code":
        return "claude-code"
    if normalized == "codex":
        return "codex"
    raise HookSettingsError(f"unsupported harness {value!r}; choose claude-code or codex")


def resolve_events(harness: HookHarness, value: str) -> tuple[str, ...]:
    """Resolve ``recommended``, ``all``, or a comma-separated event list."""

    normalized = value.strip()
    if normalized == "recommended":
        return tuple(event for event in RECOMMENDED_EVENTS if event in EVENTS_BY_HARNESS[harness])
    if normalized == "all":
        return EVENTS_BY_HARNESS[harness]
    requested = tuple(dict.fromkeys(part.strip() for part in normalized.split(",") if part.strip()))
    if not requested:
        raise HookSettingsError("--events must be recommended, all, or a comma-separated event list")
    unsupported = sorted(set(requested) - set(EVENTS_BY_HARNESS[harness]))
    if unsupported:
        supported = ", ".join(EVENTS_BY_HARNESS[harness])
        raise HookSettingsError(f"unsupported {harness} event(s): {', '.join(unsupported)}; supported: {supported}")
    return requested


def settings_path(harness: HookHarness) -> Path:
    if harness == "claude-code":
        root = Path(os.environ.get("CLAUDE_CONFIG_DIR", str(Path.home() / ".claude")))
        return root / "settings.json"
    root = Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex")))
    return root / "hooks.json"


def _codex_config_path() -> Path:
    return Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex"))) / "config.toml"


def _read_json_document(path: Path) -> tuple[dict[str, object], str]:
    if not path.exists():
        return {}, ""
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise HookSettingsError(f"cannot read {path}: {exc}") from exc
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HookSettingsError(f"cannot safely merge invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise HookSettingsError(f"cannot safely merge {path}: top-level JSON value must be an object")
    return cast(dict[str, object], value), raw


def _hooks_table(document: dict[str, object], *, create: bool) -> dict[str, object]:
    existing = document.get("hooks")
    if existing is None and create:
        table: dict[str, object] = {}
        document["hooks"] = table
        return table
    if existing is None:
        return {}
    if not isinstance(existing, dict):
        raise HookSettingsError("cannot safely merge settings: 'hooks' must be an object")
    return cast(dict[str, object], existing)


def _polylogue_event_from_command(command: object) -> str | None:
    if not isinstance(command, str):
        return None
    parts = command.replace("=", " ").split()
    for index, part in enumerate(parts):
        executable = Path(part.strip("'\"")).name
        if executable != "polylogue-hook" or index + 1 >= len(parts):
            continue
        candidate = parts[index + 1].strip("'\"")
        if candidate in set(CLAUDE_CODE_EVENTS) | set(CODEX_EVENTS):
            return candidate
    return None


def _handler_event(handler: object) -> str | None:
    if not isinstance(handler, dict):
        return None
    return _polylogue_event_from_command(handler.get("command"))


def _wired_events_from_hooks(hooks: dict[str, object]) -> set[str]:
    wired: set[str] = set()
    for event, groups in hooks.items():
        if not isinstance(groups, list):
            continue
        for group in groups:
            if not isinstance(group, dict):
                continue
            handlers = group.get("hooks")
            if not isinstance(handlers, list):
                continue
            if any(_handler_event(handler) == event for handler in handlers):
                wired.add(event)
    return wired


def _managed_handler(harness: HookHarness, event: str) -> dict[str, object]:
    return {
        "type": "command",
        "command": f"polylogue-hook {event} --provider {harness}",
        "timeout": 5,
    }


def _add_event_handler(hooks: dict[str, object], harness: HookHarness, event: str) -> None:
    groups = hooks.setdefault(event, [])
    if not isinstance(groups, list):
        raise HookSettingsError(f"cannot safely merge settings: hooks.{event} must be an array")
    groups.append({"hooks": [_managed_handler(harness, event)]})


def _remove_event_handlers(hooks: dict[str, object], event: str) -> bool:
    groups = hooks.get(event)
    if not isinstance(groups, list):
        return False
    changed = False
    kept_groups: list[object] = []
    for group in groups:
        if not isinstance(group, dict):
            kept_groups.append(group)
            continue
        handlers = group.get("hooks")
        if not isinstance(handlers, list):
            kept_groups.append(group)
            continue
        kept_handlers = [handler for handler in handlers if _handler_event(handler) != event]
        if len(kept_handlers) == len(handlers):
            kept_groups.append(group)
            continue
        changed = True
        if kept_handlers:
            updated_group = dict(group)
            updated_group["hooks"] = kept_handlers
            kept_groups.append(updated_group)
    if kept_groups:
        hooks[event] = kept_groups
    elif changed:
        hooks.pop(event, None)
    return changed


def _render_json(document: dict[str, object]) -> str:
    return json.dumps(document, indent=2, ensure_ascii=False) + "\n"


def _atomic_write(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = path.stat().st_mode & 0o777 if path.exists() else 0o600
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temp_path = Path(temporary)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        temp_path.chmod(mode)
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def _unified_diff(path: Path, before: str, after: str) -> str:
    return "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=str(path),
            tofile=str(path),
        )
    )


def _codex_inline_hooks() -> tuple[set[str], bool, HookSettingsSource]:
    path = _codex_config_path()
    if not path.exists():
        return set(), True, HookSettingsSource(str(path), False, False, "toml", False)
    try:
        raw = path.read_bytes()
        document = tomllib.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise HookSettingsError(f"cannot inspect {path}: {exc}") from exc
    hooks_value = document.get("hooks")
    hooks = cast(dict[str, object], hooks_value) if isinstance(hooks_value, dict) else {}
    features_value = document.get("features")
    features = cast(dict[str, object], features_value) if isinstance(features_value, dict) else {}
    enabled = features.get("hooks", features.get("codex_hooks", True)) is not False
    wired = _wired_events_from_hooks(hooks)
    return (
        wired,
        enabled,
        HookSettingsSource(
            str(path),
            True,
            os.access(path, os.W_OK),
            "toml",
            bool(wired),
        ),
    )


def _wiring_snapshot(harness: HookHarness) -> tuple[set[str], bool, tuple[HookSettingsSource, ...], int | None]:
    target = settings_path(harness)
    document, _ = _read_json_document(target)
    wired = _wired_events_from_hooks(_hooks_table(document, create=False))
    sources = [
        HookSettingsSource(
            str(target),
            target.exists(),
            os.access(target, os.W_OK) if target.exists() else os.access(target.parent, os.W_OK),
            "json",
            bool(wired),
        )
    ]
    feature_enabled = True
    mtimes: list[int] = []
    if wired and target.exists():
        mtimes.append(int(target.stat().st_mtime * 1000))
    if harness == "codex":
        inline_wired, feature_enabled, inline_source = _codex_inline_hooks()
        wired.update(inline_wired)
        sources.append(inline_source)
        inline_path = Path(inline_source.path)
        if inline_wired and inline_path.exists():
            mtimes.append(int(inline_path.stat().st_mtime * 1000))
    return wired, feature_enabled, tuple(sources), max(mtimes) if mtimes else None


def plan_hook_change(
    action: HookChangeAction,
    harness: HookHarness,
    events: tuple[str, ...],
    *,
    dry_run: bool,
) -> HookChangePlan:
    """Plan and optionally apply one idempotent settings mutation."""

    target = settings_path(harness)
    document, before = _read_json_document(target)
    hooks = _hooks_table(document, create=action == "install")
    all_wired, _, _, _ = _wiring_snapshot(harness)
    changed_events: list[str] = []
    if action == "install":
        for event in events:
            if event in all_wired:
                continue
            _add_event_handler(hooks, harness, event)
            changed_events.append(event)
    else:
        for event in events:
            if _remove_event_handlers(hooks, event):
                changed_events.append(event)
        if not hooks and document.get("hooks") == {}:
            document.pop("hooks", None)
    after = _render_json(document) if document else "{}\n"
    changed = bool(changed_events) and after != before
    if not changed:
        after = before
    written = changed and not dry_run
    if written:
        _atomic_write(target, after)
    return HookChangePlan(
        action=action,
        harness=harness,
        settings_path=str(target),
        selected_events=events,
        changed_events=tuple(changed_events),
        changed=changed,
        written=written,
        before=before,
        after=after,
        diff=_unified_diff(target, before, after) if changed else "",
    )


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
            (table,),
        ).fetchone()
        is not None
    )


def _recent_session_opportunities(
    index_db: Path,
    *,
    origin: str,
    cutoff_ms: int,
) -> tuple[_SessionOpportunity, ...] | None:
    if not index_db.exists():
        return None
    try:
        conn = open_readonly_connection(index_db)
        try:
            if not _table_exists(conn, "sessions"):
                return None
            rows = conn.execute(
                """
                SELECT native_id, authored_user_message_count, tool_use_count
                FROM sessions
                WHERE origin = ?
                  AND COALESCE(sort_key_ms, updated_at_ms, created_at_ms, 0) >= ?
                ORDER BY COALESCE(sort_key_ms, updated_at_ms, created_at_ms, 0), native_id
                """,
                (origin, cutoff_ms),
            ).fetchall()
        finally:
            conn.close()
    except sqlite3.Error:
        return None
    return tuple(
        _SessionOpportunity(
            native_id=str(row[0]),
            authored_prompts=int(row[1] or 0) > 0,
            tool_uses=int(row[2] or 0) > 0,
        )
        for row in rows
    )


def _recent_hook_events(
    source_db: Path,
    *,
    origin: str,
    cutoff_ms: int,
) -> dict[str, set[str]] | None:
    if not source_db.exists():
        return None
    try:
        conn = open_readonly_connection(source_db)
        try:
            if not _table_exists(conn, "raw_hook_events"):
                return None
            rows = conn.execute(
                """
                SELECT session_native_id, event_type
                FROM raw_hook_events
                WHERE origin = ?
                  AND observed_at_ms >= ?
                  AND session_native_id IS NOT NULL
                """,
                (origin, cutoff_ms),
            ).fetchall()
        finally:
            conn.close()
    except sqlite3.Error:
        return None
    by_event: dict[str, set[str]] = {}
    for session_native_id, event_type in rows:
        by_event.setdefault(str(event_type), set()).add(str(session_native_id))
    return by_event


def _expected_sessions(event: str, sessions: tuple[_SessionOpportunity, ...]) -> set[str] | None:
    if event == "SessionStart":
        return {session.native_id for session in sessions}
    if event == "UserPromptSubmit":
        return {session.native_id for session in sessions if session.authored_prompts}
    if event == "PreToolUse":
        return {session.native_id for session in sessions if session.tool_uses}
    return None


def hook_status(
    harness: HookHarness,
    *,
    coverage: bool = True,
    archive_root_path: Path | None = None,
    now_ms: int | None = None,
) -> HookHarnessStatus:
    """Return wiring plus bounded trailing-seven-day hook-flow evidence."""

    wired, feature_enabled, sources, wiring_mtime_ms = _wiring_snapshot(harness)
    supported = EVENTS_BY_HARNESS[harness]
    wired_ordered = tuple(event for event in supported if event in wired)
    recommended = tuple(event for event in RECOMMENDED_EVENTS if event in supported)
    missing_recommended = tuple(event for event in recommended if event not in wired)
    executable_available = shutil.which("polylogue-hook") is not None

    if not coverage:
        state = (
            "not-wired"
            if not wired
            else "disabled"
            if not feature_enabled
            else "incomplete"
            if missing_recommended
            else "unknown"
        )
        return HookHarnessStatus(
            harness=harness,
            settings_path=str(settings_path(harness)),
            settings_sources=sources,
            feature_enabled=feature_enabled,
            executable_available=executable_available,
            supported_events=supported,
            recommended_events=recommended,
            wired_events=wired_ordered,
            missing_recommended_events=missing_recommended,
            observed_last_7d=(),
            eligible_session_count=0,
            sessions_with_hook_events=0,
            sessions_without_hook_events=0,
            flow_state=state,
            coverage_checked=False,
            coverage=(),
        )

    current_ms = now_ms if now_ms is not None else int(datetime.now(UTC).timestamp() * 1000)
    window_cutoff = current_ms - _WINDOW_MS
    cutoff_ms = max(window_cutoff, wiring_mtime_ms or window_cutoff)
    root = archive_root_path
    if root is None:
        from polylogue.paths import archive_root

        root = archive_root()
    sessions = _recent_session_opportunities(root / "index.db", origin=ORIGIN_BY_HARNESS[harness], cutoff_ms=cutoff_ms)
    observed = _recent_hook_events(root / "source.db", origin=ORIGIN_BY_HARNESS[harness], cutoff_ms=cutoff_ms)
    if sessions is None or observed is None:
        state = (
            "not-wired"
            if not wired
            else "disabled"
            if not feature_enabled
            else "incomplete"
            if missing_recommended
            else "unknown"
        )
        return HookHarnessStatus(
            harness=harness,
            settings_path=str(settings_path(harness)),
            settings_sources=sources,
            feature_enabled=feature_enabled,
            executable_available=executable_available,
            supported_events=supported,
            recommended_events=recommended,
            wired_events=wired_ordered,
            missing_recommended_events=missing_recommended,
            observed_last_7d=tuple(event for event in supported if observed and event in observed),
            eligible_session_count=0,
            sessions_with_hook_events=0,
            sessions_without_hook_events=0,
            flow_state=state,
            coverage_checked=False,
            coverage=(),
            evidence_note="index.db sessions or source.db raw_hook_events unavailable",
        )

    eligible_ids = {session.native_id for session in sessions}
    sessions_with_events = set().union(*(ids & eligible_ids for ids in observed.values())) if observed else set()
    coverage_rows: list[HookEventCoverage] = []
    partial_gap = False
    for event in supported:
        observed_ids = observed.get(event, set()) & eligible_ids
        expected_ids = _expected_sessions(event, sessions)
        missing_expected = None if expected_ids is None else len(expected_ids - observed_ids)
        if event in wired and expected_ids and missing_expected:
            partial_gap = True
        coverage_rows.append(
            HookEventCoverage(
                event=event,
                wired=event in wired,
                recommended=event in recommended,
                observed_session_count=len(observed_ids),
                eligible_session_count=len(sessions),
                expected_session_count=None if expected_ids is None else len(expected_ids),
                missing_expected_count=missing_expected,
                observed_rate=(len(observed_ids) / len(sessions)) if sessions else None,
                enrichment=EVENT_ENRICHMENT.get(event, "harness lifecycle evidence"),
            )
        )

    without_events = eligible_ids - sessions_with_events
    if not wired:
        flow_state = "not-wired"
    elif not feature_enabled:
        flow_state = "disabled"
    elif missing_recommended:
        flow_state = "incomplete"
    elif not sessions:
        flow_state = "inactive"
    elif without_events:
        flow_state = "gap"
    elif partial_gap:
        flow_state = "partial"
    else:
        flow_state = "healthy"

    return HookHarnessStatus(
        harness=harness,
        settings_path=str(settings_path(harness)),
        settings_sources=sources,
        feature_enabled=feature_enabled,
        executable_available=executable_available,
        supported_events=supported,
        recommended_events=recommended,
        wired_events=wired_ordered,
        missing_recommended_events=missing_recommended,
        observed_last_7d=tuple(event for event in supported if event in observed and observed[event] & eligible_ids),
        eligible_session_count=len(sessions),
        sessions_with_hook_events=len(sessions_with_events),
        sessions_without_hook_events=len(without_events),
        flow_state=flow_state,
        coverage_checked=True,
        coverage=tuple(coverage_rows),
        evidence_note=f"sessions and hook events observed since {datetime.fromtimestamp(cutoff_ms / 1000, UTC).isoformat()}",
    )


def hook_statuses(
    *,
    harness: HookHarness | None = None,
    coverage: bool = True,
    archive_root_path: Path | None = None,
    now_ms: int | None = None,
) -> tuple[HookHarnessStatus, ...]:
    harnesses: tuple[HookHarness, ...] = (harness,) if harness is not None else ("claude-code", "codex")
    return tuple(
        hook_status(item, coverage=coverage, archive_root_path=archive_root_path, now_ms=now_ms) for item in harnesses
    )


def flow_states() -> tuple[str, ...]:
    return _FLOW_STATES


def _default_sidecar_dir() -> Path:
    override = os.environ.get("POLYLOGUE_HOOK_SIDECAR_DIR")
    if override:
        return Path(override)
    xdg_data_home = os.environ.get("XDG_DATA_HOME")
    base = Path(xdg_data_home) if xdg_data_home else Path.home() / ".local" / "share"
    return base / "polylogue" / "hooks"


def _detect_hook_provider(payload: dict[str, object]) -> HookHarness | None:
    forced = os.environ.get("POLYLOGUE_HOOK_PROVIDER")
    if forced in EVENTS_BY_HARNESS:
        return cast(HookHarness, forced)
    if "turn_id" in payload:
        return "codex"
    if "permission_mode" in payload or "model" in payload:
        return "claude-code"
    if "source" in payload:
        return "codex"
    return None


def _hook_provider_arg(args: list[str]) -> HookHarness | None:
    if "--provider" not in args:
        return None
    index = args.index("--provider")
    if index + 1 >= len(args):
        return None
    value = args[index + 1]
    if value == "claude-code":
        return "claude-code"
    if value == "codex":
        return "codex"
    return None


def hook_main(argv: list[str] | None = None) -> int:
    """Record one harness hook event without loading the archive runtime."""

    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print("Usage: polylogue-hook <event-type> [--provider claude-code|codex]", file=sys.stderr)
        return 1
    event_type = args[0]
    provider_arg = _hook_provider_arg(args[1:])
    if "--provider" in args[1:] and provider_arg is None:
        print("polylogue-hook: --provider must be claude-code or codex", file=sys.stderr)
        return 2
    allowed_events = (
        EVENTS_BY_HARNESS[provider_arg]
        if provider_arg is not None
        else tuple(dict.fromkeys((*CLAUDE_CODE_EVENTS, *CODEX_EVENTS)))
    )
    if event_type not in allowed_events:
        print(f"polylogue-hook: unsupported event type: {event_type}", file=sys.stderr)
        return 2
    try:
        payload = json.loads(sys.stdin.read())
    except (OSError, json.JSONDecodeError) as exc:
        print(f"polylogue-hook: invalid JSON payload: {exc}", file=sys.stderr)
        return 1
    if not isinstance(payload, dict):
        print("polylogue-hook: payload must be a JSON object", file=sys.stderr)
        return 1
    session_id = next(
        (
            value.strip()
            for key in ("session_id", "sessionId", "session")
            if isinstance((value := payload.get(key)), str) and value.strip()
        ),
        None,
    )
    if not session_id:
        print("polylogue-hook: could not extract session_id from payload", file=sys.stderr)
        return 1
    provider = provider_arg or _detect_hook_provider(cast(dict[str, object], payload))
    if provider is None:
        print(
            "polylogue-hook: could not detect provider; pass --provider claude-code|codex",
            file=sys.stderr,
        )
        return 1
    record = {
        "event_type": event_type,
        "session_id": session_id,
        "timestamp": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "provider": provider,
        "payload": payload,
    }
    from polylogue.sources.hooks import enqueue_hook_event

    sidecar_dir = _default_sidecar_dir()
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    enqueue_hook_event(
        event_type=event_type,
        session_id=session_id,
        provider=provider,
        timestamp=str(record["timestamp"]),
        payload=payload,
        root=sidecar_dir,
    )
    # Keep the established session journal available to older local tooling.
    # The daemon's durable path consumes only immutable pending envelopes.
    outfile = sidecar_dir / f"{provider}-{session_id}.jsonl"
    with outfile.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    return 0


__all__ = [
    "CLAUDE_CODE_EVENTS",
    "CODEX_EVENTS",
    "EVENTS_BY_HARNESS",
    "HookChangePlan",
    "HookEventCoverage",
    "HookHarness",
    "HookHarnessStatus",
    "HookSettingsError",
    "RECOMMENDED_EVENTS",
    "flow_states",
    "hook_main",
    "hook_status",
    "hook_statuses",
    "normalize_harness",
    "plan_hook_change",
    "resolve_events",
    "settings_path",
]
