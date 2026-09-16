"""``polylogue-hook`` console-script entrypoint.

Receives a hook event type as ``argv[1]`` and the event payload on stdin as
JSON. Appends one enriched envelope as a single newline-terminated JSON line to
this process's own carrier under ``carriers/<provider>/<day>/<pid>.ndjson``. The
archive acquires that carrier's bytes like any other append-only source and
materializes the events out of them, so the producer does not fsync.

Mirrors the behaviour of ``contrib/polylogue-hook`` in the main repository so
that ``pip install polylogue-hooks`` provides the same surface without any
dependency on the main ``polylogue`` distribution.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

# Supported event types — kept in sync with docs/hooks.md in the main repo.
_CLAUDE_CODE_EVENTS = frozenset(
    {
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
    }
)
_CODEX_EVENTS = frozenset(
    {
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
    }
)
_HERMES_EVENTS = frozenset(
    {
        "model_attempt",
        "model_failure",
        "model_retry",
        "model_fallback",
        "tool_start",
        "tool_finish",
        "tool_failure",
        "tool_denial",
        "approval_request",
        "approval_response",
        "subagent_start",
        "subagent_finish",
        "compaction",
        "rewind",
        "on_session_end",
        "on_session_finalize",
        "context_injected",
    }
)
_ALL_EVENTS = _CLAUDE_CODE_EVENTS | _CODEX_EVENTS | _HERMES_EVENTS
_TRANSCRIPT_LIKE_KEYS = ("text", "content", "transcript", "messages", "message_body", "reasoning")
_MAX_TRANSCRIPT_LIKE_FIELD_CHARS = 2000


def _default_sidecar_dir() -> Path:
    """Resolve the sidecar dir when the command carries no ``--sidecar-dir``.

    Falls back through ``POLYLOGUE_ARCHIVE_ROOT`` (mirroring the main
    package's ``archive_root()``/``hooks_sidecar_dir()``) rather than a
    separate, hook-specific env override: any environment that isolates a
    scratch/test daemon must already set ``POLYLOGUE_ARCHIVE_ROOT``, so this
    package needs no additional knob to stay isolated (polylogue-o7hx).
    Prefer ``--sidecar-dir`` (baked in by ``polylogue hooks install``) when
    the invoking environment cannot be trusted to carry that var at all.
    """
    archive_root_override = os.environ.get("POLYLOGUE_ARCHIVE_ROOT", "").strip()
    if archive_root_override:
        return Path(archive_root_override).expanduser() / "hooks"
    xdg_data_home = os.environ.get("XDG_DATA_HOME")
    base = Path(xdg_data_home) if xdg_data_home else Path.home() / ".local" / "share"
    return base / "polylogue" / "hooks"


def _sidecar_dir_arg(args: list[str]) -> Path | None:
    """Parse an explicit ``--sidecar-dir PATH`` baked in at install time."""
    if "--sidecar-dir" not in args:
        return None
    index = args.index("--sidecar-dir")
    if index + 1 >= len(args):
        return None
    return Path(args[index + 1]).expanduser()


def _detect_provider(payload: dict[str, object], *, event_type: str | None = None) -> str | None:
    forced = os.environ.get("POLYLOGUE_HOOK_PROVIDER")
    if forced:
        return forced
    if event_type in _HERMES_EVENTS and event_type not in (_CLAUDE_CODE_EVENTS | _CODEX_EVENTS):
        return "hermes"
    if "turn_id" in payload:
        return "codex"
    if "permission_mode" in payload or "model" in payload:
        return "claude-code"
    if "source" in payload:
        return "codex"
    return None


def _reject_duplicated_transcript(payload: dict[str, object]) -> str | None:
    """Return an error message if the payload looks like a duplicated transcript."""
    for key in _TRANSCRIPT_LIKE_KEYS:
        size = _transcript_like_chars(payload.get(key))
        if size > _MAX_TRANSCRIPT_LIKE_FIELD_CHARS:
            return (
                f"polylogue-hook: payload field {key!r} looks like a duplicated transcript "
                f"({size} chars > {_MAX_TRANSCRIPT_LIKE_FIELD_CHARS})"
            )
    return None


_TRANSCRIPT_LIKE_MAX_DEPTH = 6


def _transcript_like_chars(value: object, depth: int = 0) -> int:
    """Characters of string content under ``value``, depth-capped (mirrors sources/hook_producer.py)."""
    if isinstance(value, str):
        return len(value)
    if depth >= _TRANSCRIPT_LIKE_MAX_DEPTH:
        return 0
    if isinstance(value, dict):
        return sum(_transcript_like_chars(item, depth + 1) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_transcript_like_chars(item, depth + 1) for item in value)
    return 0


def _extract_session_id(payload: dict[str, object]) -> str | None:
    for key in ("session_id", "sessionId", "session"):
        val = payload.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print(
            "Usage: polylogue-hook <event-type> [--provider claude-code|codex|hermes] [--sidecar-dir PATH]",
            file=sys.stderr,
        )
        return 1

    event_type = args[0]
    sidecar_dir_arg = _sidecar_dir_arg(args[1:])
    provider_arg: str | None = None
    if "--provider" in args[1:]:
        index = args.index("--provider")
        provider_arg = args[index + 1] if index + 1 < len(args) else None
        if provider_arg not in ("claude-code", "codex", "hermes"):
            print("polylogue-hook: --provider must be claude-code, codex, or hermes", file=sys.stderr)
            return 2
    allowed_events = (
        _CLAUDE_CODE_EVENTS
        if provider_arg == "claude-code"
        else _CODEX_EVENTS
        if provider_arg == "codex"
        else _HERMES_EVENTS
        if provider_arg == "hermes"
        else _ALL_EVENTS
    )
    if event_type not in allowed_events:
        print(f"polylogue-hook: unsupported event type: {event_type}", file=sys.stderr)
        return 2

    try:
        payload_text = sys.stdin.read()
    except (OSError, KeyboardInterrupt) as exc:
        print(f"polylogue-hook: failed to read stdin: {exc}", file=sys.stderr)
        return 1

    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError as exc:
        print(f"polylogue-hook: invalid JSON payload: {exc}", file=sys.stderr)
        return 1

    if not isinstance(payload, dict):
        print("polylogue-hook: payload must be a JSON object", file=sys.stderr)
        return 1

    session_id = _extract_session_id(payload)
    if not session_id:
        print("polylogue-hook: could not extract session_id from payload", file=sys.stderr)
        return 1

    transcript_error = _reject_duplicated_transcript(payload)
    if transcript_error is not None:
        print(transcript_error, file=sys.stderr)
        return 1

    provider = provider_arg or _detect_provider(payload, event_type=event_type)
    if provider not in ("claude-code", "codex", "hermes"):
        print(
            "polylogue-hook: could not detect provider; pass --provider claude-code|codex|hermes",
            file=sys.stderr,
        )
        return 1

    record: dict[str, object] = {
        "event_id": uuid4().hex,
        "event_type": event_type,
        "session_id": session_id,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "provider": provider,
        "payload": payload,
    }

    sidecar_dir = sidecar_dir_arg or _default_sidecar_dir()
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    carrier = sidecar_dir / "carriers" / provider / day / f"{os.getpid()}.ndjson"
    os.makedirs(carrier.parent, exist_ok=True)
    _append_carrier_line(carrier, record)

    return 0


def _append_carrier_line(path: Path, record: dict[str, object]) -> None:
    """Append one newline-terminated JSON line with a single ``O_APPEND`` write.

    ``O_APPEND`` makes the seek-to-end and the write one atomic step, and only
    this process writes this carrier, so a short write can only ever be
    completed here. No tempfile, no rename, no fsync: the archive acquires the
    carrier's retained bytes, so the worst loss is a missing event, never a
    corrupt one. Mirrors ``polylogue.sources.hook_producer.append_carrier_line``.
    """
    line = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    try:
        written = 0
        while written < len(line):
            written += os.write(descriptor, line[written:])
    finally:
        os.close(descriptor)


if __name__ == "__main__":
    raise SystemExit(main())
