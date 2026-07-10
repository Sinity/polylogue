"""``polylogue-hook`` console-script entrypoint.

Receives a hook event type as ``argv[1]`` and the event payload on stdin as
JSON. Emits one enriched JSONL record to the Polylogue hook sidecar directory
where the daemon watcher picks it up.

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

# Supported event types — kept in sync with docs/hooks.md in the main repo.
_CLAUDE_CODE_EVENTS = frozenset(
    {
        "SessionStart",
        "Setup",
        "UserPromptSubmit",
        "PreToolUse",
        "PostToolUse",
        "PostToolUseFailure",
        "PermissionRequest",
        "PermissionDenied",
        "Notification",
        "Elicitation",
        "ElicitationResult",
        "CwdChanged",
        "FileChanged",
        "WorktreeCreate",
        "SubagentStart",
        "Stop",
    }
)
_CODEX_EVENTS = frozenset(
    {
        "SessionStart",
        "UserPromptSubmit",
        "PreToolUse",
        "PostToolUse",
        "PermissionRequest",
        "Stop",
    }
)
_ALL_EVENTS = _CLAUDE_CODE_EVENTS | _CODEX_EVENTS


def _default_sidecar_dir() -> Path:
    override = os.environ.get("POLYLOGUE_HOOK_SIDECAR_DIR")
    if override:
        return Path(override)
    xdg_data_home = os.environ.get("XDG_DATA_HOME")
    base = Path(xdg_data_home) if xdg_data_home else Path.home() / ".local" / "share"
    return base / "polylogue" / "hooks"


def _detect_provider(payload: dict[str, object]) -> str | None:
    forced = os.environ.get("POLYLOGUE_HOOK_PROVIDER")
    if forced:
        return forced
    if "turn_id" in payload:
        return "codex"
    if "permission_mode" in payload or "model" in payload:
        return "claude-code"
    if "source" in payload:
        return "codex"
    return None


def _extract_session_id(payload: dict[str, object]) -> str | None:
    for key in ("session_id", "sessionId", "session"):
        val = payload.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print("Usage: polylogue-hook <event-type> [--provider claude-code|codex]", file=sys.stderr)
        return 1

    event_type = args[0]
    provider_arg: str | None = None
    if "--provider" in args[1:]:
        index = args.index("--provider")
        provider_arg = args[index + 1] if index + 1 < len(args) else None
        if provider_arg not in ("claude-code", "codex"):
            print("polylogue-hook: --provider must be claude-code or codex", file=sys.stderr)
            return 2
    allowed_events = (
        _CLAUDE_CODE_EVENTS
        if provider_arg == "claude-code"
        else _CODEX_EVENTS
        if provider_arg == "codex"
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

    provider = provider_arg or _detect_provider(payload)
    if provider not in ("claude-code", "codex"):
        print(
            "polylogue-hook: could not detect provider; pass --provider claude-code|codex",
            file=sys.stderr,
        )
        return 1

    record = {
        "event_type": event_type,
        "session_id": session_id,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "provider": provider,
        "payload": payload,
    }

    sidecar_dir = _default_sidecar_dir()
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    outfile = sidecar_dir / f"{provider}-{session_id}.jsonl"
    with outfile.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
