"""Hook-event producer: validate one harness event and spool it.

This runs on the harness's critical path -- two invocations per tool call,
multiplied by every concurrent agent -- so it must stay runnable *without*
the polylogue package. ``polylogue hooks install`` renders it as a bare
script path under ``python -I -S``, which skips ``site`` and never puts
site-packages on ``sys.path``; the module therefore imports stdlib only, and
only stdlib that is cheap to load. ``uuid`` and ``re`` each cost more to
import than they save here, so the event id comes from ``os.urandom`` and the
id charset is a frozenset. ``pathlib`` and ``tempfile`` are the exception:
they carry the atomic-publish shape (``mkstemp`` for exclusive creation,
``os.replace`` for the rename); hand-rolling those to save their import is not
worth owning a second unreviewed publish path. The producer does not fsync:
see :func:`atomic_json_write`.

It is also the single implementation of the pending envelope:
:mod:`polylogue.sources.hooks` imports this module's validation and atomic
write rather than keeping a second copy that could drift from what the drain
reads back.

Package-backed resolution (the configured hook provider, the configured
archive root) is consulted only through :func:`_import_optional`, which
returns ``None`` when the polylogue package is not importable. The installed
command bakes ``--provider`` and ``--sidecar-dir``, so the fast path never
reaches for it.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType

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

EVENTS_BY_HARNESS: dict[str, tuple[str, ...]] = {
    "claude-code": CLAUDE_CODE_EVENTS,
    "codex": CODEX_EVENTS,
}

# The drain accepts Hermes envelopes from the standalone adapters
# (``contrib/polylogue-hook``, ``packaging/polylogue-hooks``); this command
# produces only the two harnesses ``polylogue hooks install`` can wire.
SUPPORTED_PROVIDERS = frozenset({"claude-code", "codex", "hermes"})

PENDING_DIRNAME = "pending"

# Event bodies carry ids/hashes/timings/outcomes, never a duplicate transcript
# (fs1.7 AC: "event bodies contain no duplicated transcript"). Enforced at the
# validation boundary so a violation fails loudly at enqueue/drain time
# instead of silently bloating source.db with a second copy of conversation
# content. The threshold is generous (short tool argument previews, error
# messages, and ids are all well under it) but catches an accidental full
# message/turn body.
TRANSCRIPT_LIKE_KEYS = ("text", "content", "transcript", "messages", "message_body", "reasoning")
MAX_TRANSCRIPT_LIKE_FIELD_CHARS = 2000

# Equivalent to ``^[A-Za-z0-9_-]+$`` without paying for ``re``.
_EVENT_ID_ALPHABET = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-")

_USAGE = "Usage: polylogue-hook <event-type> [--provider claude-code|codex] [--sidecar-dir PATH]"


class HookSpoolRecordError(ValueError):
    """A hook envelope is not a valid Claude Code/Codex/Hermes spool record."""


def _import_optional(module_name: str) -> ModuleType | None:
    """Import a polylogue module, or return ``None`` when it is unreachable.

    The installed hook command runs under ``python -I -S``: site-packages is
    absent from ``sys.path`` by design, so every package-backed fallback in
    this module is optional rather than required.
    """
    try:
        return __import__(module_name, fromlist=["_"])
    except ImportError:
        return None


def reject_duplicated_transcript(payload: dict[str, object]) -> None:
    """Reject a hook payload that looks like it duplicates transcript content.

    Applies to every provider: hook events are evidence records, not a second
    copy of the conversation the archive already retains in full through
    session parsing.
    """
    for key in TRANSCRIPT_LIKE_KEYS:
        value = payload.get(key)
        if isinstance(value, str) and len(value) > MAX_TRANSCRIPT_LIKE_FIELD_CHARS:
            raise HookSpoolRecordError(
                f"hook spool payload field {key!r} looks like a duplicated transcript "
                f"({len(value)} chars > {MAX_TRANSCRIPT_LIKE_FIELD_CHARS})"
            )


def timestamp_ms(value: str) -> int:
    try:
        return int(datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC).timestamp() * 1000)
    except ValueError as exc:
        raise HookSpoolRecordError(f"invalid hook timestamp: {value!r}") from exc


def validated_record(value: dict[str, object]) -> dict[str, object]:
    """Normalize one envelope, deriving the observation instant the drain stores."""
    for key in ("event_id", "event_type", "session_id", "timestamp", "provider"):
        item = value.get(key)
        if not isinstance(item, str) or not item.strip():
            raise HookSpoolRecordError(f"hook spool envelope has no {key}")
    provider = str(value["provider"])
    if provider not in SUPPORTED_PROVIDERS:
        raise HookSpoolRecordError(f"unsupported hook provider: {provider}")
    payload = value.get("payload")
    if not isinstance(payload, dict):
        raise HookSpoolRecordError("hook spool envelope payload must be an object")
    reject_duplicated_transcript(payload)
    observed_at_ms = timestamp_ms(str(value["timestamp"]))
    return {
        "event_id": str(value["event_id"]),
        "event_type": str(value["event_type"]),
        "session_id": str(value["session_id"]),
        "timestamp": str(value["timestamp"]),
        "provider": provider,
        "payload": dict(payload),
        "observed_at_ms": observed_at_ms,
    }


def day_shard(moment: datetime | None = None) -> str:
    """UTC ``YYYY-MM-DD`` bucket name a pending/acknowledged file lands under."""

    return (moment or datetime.now(UTC)).strftime("%Y-%m-%d")


def enqueue_event(
    *,
    event_type: str,
    session_id: str,
    provider: str,
    timestamp: str,
    payload: dict[str, object],
    root: str,
    event_id: str | None = None,
) -> str:
    """Atomically place one validated envelope in the day-sharded pending spool."""

    record: dict[str, object] = {
        "event_id": event_id or os.urandom(16).hex(),
        "event_type": event_type,
        "session_id": session_id,
        "timestamp": timestamp,
        "provider": provider,
        "payload": payload,
    }
    normalized = validated_record(record)
    resolved_id = str(normalized["event_id"])
    if not resolved_id or not _EVENT_ID_ALPHABET.issuperset(resolved_id):
        raise HookSpoolRecordError("hook spool event_id must contain only letters, digits, '_' or '-'")
    shard = Path(root) / PENDING_DIRNAME / day_shard()
    shard.mkdir(parents=True, exist_ok=True)
    target = shard / f"{resolved_id}.json"
    if target.exists():
        return str(target)
    atomic_json_write(target, normalized)
    return str(target)


def atomic_json_write(path: Path, payload: dict[str, object]) -> None:
    """Publish one envelope by rename. No fsync: this runs inside the harness's
    hook budget on every tool call, and a directory fsync under host I/O
    pressure has measured in seconds. The rename is atomic against every
    reader; the only loss window is a power failure before the page cache
    drains, which the drain path tolerates (a missing envelope is a missing
    event, never a corrupt one). Durability of the retained bytes is the
    consumer's job, taken under its own fsync when it acknowledges."""

    handle, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as output:
            output.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
            output.write("\n")
        # ast-grep-ignore: replace-without-parent-fsync
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _fsync_directory(path: Path) -> None:
    """Persist a rename's directory entry; used by the consumer's acknowledgement,
    never on the producer path."""

    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _option_value(args: list[str], name: str) -> str | None:
    if name not in args:
        return None
    index = args.index(name)
    return args[index + 1] if index + 1 < len(args) else None


def _configured_provider() -> str | None:
    """The operator-forced harness, from the environment or the config file."""

    forced = os.environ.get("POLYLOGUE_HOOK_PROVIDER", "").strip()
    if forced:
        return forced
    config = _import_optional("polylogue.config")
    if config is None:
        return None
    provider = config.load_polylogue_config().hook_provider
    return str(provider) if provider else None


def detect_provider(payload: dict[str, object]) -> str | None:
    """Resolve the harness from the operator's setting, else the payload shape."""

    forced = _configured_provider()
    if forced in EVENTS_BY_HARNESS:
        return forced
    if "turn_id" in payload:
        return "codex"
    if "permission_mode" in payload or "model" in payload:
        return "claude-code"
    if "source" in payload:
        return "codex"
    return None


def default_sidecar_dir() -> str:
    """Resolve the spool root when the command carries no ``--sidecar-dir``.

    ``polylogue hooks install`` bakes the concrete resolved path into the
    rendered command (polylogue-o7hx), so this is the manual-invocation
    fallback: the package's own archive-root resolution where it is
    importable, otherwise ``POLYLOGUE_ARCHIVE_ROOT`` or the XDG default --
    which is byte-identical to the default archive root's hooks directory.
    """
    paths = _import_optional("polylogue.paths")
    if paths is not None:
        return str(paths.hooks_sidecar_dir())
    archive_root = os.environ.get("POLYLOGUE_ARCHIVE_ROOT", "").strip()
    if archive_root:
        return os.path.join(os.path.expanduser(archive_root), "hooks")
    data_home = os.environ.get("XDG_DATA_HOME") or os.path.join(os.path.expanduser("~"), ".local", "share")
    return os.path.join(data_home, "polylogue", "hooks")


def main(argv: list[str] | None = None) -> int:
    """Record one harness hook event without loading the archive runtime."""

    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print(_USAGE, file=sys.stderr)
        return 1

    event_type = args[0]
    options = args[1:]
    provider_arg = _option_value(options, "--provider")
    if "--provider" in options and provider_arg not in EVENTS_BY_HARNESS:
        print("polylogue-hook: --provider must be claude-code or codex", file=sys.stderr)
        return 2
    sidecar_dir_arg = _option_value(options, "--sidecar-dir")

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

    provider = provider_arg or detect_provider(payload)
    if provider is None:
        print(
            "polylogue-hook: could not detect provider; pass --provider claude-code|codex",
            file=sys.stderr,
        )
        return 1

    root = os.path.expanduser(sidecar_dir_arg) if sidecar_dir_arg else default_sidecar_dir()
    try:
        enqueue_event(
            event_type=event_type,
            session_id=session_id,
            provider=provider,
            timestamp=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            payload=payload,
            root=root,
        )
    except HookSpoolRecordError as exc:
        print(f"polylogue-hook: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
