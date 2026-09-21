"""Hook-event producer: validate one harness event and spool it.

This runs on the harness's critical path -- two invocations per tool call,
multiplied by every concurrent agent -- so it must stay runnable *without*
the polylogue package. ``polylogue hooks install`` renders it as a bare
script path under ``python -I -S``, which skips ``site`` and never puts
site-packages on ``sys.path``; the module therefore imports stdlib only, and
only stdlib that is cheap to load. ``uuid`` and ``re`` each cost more to
import than they save here, so the event id comes from ``os.urandom`` and the
id charset is a frozenset. ``pathlib`` is the exception: it carries the
carrier path arithmetic and is already imported by every consumer of this
module. The producer does not fsync: see :func:`append_event`.

It is also the single implementation of the carrier line:
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

#: The append-only carrier topology this producer writes. One file per
#: producer process per UTC day per harness:
#: ``carriers/<provider>/<YYYY-MM-DD>/<pid>.ndjson``. A hook process only ever
#: appends to its own file, so a partial write can only ever be completed by
#: the process that started it and no other producer can interleave a line
#: into the gap.
CARRIERS_DIRNAME = "carriers"

#: The retired file-per-event spool. Retained as a name only so the one-shot
#: ``--compact`` fold can find what the old producer left behind; nothing
#: writes here any more.
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

_USAGE = (
    "Usage: polylogue-hook <event-type> [--provider claude-code|codex] [--sidecar-dir PATH]\n"
    "       polylogue-hook --compact [--sidecar-dir PATH]"
)


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
    copy of the conversation. The one thing the archive derives out of a hook
    payload into ``blocks`` is a tool result whose own overflow sidecar is
    unreachable -- see ``sources/live/hook_tool_response.py``.
    """
    for key in TRANSCRIPT_LIKE_KEYS:
        size = transcript_like_chars(payload.get(key))
        if size > MAX_TRANSCRIPT_LIKE_FIELD_CHARS:
            raise HookSpoolRecordError(
                f"hook spool payload field {key!r} looks like a duplicated transcript "
                f"({size} chars > {MAX_TRANSCRIPT_LIKE_FIELD_CHARS})"
            )


TRANSCRIPT_LIKE_MAX_DEPTH = 6


def transcript_like_chars(value: object, *, depth: int = 0) -> int:
    """Characters of string content reachable under ``value``, depth-capped.

    A bare ``isinstance(value, str)`` check let ``{"messages": ["<1 MB>"]}``
    spool a transcript verbatim into durable source.db.
    """
    if isinstance(value, str):
        return len(value)
    if depth >= TRANSCRIPT_LIKE_MAX_DEPTH:
        return 0
    if isinstance(value, dict):
        return sum(transcript_like_chars(item, depth=depth + 1) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(transcript_like_chars(item, depth=depth + 1) for item in value)
    return 0


def timestamp_ms(value: str) -> int:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        # Naive means UTC here too; ``astimezone`` alone would read it in the
        # producing host's local zone.
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return int(parsed.astimezone(UTC).timestamp() * 1000)
    except ValueError as exc:
        raise HookSpoolRecordError(f"invalid hook timestamp: {value!r}") from exc


# Both generations of every envelope key the drain requires. The harness emits
# a camelCase generation of its payloads, and an adapter that mirrors that
# spelling into the envelope must not be read as an envelope missing the field.
# Spelled out here rather than imported from ``core.hook_payload`` because this
# module runs under ``python -I -S`` with the package off ``sys.path``.
_ENVELOPE_KEY_SPELLINGS: dict[str, tuple[str, ...]] = {
    "event_id": ("event_id", "eventId"),
    "event_type": ("event_type", "eventType"),
    "session_id": ("session_id", "sessionId"),
    "timestamp": ("timestamp",),
    "provider": ("provider",),
}


def _envelope_text(value: dict[str, object], key: str) -> str:
    for spelling in _ENVELOPE_KEY_SPELLINGS[key]:
        item = value.get(spelling)
        if isinstance(item, str) and item.strip():
            return item
    raise HookSpoolRecordError(f"hook spool envelope has no {key}")


def validated_record(value: dict[str, object]) -> dict[str, object]:
    """Normalize one envelope, deriving the observation instant the drain stores.

    The returned envelope is snake_case whichever generation arrived; the
    payload is passed through verbatim because it is the harness's own evidence.
    """
    fields = {key: _envelope_text(value, key) for key in _ENVELOPE_KEY_SPELLINGS}
    provider = fields["provider"]
    if provider not in SUPPORTED_PROVIDERS:
        raise HookSpoolRecordError(f"unsupported hook provider: {provider}")
    payload = value.get("payload")
    if not isinstance(payload, dict):
        raise HookSpoolRecordError("hook spool envelope payload must be an object")
    reject_duplicated_transcript(payload)
    return {
        **fields,
        "payload": dict(payload),
        "observed_at_ms": timestamp_ms(fields["timestamp"]),
    }


def day_shard(moment: datetime | None = None) -> str:
    """UTC ``YYYY-MM-DD`` bucket name a carrier lands under."""

    return (moment or datetime.now(UTC)).strftime("%Y-%m-%d")


def carrier_path(root: str | Path, provider: str, *, moment: datetime | None = None, pid: int | None = None) -> Path:
    """Return the carrier this process appends to for ``provider`` today.

    One file per producer process per UTC day per harness. The pid segment is
    what makes a bare ``O_APPEND`` write sufficient: two harness processes
    never share a carrier, so interleaving is impossible by construction
    rather than by a size bound nobody can enforce on a tool-output preview.
    """

    return Path(root) / CARRIERS_DIRNAME / provider / day_shard(moment) / f"{pid or os.getpid()}.ndjson"


def append_event(
    *,
    event_type: str,
    session_id: str,
    provider: str,
    timestamp: str,
    payload: dict[str, object],
    root: str,
    event_id: str | None = None,
) -> str:
    """Append one validated envelope to this process's NDJSON carrier.

    No temporary file, no rename, no fsync: one ``O_APPEND`` write of one
    newline-terminated line. The consumer acquires the carrier's bytes like
    any other append-only source and materializes the events out of the
    retained bytes, so the producer owes durability to nobody -- the only loss
    window is a power failure before the page cache drains, which costs a
    missing event, never a corrupt one.
    """

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
        raise HookSpoolRecordError("hook carrier event_id must contain only letters, digits, '_' or '-'")
    target = carrier_path(root, str(normalized["provider"]))
    target.parent.mkdir(parents=True, exist_ok=True)
    append_carrier_line(target, normalized)
    return str(target)


def carrier_line(record: dict[str, object]) -> bytes:
    """Serialize one validated envelope as its carrier line.

    Compact, key-sorted, newline-terminated, and never containing a raw
    newline of its own (``json.dumps`` escapes them), so a line is a record
    and a byte offset into the carrier is a stable event coordinate.
    """

    return json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"


def append_carrier_line(path: Path, record: dict[str, object]) -> None:
    """Append one line with a single ``O_APPEND`` write.

    ``O_APPEND`` makes the seek-to-end and the write one atomic step against
    every other writer of the same file, so a concurrent appender can never
    land inside this line. A short write is completed here rather than left
    torn; only this process writes this carrier, so the continuation cannot be
    interleaved either.
    """

    line = carrier_line(record)
    descriptor = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    try:
        written = 0
        while written < len(line):
            written += os.write(descriptor, line[written:])
    finally:
        os.close(descriptor)


def fsync_directory(path: Path) -> None:
    """Persist a directory entry.

    Never used on the producer path. The one-shot ``--compact`` fold of the
    retired file-per-event spool uses it to prove a carrier is durable before
    it retires the files it folded.
    """

    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


#: One compacted carrier stays small enough to acquire, materialize and
#: replay as a single unit. 64 MiB is roughly 60k real envelopes.
MAX_COMPACTED_CARRIER_BYTES = 64 * 1024 * 1024

ACKNOWLEDGED_DIRNAME = "acknowledged"


class _CompactionSink:
    """Append folded envelopes to size-bounded compacted carriers."""

    def __init__(self, root: Path, *, max_bytes: int = MAX_COMPACTED_CARRIER_BYTES) -> None:
        self._root = root
        self._max_bytes = max_bytes
        self._open: dict[tuple[str, str], tuple[Path | None, int, int]] = {}
        self.carriers: list[Path] = []

    def append(self, record: dict[str, object]) -> Path:
        provider = str(record["provider"])
        day = str(record["timestamp"])[:10]
        if len(day) != 10 or not day[:4].isdigit():
            day = day_shard()
        key = (provider, day)
        line = carrier_line(record)
        path, index, size = self._open.get(key, (None, 0, 0))
        if path is None or size + len(line) > self._max_bytes:
            index = index + 1 if path is not None else 0
            directory = self._root / CARRIERS_DIRNAME / provider / day
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"compacted-{index}.ndjson"
            size = path.stat().st_size if path.exists() else 0
            self.carriers.append(path)
        append_carrier_line(path, record)
        self._open[key] = (path, index, size + len(line))
        return path

    def seal(self) -> None:
        """Fsync every carrier written, and the directories holding them."""

        directories: set[Path] = set()
        for path in self.carriers:
            descriptor = os.open(path, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            directories.add(path.parent)
        for directory in directories:
            fsync_directory(directory)


def _retire(path: Path, root: Path, bucket: str) -> None:
    """Move one folded envelope aside, without fsyncing the directory entry.

    Nothing depends on this rename surviving a power failure: the carrier the
    envelope was folded into is already fsynced before any retirement happens,
    so a lost rename costs a re-fold, and a re-folded envelope is the same
    content-derived event the archive already holds. Paying a directory fsync
    per file here would put back exactly the per-file cost this fold exists to
    remove.
    """

    acknowledged = root / ACKNOWLEDGED_DIRNAME / bucket
    acknowledged.mkdir(parents=True, exist_ok=True)
    # ast-grep-ignore: replace-without-parent-fsync
    os.replace(path, acknowledged / path.name)


#: Envelopes folded between durability checkpoints. A checkpoint fsyncs the
#: carriers and retires exactly the envelopes already in them, so an
#: interrupted fold re-folds at most this many -- against the ~716k-envelope
#: legacy spool, where a single unbounded pass re-folds everything and appends
#: a second full copy of every event to the carriers.
COMPACTION_CHECKPOINT_EVENTS = 10_000


def _sorted_directory(directory: Path) -> list[os.DirEntry[str]]:
    """One directory's entries in name order, or nothing when it is absent.

    One directory at a time, never the whole tree: ``sorted(rglob("*"))`` holds
    and sorts every path at once, measured at 1.37 GB resident for the
    716,754-file legacy spool -- on the machine that also has to hold the
    rebuild's headroom. This costs the widest single day shard instead.
    """

    try:
        with os.scandir(directory) as scan:
            return sorted(scan, key=lambda entry: entry.name)
    except FileNotFoundError:
        return []


def compact_legacy_spool(
    root: Path,
    *,
    max_bytes: int = MAX_COMPACTED_CARRIER_BYTES,
    checkpoint_events: int = COMPACTION_CHECKPOINT_EVENTS,
) -> dict[str, object]:
    """Fold the retired file-per-event spool into append-only carriers, once.

    Reads every ``pending/**/*.json`` envelope and every root-level
    ``<provider>-<session>.jsonl`` mirror, appends each as one carrier line,
    then checkpoints: fsync the carriers, and only then retire the envelopes
    already in them under ``acknowledged/``. Ordering within a checkpoint is
    durable carrier first, retirement second, so an interrupted run re-folds
    at most one checkpoint's worth -- and a re-folded envelope is the same
    content-derived event the archive already holds.

    Refusals are counted and named, never silently dropped: a hidden
    atomic-write tempname is not a published envelope, a zero-byte file is not
    a record, a member that is not a regular file is not a spool entry, and a
    payload this producer would refuse to write is not one it will fold in
    through the back door.

    Every inspected member lands in exactly one of ``folded`` or ``refused``,
    and its bytes in exactly one of ``folded_bytes`` or ``refused_bytes``, so
    ``scanned``/``scanned_bytes`` conserve against the frozen manifest by
    count *and* by size (k8wv AC4) rather than by filename count alone.
    """

    folded = 0
    folded_bytes = 0
    scanned = 0
    scanned_bytes = 0
    refused: dict[str, int] = {}
    refused_bytes: dict[str, int] = {}

    def refuse(reason: str, size: int) -> None:
        refused[reason] = refused.get(reason, 0) + 1
        refused_bytes[reason] = refused_bytes.get(reason, 0) + size

    sink = _CompactionSink(root, max_bytes=max_bytes)
    retire: list[tuple[Path, str]] = []
    retired = 0

    def checkpoint() -> None:
        """Make this batch's carriers durable, then retire what is in them."""
        nonlocal retired
        if not retire:
            return
        sink.seal()
        for path, bucket in retire:
            _retire(path, root, bucket)
        retired += len(retire)
        retire.clear()

    def member_size(path: Path) -> int:
        try:
            return path.lstat().st_size
        except OSError:
            return 0

    # Explicit stack, one directory's entries resident at a time. Descend into
    # real subdirectories only: a symlink is classified as a member below
    # instead of being followed out of the spool.
    directories = [root / PENDING_DIRNAME]
    while directories:
        entries = _sorted_directory(directories.pop())
        subdirectories = [Path(entry.path) for entry in entries if entry.is_dir(follow_symlinks=False)]
        directories.extend(reversed(subdirectories))
        for entry in entries:
            if entry.is_dir(follow_symlinks=False):
                continue
            path = Path(entry.path)
            size = member_size(path)
            scanned += 1
            scanned_bytes += size
            if not path.is_file() or path.is_symlink():
                refuse("spool member is not a regular file", size)
                continue
            if path.name.startswith("."):
                refuse("hidden atomic-write tempname is not a published envelope", size)
                continue
            if path.suffix != ".json":
                refuse(f"unrecognized spool member suffix: {path.suffix or '(none)'}", size)
                continue
            try:
                raw = path.read_text(encoding="utf-8")
            except OSError:
                refuse("unreadable", size)
                continue
            if not raw.strip():
                refuse("zero-byte file carries no record", size)
                continue
            try:
                value = json.loads(raw)
                if not isinstance(value, dict):
                    raise HookSpoolRecordError("envelope must be an object")
                record = validated_record(value)
            except (json.JSONDecodeError, HookSpoolRecordError) as exc:
                refuse(f"invalid envelope: {type(exc).__name__}", size)
                continue
            sink.append(record)
            folded += 1
            folded_bytes += size
            retire.append((path, day_shard()))
            if len(retire) >= checkpoint_events:
                checkpoint()

    # The root-level ``<provider>-<session>.jsonl`` per-session journals are
    # the one remaining unowned member of the retired spool tree (k8wv AC6).
    # They are NOT folded: a journal record carries no event identity, every
    # record in one is already enveloped in the spool under its producer's own
    # ``event_id``, and deriving an id here would mint a second identity for
    # an event the archive already holds. Naming the refusal is what AC6 asks
    # for; reversing it is a decision, not a compaction detail.
    for path in sorted(root.glob("*.jsonl")):
        if path.is_file():
            size = member_size(path)
            scanned += 1
            scanned_bytes += size
            refuse("per-session journal mirror is not an ingest surface (docs/hooks.md)", size)

    checkpoint()
    return {
        "folded": folded,
        "folded_bytes": folded_bytes,
        "scanned": scanned,
        "scanned_bytes": scanned_bytes,
        "carriers": sorted({str(path) for path in sink.carriers}),
        "refused": dict(sorted(refused.items())),
        "refused_bytes": dict(sorted(refused_bytes.items())),
        "retired": retired,
    }


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
    if "turn_id" in payload or "turnId" in payload:
        return "codex"
    if "permission_mode" in payload or "permissionMode" in payload or "model" in payload:
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

    if "--compact" in args:
        sidecar = _option_value(args, "--sidecar-dir")
        compact_root = Path(os.path.expanduser(sidecar) if sidecar else default_sidecar_dir())
        print(json.dumps(compact_legacy_spool(compact_root), sort_keys=True))
        return 0

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
        append_event(
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
