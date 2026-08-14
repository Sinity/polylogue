"""Durable local spool for Claude Code, Codex, and Hermes hook events.

Hook commands must return promptly and cannot rely on the archive daemon being
up.  They therefore atomically place one immutable envelope in ``pending``.
The daemon drains those envelopes into ``source.db`` and only moves a file to
``acknowledged`` after its ``raw_hook_events`` row has committed.  A crash in
between is safe: replay uses the stable event id as the source-tier key.

Both ``pending`` and ``acknowledged`` are sharded into UTC day-of-arrival
subdirectories (``pending/2026-07-31/<event_id>.json``) so a stalled consumer
cannot silently accumulate six figures of dentries in one directory again
(polylogue-31r1 follow-up: a hook drain root pointed at a stale archive path
sat un-drained for ~17 days and grew to 108K+ flat files, making every
liveness/glob pass over the directory increasingly expensive). A day shard is
a natural retention unit -- it self-bounds to roughly one day's arrival
volume, and an emptied shard directory is removed opportunistically so the
shard count itself stays small. File-per-event (not an append-only journal)
is kept deliberately: an append-only journal needs either a cross-process
file lock or a write smaller than ``PIPE_BUF`` to guarantee atomic concurrent
appends, and hook payloads (tool output previews, etc.) are not reliably
under that bound -- a torn concurrent write would corrupt evidence. Atomic
``mkstemp`` + ``os.replace`` per event avoids that risk entirely and is what
this module already had proven safe.

A *legacy* flat ``pending/<event_id>.json`` layout (no day subdirectory) is
still recognized on drain for backward compatibility with envelopes enqueued
before day-sharding landed, or migrated in from a stale spool root.

Hermes support (fs1.7) reuses this exact mechanism rather than inventing a
parallel spool: Hermes lifecycle hooks are best-effort in the same way Claude
Code/Codex hooks are (a synchronous call can be lost during an outage), so the
same atomic-enqueue/idempotent-drain contract applies unchanged. The one
Hermes-specific addition is a payload hygiene guard
(``_reject_duplicated_transcript``) enforcing that lifecycle events carry
ids/hashes/timings/outcomes, never a second copy of message text. See
``polylogue.sources.parsers.hermes_lifecycle`` for the event-type taxonomy and
snapshot reconciliation, and ``docs/design/hermes-archival-export-contract.md``
for the durability/finalization semantics this spool exists to capture.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import sqlite3
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from polylogue.core.enums import Origin, Provider
from polylogue.logging import get_logger
from polylogue.paths import hooks_sidecar_dir
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from polylogue.storage.sqlite.archive_tiers.source_write import ArchiveHookEvent

logger = get_logger(__name__)

_SUPPORTED_PROVIDERS = frozenset({"claude-code", "codex", "hermes"})
_ORIGIN_TOKEN_BY_PROVIDER: dict[str, str] = {
    "claude-code": "claude-code-session",
    "codex": "codex-session",
    "hermes": "hermes-session",
}
_PENDING_DIRNAME = "pending"
_ACKNOWLEDGED_DIRNAME = "acknowledged"
_SAFE_EVENT_ID = re.compile(r"^[A-Za-z0-9_-]+$")

# Event bodies carry ids/hashes/timings/outcomes, never a duplicate transcript
# (fs1.7 AC: "event bodies contain no duplicated transcript"). Enforced at the
# validation boundary so a violation fails loudly at enqueue/drain time
# instead of silently bloating source.db with a second copy of conversation
# content. The threshold is generous (short tool argument previews, error
# messages, and ids are all well under it) but catches an accidental full
# message/turn body.
_TRANSCRIPT_LIKE_KEYS = ("text", "content", "transcript", "messages", "message_body", "reasoning")
_MAX_TRANSCRIPT_LIKE_FIELD_CHARS = 2000


class HookSpoolRecordError(ValueError):
    """A pending spool file is not a valid Claude Code/Codex/Hermes hook envelope."""


@dataclass(frozen=True, slots=True)
class HookSpoolDrainResult:
    """Outcome of one durable hook-spool drain attempt.

    ``remaining`` is a lower-bound signal, not an exact backlog count: it
    equals the batch's own unacknowledged count, plus 1 if collection proved
    at least one more path exists beyond this batch. A bounded caller only
    needs ``remaining <= failed`` to know whether draining again would make
    progress; computing an *exact* backlog size would require a full listing
    of the pending directory, which is exactly the O(n) cost this module
    exists to avoid at scale.
    """

    acknowledged: int
    failed: int
    remaining: int = 0


def pending_hook_spool_dir(root: Path | None = None) -> Path:
    """Return the directory that hook commands append to atomically."""

    return (root or hook_spool_root()) / _PENDING_DIRNAME


def acknowledged_hook_spool_dir(root: Path | None = None) -> Path:
    """Return the receipt directory for source-tier-acknowledged events."""

    return (root or hook_spool_root()) / _ACKNOWLEDGED_DIRNAME


def hook_spool_root() -> Path:
    """Resolve the hook spool root shared by producer and daemon.

    Derives from ``hooks_sidecar_dir()``, which is itself scoped under the
    resolved archive root (polylogue-o7hx). There is no separate ad hoc env
    override at this layer: ``POLYLOGUE_ARCHIVE_ROOT`` is the one knob that
    already has to be set to isolate a scratch/test daemon, so a second,
    hook-specific override was a manual escape hatch operators had to
    remember on top of it -- and repeatedly didn't.
    """

    return hooks_sidecar_dir()


def _day_shard(moment: datetime | None = None) -> str:
    """UTC ``YYYY-MM-DD`` bucket name a pending/acknowledged file lands under."""

    return (moment or datetime.now(UTC)).strftime("%Y-%m-%d")


_DAY_SHARD_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _iter_pending_event_paths(pending: Path, *, limit: int | None = None) -> list[Path]:
    """Collect up to ``limit`` pending envelope paths without enumerating the
    entire backlog.

    Walks day-shard subdirectories oldest-first (bounded by day count, not
    event count) plus any legacy flat ``*.json`` files sitting directly under
    ``pending`` (pre-sharding envelopes, or files migrated in from a stale
    root). Stops as soon as ``limit`` paths are collected: a 108K-file
    backlog no longer costs an ``O(n log n)`` full listing+sort on every
    bounded drain call, only ``O(limit)`` plus one cheap directory listing per
    shard actually visited.
    """

    collected: list[Path] = []

    def want_more() -> bool:
        return limit is None or len(collected) < limit

    try:
        entries = sorted(pending.iterdir())
    except OSError:
        return collected
    shard_dirs = [entry for entry in entries if entry.is_dir() and _DAY_SHARD_RE.match(entry.name)]
    legacy_files = [entry for entry in entries if entry.is_file() and entry.suffix == ".json"]
    for legacy in sorted(legacy_files):
        if not want_more():
            return collected
        collected.append(legacy)
    for shard in shard_dirs:
        if not want_more():
            return collected
        try:
            with os.scandir(shard) as it:
                for dirent in sorted(it, key=lambda e: e.name):
                    if not want_more():
                        break
                    if dirent.is_file() and dirent.name.endswith(".json"):
                        collected.append(Path(dirent.path))
        except OSError:
            continue
    return collected


def hook_spool_pending_depth(root: Path | None = None, *, cap: int = 5000) -> int:
    """Bounded pending-event count for observability, never an exact backlog size.

    Returns ``cap`` (not the true count) once the backlog reaches ``cap``, so
    a daemon health/heartbeat pass can log "queue depth >= N, growing
    unbounded" without itself paying for the O(n) listing that caused the
    original incident (a stalled consumer let 108K+ files accumulate silently
    for ~17 days before a human happened to notice a directory listing).
    Callers should treat ``depth >= cap`` as "alert now", not "count later".
    """

    pending = pending_hook_spool_dir(root)
    return len(_iter_pending_event_paths(pending, limit=cap))


def hook_spool_has_pending_events(root: Path | None = None) -> bool:
    """O(1)-ish liveness check: is there at least one pending hook envelope?

    Unlike a full listing, this stops at the first ``*.json`` dentry it finds
    -- an existence probe, not an enumeration -- so it stays cheap regardless
    of how large the backlog is. Used by the daemon's hook-aware maintenance
    passes instead of materializing the whole pending directory just to ask
    a yes/no question.
    """

    pending = pending_hook_spool_dir(root)
    try:
        entries = sorted(pending.iterdir())
    except OSError:
        return False
    for entry in entries:
        if entry.is_file() and entry.suffix == ".json":
            return True
    for entry in entries:
        if not (entry.is_dir() and _DAY_SHARD_RE.match(entry.name)):
            continue
        try:
            with os.scandir(entry) as it:
                if any(dirent.is_file() and dirent.name.endswith(".json") for dirent in it):
                    return True
        except OSError:
            continue
    return False


def enqueue_hook_event(
    *,
    event_type: str,
    session_id: str,
    provider: str,
    timestamp: str,
    payload: dict[str, object],
    root: Path | None = None,
    event_id: str | None = None,
) -> Path:
    """Atomically enqueue one hook event before the daemon receives it."""

    record: dict[str, object] = {
        "event_id": event_id or uuid4().hex,
        "event_type": event_type,
        "session_id": session_id,
        "timestamp": timestamp,
        "provider": provider,
        "payload": payload,
    }
    normalized = _validated_record(record)
    if not _SAFE_EVENT_ID.fullmatch(str(normalized["event_id"])):
        raise HookSpoolRecordError("hook spool event_id must contain only letters, digits, '_' or '-'")
    pending = pending_hook_spool_dir(root) / _day_shard()
    pending.mkdir(parents=True, exist_ok=True)
    target = pending / f"{normalized['event_id']}.json"
    if target.exists():
        return target
    _atomic_json_write(target, normalized)
    return target


def drain_hook_event_spool(
    archive_root: Path,
    *,
    root: Path | None = None,
    limit: int | None = None,
) -> HookSpoolDrainResult:
    """Persist pending events and acknowledge only committed records.

    Pending files deliberately remain in place when archive writes or envelope
    validation fail.  The next daemon pass can retry a transient write failure;
    a malformed producer record remains inspectable rather than disappearing.

    The archive is opened ONCE per drain, not per record: this runs on the
    daemon's single writer, and a per-record open/initialize turned a few
    thousand spooled events into a multi-minute writer monopoly that starved
    every other ingest path (observed live 2026-07-18). ``limit`` bounds one
    writer hold; the caller loops on ``remaining``.

    Path collection is bounded (:func:`_iter_pending_event_paths`), not a full
    listing, so a large backlog cannot inflate the cost of asking for one
    bounded batch. ``remaining`` therefore reports "at least this many are
    still pending" rather than an exact backlog size when the backlog exceeds
    what one collection pass inspected.
    """

    pending = pending_hook_spool_dir(root)
    if not pending.exists():
        return HookSpoolDrainResult(acknowledged=0, failed=0)
    # Collect one extra path beyond the limit (when bounded) purely to learn
    # whether more remain after this batch, without paying for a full count.
    probe_limit = None if limit is None else limit + 1
    paths = _iter_pending_event_paths(pending, limit=probe_limit)
    more_remain_beyond_batch = limit is not None and len(paths) > limit
    selected = paths if limit is None else paths[:limit]
    if not selected:
        return HookSpoolDrainResult(acknowledged=0, failed=0)
    acknowledged = 0
    failed = 0
    try:
        # Import after this module has initialized: ``sources.live.__init__``
        # exposes the watcher, and the watcher imports this spool module.
        from polylogue.sources.live.archive_open import (
            _open_archive_for_live_write,
            _source_tier_acquisition_required,
        )

        if not _source_tier_acquisition_required():
            initialize_active_archive_root(archive_root)
        store = _open_archive_for_live_write(archive_root)
    except (OSError, sqlite3.Error, ValueError):
        logger.warning("hook spool drain could not open the archive; all events remain pending", exc_info=True)
        return HookSpoolDrainResult(
            acknowledged=0,
            failed=len(selected),
            remaining=len(selected) + (1 if more_remain_beyond_batch else 0),
        )
    touched_shards: set[Path] = set()
    with store as archive:
        for path in selected:
            touched_shards.add(path.parent)
            try:
                record = _read_record(path)
                _persist_record(archive, path, record)
                archive.commit()
                _acknowledge(path, root=root)
            except (HookSpoolRecordError, OSError, sqlite3.Error, ValueError):
                with contextlib.suppress(Exception):
                    archive.rollback()
                failed += 1
                logger.warning("hook spool event remains pending: %s", path, exc_info=True)
            else:
                acknowledged += 1
    _prune_empty_shards(touched_shards, pending)
    return HookSpoolDrainResult(
        acknowledged=acknowledged,
        failed=failed,
        remaining=(len(selected) - acknowledged) + (1 if more_remain_beyond_batch else 0),
    )


def _prune_empty_shards(shards: set[Path], pending_root: Path) -> None:
    """Remove a day-shard directory once it has been fully drained.

    Keeps the shard count itself bounded (only recent/still-arriving days
    persist) instead of accumulating one empty directory per day forever.
    Best-effort: a shard that still has a legacy-format neighbor or a
    concurrent late arrival simply fails ``rmdir`` and is left for the next
    pass.
    """

    for shard in shards:
        if shard == pending_root:
            continue
        with contextlib.suppress(OSError):
            shard.rmdir()


def _read_record(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HookSpoolRecordError(f"invalid hook spool JSON: {path.name}") from exc
    if not isinstance(value, dict):
        raise HookSpoolRecordError(f"hook spool envelope must be an object: {path.name}")
    return _validated_record(value)


def _validated_record(value: dict[str, object]) -> dict[str, object]:
    required_text = ("event_id", "event_type", "session_id", "timestamp", "provider")
    for key in required_text:
        item = value.get(key)
        if not isinstance(item, str) or not item.strip():
            raise HookSpoolRecordError(f"hook spool envelope has no {key}")
    provider = str(value["provider"])
    if provider not in _SUPPORTED_PROVIDERS:
        raise HookSpoolRecordError(f"unsupported hook provider: {provider}")
    payload = value.get("payload")
    if not isinstance(payload, dict):
        raise HookSpoolRecordError("hook spool envelope payload must be an object")
    _reject_duplicated_transcript(payload)
    observed_at_ms = _timestamp_ms(str(value["timestamp"]))
    return {
        "event_id": str(value["event_id"]),
        "event_type": str(value["event_type"]),
        "session_id": str(value["session_id"]),
        "timestamp": str(value["timestamp"]),
        "provider": provider,
        "payload": dict(payload),
        "observed_at_ms": observed_at_ms,
    }


def _reject_duplicated_transcript(payload: dict[str, object]) -> None:
    """Reject a hook payload that looks like it duplicates transcript content.

    Applies to every provider, not only Hermes: hook events are evidence
    records, not a second copy of the conversation the archive already
    retains in full through session parsing.
    """
    for key in _TRANSCRIPT_LIKE_KEYS:
        value = payload.get(key)
        if isinstance(value, str) and len(value) > _MAX_TRANSCRIPT_LIKE_FIELD_CHARS:
            raise HookSpoolRecordError(
                f"hook spool payload field {key!r} looks like a duplicated transcript "
                f"({len(value)} chars > {_MAX_TRANSCRIPT_LIKE_FIELD_CHARS})"
            )


def _timestamp_ms(value: str) -> int:
    try:
        return int(datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC).timestamp() * 1000)
    except ValueError as exc:
        raise HookSpoolRecordError(f"invalid hook timestamp: {value!r}") from exc


def _persist_record(archive: ArchiveStore, path: Path, record: dict[str, object]) -> None:
    provider_token = str(record["provider"])
    provider = Provider.from_string(provider_token)
    try:
        origin_token = _ORIGIN_TOKEN_BY_PROVIDER[provider_token]
    except KeyError as exc:
        # ``_validated_record`` already rejects any provider outside
        # ``_SUPPORTED_PROVIDERS`` before a record reaches this point, so this
        # should be unreachable in the current call path -- but silently
        # defaulting an unrecognized provider to "codex-session" would
        # misclassify genuinely-unknown providers as Codex if that upstream
        # invariant ever drifts (e.g. a provider added to one set but not the
        # other). Raise instead of guessing.
        raise HookSpoolRecordError(f"no origin mapping for hook provider: {provider_token!r}") from exc
    origin = Origin.from_string(origin_token)
    payload = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    observed_at_ms_value = record["observed_at_ms"]
    if not isinstance(observed_at_ms_value, int):
        raise HookSpoolRecordError("hook spool envelope has an invalid observed timestamp")
    observed_at_ms = observed_at_ms_value
    source_path = str(path)
    # A hook event is evidence WITHIN a session, keyed to it by
    # ``session_native_id`` -- never a session of its own. Persisting it as a
    # raw_sessions row (as this path once did) minted an empty standalone
    # session per hook and inflated the archive with tens of thousands of
    # content-less session shells (polylogue-31r1). ``write_hook_event`` keeps
    # the durable blob + raw_hook_events row and skips the raw_sessions insert.
    archive.write_hook_event(
        provider=provider,
        payload=payload,
        source_path=source_path,
        acquired_at_ms=observed_at_ms,
        hook_event=ArchiveHookEvent(
            hook_event_id=f"hook:{record['event_id']}",
            origin=origin,
            source_path=source_path,
            event_type=str(record["event_type"]),
            payload=record,
            observed_at_ms=observed_at_ms,
            native_id=f"{record['session_id']}:{record['event_type']}:{record['event_id']}",
            session_native_id=str(record["session_id"]),
        ),
    )


def _acknowledge(path: Path, *, root: Path | None) -> None:
    # Shard acknowledged receipts by day-of-acknowledgment (not the pending
    # file's original arrival day): acknowledgment always has a well-defined
    # "now", whereas a replayed/migrated pending file may carry no shard
    # context at all (legacy flat layout). Keeps ``acknowledged`` bounded the
    # same way ``pending`` is, without needing to parse the source path.
    acknowledged = acknowledged_hook_spool_dir(root) / _day_shard()
    acknowledged.mkdir(parents=True, exist_ok=True)
    os.replace(path, acknowledged / path.name)
    _fsync_directory(acknowledged)


def _atomic_json_write(path: Path, payload: dict[str, object]) -> None:
    handle, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as output:
            output.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
        _fsync_directory(path.parent)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _fsync_directory(path: Path) -> None:
    """Persist an atomic rename's directory entry before returning success."""

    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "HookSpoolDrainResult",
    "HookSpoolRecordError",
    "acknowledged_hook_spool_dir",
    "drain_hook_event_spool",
    "enqueue_hook_event",
    "hook_spool_root",
    "pending_hook_spool_dir",
]
