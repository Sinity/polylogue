"""Route per-record maintenance failures to the daemon raw-failure surface.

This module is the bridge between :mod:`polylogue.maintenance.replay`
and the daemon raw-failure surface implemented by
:func:`polylogue.daemon.status._raw_failure_info` and
:func:`polylogue.daemon.health._check_raw_failures_medium` (#844).

Why a separate substrate
========================

Maintenance failures are operationally distinct from ingest failures:

* ingest failures land in ``raw_sessions.parse_error`` /
  ``raw_sessions.validation_status`` because they describe the
  state of a *row* in the archive — the raw acquisition exists but
  could not be parsed or validated;
* maintenance failures describe an *attempt* to repair or replay
  archive state. They do not always correspond to a single
  ``raw_sessions`` row (e.g. a session-insight rebuild that fails
  on one ``session_id``), and they carry a maintenance-specific
  ``operation_id`` that should be visible to operators.

The two #1198 design alternatives were:

A. Extend ``raw_sessions.validation_status`` to a third value
   ``"maintenance_failed"``. Reuses existing daemon read paths but
   conflates two semantically different failure surfaces and forces a
   schema-version bump for every existing archive.
B. Persist routed failures to a small append-only JSONL file under
   ``<archive_root>/.maintenance-state/failures.jsonl`` and have the
   daemon read it alongside the SQL raw-failure query.

We chose (B). The maintenance state directory already exists
(``polylogue/maintenance/replay.py:_STATE_DIRNAME``) and is the
canonical location for resume cursors and operation snapshots. A
single JSONL file fits cleanly there and avoids any schema rebuild,
which is important because Polylogue intentionally has no in-place schema upgrade
chain — a schema bump forces every operator to rebuild their archive.

The file is **bounded**: writes append, and reads cap the returned
sample list at :data:`MAINTENANCE_FAILURE_SAMPLE_LIMIT`. A separate
maintenance task may rotate or truncate the file; the daemon surface
treats older entries as background context, not as a queue.

Redaction
=========

Every routed sample passes through redaction at construction time:

* ``message`` is truncated to :data:`MAX_MESSAGE_LEN` characters;
* absolute Unix paths in ``message`` are replaced with ``[redacted]``
  using the same heuristic as :class:`RawFailureSample` so the daemon
  surface never exposes operator filesystem layout;
* ``locator`` is preserved verbatim because it is already a typed,
  short identifier (e.g. ``target:session_insights``); a separate
  pass strips absolute path segments out of any trailing ``:path``
  segment for the same reason.

Raw message bodies and blob bytes are never routed through this
surface — only the bounded ``FailureSample.message`` from the planner
envelope is persisted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from polylogue.core.json import JSONDocument, dumps, json_document, loads
from polylogue.logging import get_logger
from polylogue.maintenance.planner import FailureSample

logger = get_logger(__name__)

#: Maximum stored characters in a routed ``message`` field. Keeps the
#: JSONL file size predictable even when a repair surfaces a long
#: traceback excerpt.
MAX_MESSAGE_LEN: Final[int] = 500

#: Cap on routed samples returned by :func:`read_maintenance_failures`.
#: Mirrors the per-failure surface limit used by the daemon status
#: payload so the two sources of failure samples are comparable.
MAINTENANCE_FAILURE_SAMPLE_LIMIT: Final[int] = 50

#: Subdirectory under ``<archive_root>`` that holds maintenance state
#: files. Matches ``polylogue.maintenance.replay._STATE_DIRNAME`` so
#: routed failures share a directory with operation snapshots and
#: resume cursors. Kept in sync deliberately; an import from the
#: ``replay`` module would create a cycle.
_STATE_DIRNAME: Final[str] = ".maintenance-state"

#: Name of the JSONL file under the state directory. One JSON object
#: per line, newest entries appended at the end. Designed so a partial
#: write (process kill mid-line) only loses the final entry — every
#: prior entry is parseable on its own.
_FAILURE_FILE_NAME: Final[str] = "failures.jsonl"

#: Matches candidate absolute Unix paths in the same shape as
#: :func:`polylogue.daemon.status._PATH_REDACTION_RE`. We keep the
#: definition local so :mod:`polylogue.maintenance` does not need to
#: import from :mod:`polylogue.daemon` (and create a layering cycle).
_PATH_REDACTION_RE: Final[re.Pattern[str]] = re.compile(r"/(?:[a-zA-Z0-9._\-]+/)*[a-zA-Z0-9._\-]+")


def _redact_paths(text: str) -> str:
    """Replace absolute Unix paths in ``text`` with ``[redacted]``.

    The redaction strategy mirrors
    :func:`polylogue.daemon.status.RawFailureSample._redact_file_paths`
    so daemon status text and routed maintenance text remain
    consistent. URL path segments (preceded by alphanumerics, dots,
    colons, or containing ``://``) are preserved.
    """

    def _replace(m: re.Match[str]) -> str:
        start = m.start()
        if start == 0:
            return "[redacted]"
        prev = text[start - 1]
        if prev.isalnum() or prev in (".", ":"):
            return m.group(0)
        prefix = text[max(0, start - 16) : start + 1]
        if "://" in prefix:
            return m.group(0)
        return "[redacted]"

    return _PATH_REDACTION_RE.sub(_replace, text)


def _redact_locator(locator: str) -> str:
    """Redact absolute paths embedded in a ``target:...`` locator string.

    Some locators emitted by :mod:`polylogue.maintenance.replay` embed
    an absolute path in a trailing segment. The redactor strips any
    absolute path unconditionally for the locator form, because a
    structured colon-separated suffix would otherwise look like a URL
    host separator and prevent path-aware redaction.
    """
    # Always replace the first absolute path token in the locator
    # suffix. The URL/host heuristic used for free-form messages is
    # inappropriate here because the locator is a structured colon-
    # separated identifier.
    return _PATH_REDACTION_RE.sub("[redacted]", locator)


@dataclass(frozen=True)
class MaintenanceFailureRecord:
    """One routed maintenance failure persisted to the JSONL surface.

    Fields are a superset of :class:`FailureSample` plus the
    operation identity needed by the daemon raw-failure surface so it
    can attribute failures back to the originating replay run.
    """

    operation_id: str
    target: str
    kind: str
    locator: str
    message: str
    routed_at: str

    def to_dict(self) -> JSONDocument:
        return json_document(
            {
                "operation_id": self.operation_id,
                "target": self.target,
                "kind": self.kind,
                "locator": self.locator,
                "message": self.message,
                "routed_at": self.routed_at,
            }
        )

    @classmethod
    def from_dict(cls, data: object) -> MaintenanceFailureRecord | None:
        if not isinstance(data, dict):
            return None
        try:
            return cls(
                operation_id=str(data.get("operation_id", "")),
                target=str(data.get("target", "")),
                kind=str(data.get("kind", "")),
                locator=str(data.get("locator", "")),
                message=str(data.get("message", "")),
                routed_at=str(data.get("routed_at", "")),
            )
        except (TypeError, ValueError):
            return None


def _failure_file_path(archive_root: Path) -> Path:
    """Return the canonical JSONL path under ``archive_root``."""
    return Path(archive_root) / _STATE_DIRNAME / _FAILURE_FILE_NAME


def _truncate_message(message: str) -> str:
    if len(message) <= MAX_MESSAGE_LEN:
        return message
    return message[: MAX_MESSAGE_LEN - 3] + "..."


def route_failure_sample(
    sample: FailureSample,
    *,
    operation_id: str,
    archive_root: Path,
    target: str | None = None,
    now: datetime | None = None,
) -> MaintenanceFailureRecord:
    """Persist one :class:`FailureSample` to the daemon-visible surface.

    Parameters
    ----------
    sample:
        The per-record failure from a replay or repair function.
    operation_id:
        Identity of the originating :class:`BackfillOperation` so the
        daemon surface can attribute the failure back to a specific
        replay run.
    archive_root:
        Archive root under which the JSONL file lives. Pass the value
        from :func:`Config.archive_root` (or the test fixture) so the
        write is scoped to the correct archive.
    target:
        Optional logical target name. When omitted, the function
        infers it from the locator prefix (``target:<name>:...``).
    now:
        Optional clock injection point used by tests; defaults to
        ``datetime.now(UTC)``.

    Returns
    -------
    MaintenanceFailureRecord
        The persisted record (also returned so callers can log it).
    """

    inferred_target = target
    if inferred_target is None:
        inferred_target = _infer_target_from_locator(sample.locator)

    routed_at = (now or datetime.now(UTC)).isoformat()
    record = MaintenanceFailureRecord(
        operation_id=operation_id,
        target=inferred_target,
        kind=sample.kind,
        locator=_redact_locator(sample.locator),
        message=_redact_paths(_truncate_message(sample.message)),
        routed_at=routed_at,
    )

    path = _failure_file_path(archive_root)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(dumps(record.to_dict()))
            handle.write("\n")
    except OSError as exc:
        # Routing must never abort the replay loop. Log and move on;
        # the failure stays in the in-memory ``BoundedFailureSamples``
        # envelope on the returned ``BackfillOperation``.
        logger.warning(
            "maintenance_failure_route_failed",
            operation_id=operation_id,
            target=inferred_target,
            error=str(exc),
        )
    return record


def _infer_target_from_locator(locator: str) -> str:
    """Pull the target token out of a ``target:<name>...`` locator."""
    if not locator.startswith("target:"):
        return "unknown"
    suffix = locator[len("target:") :]
    head = suffix.split(":", 1)[0]
    return head or "unknown"


def read_maintenance_failures(
    archive_root: Path,
    *,
    limit: int = MAINTENANCE_FAILURE_SAMPLE_LIMIT,
) -> list[MaintenanceFailureRecord]:
    """Read the routed failures JSONL and return the most recent ``limit``.

    Lines that fail to parse are skipped with a warning; a partial
    write at the tail of the file (process killed mid-line) does not
    prevent earlier records from being read.

    Returns an empty list when the file does not exist, which is the
    expected steady state for an archive with no maintenance
    failures.
    """

    path = _failure_file_path(archive_root)
    if not path.exists():
        return []
    records: list[MaintenanceFailureRecord] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    payload = loads(stripped)
                except (ValueError, TypeError):
                    logger.warning(
                        "maintenance_failure_unparseable_line",
                        path=str(path),
                    )
                    continue
                rec = MaintenanceFailureRecord.from_dict(payload)
                if rec is not None:
                    records.append(rec)
    except OSError as exc:
        logger.warning(
            "maintenance_failure_read_failed",
            path=str(path),
            error=str(exc),
        )
        return []
    if len(records) <= limit:
        return records
    return records[-limit:]


def resolve_maintenance_failures(
    archive_root: Path,
    *,
    target: str,
    kinds: tuple[str, ...] = (),
) -> int:
    """Remove routed failure records that a later successful run supersedes.

    Routed failures are append-only evidence of failed attempts, but the
    daemon health surface is an active backlog signal. Once a target has
    been successfully replayed, older failures for that target should no
    longer keep health red. ``kinds`` narrows resolution when only a
    class of failure was proven stale, such as a dry-run proving an
    ``UnsupportedReplayTargetError`` target is now wired.
    """

    path = _failure_file_path(archive_root)
    if not path.exists():
        return 0
    kept: list[MaintenanceFailureRecord] = []
    removed = 0
    kind_filter = set(kinds)
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    payload = loads(stripped)
                except (ValueError, TypeError):
                    logger.warning(
                        "maintenance_failure_unparseable_line",
                        path=str(path),
                    )
                    continue
                rec = MaintenanceFailureRecord.from_dict(payload)
                if rec is None:
                    continue
                if rec.target == target and (not kind_filter or rec.kind in kind_filter):
                    removed += 1
                    continue
                kept.append(rec)
        if removed == 0:
            return 0
        if kept:
            with path.open("w", encoding="utf-8") as handle:
                for rec in kept:
                    handle.write(dumps(rec.to_dict()))
                    handle.write("\n")
        else:
            path.unlink(missing_ok=True)
    except OSError as exc:
        logger.warning(
            "maintenance_failure_resolve_failed",
            path=str(path),
            target=target,
            error=str(exc),
        )
        return 0
    return removed


def count_maintenance_failures(archive_root: Path) -> int:
    """Return the total number of routed maintenance failures on disk.

    The count is used by :func:`polylogue.daemon.health._check_raw_failures_medium`
    to escalate raw-failure alerts; cheap to call even when the file
    is moderately large because we only count non-empty lines.
    """

    path = _failure_file_path(archive_root)
    if not path.exists():
        return 0
    try:
        with path.open("r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
    except OSError as exc:
        logger.warning(
            "maintenance_failure_count_failed",
            path=str(path),
            error=str(exc),
        )
        return 0


def clear_maintenance_failures(archive_root: Path) -> None:
    """Remove the routed failures JSONL file.

    Used by tests and by operators who have addressed a backlog of
    failed maintenance attempts and want the daemon raw-failure surface
    to reset its maintenance bucket.
    """

    path = _failure_file_path(archive_root)
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        logger.warning(
            "maintenance_failure_clear_failed",
            path=str(path),
            error=str(exc),
        )


__all__ = [
    "MAINTENANCE_FAILURE_SAMPLE_LIMIT",
    "MAX_MESSAGE_LEN",
    "MaintenanceFailureRecord",
    "clear_maintenance_failures",
    "count_maintenance_failures",
    "read_maintenance_failures",
    "resolve_maintenance_failures",
    "route_failure_sample",
]
