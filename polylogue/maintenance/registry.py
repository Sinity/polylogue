"""Persistent maintenance operation registry (issue #1197).

The :mod:`polylogue.maintenance.replay` executor checkpoints in-flight
backfill operations to JSON state files under
``<archive_root>/.maintenance-state/<op_id>.json``. Before #1197 those
files were write-only — there was no surface that could list them, tail
one specific operation, or prune the completed ones. The replay state
payload also only carried ``{cursor, results, repaired_count,
failure_count}``, which was not enough to reconstruct the full
:class:`~polylogue.maintenance.planner.BackfillOperation` snapshot the
shared envelope expects.

This module is the durable read surface for that state directory:

* :meth:`MaintenanceOperationRegistry.list_operations` returns one
  :class:`OperationRecord` per persisted state file, sorted newest-first
  by ``updated_at``;
* :meth:`MaintenanceOperationRegistry.get_operation` returns one record
  by id (or ``None`` when the file does not exist);
* :meth:`MaintenanceOperationRegistry.prune_completed` removes
  state files for operations whose ``status == "completed"`` and whose
  ``updated_at`` is older than the configured TTL. Failed operations
  stay on disk until manually cleared (operators want them for
  diagnosis).

The state file payload is upgraded to carry the full
``BackfillOperation.to_dict()`` under a ``"operation"`` key. The legacy
top-level fields (``cursor``, ``targets``, ``repaired_count``,
``failure_count``) are still written so old readers and the existing
resume code path keep working unchanged.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Final

from polylogue.config import Config
from polylogue.core.json import JSONDocument, json_document, loads
from polylogue.logging import get_logger
from polylogue.maintenance.planner import BackfillOperation, BackfillStatus

logger = get_logger(__name__)

#: Default TTL for completed-successful operations: 7 days.
DEFAULT_COMPLETED_TTL: Final[timedelta] = timedelta(days=7)

#: Subdirectory under :attr:`Config.archive_root` used for replay state
#: files. Mirrors :data:`polylogue.maintenance.replay._STATE_DIRNAME`;
#: kept private there so the path stays an implementation detail of the
#: replay module, and duplicated here so the registry does not need to
#: import the replay module (avoids an import cycle).
_STATE_DIRNAME: Final[str] = ".maintenance-state"


def _state_dir(config: Config) -> Path:
    """Return the on-disk state directory under ``archive_root``."""
    return Path(config.archive_root) / _STATE_DIRNAME


@dataclass(frozen=True)
class OperationRecord:
    """One persisted operation snapshot, projected for the read surface.

    Carries the fully reconstructed :class:`BackfillOperation` along
    with the file-level timestamps the operator actually wants when
    listing or pruning: ``updated_at`` (when the executor last
    checkpointed) and ``state_path`` (so a user can locate the raw
    file).
    """

    operation: BackfillOperation
    updated_at: str
    state_path: Path

    @property
    def operation_id(self) -> str:
        return self.operation.operation_id

    @property
    def status(self) -> BackfillStatus:
        return self.operation.status

    def to_dict(self) -> JSONDocument:
        return json_document(
            {
                "operation": self.operation.to_dict(),
                "updated_at": self.updated_at,
                "state_path": str(self.state_path),
            }
        )


def _parse_updated_at(value: object) -> datetime | None:
    """Parse the ``updated_at`` ISO timestamp recorded in the state file."""
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _load_record(path: Path) -> OperationRecord | None:
    """Load one state file and project it into an :class:`OperationRecord`.

    Returns ``None`` for unparseable files. State directories are
    operator-visible so anything is possible there — partial writes
    after a SIGKILL, stray files, hand-edited JSON. Any failure is
    logged at warning level and skipped, never raised, so a single bad
    file never poisons the whole listing.
    """
    try:
        raw_text = path.read_text()
    except OSError as exc:
        logger.warning("maintenance_registry_read_failed", path=str(path), error=str(exc))
        return None
    try:
        raw = loads(raw_text)
    except ValueError as exc:
        logger.warning("maintenance_registry_parse_failed", path=str(path), error=str(exc))
        return None
    if not isinstance(raw, dict):
        logger.warning("maintenance_registry_payload_not_object", path=str(path))
        return None
    # ``loads`` returns the recursive ``JSONValue`` alias. Project it
    # onto ``dict[str, object]`` for the rehydration helpers — the
    # nested validators (``BackfillOperation.from_dict``,
    # ``_legacy_record_to_operation``) treat untrusted values
    # defensively, so the cast cannot widen the trust boundary.
    payload: dict[str, object] = {str(k): v for k, v in raw.items()}
    op_payload = payload.get("operation")
    if isinstance(op_payload, dict):
        operation = BackfillOperation.from_dict({str(k): v for k, v in op_payload.items()})
    else:
        # Legacy state file shape (pre-#1197): synthesize a minimal
        # BackfillOperation from the top-level fields so the registry
        # still surfaces in-flight pre-upgrade operations.
        operation = _legacy_record_to_operation(payload, path)
    updated_at_raw = payload.get("updated_at") or payload.get("started_at") or ""
    updated_at = updated_at_raw if isinstance(updated_at_raw, str) else ""
    return OperationRecord(operation=operation, updated_at=updated_at, state_path=path)


def _legacy_record_to_operation(raw: dict[str, object], path: Path) -> BackfillOperation:
    """Best-effort projection of a pre-#1197 state payload onto :class:`BackfillOperation`."""
    from polylogue.maintenance.planner import BackfillKind, MaintenanceScope

    targets_raw = raw.get("targets") or ()
    if not isinstance(targets_raw, (list, tuple)):
        targets_raw = ()
    targets = tuple(str(t) for t in targets_raw)
    op_id_raw = raw.get("operation_id")
    op_id = op_id_raw if isinstance(op_id_raw, str) else path.stem
    cursor_raw = raw.get("cursor")
    cursor = cursor_raw if isinstance(cursor_raw, str) else None
    started_raw = raw.get("started_at")
    started = started_raw if isinstance(started_raw, str) else None
    return BackfillOperation(
        operation_id=op_id,
        kind=BackfillKind.DERIVED_REBUILD,
        targets=targets,
        status=BackfillStatus.RUNNING,
        started_at=started,
        resume_cursor=cursor,
        scope=MaintenanceScope(targets=targets),
    )


@dataclass(frozen=True)
class MaintenanceOperationRegistry:
    """Read-side registry over the on-disk replay state directory.

    The registry is intentionally stateless and re-reads the directory
    on every call. There is no in-process cache — concurrent
    daemon/CLI/MCP readers must agree with the filesystem, not with one
    another, and the file count is bounded by the operator's
    in-flight + recent operations (typically << 100), so a fresh
    listdir per call is cheap.
    """

    config: Config

    @property
    def state_dir(self) -> Path:
        return _state_dir(self.config)

    def _state_file_paths(self) -> list[Path]:
        directory = self.state_dir
        if not directory.exists():
            return []
        out: list[Path] = []
        try:
            for entry in directory.iterdir():
                if entry.is_file() and entry.suffix == ".json":
                    out.append(entry)
        except OSError as exc:
            logger.warning(
                "maintenance_registry_iterdir_failed",
                directory=str(directory),
                error=str(exc),
            )
            return []
        return out

    def list_operations(self) -> tuple[OperationRecord, ...]:
        """Return every persisted operation snapshot, newest first."""
        records: list[OperationRecord] = []
        for path in self._state_file_paths():
            record = _load_record(path)
            if record is not None:
                records.append(record)
        records.sort(key=lambda r: r.updated_at, reverse=True)
        return tuple(records)

    def get_operation(self, operation_id: str) -> OperationRecord | None:
        """Return the snapshot for one operation, or ``None`` when absent."""
        path = self.state_dir / f"{operation_id}.json"
        if not path.exists():
            return None
        return _load_record(path)

    def prune_completed(
        self,
        *,
        older_than: timedelta = DEFAULT_COMPLETED_TTL,
        now: datetime | None = None,
    ) -> tuple[str, ...]:
        """Remove state files for completed-successful operations.

        Failed operations are deliberately retained — operators need
        them for diagnostics. Operations that are still running, have
        no ``updated_at`` timestamp, or whose ``updated_at`` is younger
        than the TTL are skipped.

        Returns the operation ids that were pruned (for logging /
        tests).
        """
        reference = now if now is not None else datetime.now(timezone.utc)
        cutoff = reference - older_than
        pruned: list[str] = []
        for path in self._state_file_paths():
            record = _load_record(path)
            if record is None:
                continue
            if record.status is not BackfillStatus.COMPLETED:
                continue
            updated_dt = _parse_updated_at(record.updated_at)
            if updated_dt is None:
                continue
            # Normalize naive datetimes to UTC so the comparison is
            # always between timezone-aware datetimes.
            if updated_dt.tzinfo is None:
                updated_dt = updated_dt.replace(tzinfo=timezone.utc)
            if updated_dt > cutoff:
                continue
            try:
                os.unlink(path)
            except OSError as exc:
                logger.warning(
                    "maintenance_registry_prune_failed",
                    operation_id=record.operation_id,
                    error=str(exc),
                )
                continue
            pruned.append(record.operation_id)
        return tuple(pruned)


__all__ = [
    "DEFAULT_COMPLETED_TTL",
    "MaintenanceOperationRegistry",
    "OperationRecord",
]
