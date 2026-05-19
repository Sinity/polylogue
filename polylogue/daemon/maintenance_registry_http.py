"""Daemon HTTP handlers for the persistent maintenance operation registry (#1197).

Lives outside :mod:`polylogue.daemon.http` so the main HTTP module stays
within its file-size budget. The registry endpoints are pure read
surfaces — they only consult the on-disk replay state directory under
``<archive_root>/.maintenance-state/`` — so they do not need access to
the daemon's runtime state.

Two endpoints:

* ``GET /api/maintenance/status/<op_id>`` — one persisted operation
  snapshot, wrapped in the shared
  :class:`~polylogue.maintenance.envelope.MaintenanceOperationEnvelope`
  plus state-file metadata (``updated_at``, ``state_path``);
* ``GET /api/maintenance/operations`` — every persisted snapshot under
  a single ``{"operations": [...], "total": N}`` envelope.

The handlers receive the live :class:`DaemonAPIHandler` instance and
use its :meth:`_send_json` / :meth:`_send_error` primitives so the
response shape and error semantics match the rest of the daemon API.
"""

from __future__ import annotations

from http import HTTPStatus
from typing import TYPE_CHECKING

from polylogue.config import Config
from polylogue.maintenance.envelope import envelope_from_operation
from polylogue.maintenance.registry import MaintenanceOperationRegistry
from polylogue.paths import archive_root, render_root

if TYPE_CHECKING:
    from polylogue.daemon.http import DaemonAPIHandler


def _build_config() -> Config:
    return Config(
        archive_root=archive_root(),
        render_root=render_root(),
        sources=[],
    )


def handle_status(handler: DaemonAPIHandler, operation_id: str) -> None:
    """GET /api/maintenance/status/<op_id> — one persisted operation snapshot."""
    registry = MaintenanceOperationRegistry(config=_build_config())
    record = registry.get_operation(operation_id)
    if record is None:
        handler._send_error(HTTPStatus.NOT_FOUND, "not_found")
        return
    envelope = envelope_from_operation(record.operation, origin="daemon", mode="execute")
    handler._send_json(
        HTTPStatus.OK,
        {
            "envelope": envelope.to_dict(),
            "updated_at": record.updated_at,
            "state_path": str(record.state_path),
        },
    )


def handle_operations(handler: DaemonAPIHandler) -> None:
    """GET /api/maintenance/operations — list every persisted operation snapshot."""
    registry = MaintenanceOperationRegistry(config=_build_config())
    records = registry.list_operations()
    handler._send_json(
        HTTPStatus.OK,
        {
            "operations": [
                {
                    "envelope": envelope_from_operation(r.operation, origin="daemon", mode="execute").to_dict(),
                    "updated_at": r.updated_at,
                    "state_path": str(r.state_path),
                }
                for r in records
            ],
            "total": len(records),
        },
    )


__all__ = ["handle_operations", "handle_status"]
