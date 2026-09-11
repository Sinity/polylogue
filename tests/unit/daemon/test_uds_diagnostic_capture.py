"""Diagnostic capture for unhandled machine-operation handler exceptions."""

from __future__ import annotations

import queue
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from polylogue.daemon_client import DaemonMutationIndeterminateError
from polylogue.operations.audit import AuditContinuityPendingError, AuditRepository
from tests.infra.daemon_operations import running_daemon_operations


def test_machine_operation_fixture_captures_unhandled_handler_exception(tmp_path: Path) -> None:
    """The capture sink records the real socketserver error path, not a simulated report."""

    sink: queue.SimpleQueue[str] = queue.SimpleQueue()

    class _SentinelRuntime:
        def call(self, *_args: object, **_kwargs: object) -> dict[str, object]:
            raise RuntimeError("capture sentinel")

    with running_daemon_operations(tmp_path / "archive", server_error_sink=sink) as stack:
        object.__setattr__(stack.server, "operation_runtime", _SentinelRuntime())
        with pytest.raises(DaemonMutationIndeterminateError):
            stack.client.operation(
                "mutation.session.tag",
                {"session_ids": ["codex-session:sentinel"], "tags": ["sentinel"]},
                archive_root=str(stack.archive_root),
            )

    assert "RuntimeError: capture sentinel" in sink.get(timeout=1)


def test_machine_operation_await_reports_unavailable_audit_without_dropping_transport(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An unavailable settled audit read stays a typed indeterminate lifecycle state."""

    sink: queue.SimpleQueue[str] = queue.SimpleQueue()

    @contextmanager
    def unavailable_settled_read(_audit: AuditRepository) -> Iterator[dict[str, int]]:
        raise AuditContinuityPendingError("audit reader unavailable")
        yield {}

    monkeypatch.setattr(AuditRepository, "settled_machine_read", unavailable_settled_read)
    with running_daemon_operations(tmp_path / "archive", server_error_sink=sink) as stack:
        envelope = stack.client.operation(
            "operation.await",
            {"request_id": "unavailable-audit", "after_sequence": 0, "timeout_ms": 1},
            archive_root=str(stack.archive_root),
        )

    assert envelope is not None
    result = envelope["result"]
    assert isinstance(result, dict)
    assert result == {"outcome": "indeterminate", "sequence": 0}
    assert sink.empty()
