"""Diagnostic capture for unhandled machine-operation handler exceptions."""

from __future__ import annotations

import queue
from pathlib import Path

import pytest

from polylogue.daemon_client import DaemonMutationIndeterminateError
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
