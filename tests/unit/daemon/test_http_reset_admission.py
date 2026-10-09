"""Actual HTTP reset and resident writer admission regression."""

from __future__ import annotations

import asyncio
import json
import threading
from http import HTTPStatus
from pathlib import Path
from typing import Any

import pytest

from polylogue.daemon.http import DaemonAPIHandler, DaemonAPIHTTPServer
from tests.infra.daemon_operations import running_daemon_operations
from tests.unit.daemon.test_http_write_coordination import _seed_delete_authority_archive


def test_http_reset_completes_through_shared_daemon_writer(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A real authenticated reset must not hold the writer while resubmitting.

    If the facade resubmission queues behind this reset's own admission,
    cancel that exact synthetic queued execution to settle the probe. This
    observes the actual cycle without waiting for a product timeout.
    """
    from http.client import HTTPConnection

    root = tmp_path / "archive"
    (session_id,) = _seed_delete_authority_archive(root, 1)
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(root))
    with running_daemon_operations(root) as stack:
        token = "synthetic-reset-token"
        stack.server.auth_token = token
        stack.client.auth_token = token
        monkeypatch.setattr("polylogue.daemon.socket_path.daemon_socket_path", lambda _root: stack.socket_path)
        monkeypatch.setattr("polylogue.daemon.api_auth.resolve_api_auth_token", lambda *_args, **_kwargs: token)
        preview = stack.client.operation_to_completion(
            "mutation.session.delete.preview", {"session_ids": [session_id]}, archive_root=str(root)
        )
        assert preview is not None and preview["outcome"] == "completed", preview
        preview_ref = preview["result"]["preview_ref"]
        coordinator = stack.write_coordinator
        original_execute = coordinator._execute
        cycles: list[tuple[str, str | None, bool]] = []

        async def observe_execution(request: Any, operation: Any, on_admit: Any) -> Any:
            if (
                request.actor == "operation.mutation.facade.delete_session"
                and coordinator._active_actor == "http.reset"
            ):
                task = asyncio.current_task()
                assert task is not None

                def cancel_exact_waiter() -> None:
                    cycles.append((request.actor, coordinator._active_actor, request.acquired))
                    task.cancel()

                asyncio.get_running_loop().call_soon(cancel_exact_waiter)
            return await original_execute(request, operation, on_admit)

        monkeypatch.setattr(coordinator, "_execute", observe_execution)
        http_server = DaemonAPIHTTPServer(
            ("127.0.0.1", 0),
            DaemonAPIHandler,
            auth_token=token,
            archive_root=root,
            write_bridge=stack.write_bridge,
            execution_kernel=stack.execution_kernel,
        )
        serving = threading.Thread(target=http_server.serve_forever, daemon=True)
        serving.start()
        connection = HTTPConnection(*http_server.server_address)
        try:
            body = json.dumps({"scope": "session", "session_id": session_id, "preview_ref": preview_ref})
            connection.request(
                "POST",
                "/api/reset",
                body=body,
                headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            )
            response = connection.getresponse()
            payload = json.loads(response.read())
            assert cycles == [], f"reset admission cycle: {cycles}; HTTP {response.status}: {payload}"
            assert response.status == HTTPStatus.OK, payload
            assert payload["status"] == "deleted", payload
            assert not stack.session_exists(session_id)
            # The original coordinator must admit a subsequent operation.
            assert stack.write_bridge.run_sync("test.reset.after", lambda: "settled") == "settled"
        finally:
            connection.close()
            http_server.shutdown()
            http_server.server_close()
            serving.join()
