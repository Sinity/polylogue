"""Mutation-sensitive route proofs for daemon HTTP writer coordination."""

from __future__ import annotations

import contextlib
from collections.abc import Awaitable, Callable, Iterator
from http import HTTPStatus
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from polylogue.daemon.http import _REBUILD_INDEX_WRITE_TIMEOUT_S, DaemonAPIHandler, DaemonAPIHTTPServer
from polylogue.daemon.web_auth import WebCredentialScope


class _RecordingBridge:
    def __init__(self, timeline: list[str]) -> None:
        self.timeline = timeline

    @contextlib.contextmanager
    def hold(self, actor: str) -> Iterator[None]:
        self.timeline.append(f"enter:{actor}")
        try:
            yield
        finally:
            self.timeline.append(f"exit:{actor}")

    def run_sync(self, actor: str, function: Callable[..., object], *args: object) -> object:
        self.timeline.append(f"run_sync:{actor}")
        return function(*args)


def _handler(path: list[str], timeline: list[str]) -> DaemonAPIHandler:
    def allow_auth(required_scope: WebCredentialScope = "read", *, allow_web: bool = True) -> bool:
        del required_scope, allow_web
        return True

    def allow_host(*, credential_request: bool = False) -> bool:
        del credential_request
        return True

    handler = object.__new__(DaemonAPIHandler)
    handler.server = SimpleNamespace(write_bridge=_RecordingBridge(timeline))  # type: ignore[assignment]
    handler.path = "/" + "/".join(path)
    handler._parse_path = lambda: (path, {})  # type: ignore[method-assign]
    handler._check_host_admission = allow_host  # type: ignore[method-assign]
    handler._check_auth = allow_auth  # type: ignore[method-assign]
    handler._check_cross_origin = lambda: True  # type: ignore[method-assign]
    handler._send_error = lambda *_args: timeline.append("error")  # type: ignore[method-assign]
    return handler


@pytest.mark.parametrize(
    ("path", "handler_name", "actor"),
    [
        (["api", "reset"], "_handle_reset", "http.reset"),
        (["api", "ingest"], "_handle_ingest", "http.ingest"),
        (["api", "maintenance", "run"], "_handle_maintenance_run", "http.maintenance.run"),
    ],
)
def test_authenticated_write_route_holds_gate_around_handler(path: list[str], handler_name: str, actor: str) -> None:
    timeline: list[str] = []
    handler = _handler(path, timeline)
    setattr(handler, handler_name, lambda: timeline.append("body"))

    handler._do_post_impl()

    assert timeline == [f"enter:{actor}", "body", f"exit:{actor}"]


def test_user_post_and_delete_hold_named_gates_around_dispatch() -> None:
    post_timeline: list[str] = []
    post_handler = _handler(["api", "user", "marks"], post_timeline)

    def dispatch_post(*_args: object) -> bool:
        post_timeline.append("body")
        return True

    with patch("polylogue.daemon.http.user_state_http.dispatch_post", side_effect=dispatch_post):
        post_handler._do_post_impl()
    assert post_timeline == ["enter:http.user.marks.post", "body", "exit:http.user.marks.post"]

    delete_timeline: list[str] = []
    delete_handler = _handler(["api", "user", "annotations", "ann-1"], delete_timeline)

    def dispatch_delete(*_args: object) -> bool:
        delete_timeline.append("body")
        return True

    with patch("polylogue.daemon.http.user_state_http.dispatch_delete", side_effect=dispatch_delete):
        delete_handler._do_delete_impl()
    assert delete_timeline == [
        "enter:http.user.annotations.delete",
        "body",
        "exit:http.user.annotations.delete",
    ]


class _RecordingRebuildBridge(_RecordingBridge):
    """Adds ``run_sync_with_timeout`` so real ``_handle_rebuild_index`` can run.

    polylogue-ogn1: rebuild-index uses ``run_sync_with_timeout`` (not
    ``run_sync``) so a long rebuild pass isn't killed by the bridge's much
    shorter default request timeout -- see ``DaemonAPIHandler._handle_rebuild_index``
    and ``DaemonWriteThreadBridge.run_sync_with_timeout``.
    """

    def run_sync_with_timeout(
        self, actor: str, timeout: float | None, function: Callable[..., object], *args: object
    ) -> object:
        self.timeline.append(f"run_sync_with_timeout:{actor}:{timeout}")
        return function(*args)


def test_rebuild_index_route_uses_the_bridge_run_sync_with_timeout_writer_path(monkeypatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Drive the request through the real production dispatch, not a stand-in.

    The previous version of this test replaced ``_handle_rebuild_index``
    wholesale with a body that itself called ``bridge.run_sync`` -- it only
    proved the test's own stand-in called ``run_sync``, never that the real
    production handler does anything of the kind (polylogue-ogn1 finding
    #10). This exercises the real ``_do_post_impl`` route dispatch and the
    real ``_handle_rebuild_index`` implementation end to end, with only the
    typed rebuild service itself stubbed out.
    """
    import json
    from io import BytesIO

    from polylogue.maintenance.rebuild_index import RebuildIndexReceipt

    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(tmp_path))
    timeline: list[str] = []
    handler = _handler(["api", "maintenance", "rebuild-index"], timeline)
    handler.server.write_bridge = _RecordingRebuildBridge(timeline)  # type: ignore[assignment]
    body = json.dumps({"promote": False, "raw_ids": ["raw-1"]}).encode("utf-8")
    handler.headers = {"Content-Length": str(len(body))}  # type: ignore[assignment]
    handler.rfile = BytesIO(body)

    receipt = RebuildIndexReceipt(
        archive_root=str(tmp_path),
        raw_session_count=1,
        selected_raw_count=1,
        skipped_by_blob_limit_count=0,
        status="replayed",
        materialized=True,
        materialization={},
        generation={"generation_id": "candidate-1", "active": False},
        readiness={"checked": True, "blocked_surface_count": 0},
        replay={"classified_full_count": 1, "replayed_logical_source_count": 1, "quarantined_raw_count": 0},
    )
    with patch("polylogue.maintenance.rebuild_index.rebuild_index_from_source_sync", return_value=receipt) as rebuild:
        with patch.object(handler, "_send_json") as send_json:
            handler._do_post_impl()

    assert timeline == [f"run_sync_with_timeout:http.maintenance.rebuild-index:{_REBUILD_INDEX_WRITE_TIMEOUT_S}"]
    request = rebuild.call_args.args[0]
    assert request.raw_ids == ("raw-1",)
    assert request.promote is False
    assert send_json.call_args.args == (HTTPStatus.OK, receipt.to_dict())


def test_standalone_http_server_owns_and_idempotently_closes_writer_runtime() -> None:
    server = DaemonAPIHTTPServer(("127.0.0.1", 0), DaemonAPIHandler)
    runtime = server._owned_write_runtime
    assert runtime is not None
    assert runtime.thread.is_alive()

    server.server_close()
    server.server_close()

    assert not runtime.thread.is_alive()


def test_standalone_http_server_stops_loop_after_late_writer_drain() -> None:
    server = DaemonAPIHTTPServer(("127.0.0.1", 0), DaemonAPIHandler)
    runtime = server._owned_write_runtime
    assert runtime is not None
    assert runtime.coordinator is not None
    shutdown = AsyncMock(side_effect=[False, True])

    with patch.object(runtime.coordinator, "shutdown", shutdown):
        server.server_close()
        runtime.thread.join(timeout=1.0)

    assert not runtime.thread.is_alive()
    assert shutdown.await_count == 2


def test_coordinated_mutation_does_not_use_timeout_detaching_read_executor() -> None:
    handler = object.__new__(DaemonAPIHandler)
    handler._write_gate_depth = 1

    async def run_direct(operation: Callable[[object], Awaitable[object]]) -> object:
        return await operation(None)

    async def mutation(_polylogue: object) -> str:
        return "persisted"

    handler._run_archive_query = run_direct  # type: ignore[assignment]
    handler.server = SimpleNamespace(  # type: ignore[assignment]
        archive_query_admission=SimpleNamespace(acquire=lambda **_kwargs: (_ for _ in ()).throw(AssertionError())),
        archive_query_executor=SimpleNamespace(submit=lambda *_args: (_ for _ in ()).throw(AssertionError())),
    )

    assert handler._sync_run(mutation) == "persisted"
