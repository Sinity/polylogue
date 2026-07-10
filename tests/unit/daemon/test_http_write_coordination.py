"""Mutation-sensitive route proofs for daemon HTTP writer coordination."""

from __future__ import annotations

import contextlib
from collections.abc import Awaitable, Callable, Iterator
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from polylogue.daemon.http import DaemonAPIHandler, DaemonAPIHTTPServer


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


def _handler(path: list[str], timeline: list[str]) -> DaemonAPIHandler:
    handler = object.__new__(DaemonAPIHandler)
    handler.server = SimpleNamespace(write_bridge=_RecordingBridge(timeline))  # type: ignore[assignment]
    handler._parse_path = lambda: (path, {})  # type: ignore[method-assign]
    handler._check_host_admission = lambda: True  # type: ignore[method-assign]
    handler._check_auth = lambda: True  # type: ignore[method-assign]
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


@pytest.mark.parametrize("signal", ["traces", "metrics", "logs"])
def test_otlp_persistence_route_holds_gate_around_receiver(signal: str) -> None:
    timeline: list[str] = []
    handler = _handler(["v1", signal], timeline)
    handler._handle_otlp_post = lambda _path: timeline.append("body")  # type: ignore[method-assign]
    with patch(
        "polylogue.config.load_polylogue_config",
        return_value=SimpleNamespace(observability_enabled=True),
    ):
        handler._do_post_impl()

    assert timeline == [f"enter:http.otlp.{signal}", "body", f"exit:http.otlp.{signal}"]


def test_standalone_http_server_owns_and_idempotently_closes_writer_runtime() -> None:
    server = DaemonAPIHTTPServer(("127.0.0.1", 0), DaemonAPIHandler)
    runtime = server._owned_write_runtime
    assert runtime is not None
    assert runtime.thread.is_alive()

    server.server_close()
    server.server_close()

    assert not runtime.thread.is_alive()


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
