from __future__ import annotations

import inspect
import socket
import threading
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar, cast
from unittest.mock import patch

import pytest

from polylogue.archive.query.execution_control import QueryCancelledError, QueryExecutionContext
from polylogue.daemon.http import DaemonAPIHandler, DaemonAPIHTTPServer
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

T = TypeVar("T")


@pytest.mark.parametrize("route", ["query-units", "messages"])
def test_peer_eof_interrupts_actual_native_read(tmp_path: Path, route: str) -> None:
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    initialize_active_archive_root(tmp_path)
    server_socket, peer = socket.socketpair()
    handler = DaemonAPIHandler.__new__(DaemonAPIHandler)
    handler.connection = server_socket
    handler.path = "/api/query-units?expression=messages"
    entered = threading.Event()
    failures: list[BaseException] = []
    contexts: list[QueryExecutionContext] = []

    from polylogue.archive.query.transaction import QueryTransaction

    original_run = QueryTransaction.run_sync

    def track_context(self: QueryTransaction, work: Callable[[ArchiveStore], T]) -> T:
        contexts.append(self.context)
        return original_run(self, work)

    def actual_sql(
        _params: object, *, archive: ArchiveStore, execution_context: object = None, **_kwargs: object
    ) -> object:
        entered.set()
        return archive._conn.execute(
            "WITH RECURSIVE n(x) AS (VALUES(0) UNION ALL SELECT x+1 FROM n WHERE x<1000000000) SELECT sum(x) FROM n"
        ).fetchone()

    def run() -> None:
        try:
            if route == "query-units":
                handle_query_units = cast(
                    Callable[[DaemonAPIHandler, dict[str, list[str]]], None],
                    inspect.unwrap(DaemonAPIHandler._handle_query_units),
                )
                handle_query_units(handler, {"expression": ["messages where role:user"]})
            else:
                handler._do_archive_get_messages(tmp_path, "neutral", limit=1, offset=0)
        except BaseException as exc:
            failures.append(exc)

    try:
        with (
            patch("polylogue.daemon.http._web_reader_archive_root", return_value=tmp_path),
            patch("polylogue.operations.daemon_reads._query_units_payload", side_effect=actual_sql),
            patch("polylogue.daemon.http.execute_http_session_messages", side_effect=actual_sql),
            patch.object(QueryTransaction, "run_sync", track_context),
        ):
            worker = threading.Thread(target=run)
            worker.start()
            assert entered.wait(10), failures
            peer.close()
            worker.join(10)
            assert not worker.is_alive()
        assert len(failures) == 1 and isinstance(failures[0], QueryCancelledError)
        assert contexts[0].cancelled
        assert contexts[0].receipt.cleanup_complete
        assert contexts[0].receipt.state == "cancelled"
    finally:
        peer.close()
        server_socket.close()


def test_scheduled_http_read_keeps_peer_cancellation_in_nested_query(tmp_path: Path) -> None:
    from polylogue.archive.query.transaction import QueryTransaction, QueryTransactionRequest
    from polylogue.core.compute import BoundedComputeAdapter, DaemonOperationCancelled
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    initialize_active_archive_root(tmp_path)
    entered = threading.Event()
    failures: list[BaseException] = []
    contexts: list[QueryExecutionContext] = []
    server_socket, peer = socket.socketpair()
    kernel = BoundedComputeAdapter(max_workers=2, queue_units=4)
    handler = DaemonAPIHandler.__new__(DaemonAPIHandler)
    server = object.__new__(DaemonAPIHTTPServer)
    server.execution_kernel = kernel
    handler.server = server
    handler.connection = server_socket
    handler.path = "/api/sessions"

    async def read(_self: DaemonAPIHandler, _handler: Callable[[ArchiveStore], object]) -> object:
        transaction = QueryTransaction(
            tmp_path, QueryTransactionRequest(operation="http.archive.read", arguments={}, page_size=1)
        )
        contexts.append(transaction.context)

        def sql(archive: ArchiveStore) -> object:
            entered.set()
            return archive._conn.execute(
                "WITH RECURSIVE n(x) AS (VALUES(0) UNION ALL SELECT x+1 FROM n WHERE x<1000000000) SELECT sum(x) FROM n"
            ).fetchone()

        return await transaction.run(sql)

    def run() -> None:
        try:
            handler._sync_run(lambda _archive: None)
        except BaseException as exc:
            failures.append(exc)

    try:
        with patch.object(DaemonAPIHandler, "_run_archive_query", read):
            worker = threading.Thread(target=run)
            worker.start()
            assert entered.wait(10), failures
            peer.close()
            worker.join(10)
            assert not worker.is_alive()
        assert len(failures) == 1 and isinstance(failures[0], (QueryCancelledError, DaemonOperationCancelled))
        assert contexts[0].cancelled
        assert contexts[0].receipt.cleanup_complete
        assert kernel.snapshot().active_units == 0
    finally:
        peer.close()
        server_socket.close()
        kernel.shutdown(wait=True)
