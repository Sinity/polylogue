from __future__ import annotations

import socket
import threading
from pathlib import Path
from unittest.mock import patch

import pytest

from polylogue.archive.query.execution_control import QueryCancelledError
from polylogue.daemon.http import DaemonAPIHandler
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore


@pytest.mark.parametrize("route", ["query-units", "messages"])
def test_peer_eof_interrupts_actual_native_read(tmp_path: Path, route: str) -> None:
    with ArchiveStore(tmp_path):
        pass
    server_socket, peer = socket.socketpair()
    handler = DaemonAPIHandler.__new__(DaemonAPIHandler)
    handler.connection = server_socket
    handler.path = "/api/query-units?expression=messages"
    entered = threading.Event()
    failures = []
    contexts = []

    from polylogue.archive.query.transaction import QueryTransaction

    original_run = QueryTransaction.run_sync

    def track_context(self: QueryTransaction, work: object) -> object:
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
                DaemonAPIHandler._handle_query_units.__wrapped__(handler, {"expression": ["messages"]})
            else:
                handler._do_archive_get_messages(tmp_path, "neutral", limit=1, offset=0)
        except BaseException as exc:
            failures.append(exc)

    try:
        with (
            patch("polylogue.daemon.http._web_reader_archive_root", return_value=tmp_path),
            patch("polylogue.operations.daemon_reads._query_units_payload", side_effect=actual_sql),
        ):
            worker = threading.Thread(target=run)
            worker.start()
            assert entered.wait(10)
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
