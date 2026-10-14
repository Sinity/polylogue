"""Actual HTTP exchange custody and typed native receipt adaptation."""

from __future__ import annotations

import asyncio
import json
import shutil
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from polylogue.sinex.models import PublicationAcquisition, PublicationOriginal, ReceiptState
from polylogue.sinex.network_transport import SinexHTTPTransport, SinexPublicationPendingError, SinexRPCError
from polylogue.storage.blob_store import BlobStore


@pytest.mark.parametrize(
    "fault",
    [
        "disconnect",
        "timeout",
        "native_io",
        "native_nats",
        "native_blob",
        "native_service",
        "native_processing",
        "refused",
    ],
)
def test_lost_http_reply_reconciles_receipt_after_client_files_retire(tmp_path: Path, fault: str) -> None:
    cas = BlobStore(tmp_path / "source-cas")
    digest, size = cas.write_from_bytes(b"neutral acquired provider bytes")
    original = PublicationOriginal(
        "provider_native",
        "polylogue.raw:neutral",
        digest,
        size,
        "application/octet-stream",
        PublicationAcquisition("full_snapshot"),
    )
    calls: list[dict[str, Any]] = []
    retained = tmp_path / "native-cas"
    retained.mkdir()
    status_served = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args: object) -> None:
            pass

        def do_POST(self) -> None:
            assert self.headers["Authorization"] == "Bearer neutral-test-token"
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append(request)
            if request["method"] == "polylogue.revision.publish":
                params = request["params"]
                for i, descriptor in enumerate([params["manifest"], *params["segments"], *params["originals"]]):
                    shutil.copyfile(descriptor["file_path"], retained / str(i))
                # Actual client, synthetic custody: loss and timeout stay unknown
                # until the same-request status returns the stored outcome.
                if fault in {
                    "native_io",
                    "native_nats",
                    "native_blob",
                    "native_service",
                    "native_processing",
                    "refused",
                }:
                    code = {
                        "native_io": -32821,
                        "native_nats": -32870,
                        "native_blob": -32860,
                        "native_service": -32820,
                        "native_processing": -32850,
                        "refused": -32800,
                    }[fault]
                    data = json.dumps(
                        {
                            "jsonrpc": "2.0",
                            "id": request["id"],
                            "error": {
                                "code": code,
                                "message": "neutral native outcome",
                                "data": {"kind_name": "processing"},
                            },
                        }
                    ).encode()
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return
                if fault == "timeout":
                    assert status_served.wait(5)
                self.close_connection = True
                return
            assert request["method"] == "polylogue.revision.status"
            status_served.set()
            response = {
                "jsonrpc": "2.0",
                "id": request["id"],
                "result": {
                    "request_id": request["params"]["request_id"],
                    "state": "persisted_confirmed",
                    "detail": "neutral",
                },
            }
            data = json.dumps(response).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        transport = SinexHTTPTransport(
            f"http://127.0.0.1:{server.server_port}/rpc",
            "neutral-test-token",
            cas,
            tmp_path / "files",
            exchange_timeout_s=0.2,
        )
        receipt = asyncio.run(
            transport.publish_revision(
                request_id="neutral-request",
                manifest_bytes=b"{}\n",
                segment_bytes={"head.ndjson": b"neutral normalized bytes\n"},
                originals=(original,),
            )
        )
        assert receipt.request_id == "neutral-request"
        if fault == "refused":
            assert receipt.state is ReceiptState.REJECTED
            assert len(calls) == 1
        else:
            assert receipt.state is ReceiptState.PERSISTED_CONFIRMED
            assert [call["params"]["request_id"] for call in calls] == ["neutral-request", "neutral-request"]
        assert (retained / "2").read_bytes() == b"neutral acquired provider bytes"
        assert not tuple((tmp_path / "files").iterdir())
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_pending_native_custody_does_not_fabricate_a_puback() -> None:
    with pytest.raises(SinexPublicationPendingError):
        SinexHTTPTransport._receipt("neutral", {"request_id": "neutral", "state": None, "phase": "materials_ready"})


def test_authentication_failure_and_foreign_receipt_are_refusals(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="identity_mismatch"):
        SinexHTTPTransport._receipt("neutral", {"request_id": "foreign", "state": "persisted_confirmed"})
    with pytest.raises(ValueError, match="requires_tls"):
        SinexHTTPTransport("http://remote.invalid/rpc", "neutral-test-token", BlobStore(tmp_path), tmp_path)
    assert not SinexRPCError(-32002, retryable=False).retryable


def test_cancelled_await_keeps_files_until_blocking_http_exchange_finishes(tmp_path: Path) -> None:
    cas = BlobStore(tmp_path / "cas")
    digest, size = cas.write_from_bytes(b"neutral acquired bytes")
    original = PublicationOriginal("provider_native", "polylogue.raw:neutral", digest, size, "application/octet-stream")
    acquired = threading.Event()
    release = threading.Event()
    paths: list[Path] = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args: object) -> None:
            pass

        def do_POST(self) -> None:
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            params = request["params"]
            paths.extend(
                Path(item["file_path"]) for item in [params["manifest"], *params["segments"], *params["originals"]]
            )
            acquired.set()
            assert release.wait(5)
            assert all(path.is_file() for path in paths)
            data = json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": request["id"],
                    "result": {"request_id": request["id"], "state": "raw_accepted"},
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    transport = SinexHTTPTransport(f"http://127.0.0.1:{server.server_port}/rpc", "neutral", cas, tmp_path / "files")

    async def cancel_waiter() -> None:
        task = asyncio.create_task(
            transport.publish_revision(
                request_id="neutral",
                manifest_bytes=b"{}\n",
                segment_bytes={"head.ndjson": b"normalized\n"},
                originals=(original,),
            )
        )
        assert await asyncio.to_thread(acquired.wait, 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert all(path.is_file() for path in paths)
        release.set()

    try:
        asyncio.run(cancel_waiter())  # executor shutdown waits for physical completion
        assert paths and not any(path.exists() for path in paths)
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        thread.join()
