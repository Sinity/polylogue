"""Annotation input framing seals the exact client bytes before dispatch."""

from __future__ import annotations

import hashlib
import inspect
import io
import json
import struct
from collections.abc import Callable
from http import HTTPStatus
from pathlib import Path
from typing import cast

import pytest

from polylogue.core.staged_body import BodyIncompleteError, StagedBody
from polylogue.operations.daemon_protocol import DaemonOperationRequest
from polylogue.operations.request_body_transport import UPLOAD_MEDIA_TYPE, read_operation_body


def _control(raw: bytes, **descriptor: object) -> dict[str, object]:
    return DaemonOperationRequest(
        "mutation.annotation.import_batch",
        {
            "input": {"sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw), **descriptor},
            "batch_id": "neutral",
            "schema_id": "seed.activity",
            "schema_version": 2,
            "target_ref": "session:neutral",
            "source_result_ref": "result-set:neutral",
            "actor_ref": "agent:neutral",
            "model_ref": "agent:model",
            "prompt_ref": "block:prompt",
        },
        request_id="neutral-upload",
    ).to_dict()


def _payload(control: dict[str, object]) -> dict[str, object]:
    payload = control["payload"]
    assert isinstance(payload, dict)
    return cast(dict[str, object], payload)


def _framed(raw: bytes, **descriptor: object) -> bytes:
    control = json.dumps(_control(raw, **descriptor)).encode()
    return struct.pack("!Q", len(control)) + control + raw


def test_exact_bytes_beyond_old_scalar_limit_are_staged_in_bounded_reads(tmp_path: Path) -> None:
    raw = b'{"text":"cafe\\u0301"}\n' * 100_000
    body = _framed(raw)
    reads: list[int] = []

    class Observed(io.BytesIO):
        def read(self, size: int | None = -1) -> bytes:
            assert size is not None and 0 < size <= 1024 * 1024
            reads.append(size)
            return super().read(size)

    request, staged, control_size = read_operation_body(
        Observed(body), len(body), UPLOAD_MEDIA_TYPE, spool_root=tmp_path
    )
    assert staged is not None
    try:
        assert staged.path.read_bytes() == raw
        assert request.payload["input"] == {"sha256": staged.sha256, "size_bytes": staged.size_bytes}
        assert control_size < 4096 and max(reads) < len(raw)
    finally:
        staged.discard()
    assert not list((tmp_path / ".staging").iterdir())


@pytest.mark.parametrize("mutation", ["short", "digest", "length"])
def test_invalid_custody_never_returns_an_admitted_body(tmp_path: Path, mutation: str) -> None:
    raw = b'{"row_key":"neutral"}\n'
    body = _framed(
        raw,
        **(
            {"sha256": "0" * 64}
            if mutation == "digest"
            else {"size_bytes": len(raw) + 1}
            if mutation == "length"
            else {}
        ),
    )
    actual = body[:-1] if mutation == "short" else body
    with pytest.raises((ValueError, BodyIncompleteError)):
        read_operation_body(io.BytesIO(actual), len(body), UPLOAD_MEDIA_TYPE, spool_root=tmp_path)
    assert not list(tmp_path.rglob("*.tmp"))


def test_old_whole_scalar_annotation_contract_is_refused(tmp_path: Path) -> None:
    control = _control(b"")
    control["payload"] = {**_payload(control), "jsonl": "{}"}
    raw = json.dumps(control).encode()
    with pytest.raises(ValueError):
        read_operation_body(io.BytesIO(raw), len(raw), "application/json", spool_root=tmp_path)
    assert not list(tmp_path.iterdir())


def test_streamed_controls_preserve_long_legal_refs_and_numeric_metadata(tmp_path: Path) -> None:
    raw = b"{}\n"
    control = _control(raw)
    _payload(control)["prompt_ref"] = "block:" + "opaque" * 20_000
    _payload(control)["metadata"] = {"wide_integer": 2**128, "path": "cafe\u0301", "cost": 1.0}
    encoded = json.dumps(control).encode()
    body = struct.pack("!Q", len(encoded)) + encoded + raw
    request, staged, size = read_operation_body(io.BytesIO(body), len(body), UPLOAD_MEDIA_TYPE, spool_root=tmp_path)
    assert staged is not None
    try:
        assert size > 65_536
        assert {key: request.payload[key] for key in _payload(control)} == _payload(control)
        metadata = request.payload["metadata"]
        assert isinstance(metadata, dict)
        assert type(metadata["cost"]) is float
    finally:
        staged.discard()


@pytest.mark.parametrize("authenticated", [False, True])
def test_http_annotation_route_authenticates_before_sealing_exact_input(tmp_path: Path, authenticated: bool) -> None:
    from email.message import Message

    from polylogue.daemon.http import DaemonAPIHandler, DaemonAPIHTTPServer
    from polylogue.daemon.web_auth import WebCredentialScope

    raw = b'{"row_key":"neutral"}\n'
    body = _framed(raw)
    replies: list[tuple[HTTPStatus, str] | dict[str, object]] = []

    class Handler(DaemonAPIHandler):
        def __init__(self) -> None:
            self.headers = Message()
            self.headers["Content-Length"] = str(len(body))
            self.headers["Content-Type"] = UPLOAD_MEDIA_TYPE
            self.rfile = io.BytesIO(body)
            self.server = object.__new__(DaemonAPIHTTPServer)
            self.server.archive_root = tmp_path

        def _check_auth(
            self,
            required_scope: WebCredentialScope = "read",
            *,
            allow_web: bool = True,
            refuse: Callable[[HTTPStatus, str], None] | None = None,
        ) -> bool:
            if not authenticated:
                assert refuse is not None
                refuse(HTTPStatus.UNAUTHORIZED, "unauthorized")
            return authenticated

        def _check_cross_origin(self, *, refuse: Callable[[HTTPStatus, str], None] | None = None) -> bool:
            return True

        def _reject_operation(
            self, status: HTTPStatus, code: str, detail: str | None = None, *, retryable: bool = False
        ) -> None:
            replies.append((status, code))

        def _execute_daemon_operation(
            self,
            request: DaemonOperationRequest,
            *,
            input_body: StagedBody | None = None,
            request_body_bytes: int | None = None,
        ) -> dict[str, object]:
            assert input_body is not None
            assert input_body.path.read_bytes() == raw
            descriptor = request.payload["input"]
            assert isinstance(descriptor, dict)
            assert descriptor["sha256"] == input_body.sha256
            input_body.discard()
            return {"outcome": "completed"}

        def _send_daemon_operation(self, payload: dict[str, object]) -> None:
            replies.append(payload)

    handler = Handler()
    cast(Callable[[DaemonAPIHandler], None], inspect.unwrap(DaemonAPIHandler._handle_daemon_operation))(handler)
    if authenticated:
        assert replies == [{"outcome": "completed"}]
        assert not list(tmp_path.rglob("*.tmp"))
    else:
        assert handler.rfile.tell() == 0
        assert replies == [(401, "unauthorized")]
        assert not list(tmp_path.iterdir())


@pytest.mark.uses_real_clock("production UDS listener and coordinator thread own execution custody")
def test_client_streams_large_input_to_real_uds_worker_and_retires_custody(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from polylogue.archive.message.roles import Role
    from polylogue.core.enums import Provider
    from polylogue.sources.parsers.base import ParsedMessage, ParsedSession
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from tests.infra.daemon_operations import running_daemon_operations
    from tests.infra.live_ingest import write_index_session

    def seed(root: Path) -> None:
        with ArchiveStore.open_existing(root, read_only=False) as archive:
            write_index_session(
                archive,
                ParsedSession(
                    source_name=Provider.CODEX,
                    provider_session_id="annotation-stream",
                    messages=[ParsedMessage(provider_message_id="m", role=Role.USER, text="neutral")],
                ),
            )

    raw = (
        json.dumps(
            {"row_key": "neutral", "value": {"abstain": True}, "evidence_refs": ["codex-session:annotation-stream"]}
        ).encode()
        + b" " * (2 * 1024 * 1024)
        + b"\n"
    )
    control = _payload(_control(raw))
    control.pop("input")
    control["target_ref"] = "session:codex-session:annotation-stream"
    with running_daemon_operations(tmp_path / "archive", seed_archive=seed) as stack:
        actual = stack.runtime.call
        paths = []

        from polylogue.core.compute import CancellationHandle
        from polylogue.operations.mutation_transaction import MutationPrincipal

        def observed(
            request: DaemonOperationRequest,
            principal: MutationPrincipal,
            *,
            started_at: float | None = None,
            client_disconnect: CancellationHandle | None = None,
            request_body_bytes: int | None = None,
            input_body: StagedBody | None = None,
        ) -> dict[str, object]:
            if request.operation == "mutation.annotation.import_batch":
                staged = input_body
                assert staged is not None
                paths.append(staged.path)
                assert staged.size_bytes == len(raw)
                assert staged.sha256 == hashlib.sha256(raw).hexdigest()
                assert request.payload["input"] == {"sha256": staged.sha256, "size_bytes": len(raw)}
            return actual(
                request,
                principal,
                started_at=started_at,
                client_disconnect=client_disconnect,
                request_body_bytes=request_body_bytes,
                input_body=input_body,
            )

        monkeypatch.setattr(stack.runtime, "call", observed)
        result = stack.client.operation_to_completion(
            "mutation.annotation.import_batch",
            control,
            archive_root=str(stack.archive_root),
            request_id="neutral-streamed-import",
            input=io.BytesIO(raw),
        )
        assert result is not None and result["outcome"] == "completed"
        assert result["result"]["result"]["valid_count"] == 1
        assert paths and not any(path.exists() for path in paths)
        assert not list((stack.archive_root / "operation-inputs").rglob("*.tmp"))
