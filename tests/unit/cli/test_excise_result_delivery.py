"""Excision delivery controls. Transport doubles do not prove resident authorization."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace
from typing import BinaryIO, cast

import click
import pytest
from click.testing import CliRunner, Result

from polylogue.cli.commands.excise import _emit_complete_receipt, _receipt_summary
from polylogue.cli.operation_kernel import OperationFailedError
from polylogue.cli.shared.types import AppEnv


def _invoke(
    monkeypatch: pytest.MonkeyPatch, chunks: Iterator[bytes], document: object
) -> tuple[Result, list[tuple[object, object]], object]:
    from polylogue.cli import operation_kernel

    config = object()
    env = cast("AppEnv", SimpleNamespace(config=config))
    observed: list[tuple[object, object]] = []

    def delivery(actual_config: object, actual_document: object) -> Iterator[bytes]:
        observed.append((actual_config, actual_document))
        yield from chunks

    monkeypatch.setattr(operation_kernel, "iter_configured_operation_result", delivery)

    @click.command()
    def render() -> None:
        _emit_complete_receipt(
            env,
            {
                "effect": "committed",
                "result_document": document,
                "result": {"receipt_assertion_id": "receipt:original"},
            },
            session_id="session:original",
            affected_count=3,
        )

    return CliRunner().invoke(render), observed, config


def _document(body: bytes) -> dict[str, object]:
    return {"request_id": "request:original", "byte_length": len(body), "sha256": hashlib.sha256(body).hexdigest()}


def test_complete_document_preserves_all_arrays_and_exact_utf8_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    arrays = (
        "removed_blob_hashes",
        "shared_blob_hashes",
        "marker_input_digests",
        "cascaded_session_ids",
        "retained_hook_events",
        "retained_source_containers",
    )
    body = json.dumps({name: ["界[markup]" * 10000] for name in arrays}, ensure_ascii=False).encode()
    document = _document(body)
    closed: list[bool] = []

    def chunks() -> Iterator[bytes]:
        try:
            for start in range(0, len(body), 65536):
                yield body[start : start + 65536]
        finally:
            closed.append(True)

    result, observed, config = _invoke(monkeypatch, chunks(), document)
    assert result.exit_code == 0, result.exception
    assert result.stdout_bytes.endswith(b'"domain_receipt":' + body + b"}\n")
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["detail"] == "receipt:original"
    assert payload["affected_count"] == 3
    assert set(payload["domain_receipt"]) == set(arrays)
    assert observed == [(config, document)]
    assert closed == [True]


@pytest.mark.parametrize("fault", ["missing", "length", "hash", "utf8", "stale", "cancel"])
def test_failed_delivery_emits_no_completed_json_and_closes_stage(
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
) -> None:
    from polylogue.core import staged_content

    body = b'{"cascaded_session_ids":[]}'
    document: object = _document(body)
    if fault == "missing":
        document = None
    elif fault == "length":
        document = {**_document(body), "byte_length": len(body) + 1}
    elif fault == "hash":
        document = {**_document(body), "sha256": "0" * 64}
    elif fault == "utf8":
        body = b'{"value":"\xff"}'
        document = _document(body)
    stages: list[BinaryIO] = []
    original_stage = staged_content.staged_binary_content

    @contextmanager
    def observe_stage() -> Iterator[BinaryIO]:
        with original_stage() as stage:
            stages.append(stage)
            yield stage

    monkeypatch.setattr(staged_content, "staged_binary_content", observe_stage)
    closed: list[bool] = []

    def chunks() -> Iterator[bytes]:
        try:
            yield body
            if fault == "stale":
                raise OperationFailedError("operation_result_delivery_failed", "document identity changed")
            if fault == "cancel":
                raise KeyboardInterrupt
        finally:
            closed.append(True)

    result, observed, _config = _invoke(monkeypatch, chunks(), document)
    assert result.exit_code == 1
    assert result.stdout_bytes == b""
    assert all(stage.closed for stage in stages)
    if fault != "missing":
        assert closed == [True]
        assert len(observed) == 1
    else:
        assert observed == []
    assert isinstance(result.exception, OperationFailedError)
    assert result.exception.code == (
        "operation_result_delivery_cancelled" if fault == "cancel" else "operation_result_delivery_failed"
    )
    assert result.exception.data == {"reference": "receipt:original", "effect_committed": True, "effect": "committed"}
    assert result.exception.request_id == (None if fault == "missing" else "request:original")


def test_human_summary_uses_scalar_cardinalities_without_accessing_arrays() -> None:
    class Scalars(dict[str, object]):
        def get(self, key: str, default: object = None) -> object:
            if key in {
                "cascaded_session_ids",
                "retained_hook_events",
                "retained_source_containers",
                "shared_blob_hashes",
            }:
                raise AssertionError("unbounded array was read")
            return super().get(key, default)

    summary = Scalars(
        counts={"index_sessions": 3},
        receipt_assertion_id="receipt:original",
        complete=False,
        cascaded_session_ids_count=2,
        retained_hook_events_count=4,
        retained_source_containers_count=5,
        shared_blob_hashes_count=6,
    )
    text = _receipt_summary("session:original", summary, "reference:original")
    assert "2 lineage-dependent sessions" in text
    assert "4 hook events" in text
    assert "5 source containers" in text
    assert "6 blobs" in text
    assert "INCOMPLETE" in text
    assert "receipt:original" in text


@pytest.mark.parametrize("count", [None, -1, True, "3"])
def test_human_summary_refuses_unmeasured_cardinality(count: object) -> None:
    with pytest.raises(click.ClickException):
        _receipt_summary("session:original", {"cascaded_session_ids_count": count}, "receipt:original")


@pytest.mark.parametrize("fault", [None, "identity", "offset", "no_progress", "unavailable", "hash"])
def test_actual_result_iterator_exhaustion_governs_renderer_success(
    monkeypatch: pytest.MonkeyPatch,
    fault: str | None,
) -> None:
    import base64
    from pathlib import Path

    from polylogue.daemon_client import DaemonClient

    body = b'{"retained_hook_events":["one","two"]}'
    document = _document(body)
    client = DaemonClient(Path("/synthetic/resident.sock"))
    calls: list[dict[str, object]] = []
    cut = len(body) // 2

    def operation(name: str, payload: dict[str, object], **options: object) -> dict[str, object]:
        assert name == "operation.result"
        assert options == {"archive_root": "/synthetic/archive", "expected_archive_identity": "incarnation:original"}
        calls.append(payload)
        offset = payload["offset"]
        assert isinstance(offset, int)
        chunk = body[:cut] if offset == 0 else body[cut:]
        page_document = dict(document)
        page_offset = offset
        next_offset = cut if offset == 0 else None
        if offset:
            if fault == "identity":
                page_document["request_id"] = "request:foreign"
            elif fault == "offset":
                page_offset = offset + 1
            elif fault == "no_progress":
                chunk = b""
                next_offset = offset
            elif fault == "unavailable":
                return {"outcome": "failed"}
            elif fault == "hash":
                chunk = b"x" * len(chunk)
        return {
            "outcome": "completed",
            "result": {
                "document": page_document,
                "offset": page_offset,
                "data_base64": base64.b64encode(chunk).decode("ascii"),
                "next_offset": next_offset,
            },
        }

    monkeypatch.setattr(client, "operation", operation)
    result, _observed, _config = _invoke(
        monkeypatch,
        client.iter_operation_result(
            document, archive_root="/synthetic/archive", expected_archive_identity="incarnation:original"
        ),
        document,
    )
    assert [call["offset"] for call in calls] == [0, cut]
    assert all(call["request_id"] == "request:original" for call in calls)
    if fault is None:
        assert result.exit_code == 0, result.exception
        assert json.loads(result.stdout)["domain_receipt"] == {"retained_hook_events": ["one", "two"]}
    else:
        assert result.exit_code == 1
        assert result.stdout_bytes == b""


@pytest.mark.parametrize("missing", [False, True])
def test_command_delivers_domain_or_truthful_delivery_error_without_repeating_mutation(
    monkeypatch: pytest.MonkeyPatch,
    missing: bool,
) -> None:
    from polylogue.cli import operation_kernel
    from polylogue.cli.commands import excise

    body = b'{"cascaded_session_ids":["child"],"complete":true}'
    document = _document(body)
    config = object()
    env = cast("AppEnv", SimpleNamespace(config=config))
    submitted: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(
        operation_kernel,
        "configured_read_operation",
        lambda *_args, **_kwargs: SimpleNamespace(value={"found": True, "refused": False, "plan": {"found": True}}),
    )

    def submit(_env: AppEnv, operation: str, payload: dict[str, object]) -> dict[str, object]:
        submitted.append((operation, payload))
        return {
            "affected_count": 2,
            "effect": "committed",
            "result": {"receipt_assertion_id": "receipt:original"},
            "result_document": None if missing else document,
        }

    def delivery(actual_config: object, actual_document: object) -> Iterator[bytes]:
        assert actual_config is config
        assert actual_document is document
        yield body

    monkeypatch.setattr(excise, "_submit", submit)
    monkeypatch.setattr(operation_kernel, "iter_configured_operation_result", delivery)
    result = CliRunner().invoke(
        excise.excise_command,
        ["--session", "session:original", "--reason", "neutral", "--yes", "--json"],
        obj=env,
    )
    assert len(submitted) == 1
    assert submitted[0][0] == "mutation.session.excision"
    payload = json.loads(result.stdout)
    if missing:
        assert result.exit_code == 1
        assert payload["code"] == "operation_result_delivery_failed"
        assert payload["details"] == {
            "reference": "receipt:original",
            "effect_committed": True,
            "effect": "committed",
            "request_id": None,
            "operation": "operation.result",
        }
        assert "domain_receipt" not in payload
    else:
        assert result.exit_code == 0, result.exception
        assert payload["status"] == "ok"
        assert payload["domain_receipt"] == {"cascaded_session_ids": ["child"], "complete": True}
