"""Annotation input framing seals the exact client bytes before dispatch."""

from __future__ import annotations

import hashlib
import io
import json
import struct
from pathlib import Path

import pytest

from polylogue.core.staged_body import BodyIncompleteError
from polylogue.operations.daemon_protocol import DaemonOperationRequest
from polylogue.operations.request_body_transport import UPLOAD_MEDIA_TYPE, read_operation_body


def _control(raw: bytes, **descriptor) -> dict[str, object]:
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


def _framed(raw: bytes, **descriptor) -> bytes:
    control = json.dumps(_control(raw, **descriptor)).encode()
    return struct.pack("!Q", len(control)) + control + raw


def test_exact_bytes_beyond_old_scalar_limit_are_staged_in_bounded_reads(tmp_path: Path) -> None:
    raw = b'{"text":"cafe\\u0301"}\n' * 100_000
    body = _framed(raw)
    reads = []

    class Observed(io.BytesIO):
        def read(self, size=-1):
            reads.append(size)
            assert 0 < size <= 1024 * 1024
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
    control["payload"] = {**control["payload"], "jsonl": "{}"}
    raw = json.dumps(control).encode()
    with pytest.raises(ValueError):
        read_operation_body(io.BytesIO(raw), len(raw), "application/json", spool_root=tmp_path)
    assert not list(tmp_path.iterdir())
