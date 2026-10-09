"""Exact body custody for streamed annotation machine operations."""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import BinaryIO

from polylogue.core.staged_body import BodyIncompleteError, StagedBody, stage_body
from polylogue.operations.daemon_protocol import (
    MAX_DECLARED_OPERATION_BODY_BYTES,
    DaemonOperationRequest,
    daemon_operation_spec,
)

UPLOAD_MEDIA_TYPE = "application/vnd.polylogue.operation-input"
ANNOTATION_IMPORT_OPERATION = "mutation.annotation.import_batch"


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate operation field")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"invalid JSON constant: {value}")


def _decode_control(raw):
    return DaemonOperationRequest.from_dict(
        json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    )


def _read_exact(source: BinaryIO, length: int) -> bytes:
    chunks = []
    remaining = length
    while remaining:
        part = source.read(min(remaining, 65536))
        if not part:
            raise BodyIncompleteError("incomplete operation control")
        chunks.append(part)
        remaining -= len(part)
    return b"".join(chunks)


def read_operation_body(
    source: BinaryIO, length: int, media_type: str, *, spool_root: Path
) -> tuple[DaemonOperationRequest, StagedBody | None, int]:
    """Read control then seal the body; callers authenticate before this call."""
    if media_type == "application/json":
        if length <= 0 or length > MAX_DECLARED_OPERATION_BODY_BYTES:
            raise ValueError("invalid operation control length")
        request = _decode_control(_read_exact(source, length))
        if request.operation == ANNOTATION_IMPORT_OPERATION:
            raise ValueError("annotation import requires streamed input framing")
        return request, None, length
    if media_type != UPLOAD_MEDIA_TYPE or length < 8:
        raise ValueError("unsupported operation input framing")
    control_length = struct.unpack("!Q", _read_exact(source, 8))[0]
    if control_length <= 0 or control_length > MAX_DECLARED_OPERATION_BODY_BYTES or control_length > length - 8:
        raise ValueError("invalid operation control length")
    request = _decode_control(_read_exact(source, control_length))
    if request.operation != ANNOTATION_IMPORT_OPERATION:
        raise ValueError("operation does not declare an input body")
    declared = request.payload["input"]
    if not isinstance(declared, dict) or declared.get("size_bytes") != length - 8 - control_length:
        raise ValueError("input byte length differs from HTTP framing")
    spec = daemon_operation_spec(request.operation)
    assert spec is not None
    if control_length > spec.max_body_bytes:
        raise ValueError("operation control exceeds declared framing")
    staged = stage_body(source.read, length - 8 - control_length, spool_root=spool_root)
    if staged.sha256 != declared.get("sha256"):
        staged.discard()
        raise ValueError("input digest mismatch")
    return request, staged, control_length
