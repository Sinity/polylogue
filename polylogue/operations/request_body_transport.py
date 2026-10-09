"""Exact body custody for streamed annotation machine operations."""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import NoReturn, Protocol, TypedDict, cast

from polylogue.core.staged_body import BodyIncompleteError, StagedBody, stage_body
from polylogue.operations.daemon_protocol import (
    MAX_DECLARED_OPERATION_BODY_BYTES,
    DaemonOperationRequest,
)

UPLOAD_MEDIA_TYPE = "application/vnd.polylogue.operation-input"
ANNOTATION_IMPORT_OPERATION = "mutation.annotation.import_batch"


class BinaryReader(Protocol):
    def read(self, size: int = -1, /) -> bytes: ...


class OperationInputKwargs(TypedDict, total=False):
    input: BinaryReader


class OperationBodyKwargs(TypedDict, total=False):
    input_body: StagedBody
    request_body_bytes: int


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate operation field")
        result[key] = value
    return result


def _reject_constant(value: str) -> NoReturn:
    raise ValueError(f"invalid JSON constant: {value}")


def _decode_control(raw: bytes) -> DaemonOperationRequest:
    return DaemonOperationRequest.from_dict(
        json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    )


def _decode_streamed_control(source: BinaryReader, length: int) -> DaemonOperationRequest:
    from decimal import Decimal

    from ijson.backends.python import parse
    from ijson.common import JSONError, ObjectBuilder

    class FramedReader:
        remaining = length

        def read(self, size: int = -1) -> bytes:
            if size == 0 or self.remaining == 0:
                return b""
            chunk = source.read(min(65536, self.remaining, size if size > 0 else self.remaining))
            if not chunk:
                raise BodyIncompleteError("incomplete operation control")
            self.remaining -= len(chunk)
            return chunk

    framed = FramedReader()
    builder = ObjectBuilder()
    objects: list[set[str] | None] = []
    try:
        for _prefix, event, value in parse(framed, use_float=False):
            if event == "start_map":
                objects.append(set())
            elif event == "start_array":
                objects.append(None)
            elif event in {"end_map", "end_array"}:
                objects.pop()
            elif event == "map_key":
                keys = cast(set[str], objects[-1])
                value = cast(str, value)
                if value in keys:
                    raise ValueError("duplicate operation field")
                keys.add(value)
            if isinstance(value, Decimal):
                value = float(value)
            builder.event(event, value)
        if framed.remaining:
            raise BodyIncompleteError("incomplete operation control")
        return DaemonOperationRequest.from_dict(builder.value)
    except JSONError as exc:
        raise ValueError("invalid operation control JSON") from exc


def _read_exact(source: BinaryReader, length: int) -> bytes:
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
    source: BinaryReader, length: int, media_type: str, *, spool_root: Path
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
    if control_length <= 0 or control_length > length - 8:
        raise ValueError("invalid operation control length")
    request = _decode_streamed_control(source, control_length)
    if request.operation != ANNOTATION_IMPORT_OPERATION:
        raise ValueError("operation does not declare an input body")
    declared = request.payload["input"]
    if not isinstance(declared, dict) or declared.get("size_bytes") != length - 8 - control_length:
        raise ValueError("input byte length differs from HTTP framing")
    staged = stage_body(source.read, length - 8 - control_length, spool_root=spool_root)
    if staged.sha256 != declared.get("sha256"):
        staged.discard()
        raise ValueError("input digest mismatch")
    return request, staged, control_length
