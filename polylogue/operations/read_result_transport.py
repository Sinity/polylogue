"""Finite operation JSON delivery without an outcome-changing byte ceiling."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from tempfile import TemporaryFile
from typing import BinaryIO

TRANSFER_BYTES = 64 * 1024


@contextmanager
def staged_json_response(
    payload: object, *, append_newline: bool = False, ensure_ascii: bool = False
) -> Iterator[BinaryIO]:
    """Finish encoding before publishing headers; retire scratch on every exit.

    The product still owns its Python values. Scratch avoids retaining an
    additional whole-envelope string and bytes object during delivery.
    """
    with TemporaryFile(mode="w+b") as staged:
        encoder = json.JSONEncoder(ensure_ascii=ensure_ascii, separators=(",", ":"), allow_nan=False)
        for fragment in encoder.iterencode(payload):
            # A single JSON string fragment can itself exceed the transfer size.
            for offset in range(0, len(fragment), TRANSFER_BYTES):
                staged.write(fragment[offset : offset + TRANSFER_BYTES].encode("utf-8"))
        if append_newline:
            staged.write(b"\n")
        staged.seek(0)
        yield staged


def decode_json_response(source: BinaryIO, byte_length: int) -> object:
    """Consume exactly the framed response before returning any decoded value."""
    from ijson.backends.python import items
    from ijson.common import JSONError

    with TemporaryFile(mode="w+b") as staged:
        remaining = byte_length
        while remaining:
            chunk = source.read(min(TRANSFER_BYTES, remaining))
            if not chunk:
                raise EOFError("daemon response body is incomplete")
            staged.write(chunk)
            remaining -= len(chunk)
        staged.seek(0)
        values = items(staged, "", use_float=True)
        try:
            value = next(values)
            # Exhaustion checks trailing malformed bytes, not just the root value.
            if next(values, None) is not None:
                raise ValueError("daemon response has multiple JSON values")
            return value
        except (JSONError, StopIteration) as exc:
            raise ValueError("daemon response is not complete JSON") from exc
        finally:
            values.close()
