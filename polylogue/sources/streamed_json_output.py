"""Encode borrowed JSON trees without reconstructing complete strings or keys."""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import closing
from decimal import Decimal
from pathlib import Path
from typing import BinaryIO

from polylogue.core.compute_cancel import check_compute_cancelled
from polylogue.core.json import dumps_bytes
from polylogue.schemas.observation_spill import SpilledArray, SpilledObject, _load_node, _read_row, _read_rows


def write_streamed_json(value: object, destination: Path, *, member_format: bool = False) -> None:
    """Publish complete encoded bytes while borrowing the input owner's lifetime.

    Member format preserves the original stdlib bundle-member serialization;
    compact format preserves the core JSON encoder. Failed output stays private.
    """
    stage = destination.with_name(f".{destination.name}-{uuid.uuid4().hex}.partial")
    try:
        with stage.open("xb") as output:
            _write_value(value, output, member_format)
            check_compute_cancelled()
        os.replace(stage, destination)
    finally:
        stage.unlink(missing_ok=True)


def _scalar(value: object, member_format: bool) -> bytes:
    if isinstance(value, int) and not isinstance(value, bool):
        return format(Decimal(value), "f").encode("ascii")
    return json.dumps(value, ensure_ascii=True).encode("utf-8") if member_format else dumps_bytes(value)


def _write_string(chunks: Iterator[str], output: BinaryIO, member_format: bool) -> None:
    output.write(b'"')
    for chunk in chunks:
        check_compute_cancelled()
        output.write(_scalar(chunk, member_format)[1:-1])
    output.write(b'"')


def _text_chunks(value: str) -> Iterator[str]:
    for offset in range(0, len(value), 4096):
        yield value[offset : offset + 4096]


def _write_node(connection: sqlite3.Connection, node: int, output: BinaryIO, member_format: bool) -> None:
    row = _read_row(connection, "SELECT kind, token FROM json_nodes WHERE id=?", (node,))
    if row is None:
        raise ValueError("streamed JSON tree lost a referenced node")
    kind, token = row
    if kind == "number" and token is not None:
        json_kind, decoded_bytes = _read_row(
            connection,
            "SELECT json_kind, decoded_bytes FROM json_scalar_tokens WHERE kind='number' AND token=?",
            (token,),
        )
        if json_kind == "integer":
            with closing(
                _read_rows(
                    connection,
                    "SELECT data FROM json_scalar_chunks WHERE kind='number' AND token=? ORDER BY ordinal",
                    (token,),
                )
            ) as chunks:
                if decoded_bytes <= 2:
                    number = b"".join(bytes(data) for (data,) in chunks)
                    output.write(b"0" if number == b"-0" else number)
                else:
                    for (data,) in chunks:
                        check_compute_cancelled()
                        output.write(bytes(data))
            return
    if kind == "string" and token is not None:
        with closing(
            _read_rows(
                connection,
                "SELECT data FROM json_scalar_chunks WHERE kind='string' AND token=? ORDER BY ordinal",
                (token,),
            )
        ) as chunks:
            _write_string((bytes(data).decode("utf-8", "surrogatepass") for (data,) in chunks), output, member_format)
        return
    # Numeric tokens must pass through the existing decoder/encoder: preserving
    # a wire lexeme such as 1e2 would change the canonical output's 100.0.
    _write_value(_load_node(connection, node), output, member_format)


def _write_value(value: object, output: BinaryIO, member_format: bool) -> None:
    check_compute_cancelled()
    comma, colon = (b", ", b": ") if member_format else (b",", b":")
    if isinstance(value, SpilledObject):
        output.write(b"{")
        with closing(value.key_entries()) as entries:
            for ordinal, (key, child) in enumerate(entries):
                if ordinal:
                    output.write(comma)
                with closing(key.iter_utf8_chunks()) as chunks:
                    _write_string((chunk.decode("utf-8", "surrogatepass") for chunk in chunks), output, member_format)
                output.write(colon)
                _write_node(value._connection, child, output, member_format)
        output.write(b"}")
    elif isinstance(value, SpilledArray):
        output.write(b"[")
        with closing(
            _read_rows(
                value._connection,
                "SELECT child_id FROM json_array_items WHERE parent_id=? ORDER BY ordinal",
                (value._node_id,),
            )
        ) as rows:
            for ordinal, (child,) in enumerate(rows):
                if ordinal:
                    output.write(comma)
                _write_node(value._connection, int(child), output, member_format)
        output.write(b"]")
    elif isinstance(value, dict):
        output.write(b"{")
        for ordinal, (key, child) in enumerate(value.items()):
            if ordinal:
                output.write(comma)
            if not isinstance(key, str):
                raise TypeError("JSON object keys must be strings")
            _write_string(_text_chunks(key), output, member_format)
            output.write(colon)
            _write_value(child, output, member_format)
        output.write(b"}")
    elif isinstance(value, (list, tuple)):
        output.write(b"[")
        for ordinal, child in enumerate(value):
            if ordinal:
                output.write(comma)
            _write_value(child, output, member_format)
        output.write(b"]")
    elif isinstance(value, str):
        _write_string(_text_chunks(value), output, member_format)
    else:
        output.write(_scalar(value, member_format))
