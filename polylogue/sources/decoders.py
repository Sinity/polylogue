"""Byte decoding, JSON streaming, and ZIP processing utilities."""

from __future__ import annotations

from collections.abc import Iterable
from typing import IO, BinaryIO

import ijson

from polylogue.logging import get_logger
from polylogue.sources.decoder_json import (
    JsonlDecodeError,
    JsonValue,
    decode_json_bytes_with,
    iter_json_stream_with,
)
from polylogue.sources.decoder_zip import (
    MAX_AGGREGATE_UNCOMPRESSED_SIZE,
    MAX_COMPRESSION_RATIO,
    MAX_UNCOMPRESSED_SIZE,
    open_bounded_zip_entry,
)
from polylogue.sources.decoder_zip import ZipEntryValidator as _ZipEntryValidator
from polylogue.sources.decoder_zip import process_zip as _process_zip
from polylogue.sources.decoder_zip import zip_entry_provider_hint as _zip_entry_provider_hint

logger = get_logger(__name__)


def _decode_json_bytes(blob: bytes) -> str | None:
    return decode_json_bytes_with(logger, blob)


def _iter_json_stream(
    handle: BinaryIO | IO[bytes],
    path_name: str,
    unpack_lists: bool = True,
    *,
    fail_on_decode_error: bool = False,
) -> Iterable[JsonValue]:
    yield from iter_json_stream_with(
        logger,
        ijson,
        handle,
        path_name,
        unpack_lists,
        fail_on_decode_error=fail_on_decode_error,
    )


__all__ = [
    "_decode_json_bytes",
    "_iter_json_stream",
    "JsonlDecodeError",
    "_ZipEntryValidator",
    "_zip_entry_provider_hint",
    "_process_zip",
    "MAX_AGGREGATE_UNCOMPRESSED_SIZE",
    "MAX_COMPRESSION_RATIO",
    "MAX_UNCOMPRESSED_SIZE",
    "open_bounded_zip_entry",
]
