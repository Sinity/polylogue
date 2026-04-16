"""Byte decoding, JSON streaming, and ZIP processing utilities."""

from __future__ import annotations

import polylogue.sources.decoder_json as _decoder_json
from polylogue.sources.decoder_json import decode_json_bytes_with, iter_json_stream_with
from polylogue.sources.decoder_zip import (
    MAX_COMPRESSION_RATIO,
    MAX_UNCOMPRESSED_SIZE,
)
from polylogue.sources.decoder_zip import ZipEntryValidator as _ZipEntryValidator
from polylogue.sources.decoder_zip import process_zip as _process_zip
from polylogue.sources.decoder_zip import zip_entry_provider_hint as _zip_entry_provider_hint

ijson = _decoder_json.ijson
logger = _decoder_json.logger


def _decode_json_bytes(blob: bytes) -> str | None:
    return decode_json_bytes_with(logger, blob)


def _iter_json_stream(handle, path_name: str, unpack_lists: bool = True):
    yield from iter_json_stream_with(logger, ijson, handle, path_name, unpack_lists)


__all__ = [
    "_decode_json_bytes",
    "_iter_json_stream",
    "_ZipEntryValidator",
    "_zip_entry_provider_hint",
    "_process_zip",
    "MAX_COMPRESSION_RATIO",
    "MAX_UNCOMPRESSED_SIZE",
]
