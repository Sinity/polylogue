"""Byte decoding, JSON streaming, and ZIP processing utilities."""

from __future__ import annotations

from collections.abc import Generator, Iterable, Iterator
from contextlib import closing, contextmanager
from typing import IO, BinaryIO

import ijson

from polylogue.core.compute_cancel import check_compute_cancelled
from polylogue.logging import get_logger
from polylogue.sources.decoder_json import (
    DecodedRecordSequence,
    JsonlDecodeError,
    JsonValue,
    decode_json_bytes_with,
    iter_json_stream_with,
)
from polylogue.sources.decoder_zip import ZipEntryValidator as _ZipEntryValidator
from polylogue.sources.decoder_zip import (
    open_zip_entry,
)
from polylogue.sources.decoder_zip import process_zip as _process_zip
from polylogue.sources.decoder_zip import zip_entry_provider_hint as _zip_entry_provider_hint

logger = get_logger(__name__)


@contextmanager
def owned_json_records(
    handle: BinaryIO | Iterable[bytes],
    path_name: str,
    *,
    unpack_lists: bool = True,
    fail_on_decode_error: bool = False,
) -> Iterator[Iterable[JsonValue]]:
    """Retain decoded JSONL containers through the consumer's final output."""
    from polylogue.sources.dispatch import is_jsonl_source_path

    if is_jsonl_source_path(path_name):
        with closing(
            DecodedRecordSequence.from_jsonl(
                handle, path_name, logger_obj=logger, fail_on_decode_error=fail_on_decode_error
            )
        ) as records:
            yield records
    else:
        with closing(
            _iter_json_stream(handle, path_name, unpack_lists, fail_on_decode_error=fail_on_decode_error)
        ) as records:
            yield records


def _decode_json_bytes(blob: bytes) -> str | None:
    return decode_json_bytes_with(logger, blob)


def _iter_json_stream(
    handle: BinaryIO | IO[bytes] | Iterable[bytes],
    path_name: str,
    unpack_lists: bool = True,
    *,
    fail_on_decode_error: bool = False,
) -> Generator[JsonValue, None, None]:
    for value in iter_json_stream_with(
        logger,
        ijson,
        handle,
        path_name,
        unpack_lists,
        fail_on_decode_error=fail_on_decode_error,
    ):
        check_compute_cancelled()
        yield value


__all__ = [
    "_decode_json_bytes",
    "_iter_json_stream",
    "JsonlDecodeError",
    "_ZipEntryValidator",
    "_zip_entry_provider_hint",
    "_process_zip",
    "open_zip_entry",
]
