"""Explicit eager fixture boundary for values borrowed from JSON owners."""

from collections.abc import Generator, Iterable
from typing import BinaryIO

from polylogue.core.json import JSONValue
from polylogue.sources.decoders import owned_json_records


def materialize_json(value: JSONValue) -> JSONValue:
    if isinstance(value, dict):
        return {key: materialize_json(child) for key, child in value.items()}
    if isinstance(value, list):
        return [materialize_json(child) for child in value]
    return value


def iter_owned_json_values(
    handle: BinaryIO | Iterable[bytes],
    path_name: str,
    unpack_lists: bool = True,
    *,
    fail_on_decode_error: bool = False,
) -> Generator[JSONValue, None, None]:
    """Materialize only synthetic fixtures while the production owner is live."""
    with owned_json_records(
        handle, path_name, unpack_lists=unpack_lists, fail_on_decode_error=fail_on_decode_error
    ) as records:
        for value in records:
            yield materialize_json(value)
