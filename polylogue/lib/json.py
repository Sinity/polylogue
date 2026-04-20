"""Central JSON utilities using orjson with typed JSON contracts."""

from __future__ import annotations

import json
from collections.abc import Callable
from decimal import Decimal
from typing import TypeAlias, TypeGuard, cast

import orjson

JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]
JSONDocument: TypeAlias = dict[str, JSONValue]
JSONDocumentList: TypeAlias = list[JSONDocument]
JSONEncoder: TypeAlias = Callable[[object], object]


def is_json_value(value: object) -> TypeGuard[JSONValue]:
    """Return whether *value* is representable as JSON."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return True
    if isinstance(value, list):
        return all(is_json_value(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and is_json_value(item) for key, item in value.items())
    return False


def is_json_document(value: object) -> TypeGuard[JSONDocument]:
    """Return whether *value* is a JSON object with string keys."""
    return isinstance(value, dict) and all(isinstance(key, str) for key in value)


def json_document(value: object) -> JSONDocument:
    """Coerce a value into a string-keyed JSON object when possible."""
    return value if is_json_document(value) else {}


def json_document_list(value: object) -> JSONDocumentList:
    """Coerce a value into a list of string-keyed JSON objects."""
    if not isinstance(value, list):
        return []
    return [document for item in value if (document := json_document(item))]


def _reject_non_finite_token(token: str) -> JSONValue:
    raise ValueError(f"invalid non-finite JSON token: {token}")


def _default_encoder(user_default: JSONEncoder | None = None) -> JSONEncoder:
    """Create a JSON encoder that handles Decimal values."""

    def _encoder(obj: object) -> object:
        if user_default is not None:
            try:
                return user_default(obj)
            except TypeError:
                pass
        if isinstance(obj, Decimal):
            return float(obj)
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

    return _encoder


def dumps_bytes(
    obj: object,
    *,
    default: JSONEncoder | None = None,
    option: int | None = None,
) -> bytes:
    """Dump object to UTF-8 JSON bytes."""
    encoder = _default_encoder(default)
    try:
        if option is None:
            return orjson.dumps(obj, default=encoder)
        return orjson.dumps(obj, default=encoder, option=option)
    except TypeError:
        pass

    return json.dumps(obj, default=encoder).encode("utf-8")


def dumps(obj: object, *, default: JSONEncoder | None = None, option: int | None = None) -> str:
    """Dump object to JSON string."""
    return dumps_bytes(obj, default=default, option=option).decode("utf-8")


def loads(obj: str | bytes | bytearray) -> JSONValue:
    """Load object from JSON string or bytes."""
    try:
        return cast(JSONValue, orjson.loads(obj))
    except (orjson.JSONDecodeError, ValueError) as exc:
        try:
            return cast(JSONValue, json.loads(obj, parse_constant=_reject_non_finite_token))
        except (json.JSONDecodeError, ValueError):
            raise exc from None


__all__ = [
    "JSONDocument",
    "JSONDocumentList",
    "JSONEncoder",
    "JSONScalar",
    "JSONValue",
    "dumps",
    "dumps_bytes",
    "is_json_document",
    "is_json_value",
    "json_document",
    "json_document_list",
    "loads",
]
