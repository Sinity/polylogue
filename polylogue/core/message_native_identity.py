"""Exact native-message names, stable keys, and SQLite-safe carriers."""

from __future__ import annotations

import json
import re
from collections.abc import Set

_SURROGATES = re.compile(r"[\ud800-\udfff]")


def normalized_message_native_id(value: str | None) -> str | None:
    """Only literal empty is absent; every other Source name is opaque."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("message native_id must be text or None")
    return value or None


def message_native_key(value: str | None) -> str | None:
    native = normalized_message_native_id(value)
    if native is None:
        return None
    if _SURROGATES.search(native):
        return "s:" + native.encode("utf-8", "surrogatepass").hex()
    return "n:" + native


def native_id_from_key(key: str) -> str:
    if key.startswith("n:") and key[2:]:
        return key[2:]
    if key.startswith("s:"):
        try:
            value = bytes.fromhex(key[2:]).decode("utf-8", "surrogatepass")
        except (ValueError, UnicodeError) as exc:
            raise ValueError("invalid surrogate native key") from exc
        if message_native_key(value) == key:
            return value
    raise ValueError("invalid native message key")


def sqlite_message_native_id(value: str | None) -> str | None:
    key = message_native_key(value)
    return key[2:] if key is not None else None


def source_native_id_json(value: str | None) -> str | None:
    key = message_native_key(value)
    if key is None:
        return None
    payload: object = {"encoding": "utf8-surrogatepass", "value": key[2:]} if key.startswith("s:") else value
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":"))


def source_native_id_from_json(encoded: str | None) -> str | None:
    if encoded is None:
        return None
    if not isinstance(encoded, str):
        raise ValueError("Source native carrier must be JSON text")
    payload = json.loads(encoded)
    if isinstance(payload, str):
        if not payload or _SURROGATES.search(payload):
            raise ValueError("invalid ordinary native carrier")
        return payload
    if (
        isinstance(payload, dict)
        and set(payload) == {"encoding", "value"}
        and payload["encoding"] == "utf8-surrogatepass"
        and isinstance(payload["value"], str)
    ):
        return native_id_from_key("s:" + payload["value"])
    raise ValueError("invalid Source native carrier")


def native_id_from_storage(value: str | None, carrier: str | None) -> str | None:
    if value is None:
        return None
    original = source_native_id_from_json(carrier)
    original_key = message_native_key(original)
    if original_key is not None and original_key.startswith("s:"):
        if sqlite_message_native_id(original) != value:
            raise ValueError("stored surrogate native ID disagrees with Source carrier")
        return original
    if original is not None and original != value:
        raise ValueError("stored native ID disagrees with Source carrier")
    return normalized_message_native_id(value)


def stored_message_native_id(value: str | None, duplicate_native_ids: Set[str]) -> str | None:
    native = normalized_message_native_id(value)
    return None if message_native_key(native) in duplicate_native_ids else native
