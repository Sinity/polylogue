"""Shared handling for the historical Codex append payload envelope."""

from __future__ import annotations

from json import dumps as json_dumps

from polylogue.core.json import loads as json_loads


def strip_codex_legacy_append_header(payload: bytes) -> bytes | None:
    """Return a historical Codex append delta without its synthetic header."""
    newline_at = payload.find(b"\n")
    if newline_at < 0:
        return None
    first_line = payload[:newline_at]
    try:
        record = json_loads(first_line.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None
    if not isinstance(record, dict) or record.get("type") != "session_meta":
        return None
    inner = record.get("payload")
    if not isinstance(inner, dict) or "id" not in inner or set(inner) != {"id"}:
        return None
    if set(record) != {"type", "payload"}:
        return None
    if (
        first_line
        != json_dumps({"type": "session_meta", "payload": {"id": inner["id"]}}, separators=(",", ":")).encode()
    ):
        return None
    return payload[newline_at + 1 :]
