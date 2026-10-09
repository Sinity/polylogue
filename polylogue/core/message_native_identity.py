"""Native message identity normalization shared by archive and material writers."""

from __future__ import annotations

import re
from collections.abc import Set

_SURROGATES = re.compile(r"[\ud800-\udfff]")


def normalized_message_native_id(value: str | None) -> str | None:
    """Keep opaque nonempty text; normalize only SQLite-unrepresentable surrogates."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("message native_id must be text or None")
    return _SURROGATES.sub("\ufffd", value) or None


def stored_message_native_id(value: str | None, duplicate_native_ids: Set[str]) -> str | None:
    native_id = normalized_message_native_id(value)
    return None if native_id in duplicate_native_ids else native_id
