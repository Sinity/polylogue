"""Native message identity normalization shared by archive and material writers."""

from __future__ import annotations

import re
from collections.abc import Set

_SURROGATES = re.compile(r"[\ud800-\udfff]")


def normalized_message_native_id(value: str | None) -> str | None:
    """Keep SQLite-storable opaque text, stripping the native identity boundary."""
    if value is None:
        return None
    return _SURROGATES.sub("\ufffd", value).strip() or None


def stored_message_native_id(value: str | None, duplicate_native_ids: Set[str]) -> str | None:
    native_id = normalized_message_native_id(value)
    return None if native_id in duplicate_native_ids else native_id
