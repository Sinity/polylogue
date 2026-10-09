"""Shared attachment preview states derived from read-side availability."""

from __future__ import annotations

from typing import Any

PREVIEW_SIZE_BUDGET = 8 * 1024 * 1024
UNSUPPORTED_MIME_PREFIXES = (
    "application/x-executable",
    "application/x-msdownload",
    "application/x-msdos-program",
    "application/x-sharedlib",
)
UNSUPPORTED_MIME_EXACT = frozenset(
    {"application/x-tar", "application/zip", "application/x-7z-compressed", "application/x-rar-compressed"}
)


def classify_attachment_state(
    *, path: str | None = None, size_bytes: int | None, mime_type: str | None, availability: Any = None
) -> str:
    if availability is not None:
        state = getattr(availability, "state", availability)
        state = getattr(state, "value", state)
        if state in {"missing", "unfetched", "unavailable", "unknown", "hash-mismatch", "unauthorized"}:
            return "missing-blob"
        if state != "available":
            return str(state)
    elif not path:
        return "missing-blob"
    if isinstance(mime_type, str):
        mime = mime_type.lower()
        if mime in UNSUPPORTED_MIME_EXACT or any(mime.startswith(prefix) for prefix in UNSUPPORTED_MIME_PREFIXES):
            return "unsupported-kind"
    if isinstance(size_bytes, int) and size_bytes > PREVIEW_SIZE_BUDGET:
        return "too-large"
    return "available"
