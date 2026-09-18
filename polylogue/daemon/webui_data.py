"""Typed data envelopes used by the daemon WebUI routes.

This module contains transport-neutral shaping helpers. HTML, JavaScript, and
browser presentation belong to the typed WebUI package.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, cast

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


def _attachment_name(attachment: Any) -> str:
    name = getattr(attachment, "name", None)
    if isinstance(name, str) and name.strip():
        return name
    return str(getattr(attachment, "id", None) or getattr(attachment, "attachment_id", None) or "")


def _availability_payload(availability: Any) -> object:
    if availability is None:
        return None
    if hasattr(availability, "model_dump"):
        return cast(Any, availability).model_dump(mode="json")
    if is_dataclass(availability):
        return asdict(cast(Any, availability))
    return availability


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


def attachment_to_envelope(attachment: Any, *, session_id: str, message_id: str | None = None) -> dict[str, object]:
    mime_type = getattr(attachment, "mime_type", None)
    size_bytes = getattr(attachment, "size_bytes", None)
    availability = getattr(attachment, "availability", None)
    return {
        "attachment_id": str(getattr(attachment, "id", None) or getattr(attachment, "attachment_id", "") or ""),
        "session_id": session_id,
        "message_id": str(message_id) if message_id is not None else None,
        "name": _attachment_name(attachment),
        "mime_type": mime_type if isinstance(mime_type, str) else None,
        "size_bytes": int(size_bytes) if isinstance(size_bytes, int) else None,
        "path": None,
        "state": classify_attachment_state(size_bytes=size_bytes, mime_type=mime_type, availability=availability),
        "availability": _availability_payload(availability),
        "can_fetch": bool(getattr(availability, "can_fetch", False)),
    }


@dataclass(frozen=True)
class LibraryEntry:
    envelope: dict[str, object]
    session_title: str
    origin: str | None
    message_anchor: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            **self.envelope,
            "session_title": self.session_title,
            "origin": self.origin,
            "message_anchor": self.message_anchor,
        }


def build_library_payload(
    entries: Iterable[LibraryEntry],
    *,
    total: int | None,
    total_is_exact: bool = True,
    matched_so_far: int | None = None,
) -> dict[str, object]:
    """Shape the attachment-library page with an honest total.

    polylogue-q54dt: the producing walk stops as soon as the page is full, so
    its running counter is the page size, not the archive's match count.
    Publishing that counter as ``total`` told a reader with 10,000 matches
    that it had seen all 200 of them. An unknown total is published as
    ``null`` with ``total_is_exact=false`` and the lower bound the walk did
    establish, never as a plausible-looking figure.
    """

    payload: dict[str, object] = {
        "items": [entry.to_dict() for entry in entries],
        "total": total,
        "total_is_exact": total_is_exact,
    }
    if matched_so_far is not None:
        payload["total_lower_bound"] = matched_so_far
    return payload


@dataclass(frozen=True)
class PasteBrowserEntry:
    session_id: str
    session_title: str
    origin: str | None
    message_id: str
    message_anchor: str
    role: str
    timestamp: str | None
    word_count: int
    snippet: str
    paste_spans: list[dict[str, object]]
    has_diff: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "session_id": self.session_id,
            "session_title": self.session_title,
            "origin": self.origin,
            "message_id": self.message_id,
            "message_anchor": self.message_anchor,
            "role": self.role,
            "timestamp": self.timestamp,
            "word_count": self.word_count,
            "snippet": self.snippet,
            "paste_spans": self.paste_spans,
            "has_diff": self.has_diff,
        }


def build_paste_browser_payload(
    entries: Iterable[PasteBrowserEntry],
    *,
    total: int | None,
    total_is_exact: bool = True,
    matched_so_far: int | None = None,
) -> dict[str, object]:
    """Shape the paste-browser page with an honest total (polylogue-q54dt)."""

    payload: dict[str, object] = {
        "items": [entry.to_dict() for entry in entries],
        "total": total,
        "total_is_exact": total_is_exact,
    }
    if matched_so_far is not None:
        payload["total_lower_bound"] = matched_so_far
    return payload


def detect_paste_spans(text: str) -> list[dict[str, object]]:
    """Return conservative unified-diff spans embedded in message text."""
    import re

    if not text or not re.search(r"^@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@", text, re.MULTILINE):
        return []
    lines = text.split("\n")
    offsets: list[int] = []
    cursor = 0
    for line in lines:
        offsets.append(cursor)
        cursor += len(line) + 1
    spans: list[dict[str, object]] = []
    for index, line in enumerate(lines):
        if not re.match(r"^@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@", line):
            continue
        end = index
        while end + 1 < len(lines) and (not lines[end + 1] or lines[end + 1].startswith(("+", "-", " ", "\\", "@@"))):
            end += 1
        spans.append(
            {"kind": "diff", "start": offsets[index], "end": offsets[end] + len(lines[end]), "confidence": 0.95}
        )
    return spans


def envelope_paste_spans(text: str | None, *, has_paste: bool) -> list[dict[str, object]]:
    """Return the localizable paste spans the reader can highlight.

    ``has_paste`` is the stored per-message evidence flag; this derivation
    only localizes unified-diff hunks. The two disagree for marker, size,
    base64 and fence pastes, which is why callers must publish
    :func:`paste_span_localization` alongside the list rather than let an
    empty list read as "this message contains no paste" (polylogue-7mgx).
    """

    return detect_paste_spans(text or "")


#: How to read an ``envelope_paste_spans`` result against the stored flag.
#:
#: ``localized`` — spans were derived and can be highlighted.
#: ``none`` — the message carries no paste evidence and no spans.
#: ``unlocalized`` — stored evidence says this message contains a paste that
#: the reader's diff derivation cannot place. The empty span list is a gap,
#: not an absence, and the reader must not render it as "no paste here".
PasteSpanLocalization = str


def paste_span_localization(spans: list[dict[str, object]], *, has_paste: bool) -> str:
    """Name what an empty span list means for this message."""

    if spans:
        return "localized"
    return "unlocalized" if has_paste else "none"


def snippet_for_paste(text: str, spans: list[dict[str, object]], *, limit: int = 160) -> str:
    if spans:
        start = spans[0].get("start", 0)
        end = spans[0].get("end", len(text))
        body = (
            text[int(start) : int(end)] if isinstance(start, (int, float)) and isinstance(end, (int, float)) else text
        )
    else:
        body = text
    first_line = body.strip().split("\n", 1)[0] if body else ""
    return first_line if len(first_line) <= limit else first_line[:limit] + "\u2026"
