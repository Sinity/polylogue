"""Parser ingest flags shared by parsers and archive write paths."""

from __future__ import annotations

TEMPORARY_CHAT_INGEST_FLAG = "capture:temporary-chat"
DOM_FALLBACK_INGEST_FLAG = "capture:dom-fallback"
NATIVE_BROWSER_CAPTURE_INGEST_FLAG = "capture:browser-native-payload"
COMPACT_BROWSER_CAPTURE_INGEST_FLAG = "capture:browser-native-compact"

__all__ = [
    "DOM_FALLBACK_INGEST_FLAG",
    "COMPACT_BROWSER_CAPTURE_INGEST_FLAG",
    "NATIVE_BROWSER_CAPTURE_INGEST_FLAG",
    "TEMPORARY_CHAT_INGEST_FLAG",
]
