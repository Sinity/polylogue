"""Parser ingest flags shared by parsers and archive write paths."""

from __future__ import annotations

TEMPORARY_CHAT_INGEST_FLAG = "capture:temporary-chat"
DOM_FALLBACK_INGEST_FLAG = "capture:dom-fallback"
NATIVE_BROWSER_CAPTURE_INGEST_FLAG = "capture:browser-native-payload"
COMPACT_BROWSER_CAPTURE_INGEST_FLAG = "capture:browser-native-compact"

# Both flags mark "this session's content came from a native (non-DOM-fallback)
# browser payload" -- `browser_capture.py` tags a session with one or the
# other depending on whether the capture used the compact or full native
# shape (see `_is_compact_native_capture`), never both. Any caller deriving
# "does this session have native-payload precedence" must check both, or a
# compact capture silently misclassifies as plain, non-browser-capture
# content (polylogue-z1c6 review follow-up).
NATIVE_BROWSER_CAPTURE_FLAGS = (NATIVE_BROWSER_CAPTURE_INGEST_FLAG, COMPACT_BROWSER_CAPTURE_INGEST_FLAG)

__all__ = [
    "DOM_FALLBACK_INGEST_FLAG",
    "COMPACT_BROWSER_CAPTURE_INGEST_FLAG",
    "NATIVE_BROWSER_CAPTURE_FLAGS",
    "NATIVE_BROWSER_CAPTURE_INGEST_FLAG",
    "TEMPORARY_CHAT_INGEST_FLAG",
]
