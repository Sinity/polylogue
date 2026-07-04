"""Parser ingest flags shared by parsers and archive write paths."""

from __future__ import annotations

TEMPORARY_CHAT_INGEST_FLAG = "capture:temporary-chat"
DOM_FALLBACK_INGEST_FLAG = "capture:dom-fallback"

__all__ = ["DOM_FALLBACK_INGEST_FLAG", "TEMPORARY_CHAT_INGEST_FLAG"]
