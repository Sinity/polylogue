from __future__ import annotations

from typing import Any

from .sessions import SessionError, SessionLogService
from .sources import (
    LOCAL_AUTHORITY,
    UNAVAILABLE_SOURCES,
    resolve_providers,
)


class MemoryError(ValueError):
    pass


class MemoryService:
    def __init__(self, sessions: SessionLogService):
        self.sessions = sessions

    @staticmethod
    def _query(value: Any) -> str:
        if not isinstance(value, str) or not value or len(value) > 1_000:
            raise MemoryError("query must contain 1-1000 characters")
        return value

    def search(
        self,
        query: str,
        providers: list[str] | None = None,
        limit: Any = 100,
        *,
        source_cursors: Any = None,
        cursor_key: bytes | None = None,
        scan_bytes: int = 8 * 1_024 * 1_024,
    ) -> dict[str, Any]:
        query = self._query(query)
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise MemoryError("limit must be positive")
        requested = resolve_providers(providers, error=MemoryError, noun="memory")
        if source_cursors is not None and (
            not isinstance(source_cursors, dict)
            or set(source_cursors) - {"claude-code", "codex"}
            or any(value is not None and not isinstance(value, str) for value in source_cursors.values())
        ):
            raise MemoryError("source_cursors must map raw providers to continuation tokens")
        source_cursors = source_cursors or {}
        sources: list[dict[str, Any]] = []
        matches: list[dict[str, Any]] = []
        next_cursors: dict[str, str | None] = {}
        page_full = False
        for provider in requested:
            if provider in UNAVAILABLE_SOURCES:
                sources.append(
                    {
                        "source": provider,
                        "authority": "upstream",
                        "availability": "unavailable",
                        "reason": UNAVAILABLE_SOURCES[provider],
                    }
                )
                continue
            source = next(candidate for candidate in self.sessions.sources if candidate.provider == provider)
            if not source.root.is_dir():
                sources.append(
                    {
                        "source": provider,
                        "authority": LOCAL_AUTHORITY,
                        "availability": "unavailable",
                        "reason": "session source directory is unavailable",
                    }
                )
                continue
            if provider in source_cursors and source_cursors[provider] is None:
                sources.append(
                    {
                        "source": provider,
                        "authority": LOCAL_AUTHORITY,
                        "availability": "available",
                        "coverage": {"scanned_bytes": 0, "truncated": False},
                    }
                )
                next_cursors[provider] = None
                continue
            if page_full:
                sources.append(
                    {
                        "source": provider,
                        "authority": LOCAL_AUTHORITY,
                        "availability": "available",
                        "coverage": {
                            "scanned_bytes": 0,
                            "truncated": True,
                            "pending": True,
                        },
                    }
                )
                continue
            try:
                result = self.sessions.search(
                    provider,
                    query,
                    limit - len(matches),
                    cursor=source_cursors.get(provider),
                    cursor_key=cursor_key,
                    scan_bytes=scan_bytes,
                )
            except SessionError as exc:
                raise MemoryError(str(exc)) from exc
            sources.append(
                {
                    "source": provider,
                    "authority": LOCAL_AUTHORITY,
                    "availability": "available",
                    "coverage": {
                        "scanned_bytes": result["scanned_bytes"],
                        "truncated": result["truncated"],
                    },
                }
            )
            matches.extend(
                {
                    "source": provider,
                    "authority": LOCAL_AUTHORITY,
                    "object_reference": row["reference"],
                    "line": row["line"],
                    "text": row["text"],
                    "source_observation": row["source_observation"],
                }
                for row in result["matches"]
            )
            next_cursors[provider] = result["next_cursor"]
            page_full = result["next_cursor"] is not None or len(matches) >= limit
        return {
            "query": query,
            "sources": sources,
            "matches": matches,
            "truncated": page_full or any(source.get("coverage", {}).get("truncated") is True for source in sources),
            "next_cursors": next_cursors or None,
        }

    def get(self, reference: Any, offset: int = 0, max_bytes: int = 64_000) -> dict[str, Any]:
        if not isinstance(reference, str) or not reference or len(reference) > 8_192:
            raise MemoryError("reference must be a bounded non-empty string")
        try:
            result = self.sessions.read(reference, offset, max_bytes)
        except SessionError as exc:
            raise MemoryError(str(exc)) from exc
        return {
            "source": result["provider"],
            "authority": LOCAL_AUTHORITY,
            "availability": "available",
            "object_reference": result["reference"],
            "offset": result["offset"],
            "bytes": result["bytes"],
            "truncated": result["truncated"],
            "content": result["content"],
        }
