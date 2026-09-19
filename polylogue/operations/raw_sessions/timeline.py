from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any

from .page import CompactJSONPage
from .sessions import OpaqueSessionCursor, SessionError, SessionLogService, _ObservedTimeline
from .sources import (
    LOCAL_AUTHORITY,
    UNAVAILABLE_SOURCES,
    resolve_providers,
)


class TimelineError(ValueError):
    pass


_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


class TimelineService:
    def __init__(self, sessions: SessionLogService):
        self.sessions = sessions

    @staticmethod
    def _timestamp(value: Any, name: str) -> int | None:
        if value is None:
            return None
        if not isinstance(value, str) or not value or len(value) > 64:
            raise TimelineError(f"{name} must be an RFC 3339 timestamp")
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise TimelineError(f"{name} must be an RFC 3339 timestamp") from exc
        if parsed.tzinfo is None:
            raise TimelineError(f"{name} must include a timezone")
        delta = parsed.astimezone(UTC) - _EPOCH
        return delta.days * 86_400_000_000_000 + delta.seconds * 1_000_000_000 + delta.microseconds * 1_000

    def query(
        self,
        start: str | None = None,
        end: str | None = None,
        query: Any = None,
        providers: list[str] | None = None,
        limit: Any = 100,
        *,
        cursor: str | None = None,
        cursor_key: bytes | None = None,
        scan_bytes: int = 8 * 1_024 * 1_024,
    ) -> dict[str, Any]:
        start_ns = self._timestamp(start, "start")
        end_ns = self._timestamp(end, "end")
        if start_ns is not None and end_ns is not None and start_ns > end_ns:
            raise TimelineError("start must not be after end")
        if query is not None and (not isinstance(query, str) or not query or len(query) > 1_000):
            raise TimelineError("query must contain 1-1000 characters")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise TimelineError("limit must be positive")
        requested = resolve_providers(providers, error=TimelineError, noun="timeline")
        if cursor_key is None and cursor is not None:
            raise TimelineError("timeline continuation cursor is unavailable")
        # Direct service users do not own the gateway result key.  Keep a
        # process-local equivalent only for completing this call; gateway
        # actions always supply the persistent, principal-scoped key.
        effective_cursor_key = cursor_key or hashlib.sha256(f"timeline-direct:{self.sessions.scope}".encode()).digest()
        scope = {
            "principal": self.sessions.scope,
            "start": start_ns,
            "end": end_ns,
            "query": query,
            "providers": requested,
        }
        if cursor is None:
            state: dict[str, Any] = {"current": {}, "pending": {}, "done": []}
        else:
            state = OpaqueSessionCursor(self.sessions.scope, effective_cursor_key, "timeline-query").decode(
                cursor, scope
            )
            if (
                set(state) != {"current", "pending", "done"}
                or not isinstance(state["current"], dict)
                or not isinstance(state["pending"], dict)
                or not isinstance(state["done"], list)
            ):
                raise TimelineError("timeline continuation cursor is malformed")

        sources: list[dict[str, Any]] = []
        raw = [provider for provider in requested if provider not in UNAVAILABLE_SOURCES]
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
                state["done"].append(provider)
            else:
                sources.append(
                    {
                        "source": provider,
                        "authority": LOCAL_AUTHORITY,
                        "availability": "available",
                        "coverage": {
                            "scanned_bytes": 0,
                            "truncated": provider not in state["done"],
                        },
                    }
                )

        # Every refill in this request shares the same metadata observation.
        # New external requests create new observations; this is not a cache.
        readers: dict[str, _ObservedTimeline] = {}

        def load_head(provider: str, after: str | None) -> dict[str, Any]:
            try:
                reader = readers.get(provider)
                if reader is None:
                    reader = self.sessions.observe_timeline(provider, start_ns, end_ns, query)
                    readers[provider] = reader
                result = reader.page(1, cursor=after, cursor_key=effective_cursor_key, scan_bytes=scan_bytes)
            except SessionError as exc:
                raise TimelineError(str(exc)) from exc
            if result["entries"]:
                state["pending"][provider] = {"entry": result["entries"][0], "after": result["next_cursor"]}
            elif result["next_cursor"] is None:
                state["done"].append(provider)
            else:
                state["current"][provider] = result["next_cursor"]
            return result

        # One look-ahead per provider is sufficient while each head is known.
        for provider in raw:
            if provider in state["done"] or provider in state["pending"]:
                continue
            result = load_head(provider, state["current"].get(provider))
            source_row = next(row for row in sources if row["source"] == provider)
            source_row["coverage"] = {
                "scanned_bytes": result["scanned_bytes"],
                "truncated": result["truncated"],
            }

        page = CompactJSONPage(max(1, self.sessions.max_result_bytes - 16_384))
        entries = page.items
        while state["pending"] and len(entries) < limit:
            # A bounded filtered scan may advance without finding its next
            # head. Do not emit an older head until every live source has a
            # head or is exhausted. A short or empty continued page is valid.
            if any(provider not in state["done"] and provider not in state["pending"] for provider in raw):
                break
            provider, pending = max(state["pending"].items(), key=lambda row: row[1]["entry"]["mtime_ns"])
            entry = pending["entry"]
            candidate = {
                "source": provider,
                "authority": LOCAL_AUTHORITY,
                "object_reference": entry["reference"],
                **{key: value for key, value in entry.items() if key != "reference"},
            }
            if not page.try_append(candidate):
                break
            state["pending"].pop(provider)
            if pending["after"] is None:
                state["done"].append(provider)
            else:
                state["current"][provider] = pending["after"]
                # Reuse the request-owned observation for the next head.
                if query is None:
                    load_head(provider, pending["after"])

        more = bool(state["pending"]) or any(provider not in state["done"] for provider in raw)
        next_cursor = None
        if more:
            next_cursor = OpaqueSessionCursor(self.sessions.scope, effective_cursor_key, "timeline-query").encode(
                scope, state
            )
        response = {
            "time_basis": "session-file-mtime",
            "start": start,
            "end": end,
            "query": query,
            "sources": sources,
            "entries": entries,
            "truncated": more,
            "next_cursor": next_cursor,
        }
        if len(json.dumps(response, sort_keys=True, separators=(",", ":")).encode()) > self.sessions.max_result_bytes:
            # A tiny direct-service response bound may not even accommodate a
            # signed continuation plus the provenance envelope.  Preserve the
            # honest coverage bit rather than returning an oversized payload.
            response["next_cursor"] = None
        if len(json.dumps(response, sort_keys=True, separators=(",", ":")).encode()) > self.sessions.max_result_bytes:
            return {
                "available": False,
                "reason": "timeline response metadata exceeded response bound",
            }
        return response
