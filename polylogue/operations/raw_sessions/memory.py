from __future__ import annotations

from typing import Any

from .sessions import SessionError, SessionLogService
from .sources import (
    LOCAL_AUTHORITY,
    UNAVAILABLE_SOURCES,
    resolve_providers,
)


class MemoryError(SessionError):
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
        configured = [source.provider for source in self.sessions.sources]
        requested = resolve_providers(
            providers if providers is not None else configured, error=MemoryError, noun="memory"
        )
        if source_cursors is not None and (
            not isinstance(source_cursors, dict)
            or set(source_cursors) - {"claude-code", "codex"}
            or any(value is not None and not isinstance(value, str) for value in source_cursors.values())
        ):
            raise MemoryError("source_cursors must map raw providers to continuation tokens")
        source_cursors = source_cursors or {}
        if providers is None:
            requested = list(dict.fromkeys((*requested, *source_cursors.keys())))
        sources: list[dict[str, Any]] = []
        matches: list[dict[str, Any]] = []
        next_cursors: dict[str, str | None] = {}
        gaps: list[str] = []
        # Files each provider skipped on earlier pages (plus this page for a
        # provider that finished here). Reported once, on the terminal page.
        owed_skips: dict[str, int] = {}
        # Completion tokens resumed this page; reissuing them keeps one
        # retained snapshot per finished provider instead of one per page.
        reusable: dict[str, str] = {}
        # A provider that stopped early without a cursor (its population could
        # not be retained) would read as finished in the returned map.
        continuation_lost = False
        earlier_skips = 0
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
            source = next((candidate for candidate in self.sessions.sources if candidate.provider == provider), None)
            if source is None:
                sources.append(
                    {
                        "source": provider,
                        "authority": LOCAL_AUTHORITY,
                        "availability": "unavailable",
                        "reason": "session source is not configured",
                    }
                )
                continue
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
                    summarize_skipped=False,
                )
            except SessionError as exc:
                if type(exc) is not SessionError:
                    # Typed outcomes (stale, retryable) must reach the caller
                    # with their code; a generic wrapper would hide them.
                    raise
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
                    "offset": row["offset"],
                    "text": row["text"],
                    "source_observation": row["source_observation"],
                    "match_offset": row["match_offset"],
                    "match_end": row["match_end"],
                }
                for row in result["matches"]
            )
            gaps.extend(f"{provider}: {gap}" for gap in result.get("gaps", ()))
            next_cursors[provider] = result["next_cursor"]
            if result.get("continuation_lost") or (result["truncated"] and result["next_cursor"] is None):
                continuation_lost = True
            earlier_skips += result["skipped_earlier"]
            if result["next_cursor"] is None and result["skipped_earlier"] + result["skipped_now"]:
                owed_skips[provider] = result["skipped_earlier"] + result["skipped_now"]
                if result.get("completion_cursor"):
                    reusable[provider] = result["completion_cursor"]
            page_full = result["next_cursor"] is not None or len(matches) >= limit
        # More pages exist only while some provider has a live cursor or has not
        # started; reaching ``limit`` exactly on a provider's last match is not truncation.
        truncated = any(source.get("coverage", {}).get("truncated") is True for source in sources)
        if continuation_lost:
            next_cursors = {}  # no replayable map: every slot would read as finished
        elif truncated and cursor_key is not None:
            # A finished provider's skips must survive to the fan-out's terminal
            # page, which a ``None`` slot would forget.
            for provider, owed in owed_skips.items():
                token = reusable.get(provider) or self.sessions.completed_skips_token(
                    provider, query, owed, cursor_key=cursor_key
                )
                if token is None:
                    # The count cannot be retained (for example ENOSPC). A
                    # continuation that would forget it is worse than none.
                    gaps.append(
                        f"{provider}: continuation unavailable: its {owed} skipped files could not be retained; "
                        "restart the search"
                    )
                    next_cursors = {}  # no replayable map: every slot would read as finished
                    break
                next_cursors[provider] = token
        elif not truncated and earlier_skips:
            gaps.append(f"{earlier_skips} selected files were skipped on earlier pages of this continuation")
        return {
            "query": query,
            "sources": sources,
            "matches": matches,
            "truncated": truncated,
            "next_cursors": next_cursors or None,
            "gaps": gaps,
        }

    def get(
        self,
        reference: Any,
        offset: int = 0,
        max_bytes: int = 64_000,
        *,
        expected_observation: str | None = None,
    ) -> dict[str, Any]:
        if not isinstance(reference, str) or not reference or len(reference) > 8_192:
            raise MemoryError("reference must be a bounded non-empty string")
        try:
            result = self.sessions.read(reference, offset, max_bytes, expected_observation=expected_observation)
        except SessionError as exc:
            error = MemoryError(str(exc))
            error.code = exc.code
            raise error from exc
        return {
            "source": result["provider"],
            "authority": LOCAL_AUTHORITY,
            "availability": "available",
            "object_reference": result["reference"],
            "observation": result["observation"],
            "consistency": result["consistency"],
            "next_offset": result["next_offset"],
            "offset": result["offset"],
            "bytes": result["bytes"],
            "truncated": result["truncated"],
            "content": result["content"],
        }
