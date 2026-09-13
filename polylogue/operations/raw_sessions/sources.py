"""The source fan-out shared by the memory and timeline services.

Both walk the same requested-source list: an upstream this host does not serve
is reported unavailable with its standing reason, a session provider is fetched
through `SessionLogService`, and every source yields exactly one provenance
row. Each provider may contribute up to the caller's limit; callers apply
the global limit after merging so one provider can fill the result.

Row mapping and error translation stay with the caller: memory rows and
timeline entries have different shapes, and each service raises its own error
class.
"""

from __future__ import annotations

from typing import Any

RAW_PROVIDERS = ("claude-code", "codex")
UNAVAILABLE_SOURCES = {
    "polylogue": "use indexed session operations for archive coverage",
    "sinex": "raw fallback does not query Sinex",
    "lynchpin": "raw fallback does not query Lynchpin",
}
LOCAL_AUTHORITY = "original-local-session-jsonl"


def resolve_providers(providers: Any, *, error: type[Exception], noun: str) -> list[str]:
    known = {*RAW_PROVIDERS, *UNAVAILABLE_SOURCES}
    if providers is None:
        return list(RAW_PROVIDERS)
    if not isinstance(providers, list) or not providers or any(not isinstance(provider, str) for provider in providers):
        raise error("providers must be a non-empty list of source names")
    unknown = sorted(set(providers) - known)
    if unknown:
        raise error(f"unknown {noun} source(s): {unknown}")
    return list(dict.fromkeys(providers))
