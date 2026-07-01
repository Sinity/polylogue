"""Project-reference normalization for archive SQL filters."""

from __future__ import annotations

import re
from collections.abc import Iterable

_CHATGPT_PROJECT_URL_RE = re.compile(r"/g/(?P<ref>g-p-[A-Za-z0-9-]+)(?:/|$)")
_CHATGPT_VISIBLE_SUFFIX_RE = re.compile(r"^(?P<base>g-p-[A-Za-z0-9]{8,})-[a-z]$")


def expand_project_refs(project_refs: Iterable[str]) -> tuple[str, ...]:
    """Return query aliases for provider project refs without mutating storage.

    ChatGPT browser URLs expose project refs as ``g-p-<id>-a`` while the backend
    payloads currently store ``conversation_template_id`` as ``g-p-<id>``.
    Filtering should accept both forms because both identify the same project
    from the operator's point of view.
    """

    expanded: list[str] = []
    seen: set[str] = set()
    for ref in project_refs:
        for candidate in _candidate_project_refs(ref):
            if candidate not in seen:
                expanded.append(candidate)
                seen.add(candidate)
    return tuple(expanded)


def _candidate_project_refs(ref: str) -> tuple[str, ...]:
    value = ref.strip()
    if not value:
        return ()
    url_match = _CHATGPT_PROJECT_URL_RE.search(value)
    primary = url_match.group("ref") if url_match else value
    candidates = [primary]
    suffix_match = _CHATGPT_VISIBLE_SUFFIX_RE.match(primary)
    if suffix_match:
        candidates.append(suffix_match.group("base"))
    return tuple(candidates)


__all__ = ["expand_project_refs"]
