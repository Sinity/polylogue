"""Automatic tag inference for session profiles.

Tags are prefixed strings like "project:sinex", "kind:debugging", "provider:claude-code".
They enable filtering and grouping without manual annotation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from polylogue.lib.session_profile import SessionProfile


def infer_tags(profile: "SessionProfile") -> tuple[str, ...]:
    """Infer structured tags from a session profile.

    Returns a sorted tuple of tag strings. Tags use the "prefix:value"
    convention for structured filtering (e.g., "project:sinex").
    """
    tags: list[str] = []

    # Provider tag
    tags.append(f"provider:{profile.provider}")

    # Project tags — capped at 3 to avoid noise on deeply multi-project sessions
    for project in list(profile.canonical_projects)[:3]:
        tags.append(f"project:{project}")

    # Work kind tag — dominant kind from work events
    if profile.work_events:
        from collections import Counter
        kind_counter: Counter[str] = Counter()
        for we in profile.work_events:
            kind_str = we.kind.value if hasattr(we.kind, "value") else str(we.kind)
            kind_counter[kind_str] += 1
        dominant_kind = kind_counter.most_common(1)[0][0]
        tags.append(f"kind:{dominant_kind}")

    # Continuation tags
    if profile.is_continuation:
        tags.append("continuation")
    if profile.continuation_depth >= 3:
        tags.append("deep-thread")

    # Multi-project sessions
    if len(profile.canonical_projects) > 1:
        tags.append("multi-project")

    # Cost tag — expensive sessions worth flagging
    if profile.total_cost_usd >= 1.0:
        tags.append("costly")

    return tuple(sorted(set(tags)))
