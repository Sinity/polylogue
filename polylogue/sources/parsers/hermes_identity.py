"""Shared profile-qualified session-identity join keys for the Hermes bridge.

Hermes evidence arrives from several independently-acquired artifact
families for the same logical install: the durable ``state.db`` snapshot
(``hermes_state.py``), NeMo Relay ATIF trajectory documents and raw ATOF
event streams (``hermes_spans.py``), and the lifecycle/verification spool
parsers. Each artifact only ever carries the *raw* Hermes session id (the
value Hermes itself assigns) -- there is no cross-artifact install
identifier in the wire formats. Two separate Hermes installs (profiles) can
legitimately reuse the same raw session id, so a join key built from the raw
id alone silently collapses evidence from different installs onto one
archive session identity.

This module is the single source of truth for turning "the directory a raw
Hermes artifact file lives under" into a stable, hashed profile qualifier,
and for building/parsing the qualified session id
(``<raw_session_id>@profile-<profile_key>``) every Hermes parser uses. It
existed previously as private helpers duplicated inside ``hermes_state.py``;
centralizing it here is what lets ``hermes_spans.py`` (fs1.14) qualify ATIF/
ATOF observer-evidence session identity with the *same* key the state.db
parser computes for the conversational session it correlates with, instead
of inventing a second, incompatible scheme.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

__all__ = ["profile_key", "qualified_session_id", "split_qualified_session_id"]


def profile_key(profile_root: Path) -> str:
    """Return a stable, hashed qualifier for a Hermes install root directory.

    Raw profile paths are never exposed in archive identity -- only this
    truncated SHA-256 digest of the normalized (expanded, resolved) path.
    """
    normalized = str(profile_root.expanduser().resolve(strict=False))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:12]


def qualified_session_id(raw_session_id: str, key: str) -> str:
    """Return the profile-qualified session id for a raw Hermes session id."""
    return f"{raw_session_id}@profile-{key}"


def split_qualified_session_id(qualified_id: str) -> tuple[str, str | None]:
    """Split a possibly-qualified session id into ``(raw_id, profile_key)``.

    Returns ``(qualified_id, None)`` unchanged when no ``@profile-`` marker is
    present (legacy/unqualified identity) -- callers must not silently invent
    a profile key that was never asserted by a producer.
    """
    raw_id, marker, key = qualified_id.partition("@profile-")
    return (raw_id, key) if marker else (qualified_id, None)
