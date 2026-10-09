"""Canonical resolution of explicit session selection references."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from polylogue.core.errors import SessionNotFoundError

if TYPE_CHECKING:
    from polylogue.archive.query.spec import SessionQuerySpec
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore


def resolved_scope_spec(spec: SessionQuerySpec, *, archive: ArchiveStore) -> SessionQuerySpec:
    """Resolve an explicit session scope to a full session id before filtering.

    ``--id`` accepts any reference spelling the archive can resolve — a native
    id, a prefix, a full ``origin:native`` id, or its outer ``session:`` namespace.
    The SQL filters compare against
    the full ``session_id``, so an unresolved spelling silently scopes the page
    to nothing and reports an empty result instead of the session the operator
    named. Missing scopes raise SessionNotFoundError; ambiguity remains a
    ValueError. Query-set callers may retain a missing exact-ID predicate.
    """

    scope = spec.session_id
    if not scope:
        return spec
    try:
        resolved = archive.resolve_session_id(scope.removeprefix("session:"))
    except KeyError as exc:
        raise SessionNotFoundError(f"session not found: {scope}") from exc
    return spec if resolved == scope else replace(spec, session_id=resolved)
