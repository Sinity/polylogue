"""Deterministic archive identity helpers for the archive rewrite.

These helpers are the executable reference for #1743's generated-ID law.
The archive SQLite schema will compute these IDs with generated columns; Python
uses this module as the test oracle and for pre-DDL contract checks.
"""

from __future__ import annotations


def _required_origin(value: str) -> str:
    candidate = value.strip()
    if not candidate:
        raise ValueError("origin cannot be empty")
    if ":" in candidate:
        raise ValueError("origin cannot contain ':'")
    return candidate


def _required_text(field: str, value: str) -> str:
    candidate = value.strip()
    if not candidate:
        raise ValueError(f"{field} cannot be empty")
    return candidate


def _required_non_negative(field: str, value: int) -> int:
    if value < 0:
        raise ValueError(f"{field} cannot be negative")
    return value


def session_id(origin: str, native_id: str) -> str:
    """Return archive ``session_id``: ``origin:native_id``."""
    return f"{_required_origin(origin)}:{_required_text('native_id', native_id)}"


def message_local_id(
    native_id: str | None,
    *,
    content_identity: str | None = None,
    content_occurrence: int = 0,
) -> str:
    """Return the message-local identity component.

    Provider-native message IDs and the content-derived fallback occupy
    disjoint tagged namespaces (``n:`` and ``c:``), so a provider id can never
    collide with a fallback identity while both components stay opaque.

    The fallback is derived from the message's declared semantic fields
    (``pipeline.ids.message_content_identity``), never from its ordinal. An
    ordinal fallback renumbers every later message when an upstream export
    gains or loses one, silently re-resolving a durable ``user.db`` reference
    onto a *different* message -- a mis-resolution no existence check can see
    (polylogue-eqsri). ``content_occurrence`` separates messages whose declared
    semantics are byte-identical; it counts only within one digest, so an
    unrelated insertion never moves it.
    """
    if native_id is not None and native_id.strip():
        return f"n:{_required_text('message native_id', native_id)}"
    identity = _required_text("content_identity", content_identity or "")
    return f"c:{identity}.{_required_non_negative('content_occurrence', content_occurrence)}"


def message_id(
    parent_session_id: str,
    native_id: str | None,
    *,
    content_identity: str | None = None,
    content_occurrence: int = 0,
) -> str:
    """Return archive ``message_id`` under a session."""
    session = _required_text("session_id", parent_session_id)
    local = message_local_id(
        native_id,
        content_identity=content_identity,
        content_occurrence=content_occurrence,
    )
    return f"{session}:{local}"


def split_message_local_id(stored_message_id: str) -> tuple[str | None, str | None, int]:
    """Invert ``message_local_id`` on a stored ``message_id``.

    Returns ``(native_id, content_identity, content_occurrence)`` with exactly
    one of the first two set, so a surface holding a stored id can restate the
    identity it was built from without re-deriving it from a position.
    ``ValueError`` if the id carries neither tagged namespace.
    """
    native_marker = ":n:"
    content_marker = ":c:"
    native_at = stored_message_id.rfind(native_marker)
    content_at = stored_message_id.rfind(content_marker)
    if native_at > content_at:
        return stored_message_id[native_at + len(native_marker) :], None, 0
    if content_at >= 0:
        local = stored_message_id[content_at + len(content_marker) :]
        identity, _, occurrence = local.rpartition(".")
        if identity and occurrence.isdigit():
            return None, identity, int(occurrence)
    raise ValueError(f"not a tagged archive message id: {stored_message_id!r}")


def block_id(parent_message_id: str, *, position: int) -> str:
    """Return archive ``block_id`` under a message."""
    message = _required_text("message_id", parent_message_id)
    return f"{message}:{_required_non_negative('position', position)}"


#: The coordinate a session's transcript order is stated in. Content position,
#: never the observed clock: ``occurred_at_ms`` is non-monotonic against
#: position on every origin, so it cannot decide a conversation's order. This
#: pair is the ``messages`` primary key under ``session_id``, so it is a total
#: order within one session and admits no tiebreaker.
TRANSCRIPT_ORDER_COLUMNS: tuple[str, ...] = ("position", "variant_index")


def transcript_order_sql(alias: str | None = None, *, descending: bool = False) -> str:
    """Return the ``ORDER BY`` fragment stating a session's transcript order.

    Every read that states a session's message order -- lineage composition,
    pagination, the keyset stream, batched reads, edge windows, the query
    surface and the streaming markdown export -- orders by this fragment, so
    a caller may take a total from one route and slice another.
    ``idx_messages_session_position`` serves it without a temp sort.
    """
    prefix = f"{alias}." if alias else ""
    direction = " DESC" if descending else ""
    return ", ".join(f"{prefix}{column}{direction}" for column in TRANSCRIPT_ORDER_COLUMNS)


__all__ = [
    "TRANSCRIPT_ORDER_COLUMNS",
    "block_id",
    "message_id",
    "message_local_id",
    "session_id",
    "split_message_local_id",
    "transcript_order_sql",
]
