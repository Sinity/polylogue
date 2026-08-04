"""message_identity_hash's fallback for id-less messages must not read array position.

polylogue-gysk3: the attachment-class bug (a synthetic identity seeded partly
by array index, unstable across export vintages that reorder/insert array
entries) was found and fixed for attachments (polylogue-hith/-d8al) by
excluding the (real-or-synthetic) attachment id from identity entirely and
deriving it from stable content fields instead (message_id, name, mime_type).

`message_identity_hash` has no separate field to fall back to -- a message's
provider id (or its absence) IS the sole identity input by construction. Its
one addressable-within-`pipeline/ids.py` instance is the local fallback that
used to fire when a parser could not populate `provider_message_id`:
previously `f"msg-{index}"`, positionally-derived exactly like the old
attachment synthetic id. This file proves the fix: two parses of the same
id-less message in a different array position must resolve to the same
message-content identity when a provider timestamp is available to anchor
on, instead of silently comparing wrong pairs as if their (position-shifted)
fallback ids matched.
"""

from __future__ import annotations

from polylogue.archive.message.roles import Role
from polylogue.core.enums import Provider
from polylogue.pipeline.ids import session_revision_projection
from polylogue.sources.parsers.base import ParsedMessage, ParsedSession


def _session(messages: list[ParsedMessage]) -> ParsedSession:
    return ParsedSession(
        source_name=Provider.CHATGPT,
        provider_session_id="conv-1",
        title="Test",
        created_at="2024-01-01T00:00:00Z",
        updated_at="2024-01-01T00:00:00Z",
        messages=messages,
        attachments=[],
    )


def _with_id(provider_message_id: str, role: str, text: str, timestamp: str) -> ParsedMessage:
    return ParsedMessage(
        provider_message_id=provider_message_id, role=Role.normalize(role), text=text, timestamp=timestamp
    )


def _id_less(role: str, text: str, timestamp: str | None) -> ParsedMessage:
    """A message the parser could not assign a native id to (empty string)."""
    return ParsedMessage(provider_message_id="", role=Role.normalize(role), text=text, timestamp=timestamp)


def test_id_less_message_content_identity_is_stable_across_reorder() -> None:
    """Same message set, different array order -- an id-less message with a
    provider timestamp must still resolve to the same (identity, content)
    pair in both orderings, not shift with its array position.
    """
    anchored = _id_less("assistant", "hello there", "2024-01-01T00:01:00Z")
    keyed = _with_id("m1", "user", "hi", "2024-01-01T00:00:00Z")

    forward = session_revision_projection(_session([keyed, anchored]))
    reordered = session_revision_projection(_session([anchored, keyed]))

    assert forward.message_contents == reordered.message_contents


def test_id_less_messages_with_distinct_timestamps_do_not_collide() -> None:
    """Two different id-less messages must not collapse to one identity."""
    a = _id_less("assistant", "first", "2024-01-01T00:01:00Z")
    b = _id_less("assistant", "second", "2024-01-01T00:02:00Z")

    projection = session_revision_projection(_session([a, b]))

    assert len(projection.message_contents) == 2


def test_id_less_and_timestamp_less_message_uses_content_anchor() -> None:
    bare = _id_less("user", "x", None)
    keyed = _with_id("m1", "assistant", "y", "2024-01-01T00:00:00Z")

    forward = session_revision_projection(_session([keyed, bare]))
    reordered = session_revision_projection(_session([bare, keyed]))

    assert forward.message_contents == reordered.message_contents
