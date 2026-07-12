"""Per-message semantic-card placement shared by every card-consuming surface.

``rendering/semantic_cards.py`` builds one ordered :class:`SemanticTranscript`
(pure, no archive access) that the CLI already renders to Markdown
(``cli/messages.py``). This module derives a second, equally pure projection
of that same transcript: a lookup from message id to the cards whose primary
evidence lives on that message, plus the set of messages whose entire content
was already absorbed into a card (a paired tool-result message, most often).

Surfaces that render one message at a time — the daemon web reader today —
consume this instead of re-deriving tool semantics from raw block flags. This
keeps CLI Markdown and web HTML two renderers over one shared registry
(``rendering/semantic_card_registry.py``) rather than two independent
tool-classification implementations drifting apart.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from polylogue.archive.message.models import Message
from polylogue.core.json import JSONDocument
from polylogue.rendering.semantic_cards import build_semantic_transcript


@dataclass(frozen=True, slots=True)
class SemanticCardPlacement:
    """Card documents and suppression, keyed by the message that owns them."""

    cards_by_message_id: Mapping[str, tuple[JSONDocument, ...]]
    suppressed_message_ids: frozenset[str]

    def cards_for(self, message_id: str) -> list[JSONDocument]:
        """Cards whose primary evidence is on *message_id*, in transcript order."""

        return list(self.cards_by_message_id.get(message_id, ()))

    def is_suppressed(self, message_id: str) -> bool:
        """Whether *message_id* is fully absorbed into another message's card.

        A shell/edit/task card pairs a tool-use message with its tool-result
        message by exact ``tool_id``. The card carries both message ids
        (``source.message_id`` and ``source.result_message_id``); the result
        message renders nothing of its own, or the operator would see the
        same evidence twice — once inside the card, once as a bare
        tool-result block.
        """

        return message_id in self.suppressed_message_ids


_EMPTY_PLACEMENT = SemanticCardPlacement(cards_by_message_id={}, suppressed_message_ids=frozenset())


def semantic_card_placement_for_messages(
    messages: Sequence[Message | Mapping[str, object] | object],
    *,
    session_id: str,
    provider_family: str | None = None,
) -> SemanticCardPlacement:
    """Index a session's semantic cards by the message that should render them.

    Accepts the same message shapes as
    :func:`polylogue.rendering.semantic_cards.build_semantic_transcript`
    (domain ``Message`` objects or message-shaped mappings) so DB-backed and
    archive-backed session reads can share one call.
    """

    if not messages:
        return _EMPTY_PLACEMENT
    transcript = build_semantic_transcript(
        messages,
        session_id=session_id,
        provider_family=provider_family,
    )
    cards_by_message: dict[str, list[JSONDocument]] = {}
    suppressed: set[str] = set()
    for card in transcript.cards:
        primary_id = card.source.message_id or card.source.result_message_id
        if primary_id is not None:
            cards_by_message.setdefault(primary_id, []).append(card.to_document())
        result_id = card.source.result_message_id
        if result_id is not None and result_id != primary_id:
            suppressed.add(result_id)
    if not cards_by_message and not suppressed:
        return _EMPTY_PLACEMENT
    return SemanticCardPlacement(
        cards_by_message_id={key: tuple(value) for key, value in cards_by_message.items()},
        suppressed_message_ids=frozenset(suppressed),
    )


__all__ = ["SemanticCardPlacement", "semantic_card_placement_for_messages"]
