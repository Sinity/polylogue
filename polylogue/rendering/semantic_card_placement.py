"""Per-message placement of one shared semantic transcript document.

The pure renderer emits ordered prose, notices, cards, and session-level
lineage. Message-oriented surfaces consume this placement rather than
reclassifying raw roles, block flags, or prose at the presentation leaf.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from polylogue.core.json import JSONDocument
from polylogue.rendering.semantic_card_models import LineageDescriptor
from polylogue.rendering.semantic_cards import build_semantic_transcript


@dataclass(frozen=True, slots=True)
class SemanticCardPlacement:
    """Semantic entry documents and compatibility card views by message id."""

    entries_by_message_id: Mapping[str, tuple[JSONDocument, ...]]
    session_entries: tuple[JSONDocument, ...]
    cards_by_message_id: Mapping[str, tuple[JSONDocument, ...]]
    suppressed_message_ids: frozenset[str]

    def entries_for(self, message_id: str) -> list[JSONDocument]:
        """All semantic entries owned by *message_id*, in transcript order."""

        return list(self.entries_by_message_id.get(message_id, ()))

    def cards_for(self, message_id: str) -> list[JSONDocument]:
        """Compatibility view containing only cards owned by *message_id*."""

        return list(self.cards_by_message_id.get(message_id, ()))

    def is_suppressed(self, message_id: str) -> bool:
        """Whether a pure protocol row is fully absorbed into a paired card."""

        return message_id in self.suppressed_message_ids


# The historical public name remains valid while the object now places the
# complete transcript rather than cards alone.
SemanticTranscriptPlacement = SemanticCardPlacement

_EMPTY_PLACEMENT = SemanticCardPlacement(
    entries_by_message_id={},
    session_entries=(),
    cards_by_message_id={},
    suppressed_message_ids=frozenset(),
)


def semantic_card_placement_for_messages(
    messages: Sequence[Mapping[str, object] | object],
    *,
    session_id: str,
    provider_family: str | None = None,
    lineage: LineageDescriptor | None = None,
) -> SemanticCardPlacement:
    """Place one shared semantic transcript into message/session envelopes.

    Paired result rows are suppressed only when the renderer emitted no
    independently meaningful entry for that result message. This preserves
    mixed protocol/context rows while preventing duplicate pure result output.
    """

    if not messages and lineage is None:
        return _EMPTY_PLACEMENT
    transcript = build_semantic_transcript(
        messages,
        session_id=session_id,
        provider_family=provider_family,
        lineage=lineage,
    )
    entries_by_message: dict[str, list[JSONDocument]] = {}
    cards_by_message: dict[str, list[JSONDocument]] = {}
    session_entries: list[JSONDocument] = []
    independent_message_ids: set[str] = set()
    paired_result_ids: set[str] = set()

    for entry in transcript.entries:
        document = entry.to_document()
        primary_id = entry.primary_message_id
        if primary_id is None:
            session_entries.append(document)
        else:
            entries_by_message.setdefault(primary_id, []).append(document)
            independent_message_ids.add(primary_id)

        if entry.prose is not None:
            independent_message_ids.add(entry.prose.message_id)
        elif entry.notice is not None:
            independent_message_ids.update(source.message_id for source in entry.notice.sources)
        elif entry.card is not None:
            card = entry.card
            if primary_id is not None:
                cards_by_message.setdefault(primary_id, []).append(card.to_document())
            result_id = card.source.result_message_id
            if result_id is not None and result_id != primary_id:
                paired_result_ids.add(result_id)

    suppressed = paired_result_ids - independent_message_ids
    if not entries_by_message and not session_entries and not cards_by_message and not suppressed:
        return _EMPTY_PLACEMENT
    return SemanticCardPlacement(
        entries_by_message_id={key: tuple(value) for key, value in entries_by_message.items()},
        session_entries=tuple(session_entries),
        cards_by_message_id={key: tuple(value) for key, value in cards_by_message.items()},
        suppressed_message_ids=frozenset(suppressed),
    )


__all__ = [
    "SemanticCardPlacement",
    "SemanticTranscriptPlacement",
    "semantic_card_placement_for_messages",
]
